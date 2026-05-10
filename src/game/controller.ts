import {
  CHECKERS_PER_SIDE,
  Position,
  Side,
  mirror,
  startingPosition,
} from "../engine/position";
import { Play, SubMove, applyPlay, generatePlays } from "../engine/moves";
import { GameResult, checkWin, rollDice, rollOpening } from "../engine/rules";
import { AIClient, MoveAnalysis } from "../ai/api";
import { Difficulty } from "../ai/levels";
import { saveGame, saveSettings } from "./persistence";

export type PlayerKind = "human" | "cpu";

export interface GameSettings {
  matchLength: number;
  cubeEnabled: boolean;
  whitePlayer: PlayerKind;
  blackPlayer: PlayerKind;
  whiteName: string;
  blackName: string;
  cpuDifficulty: Difficulty;
  tutorEnabled: boolean;
  hidePassAndPlay: boolean;
}

export interface TutorEntry {
  side: Side;
  // Equity loss in mEMG (millipoints per game). 0 = perfect play.
  equityLoss: number;
  classification: "good" | "doubtful" | "error" | "blunder";
  bestPlay: Play;
  yourPlay: Play;
  // Optional human-readable note
  note?: string;
}

export type Phase =
  | { kind: "menu" }
  | { kind: "opening"; whiteDie: number; blackDie: number; firstSide: Side }
  | { kind: "pass-overlay"; forSide: Side }
  | { kind: "roll" }
  | { kind: "play"; legalPlays: Play[] }
  | { kind: "cpu-thinking" }
  | { kind: "won"; result: GameResult }
  | { kind: "match-won"; winner: Side };

export interface State {
  position: Position; // canonical: always in "us" (player on roll) perspective
  phase: Phase;
  pendingPlay: Play; // sub-moves chosen so far this turn (us-coords)
  selectedFrom: number | null;
  settings: GameSettings;
  // Match-level state (also embedded in position.score, but kept here for convenience)
  whiteScore: number;
  blackScore: number;
  gameNumber: number;
  // Tutor (Phase 4)
  tutor: {
    history: TutorEntry[];
    lastEntry?: TutorEntry;
  };
}

function emptyState(settings: GameSettings): State {
  return {
    position: startingPosition({ matchLength: settings.matchLength }),
    phase: { kind: "menu" },
    pendingPlay: [],
    selectedFrom: null,
    settings,
    whiteScore: 0,
    blackScore: 0,
    gameNumber: 1,
    tutor: { history: [] },
  };
}

// Compute the working position with `pendingPlay` applied (for rendering during human's turn).
export function workingPosition(state: State): Position {
  if (state.pendingPlay.length === 0) return state.position;
  return applyPlay(state.position, state.pendingPlay);
}

// Plays (from `legalPlays`) that still match `pendingPlay` as a prefix.
export function remainingPlays(legalPlays: Play[], pendingPlay: Play): Play[] {
  return legalPlays.filter((pl) => playStartsWith(pl, pendingPlay));
}

function playStartsWith(play: Play, prefix: Play): boolean {
  if (prefix.length > play.length) return false;
  for (let i = 0; i < prefix.length; i++) {
    const a = play[i];
    const b = prefix[i];
    if (a.from !== b.from || a.to !== b.to || a.die !== b.die) return false;
  }
  return true;
}

// Map of "next sub-move from" -> set of legal "to" destinations, given pendingPlay.
export function legalNextTargets(legalPlays: Play[], pendingPlay: Play): Map<number, Set<number>> {
  const m = new Map<number, Set<number>>();
  const remaining = remainingPlays(legalPlays, pendingPlay);
  for (const play of remaining) {
    if (play.length <= pendingPlay.length) continue;
    const sub = play[pendingPlay.length];
    let s = m.get(sub.from);
    if (!s) {
      s = new Set();
      m.set(sub.from, s);
    }
    s.add(sub.to);
  }
  return m;
}

export function isPendingComplete(legalPlays: Play[], pendingPlay: Play): boolean {
  const remaining = remainingPlays(legalPlays, pendingPlay);
  return remaining.length > 0 && remaining.every((p) => p.length === pendingPlay.length);
}

export function consumedDice(pendingPlay: Play): number[] {
  return pendingPlay.map((s) => s.die);
}

export function remainingDice(all: number[], pendingPlay: Play): number[] {
  const r = [...all];
  for (const sub of pendingPlay) {
    const i = r.indexOf(sub.die);
    if (i >= 0) r.splice(i, 1);
  }
  return r;
}

export type Listener = () => void;

export class GameController {
  state: State;
  ai: AIClient;
  private listeners = new Set<Listener>();

  constructor(settings: GameSettings, ai: AIClient) {
    this.ai = ai;
    this.state = emptyState(settings);
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(): void {
    for (const l of this.listeners) l();
  }

  setSettings(s: GameSettings): void {
    this.state.settings = s;
    void saveSettings(s);
    this.emit();
  }

  /** Start a brand-new match. */
  startNewMatch(): void {
    const s = this.state.settings;
    this.state = emptyState(s);
    this.startNewGame();
  }

  /** Start a new game within the current match (or a single game). */
  startNewGame(): void {
    this.state.position = startingPosition({ matchLength: this.state.settings.matchLength });
    this.state.position.score = [this.state.whiteScore, this.state.blackScore];
    this.state.pendingPlay = [];
    this.state.selectedFrom = null;
    // Opening roll
    const [d1, d2] = rollOpening();
    const firstSide: Side = d1 > d2 ? 0 : 1;
    if (firstSide === 1) this.state.position = mirror(this.state.position);
    // Use both dice as the first roll
    this.state.position.dice = [d1, d2];
    this.state.position.turn = firstSide;
    this.state.phase = {
      kind: "opening",
      whiteDie: d1,
      blackDie: d2,
      firstSide,
    };
    this.emit();
    // Pause briefly to show the opening dice, then start the first turn.
    setTimeout(() => this.afterRoll(), 1300);
  }

  /** Player on roll rolls dice (called for non-opening turns). */
  rollDice(): void {
    if (this.state.phase.kind !== "roll") return;
    const [d1, d2] = rollDice();
    this.state.position.dice = [d1, d2];
    this.afterRoll();
  }

  /** Resolve the current dice into legal plays, then either pick (CPU) or wait (human). */
  private afterRoll(): void {
    const dice = this.state.position.dice;
    if (!dice) return;
    const [d1, d2] = dice;
    const legalPlays = generatePlays(this.state.position, d1, d2);

    const onRoll = this.state.position.turn;
    const isHuman = onRoll === 0 ? this.state.settings.whitePlayer === "human" : this.state.settings.blackPlayer === "human";

    if (legalPlays.length === 1 && legalPlays[0].length === 0) {
      // Forfeit
      this.state.phase = { kind: "play", legalPlays };
      this.state.pendingPlay = [];
      this.emit();
      setTimeout(() => this.commitPlay(), 1000);
      return;
    }

    if (isHuman) {
      this.state.phase = { kind: "play", legalPlays };
      this.state.pendingPlay = [];
      this.state.selectedFrom = null;
      this.emit();
    } else {
      this.state.phase = { kind: "cpu-thinking" };
      this.state.pendingPlay = [];
      this.emit();
      void this.ai
        .pickMove(this.state.position, legalPlays, this.state.settings.cpuDifficulty)
        .then((play) => {
          if (this.state.phase.kind !== "cpu-thinking") return;
          this.state.pendingPlay = play;
          // Brief pause so the user can see the moves
          setTimeout(() => this.commitPlay(), 250);
        });
    }
  }

  /** User clicks a source point (or BAR=24). */
  pickFrom(point: number): void {
    if (this.state.phase.kind !== "play") return;
    const targets = legalNextTargets(this.state.phase.legalPlays, this.state.pendingPlay);
    if (!targets.has(point)) {
      // If user clicks an empty point or own checker that isn't a legal source, ignore.
      this.state.selectedFrom = null;
      this.emit();
      return;
    }
    this.state.selectedFrom = point;
    this.emit();
  }

  /** User clicks a destination point (or OFF=-1). */
  pickTo(point: number): void {
    if (this.state.phase.kind !== "play") return;
    const from = this.state.selectedFrom;
    if (from === null) {
      // Allow clicking destination directly only if there's a single legal source for that target
      const targets = legalNextTargets(this.state.phase.legalPlays, this.state.pendingPlay);
      const sources: number[] = [];
      for (const [src, dests] of targets) if (dests.has(point)) sources.push(src);
      if (sources.length === 1) {
        this.state.selectedFrom = sources[0];
      } else {
        return;
      }
    }
    const sub = this.findSub(this.state.selectedFrom!, point);
    if (!sub) {
      this.state.selectedFrom = null;
      this.emit();
      return;
    }
    this.state.pendingPlay.push(sub);
    this.state.selectedFrom = null;
    // Auto-commit when complete
    if (isPendingComplete(this.state.phase.legalPlays, this.state.pendingPlay)) {
      this.commitPlay();
      return;
    }
    this.emit();
  }

  private findSub(from: number, to: number): SubMove | null {
    if (this.state.phase.kind !== "play") return null;
    const remaining = remainingPlays(this.state.phase.legalPlays, this.state.pendingPlay);
    for (const play of remaining) {
      if (play.length <= this.state.pendingPlay.length) continue;
      const sub = play[this.state.pendingPlay.length];
      if (sub.from === from && sub.to === to) return sub;
    }
    return null;
  }

  undo(): void {
    if (this.state.phase.kind !== "play") return;
    if (this.state.pendingPlay.length === 0) return;
    this.state.pendingPlay.pop();
    this.state.selectedFrom = null;
    this.emit();
  }

  /** Commit pendingPlay as the turn's final play and pass to next side. */
  commitPlay(): void {
    let pos = applyPlay(this.state.position, this.state.pendingPlay);

    // Tutor analysis
    if (this.state.phase.kind === "play" && this.state.settings.tutorEnabled) {
      const onRoll = this.state.position.turn;
      const isHuman = onRoll === 0 ? this.state.settings.whitePlayer === "human" : this.state.settings.blackPlayer === "human";
      if (isHuman) {
        // Fire-and-forget (tutor result populated later)
        const yourPlay = this.state.pendingPlay;
        const legalPlays = this.state.phase.legalPlays;
        const startPos = this.state.position;
        void this.ai
          .analyze(startPos, legalPlays, this.state.settings.cpuDifficulty)
          .then((analysis) => this.recordTutorEntry(onRoll, yourPlay, legalPlays, analysis))
          .catch(() => {});
      }
    }

    pos.dice = null;

    // Did this turn end the game?
    const result = checkWin(pos);
    if (result) {
      const stake = result.basePoints * pos.cube.value;
      if (result.winner === 0) this.state.whiteScore += stake;
      else this.state.blackScore += stake;
      this.state.position = pos;
      this.state.pendingPlay = [];
      const matchLen = this.state.settings.matchLength;
      const matchOver = matchLen > 1 && (this.state.whiteScore >= matchLen || this.state.blackScore >= matchLen);
      this.state.phase = matchOver
        ? { kind: "match-won", winner: this.state.whiteScore > this.state.blackScore ? 0 : 1 }
        : { kind: "won", result };
      void saveGame(null); // clear saved game on game end
      this.emit();
      return;
    }

    // Pass turn
    pos = mirror(pos);
    this.state.position = pos;
    this.state.pendingPlay = [];
    this.state.selectedFrom = null;

    const nextSide = pos.turn;
    const nextIsHuman = nextSide === 0 ? this.state.settings.whitePlayer === "human" : this.state.settings.blackPlayer === "human";
    const bothHuman = this.state.settings.whitePlayer === "human" && this.state.settings.blackPlayer === "human";

    if (bothHuman && this.state.settings.hidePassAndPlay) {
      this.state.phase = { kind: "pass-overlay", forSide: nextSide };
    } else {
      this.state.phase = { kind: "roll" };
    }
    this.emit();

    if (!nextIsHuman) {
      // Auto-roll & play for CPU
      setTimeout(() => this.rollDice(), 350);
    }
  }

  private recordTutorEntry(side: Side, yourPlay: Play, legalPlays: Play[], analysis: MoveAnalysis): void {
    const yourIdx = legalPlays.findIndex((p) => playEqual(p, yourPlay));
    const yourEquity = yourIdx >= 0 ? analysis.equities[yourIdx] : analysis.bestEquity;
    const eqLoss = analysis.bestEquity - yourEquity;
    const classification = classifyEquityLoss(eqLoss);
    const entry: TutorEntry = {
      side,
      equityLoss: eqLoss,
      classification,
      bestPlay: analysis.bestPlay,
      yourPlay,
    };
    this.state.tutor.history.push(entry);
    this.state.tutor.lastEntry = entry;
    this.emit();
  }

  /** User taps the pass-and-play overlay to continue. */
  ackPassOverlay(): void {
    if (this.state.phase.kind !== "pass-overlay") return;
    this.state.phase = { kind: "roll" };
    this.emit();
  }

  returnToMenu(): void {
    this.state.phase = { kind: "menu" };
    this.emit();
  }

  shouldFlipDisplay(): boolean {
    // We always store position in "us" (player-on-roll) perspective.
    // In pass-and-play (both human), don't flip — each player sees themselves at the bottom.
    // In human vs CPU, fix the human's perspective: flip when CPU is on roll.
    const s = this.state.settings;
    if (s.whitePlayer === "human" && s.blackPlayer === "human") return false;
    if (s.whitePlayer === "cpu" && s.blackPlayer === "cpu") return false;
    const humanSide: Side = s.whitePlayer === "human" ? 0 : 1;
    return this.state.position.turn !== humanSide;
  }
}

function playEqual(a: Play, b: Play): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].from !== b[i].from || a[i].to !== b[i].to || a[i].die !== b[i].die) return false;
  }
  return true;
}

function classifyEquityLoss(eq: number): TutorEntry["classification"] {
  if (eq < 0.02) return "good";
  if (eq < 0.04) return "doubtful";
  if (eq < 0.08) return "error";
  return "blunder";
}

// Sanity helper used by tests / debugging — verify total checker count.
export function checkerInvariant(p: Position): boolean {
  let us = p.barUs + p.offUs;
  let them = p.barThem + p.offThem;
  for (let i = 0; i < p.points.length; i++) {
    if (p.points[i] > 0) us += p.points[i];
    else if (p.points[i] < 0) them += -p.points[i];
  }
  return us === CHECKERS_PER_SIDE && them === CHECKERS_PER_SIDE;
}
