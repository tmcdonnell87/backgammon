import {
  CHECKERS_PER_SIDE,
  OFF,
  Position,
  Side,
  mirror,
  startingPosition,
} from "../engine/position";
import { Play, SubMove, applyPlay, generatePlays } from "../engine/moves";
import { GameResult, checkWin, rollDice, rollOpening } from "../engine/rules";
import { applyDoubleAccepted, canDouble } from "../engine/cube";
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
  showPipCount: boolean;
  showEquity: boolean;
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
  // Pre-move position, all legal plays analyzed, and the corresponding
  // equities — needed so the analysis modal can paint the pre-move board
  // and render the full ranked list of alternatives.
  startPos: Position;
  legalPlays: Play[];
  equities: number[];
}

export type Phase =
  | { kind: "menu" }
  | { kind: "opening"; whiteDie: number; blackDie: number; firstSide: Side }
  | { kind: "roll" }
  | { kind: "play"; legalPlays: Play[] }
  | { kind: "cpu-thinking" }
  | { kind: "cube-decision"; doubler: Side }
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
  // True once the Crawford game has been played this match (so subsequent
  // games are post-Crawford and the cube is live again).
  crawfordPlayed: boolean;
  // Tutor (Phase 4)
  tutor: {
    history: TutorEntry[];
    lastEntry?: TutorEntry;
  };
  // Absolute White POV equity of the position RESULTING from the last commit.
  currentEquity: number | null;
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
    crawfordPlayed: false,
    tutor: { history: [] },
    currentEquity: null,
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
  // True while we're waiting on the user to dismiss an error/blunder tutor
  // modal. The next dice roll is held off until ackTutor() is called.
  private tutorAckPending = false;

  // Eager turn analysis: kicked off the moment dice are resolved. Bundles the
  // analyze() promise with the exact legalPlays array it was given so commit-time
  // and tutor lookups can match the committed play against the same row vector
  // without depending on phase state or rebuilding plays.
  private activeAnalysis: { promise: Promise<MoveAnalysis>; legalPlays: Play[] } | null = null;

  constructor(settings: GameSettings, ai: AIClient) {
    this.ai = ai;
    this.state = emptyState(settings);
  }

  /** Retrieve the eager analysis promise for the current turn. */
  public getActiveAnalysis(): Promise<MoveAnalysis> | null {
    return this.activeAnalysis?.promise ?? null;
  }

  private triggerEagerAnalysis(p: Position, legalPlays: Play[]): void {
    const matchLen = this.state.settings.matchLength;
    const usScore = p.turn === 0 ? this.state.whiteScore : this.state.blackScore;
    const themScore = p.turn === 0 ? this.state.blackScore : this.state.whiteScore;
    const promise = this.ai.analyze(
      p,
      legalPlays,
      this.state.settings.cpuDifficulty,
      matchLen - usScore,
      matchLen - themScore,
    );
    this.activeAnalysis = { promise, legalPlays };
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
    this.activeAnalysis = null;
    this.state.currentEquity = null;
    this.state.position = startingPosition({ matchLength: this.state.settings.matchLength });
    this.state.position.score = [this.state.whiteScore, this.state.blackScore];
    // Crawford rule: the game right AFTER one side first reaches
    // matchLength - 1 points is the Crawford game (no cube allowed). The
    // following games are post-Crawford (cube live again for the trailer).
    if (
      this.state.settings.matchLength > 1
      && !this.state.crawfordPlayed
      && Math.max(this.state.whiteScore, this.state.blackScore)
         === this.state.settings.matchLength - 1
    ) {
      this.state.position.crawford = true;
    } else {
      this.state.position.crawford = false;
    }
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
    // Normalize so the larger die is on the left. Single-tap auto-exec picks
    // the leftmost remaining die, so this makes the bigger die go first by
    // default — matching backgammon convention. The user can swap mid-turn.
    if (dice.length === 2 && dice[0] !== dice[1] && dice[0] < dice[1]) {
      this.state.position.dice = [dice[1], dice[0]];
    }
    const [d1, d2] = this.state.position.dice as [number, number];
    const legalPlays = generatePlays(this.state.position, d1, d2);
    this.triggerEagerAnalysis(this.state.position, legalPlays);

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
          // Emit the pending-play state so the renderer can animate each sub-move
          // before the commit-state arrives.
          this.emit();
          setTimeout(() => this.commitPlay(), 100);
        });
    }
  }

  /** User clicks a source point (or BAR=24). When `silent` is true, no emit
   * fires — useful when an immediate `pickTo` is about to emit anyway, so the
   * intermediate "source-selected, no destination yet" frame doesn't flash. */
  pickFrom(point: number, silent = false): void {
    if (this.state.phase.kind !== "play") return;
    const targets = legalNextTargets(this.state.phase.legalPlays, this.state.pendingPlay);
    if (!targets.has(point)) {
      this.state.selectedFrom = null;
      if (!silent) this.emit();
      return;
    }
    this.state.selectedFrom = point;
    if (!silent) this.emit();
  }

  clearSelection(): void {
    if (this.state.selectedFrom === null) return;
    this.state.selectedFrom = null;
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
    // Emit on every append so the renderer can animate this sub-move. The user
    // confirms the turn by tapping the dice (see commitPlay); no auto-commit
    // EXCEPT when this sub-move bears off the 15th checker — at that point
    // the game is decided, there's nothing to confirm, so auto-commit after a
    // short delay that lets the bear-off animation play out.
    this.emit();
    if (sub.to === OFF) {
      const after = applyPlay(this.state.position, this.state.pendingPlay);
      if (after.offUs >= CHECKERS_PER_SIDE) {
        setTimeout(() => this.commitPlay(), 250);
      }
    }
  }

  /** Swap the order of the two unused dice. No-op for doubles or if any die has been consumed. */
  swapDice(): void {
    if (this.state.phase.kind !== "play") return;
    if (this.state.pendingPlay.length !== 0) return;
    const d = this.state.position.dice;
    if (!d || d.length !== 2) return;
    if (d[0] === d[1]) return;
    this.state.position.dice = [d[1], d[0]];
    // Regenerate legal plays so the canonical "first sub-move" per source
    // reflects the new ordering (the dedup in generatePlays is order-sensitive).
    const legalPlays = generatePlays(this.state.position, d[1], d[0]);
    this.triggerEagerAnalysis(this.state.position, legalPlays);
    this.state.phase = { kind: "play", legalPlays };
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

  /** Apply a full play in one shot — used by the hint modal's "play this"
   * button. The renderer animates each sub-move (the snapshot queue sees a
   * length-0 → length-N pendingPlay extension and slides each in turn), then
   * we commit after the animations have time to play.
   *
   * Any partial pending play the user had built up is discarded — the hint
   * modal's backdrop is click-through so the user may have accidentally
   * tapped checkers while it was open; their intent in clicking "Play
   * selected" is unambiguous so we just take over from a clean slate. */
  playFullPlay(play: Play): void {
    if (this.state.phase.kind !== "play") return;
    // Verify the play is among the legal options before clobbering pending.
    const legal = this.state.phase.legalPlays.some(
      (lp) => lp.length === play.length && lp.every((s, i) => s.from === play[i].from && s.to === play[i].to && s.die === play[i].die),
    );
    if (!legal) return;
    this.state.pendingPlay = [];
    this.state.selectedFrom = null;
    this.state.pendingPlay = [...play];
    this.state.selectedFrom = null;
    this.emit();
    // Schedule commit after the animations have time to play.
    const ANIM_PER_SUB = 150;
    const delay = play.length * ANIM_PER_SUB + 80;
    setTimeout(() => this.commitPlay(), delay);
  }

  /** Commit pendingPlay as the turn's final play and pass to next side.
   *
   * Caller is responsible for ensuring the pending play is a legal complete play
   * (either max-length or [] for a forfeit). Callers from within the controller
   * (CPU AI, opening forfeit) construct legal plays. The UI calls this only when
   * the user confirms via a dice tap and `isPendingComplete` is true.
   */
  commitPlay(): void {
    if (this.state.phase.kind === "play") {
      if (!isPendingComplete(this.state.phase.legalPlays, this.state.pendingPlay)) return;
    }
    let pos = applyPlay(this.state.position, this.state.pendingPlay);

    const onRoll = this.state.position.turn;
    const committedPlay = [...this.state.pendingPlay];
    const active = this.activeAnalysis;

    // Capture tutor inputs BEFORE we mutate position/phase so the analysis
    // can run after we've already moved on visibly.
    let tutorCtx: { onRoll: Side; yourPlay: Play; legalPlays: Play[]; startPos: Position } | null = null;
    if (this.state.settings.tutorEnabled && active) {
      const isHuman = onRoll === 0 ? this.state.settings.whitePlayer === "human" : this.state.settings.blackPlayer === "human";
      if (isHuman) {
        tutorCtx = {
          onRoll,
          yourPlay: committedPlay,
          legalPlays: active.legalPlays,
          startPos: this.state.position,
        };
      }
    }

    // Update currentEquity from the eager analysis. Lookup uses the cached
    // legalPlays so it works for both human AND CPU commits — the prior
    // tutor-gated lookup left CPU commits showing analysis.bestEquity, which
    // overstates equity whenever the CPU plays a suboptimal move.
    if (active) {
      const legal = active.legalPlays;
      void active.promise.then((analysis) => {
        const idx = legal.findIndex(
          (lp) =>
            lp.length === committedPlay.length
            && lp.every(
              (s, i) =>
                s.from === committedPlay[i].from
                && s.to === committedPlay[i].to
                && s.die === committedPlay[i].die,
            ),
        );
        const equityUs = idx >= 0 ? analysis.equities[idx] : analysis.bestEquity;
        this.state.currentEquity = onRoll === 0 ? equityUs : -equityUs;
        this.emit();
      });
    }

    pos.dice = null;

    // Did this turn end the game?
    const result = checkWin(pos);
    if (result) {
      this.endGame(pos, result);
      return;
    }

    // Pass turn
    pos = mirror(pos);
    this.state.position = pos;
    this.state.pendingPlay = [];
    this.state.selectedFrom = null;

    const nextSide = pos.turn;
    const nextIsHuman = nextSide === 0 ? this.state.settings.whitePlayer === "human" : this.state.settings.blackPlayer === "human";
    const bothCpu = this.state.settings.whitePlayer === "cpu" && this.state.settings.blackPlayer === "cpu";

    // Cube decision: only fires in AI-vs-AI matches (no UI for human take/drop).
    // The next mover (pos.turn) may offer the cube before rolling.
    const cubeShouldFire =
      this.state.settings.cubeEnabled
      && bothCpu
      && this.state.settings.matchLength > 1
      && !this.state.position.crawford
      && canDouble(pos, pos.turn);
    if (cubeShouldFire) {
      this.state.phase = { kind: "cube-decision", doubler: pos.turn };
      this.emit();
      void this.runCubeFlow(pos.turn, tutorCtx);
      return;
    }

    this.state.phase = { kind: "roll" };
    this.emit();
    if (tutorCtx) {
      // Tutor analysis gates the next roll — for error/blunder we keep the
      // dice off the board until the user dismisses the modal via ackTutor().
      void this.runTutorAnalysis(tutorCtx, /* gateRoll */ true);
    } else {
      setTimeout(() => this.rollDice(), 100);
    }
    void nextIsHuman; // unused: auto-roll applies uniformly now
  }

  /** Finalize a game's end: apply stake, set phase, persist. */
  private endGame(pos: Position, result: GameResult): void {
    this.activeAnalysis = null;
    const stake = result.basePoints * pos.cube.value;
    if (result.winner === 0) this.state.whiteScore += stake;
    else this.state.blackScore += stake;
    // Crawford bookkeeping: if THIS game was the Crawford game, the next
    // games are post-Crawford and the cube is live again. Set this AFTER
    // applying the stake so startNewGame sees the up-to-date `played` flag.
    if (this.state.position.crawford) this.state.crawfordPlayed = true;
    this.state.position = pos;
    this.state.pendingPlay = [];
    const matchLen = this.state.settings.matchLength;
    const matchOver = matchLen > 1 && (this.state.whiteScore >= matchLen || this.state.blackScore >= matchLen);
    this.state.phase = matchOver
      ? { kind: "match-won", winner: this.state.whiteScore > this.state.blackScore ? 0 : 1 }
      : { kind: "won", result };
    void saveGame(null); // clear saved game on game end
    this.emit();
  }

  /** Drive an AI cube decision + (if doubled) the receiver's take/drop. */
  private async runCubeFlow(
    doubler: Side,
    tutorCtx: { onRoll: Side; yourPlay: Play; legalPlays: Play[]; startPos: Position } | null,
  ): Promise<void> {
    let action;
    try {
      action = await this.ai.decideCube(this.state.position, doubler);
    } catch {
      action = "no_double" as const;
    }
    // Bail if the phase has moved on (e.g. user returned to menu mid-flight).
    if (this.state.phase.kind !== "cube-decision") return;
    if (action === "no_double") {
      this.advanceToRoll(tutorCtx);
      return;
    }

    // Receiver decides. Mirror the position so the receiver's side is on
    // roll in the encoding — decideTake operates from `side`'s perspective.
    const receiver: Side = (1 - doubler) as Side;
    const receiverPos = mirror(this.state.position);
    let response;
    try {
      response = await this.ai.decideTake(receiverPos, receiver);
    } catch {
      response = "take" as const;
    }
    if (this.state.phase.kind !== "cube-decision") return;

    if (response === "drop") {
      // Doubler wins current cube value as points; game ends without rolling.
      const stakePoints = this.state.position.cube.value;
      const result: GameResult = {
        winner: doubler,
        kind: "single",
        basePoints: 1,
      };
      // Reuse endGame; pass a synthetic position with cube_value retained
      // so the standard `stake = basePoints * cube.value` math works out.
      this.endGame(this.state.position, result);
      // The standard endGame uses basePoints * cube.value = 1 * V = V, which
      // matches the drop reward. (no-op safety: stakePoints aligns.)
      void stakePoints;
      return;
    }

    // Take: cube doubles, ownership goes to receiver.
    this.state.position = applyDoubleAccepted(this.state.position, doubler);
    this.advanceToRoll(tutorCtx);
  }

  private advanceToRoll(
    tutorCtx: { onRoll: Side; yourPlay: Play; legalPlays: Play[]; startPos: Position } | null,
  ): void {
    this.state.phase = { kind: "roll" };
    this.emit();
    if (tutorCtx) {
      void this.runTutorAnalysis(tutorCtx, /* gateRoll */ true);
    } else {
      setTimeout(() => this.rollDice(), 100);
    }
  }

  /**
   * Run tutor analysis for a just-committed human play. If gateRoll is true,
   * an error/blunder result pauses the next dice roll until ackTutor() is
   * called (so the UI can show an analysis modal before the next roll).
   */
  private async runTutorAnalysis(
    ctx: { onRoll: Side; yourPlay: Play; legalPlays: Play[]; startPos: Position },
    gateRoll: boolean,
  ): Promise<void> {
    let blocked = false;
    try {
      // Use the eager analysis triggered at the start of the turn.
      if (!this.activeAnalysis) throw new Error("no active analysis");
      const analysis = await this.activeAnalysis.promise;
      const yourIdx = ctx.legalPlays.findIndex(
        (lp) => lp.length === ctx.yourPlay.length && lp.every((s, i) => s.from === ctx.yourPlay[i].from && s.to === ctx.yourPlay[i].to),
      );

      const yourEquity = yourIdx >= 0 ? analysis.equities[yourIdx] : analysis.bestEquity;
      const eqLoss = analysis.bestEquity - yourEquity;
      const classification = classifyEquityLoss(eqLoss);
      const entry: TutorEntry = {
        side: ctx.onRoll,
        equityLoss: eqLoss,
        classification,
        bestPlay: analysis.bestPlay,
        yourPlay: ctx.yourPlay,
        startPos: ctx.startPos,
        legalPlays: ctx.legalPlays,
        equities: analysis.equities,
      };
      this.state.tutor.history.push(entry);
      this.state.tutor.lastEntry = entry;
      blocked = gateRoll && (classification === "error" || classification === "blunder");
      if (blocked) this.tutorAckPending = true;
      this.emit();
    } catch (e) {
      // ignore tutor errors
    }
    if (gateRoll && !blocked) {
      setTimeout(() => this.rollDice(), 100);
    }
  }

  /**
   * Called by the UI when the user dismisses the error/blunder tutor modal.
   * Resumes the next dice roll that commitPlay deferred.
   */
  ackTutor(): void {
    if (!this.tutorAckPending) return;
    this.tutorAckPending = false;
    if (this.state.phase.kind === "roll") {
      setTimeout(() => this.rollDice(), 100);
    }
  }

  returnToMenu(): void {
    this.activeAnalysis = null;
    this.state.currentEquity = null;
    this.state.phase = { kind: "menu" };
    this.emit();
  }

  shouldFlipDisplay(): boolean {
    // Position is stored in "us" (player-on-roll) perspective. shouldFlipDisplay
    // mirrors the view so that:
    //  - vs-CPU: the human always sits at the bottom.
    //  - 2P: white always sits at the bottom (fixed board orientation).
    //  - both-CPU (debug only): no flip.
    const s = this.state.settings;
    if (s.whitePlayer === "cpu" && s.blackPlayer === "cpu") return false;
    if (s.whitePlayer === "human" && s.blackPlayer === "human") {
      return this.state.position.turn === 1;
    }
    const humanSide: Side = s.whitePlayer === "human" ? 0 : 1;
    return this.state.position.turn !== humanSide;
  }
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
