// AI engine: stateless function that picks a move and analyzes a position.
// Used both directly (main thread) and inside the Worker.

import { Play, applyPlay } from "../engine/moves";
import { Position, Side, hashPosition, mirror } from "../engine/position";
import { checkWin } from "../engine/rules";
import { LEVELS, Difficulty } from "./levels";
import { rankPlays, pickWithNoise } from "./search";
import { heuristicEvaluator } from "./heuristic";
import { Evaluator } from "./evaluator";
import { AnyNeuralEvaluator, OutcomeProbs } from "./neural";
import { Met } from "./met";
import {
  CubeAction,
  TakeAction,
  decideCubeAction,
  decideTakeDrop,
} from "./cubeDecision";

let neuralEv: AnyNeuralEvaluator | null = null;
export function setNeuralEvaluator(ev: AnyNeuralEvaluator | null) {
  neuralEv = ev;
}
export function hasNeural(): boolean {
  return neuralEv !== null;
}

// Singleton MET for cube decisions. Loaded once at startup (see worker.ts
// and main.ts) and shared. Cube decisions return "no_double" / "take" if
// the MET hasn't been loaded yet.
let cubeMet: Met | null = null;
export function setMet(m: Met | null) {
  cubeMet = m;
}
export function hasMet(): boolean {
  return cubeMet !== null;
}

export function decideCube(p: Position, side: Side): CubeAction {
  if (!cubeMet) return "no_double";
  // Use the analysis-tier evaluator's outcome probabilities so cube
  // decisions align with what the tutor would see for this position.
  const ev = evaluatorFor(LEVELS.expert.analysisEvaluator);
  return decideCubeAction(p, side, evaluatorOutcomes(ev, p), cubeMet);
}

export function decideTake(p: Position, side: Side): TakeAction {
  if (!cubeMet) return "take";
  const ev = evaluatorFor(LEVELS.expert.analysisEvaluator);
  return decideTakeDrop(p, side, evaluatorOutcomes(ev, p), cubeMet);
}

function evaluatorFor(name: string): Evaluator {
  switch (name) {
    case "neural":
      return neuralEv ?? heuristicEvaluator;
    case "pubeval":
    case "heuristic":
    default:
      return heuristicEvaluator;
  }
}

export function pickMove(p: Position, legalPlays: Play[], difficulty: Difficulty): Play {
  if (legalPlays.length === 0) return [];
  if (legalPlays.length === 1) return legalPlays[0];
  const cfg = LEVELS[difficulty];
  const ev = evaluatorFor(cfg.evaluator);
  const ranked = rankPlays(p, legalPlays, ev, cfg.plies);
  return pickWithNoise(ranked, cfg.noise);
}

export interface AnalysisResult {
  bestPlay: Play;
  bestEquity: number;
  equities: number[]; // one per legalPlays entry, matching input order
}

// Extract outcome probabilities from any Evaluator. For evaluators with a
// native evaluateOutcomes (the 4-output neural net), use it directly. For
// scalar-only evaluators (heuristic, 1-output neural), reconstruct as a no-
// gammon outcome distribution from the scalar equity. The scalar is assumed
// to be in [-1, +1] and represents 2 * p_win - 1.
function evaluatorOutcomes(ev: Evaluator, p: Position): OutcomeProbs {
  const withOutcomes = ev as Evaluator & { evaluateOutcomes?: (p: Position) => OutcomeProbs };
  if (typeof withOutcomes.evaluateOutcomes === "function") {
    return withOutcomes.evaluateOutcomes(p);
  }
  const eq = Math.max(-1, Math.min(1, ev.evaluate(p)));
  const pWin = (eq + 1) / 2;
  return { pWin, pGammonWin: 0, pLoss: 1 - pWin, pGammonLoss: 0 };
}

// Signed game-win probability from the on-roll player's POV after applying
// `play`, range [-1, +1]. This is pWin - pLoss — a faithful "how likely am I
// to win this game" measure, independent of match score AND independent of
// gammon premium. A 75% gammon-win position reads +0.5 (75% chance of any
// win), NOT +1.0 (clamped expected points), so the bar tracks win likelihood
// rather than expected points scored.
function gameEquityForPostMove(
  pre: Position,
  play: Play,
  ev: Evaluator,
): number {
  const after = applyPlay(pre, play);
  const term = checkWin(after);
  if (term !== null) {
    // Terminal: we either won (+1) or lost (-1). basePoints (single/gammon/
    // backgammon) affects the match score, not the bar — the bar is in
    // probability-of-winning units.
    return term.winner === pre.turn ? 1 : -1;
  }
  // Non-terminal: evaluator from opponent's POV (mirror flips us<->them);
  // pWin from us POV is the mirror's pLoss.
  const oppOutcomes = evaluatorOutcomes(ev, mirror(after));
  const pWin = oppOutcomes.pLoss;
  const pLoss = oppOutcomes.pWin;
  return pWin - pLoss;
}

export function analyzeMove(
  p: Position,
  legalPlays: Play[],
  difficulty: Difficulty,
  awayUs?: number,
  awayThem?: number,
): AnalysisResult {
  if (legalPlays.length === 0) return { bestPlay: [], bestEquity: 0, equities: [] };

  const cfg = LEVELS[difficulty];
  const ev = evaluatorFor(cfg.analysisEvaluator);
  // Score every play (no noise) at analysisPlies depth. rankPlays dedupes by
  // final internally, so `scored` has one entry per unique final. We map
  // equities back to every input play via the final-position hash, so two
  // orderings reaching the same final share an equity (which is correct —
  // they ARE equivalent outcomes). The scalar score is what determines the
  // play ranking (and stays compatible with the existing 2-ply analysis).
  const scored = rankPlays(p, legalPlays, ev, cfg.analysisPlies);
  const eqByFinal = new Map<string, number>();
  for (const s of scored) {
    const h = hashPosition(applyPlay(p, s.play));
    eqByFinal.set(h, s.equity);
  }
  const scalarEquities = legalPlays.map((pl) => {
    const h = hashPosition(applyPlay(p, pl));
    return eqByFinal.get(h) ?? -Infinity;
  });
  let bestI = 0;
  for (let i = 1; i < scalarEquities.length; i++) {
    if (scalarEquities[i] > scalarEquities[bestI]) bestI = i;
  }
  // Report cubeless game equity per play (clamped to [-1, +1]). The MET-aware
  // (match-equity) version was confusing in long matches: a 95%-won race
  // showed only +0.13 because the single game's contribution to the match was
  // small. Game equity matches user intuition — "how dominant is my
  // position right now" — independent of match score.
  void awayUs; void awayThem;  // reserved for future match-aware tweaks
  const gameByFinal = new Map<string, number>();
  for (const s of scored) {
    const h = hashPosition(applyPlay(p, s.play));
    if (!gameByFinal.has(h)) {
      gameByFinal.set(h, gameEquityForPostMove(p, s.play, ev));
    }
  }
  const gameEquities = legalPlays.map((pl) => {
    const h = hashPosition(applyPlay(p, pl));
    return gameByFinal.get(h) ?? -Infinity;
  });
  return {
    bestPlay: legalPlays[bestI],
    bestEquity: gameEquities[bestI],
    equities: gameEquities,
  };
}
