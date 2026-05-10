// AI engine: stateless function that picks a move and analyzes a position.
// Used both directly (main thread) and inside the Worker.

import { Play } from "../engine/moves";
import { Position } from "../engine/position";
import { LEVELS, Difficulty } from "./levels";
import { rankPlays, pickWithNoise, score0ply } from "./search";
import { heuristicEvaluator } from "./heuristic";
import { Evaluator } from "./evaluator";

function evaluatorFor(name: string): Evaluator {
  // Until pubeval/neural weights are trained, all paths use the heuristic.
  // The ranking remains correct; quality comes from search depth + noise.
  switch (name) {
    case "heuristic":
    case "pubeval":
    case "neural":
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

export function analyzeMove(p: Position, legalPlays: Play[], difficulty: Difficulty): AnalysisResult {
  if (legalPlays.length === 0) return { bestPlay: [], bestEquity: 0, equities: [] };
  if (legalPlays.length === 1) {
    return { bestPlay: legalPlays[0], bestEquity: 0, equities: [0] };
  }
  const cfg = LEVELS[difficulty];
  const ev = evaluatorFor(cfg.analysisEvaluator);
  // Score every play (no noise) at analysisPlies depth
  const scored = cfg.analysisPlies === 2
    ? rankPlays(p, legalPlays, ev, 2)
    : score0ply(p, legalPlays, ev);
  // Map back to input order for tutor display
  const eqByHash = new Map<Play, number>();
  for (const s of scored) eqByHash.set(s.play, s.equity);
  const equities = legalPlays.map((pl) => eqByHash.get(pl) ?? -Infinity);
  let bestI = 0;
  for (let i = 1; i < equities.length; i++) {
    if (equities[i] > equities[bestI]) bestI = i;
  }
  return {
    bestPlay: legalPlays[bestI],
    bestEquity: equities[bestI],
    equities,
  };
}
