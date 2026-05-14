// Main-thread client to the AI worker. The worker isn't created here yet —
// Phase 3 wires it up. For now, all calls run on the main thread but use the
// same interface, so swapping the implementation later is mechanical.

import { Play } from "../engine/moves";
import { Position, Side } from "../engine/position";
import { Difficulty } from "./levels";
import { CubeAction, TakeAction } from "./cubeDecision";

export interface MoveAnalysis {
  bestPlay: Play;
  bestEquity: number;
  // Equity for every legal play, indexed in the same order as the input.
  equities: number[];
}

export interface AIClient {
  pickMove(p: Position, legalPlays: Play[], difficulty: Difficulty): Promise<Play>;
  // Analyze a set of legal plays. When awayUs/awayThem are supplied and the
  // worker has loaded a match equity table, the returned `equities` and
  // `bestEquity` are MET-converted match equities in [-1, +1] — matching the
  // values returned by `evaluate`. Without them, raw evaluator equities are
  // returned (range varies by evaluator).
  analyze(
    p: Position,
    legalPlays: Play[],
    difficulty: Difficulty,
    awayUs?: number,
    awayThem?: number,
  ): Promise<MoveAnalysis>;
  // Cube actions for match play. Both operate from `side`'s perspective; the
  // caller passes the position as the relevant decision-maker sees it
  // (typically `side === p.turn`, the player about to roll). Returns
  // "no_double" / "take" by default if no neural net is loaded (heuristic
  // has no gammon information; cube decisions stay conservative).
  decideCube(p: Position, side: Side): Promise<CubeAction>;
  decideTake(p: Position, side: Side): Promise<TakeAction>;
}
