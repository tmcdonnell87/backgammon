// Main-thread client to the AI worker. The worker isn't created here yet —
// Phase 3 wires it up. For now, all calls run on the main thread but use the
// same interface, so swapping the implementation later is mechanical.

import { Play } from "../engine/moves";
import { Position } from "../engine/position";
import { Difficulty } from "./levels";

export interface MoveAnalysis {
  bestPlay: Play;
  bestEquity: number;
  // Equity for every legal play, indexed in the same order as the input.
  equities: number[];
}

export interface AIClient {
  pickMove(p: Position, legalPlays: Play[], difficulty: Difficulty): Promise<Play>;
  analyze(p: Position, legalPlays: Play[], difficulty: Difficulty): Promise<MoveAnalysis>;
}
