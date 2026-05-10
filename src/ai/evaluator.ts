import { Position } from "../engine/position";

// Equity is in the [-1, 1] range under cubeless single-game scoring:
//   +1 = we are certain to win, -1 = certain to lose.
// Gammon/backgammon contributions are folded into the magnitude where relevant.
export type Equity = number;

export interface Evaluator {
  // Pure board evaluation from "us" (player on roll, ignoring whose turn it is)
  // perspective. Higher is better for "us".
  evaluate(p: Position): Equity;
  readonly name: string;
}
