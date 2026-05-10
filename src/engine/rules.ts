import { POINTS, Position, Side } from "./position";

export type WinKind = "single" | "gammon" | "backgammon";

export interface GameResult {
  winner: Side;
  kind: WinKind;
  // Base points (1/2/3). Cube multiplier applied by match logic.
  basePoints: number;
}

// Detect win from "us" perspective in `p`. Returns null if game is ongoing.
export function checkWin(p: Position): GameResult | null {
  if (p.offUs >= 15) {
    let kind: WinKind = "single";
    if (p.offThem === 0) {
      let bg = p.barThem > 0;
      if (!bg) {
        for (let i = 0; i < 6; i++) {
          if (p.points[i] < 0) {
            bg = true;
            break;
          }
        }
      }
      kind = bg ? "backgammon" : "gammon";
    }
    return {
      winner: p.turn,
      kind,
      basePoints: kind === "single" ? 1 : kind === "gammon" ? 2 : 3,
    };
  }
  if (p.offThem >= 15) {
    let kind: WinKind = "single";
    if (p.offUs === 0) {
      let bg = p.barUs > 0;
      if (!bg) {
        for (let i = 18; i < POINTS; i++) {
          if (p.points[i] > 0) {
            bg = true;
            break;
          }
        }
      }
      kind = bg ? "backgammon" : "gammon";
    }
    return {
      winner: (1 - p.turn) as Side,
      kind,
      basePoints: kind === "single" ? 1 : kind === "gammon" ? 2 : 3,
    };
  }
  return null;
}

// Roll one die, 1..6.
export function rollDie(rng: () => number = Math.random): number {
  return 1 + Math.floor(rng() * 6);
}

// Roll a pair, returned as a [d1, d2] tuple.
export function rollDice(rng: () => number = Math.random): [number, number] {
  return [rollDie(rng), rollDie(rng)];
}

// Roll an opening pair where the two dice differ (re-rolling doubles), used to
// determine who plays first.
export function rollOpening(rng: () => number = Math.random): [number, number] {
  let d1 = rollDie(rng);
  let d2 = rollDie(rng);
  while (d1 === d2) {
    d1 = rollDie(rng);
    d2 = rollDie(rng);
  }
  return [d1, d2];
}
