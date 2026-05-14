// Cube decision via the dead-cube Janowski recursion against a Match
// Equity Table. Two entry points:
//
//   decideCubeAction(p, side, outcomes, met)
//     Called before `side` rolls. Returns one of:
//       "no_double"  — don't double; play out the game at the current cube
//       "double_take" — double the cube; opponent should rationally take
//       "double_drop" — double the cube; opponent should rationally drop
//
//   decideTakeDrop(p, side, outcomes, met)
//     Called when the opponent just doubled and `side` must respond.
//     Returns "take" or "drop".
//
// Both functions take outcomes in the decision-maker's perspective —
// pWin/pGammonWin/pLoss/pGammonLoss as `side` sees the board.
//
// The cube is "dead" (i.e. the formulas degrade to no in-game cube
// movement) when matchLength <= 1 or p.crawford. In that case we always
// return "no_double" / default "take".

import { Position, Side } from "../engine/position";
import { canDouble } from "../engine/cube";
import { Met, metEntry } from "./met";

export type CubeAction = "no_double" | "double_take" | "double_drop";
export type TakeAction = "take" | "drop";

export interface Outcomes4 {
  pWin: number;
  pGammonWin: number; // gammon-or-better win prob, nested in pWin
  pLoss: number;
  pGammonLoss: number; // nested in pLoss
}

// Expected MWC for `side` after playing the current game out at cube
// value V (dead-cube: no further doubles assumed within this game).
function mwcAfterGame(
  met: Met,
  awayUs: number,
  awayThem: number,
  V: number,
  o: Outcomes4,
): number {
  const pSw = Math.max(0, o.pWin - o.pGammonWin); // single (non-gammon) win
  const pGw = Math.max(0, o.pGammonWin);
  const pSl = Math.max(0, o.pLoss - o.pGammonLoss);
  const pGl = Math.max(0, o.pGammonLoss);
  return (
    pSw * metEntry(met, awayUs - V, awayThem) +
    pGw * metEntry(met, awayUs - 2 * V, awayThem) +
    pSl * metEntry(met, awayUs, awayThem - V) +
    pGl * metEntry(met, awayUs, awayThem - 2 * V)
  );
}

export function decideCubeAction(
  p: Position,
  side: Side,
  o: Outcomes4,
  met: Met,
): CubeAction {
  if (p.matchLength <= 1) return "no_double";
  if (p.crawford) return "no_double";
  if (!canDouble(p, side)) return "no_double";

  const awayUs = p.matchLength - p.score[side];
  const awayThem = p.matchLength - p.score[1 - side];
  const V = p.cube.value;

  const mwcNoDouble = mwcAfterGame(met, awayUs, awayThem, V, o);
  const mwcTake = mwcAfterGame(met, awayUs, awayThem, 2 * V, o);
  const mwcDrop = metEntry(met, awayUs - V, awayThem);

  // Opponent picks the response that minimizes our MWC.
  const mwcDoubled = Math.min(mwcTake, mwcDrop);

  // Live-cube "waiting value" — Janowski-style. In live-cube play, holding
  // the cube has option value: you can double later at a higher equity
  // threshold. Dead-cube comparison (mwcDoubled > mwcNoDouble) is too
  // generous and produces a cube spiral. We require the doubling MWC to
  // beat no-doubling by a margin proportional to cube efficiency τ and to
  // the room left in the cube (no margin when doubling is forced by score,
  // larger margin when there's plenty of game left).
  const cubeRoom = Math.max(0, awayUs - V) / Math.max(1, met.matches);
  const liveCubeMargin = met.cube_efficiency * 0.06 * cubeRoom;
  if (mwcDoubled <= mwcNoDouble + liveCubeMargin) return "no_double";
  // Opp drops if drop is better for them (worse for us).
  return mwcDrop <= mwcTake ? "double_drop" : "double_take";
}

export function decideTakeDrop(
  p: Position,
  side: Side,
  o: Outcomes4,
  met: Met,
): TakeAction {
  if (p.matchLength <= 1) return "take";
  if (p.crawford) return "take"; // shouldn't be offered in Crawford, but be safe

  const awayUs = p.matchLength - p.score[side];
  const awayThem = p.matchLength - p.score[1 - side];
  const V = p.cube.value;

  // Take: cube goes to 2V, owned by us; play the game out.
  const mwcTake = mwcAfterGame(met, awayUs, awayThem, 2 * V, o);
  // Drop: opponent gains V points; their away decreases by V.
  const mwcDrop = metEntry(met, awayUs, awayThem - V);

  return mwcTake >= mwcDrop ? "take" : "drop";
}
