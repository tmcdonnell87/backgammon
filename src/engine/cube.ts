import { Position, Side, clonePosition } from "./position";

// Doubling cube state transitions and helpers.
// Cube is "centered" (owner === null) until the first double; then owned by the
// player who accepted the most recent double, who is the only player able to
// double next.

export function canDouble(p: Position, side: Side): boolean {
  if (p.matchLength > 1 && p.crawford) return false;
  if (p.cube.owner === null) return true; // centered cube; either player may double
  return p.cube.owner === side;
}

export function applyDoubleAccepted(p: Position, doubler: Side): Position {
  const np = clonePosition(p);
  np.cube = { value: p.cube.value * 2, owner: (1 - doubler) as Side };
  return np;
}

// When a double is dropped, the doubler wins the current cube value (pre-double)
// in points. This is handled by the match-state code; here we only signal.
export function pointsForDrop(p: Position): number {
  return p.cube.value;
}

// Maximum useful cube value given match score (no need to double past what would
// win the match in one shot — "match equity" considerations live in the AI).
export function maxUsefulCube(p: Position, side: Side): number {
  if (p.matchLength <= 1) return Infinity;
  const need = p.matchLength - p.score[side];
  let v = p.cube.value;
  while (v * 2 < need * 2) v *= 2;
  return v;
}
