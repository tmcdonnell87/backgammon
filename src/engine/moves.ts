import {
  BAR,
  OFF,
  POINTS,
  Position,
  allHome,
  clonePosition,
} from "./position";

export interface SubMove {
  from: number; // 0..23 or BAR
  to: number; // 0..23 or OFF
  die: number; // 1..6
}

export type Play = SubMove[];

// Apply a single sub-move (assumed legal) to a position. Returns a new Position.
export function applySubMove(p: Position, sub: SubMove): Position {
  const np = clonePosition(p);
  if (sub.from === BAR) {
    np.barUs--;
  } else {
    np.points[sub.from] = (np.points[sub.from] - 1) as number;
  }
  if (sub.to === OFF) {
    np.offUs++;
  } else if (np.points[sub.to] === -1) {
    // Hit a blot
    np.points[sub.to] = 1;
    np.barThem++;
  } else {
    np.points[sub.to] = (np.points[sub.to] + 1) as number;
  }
  return np;
}

// Apply a full play to a position.
export function applyPlay(p: Position, play: Play): Position {
  let cur = p;
  for (const sub of play) cur = applySubMove(cur, sub);
  return cur;
}

function canBearOffFrom(p: Position, from: number, die: number): boolean {
  // from - die < 0; legal if die exactly fits OR overshoot from highest occupied home point
  if (die === from + 1) return true;
  if (die < from + 1) return false; // shouldn't happen at call site
  // overshoot: from must be the highest occupied point in our home
  for (let j = from + 1; j < 6; j++) if (p.points[j] > 0) return false;
  return true;
}

// All legal sub-moves available with a single die in the given position.
export function legalSubMoves(p: Position, die: number): SubMove[] {
  const moves: SubMove[] = [];
  if (p.barUs > 0) {
    const dest = BAR - die; // 18..23
    if (p.points[dest] >= -1) moves.push({ from: BAR, to: dest, die });
    return moves;
  }
  for (let from = 0; from < POINTS; from++) {
    if (p.points[from] <= 0) continue;
    const dest = from - die;
    if (dest >= 0) {
      if (p.points[dest] >= -1) moves.push({ from, to: dest, die });
    } else {
      // bear-off attempt
      if (allHome(p) && canBearOffFrom(p, from, die)) {
        moves.push({ from, to: OFF, die });
      }
    }
  }
  return moves;
}

interface Found {
  play: Play;
  pos: Position;
}

function explore(p: Position, remaining: number[], partial: Play, out: Found[]): void {
  if (remaining.length === 0) {
    out.push({ play: partial, pos: p });
    return;
  }
  const die = remaining[0];
  const rest = remaining.slice(1);
  const subs = legalSubMoves(p, die);
  if (subs.length === 0) {
    out.push({ play: partial, pos: p });
    return;
  }
  for (const sub of subs) {
    const np = applySubMove(p, sub);
    explore(np, rest, [...partial, sub], out);
  }
}

// Generate the set of legal full plays for the given dice in the given position.
// Enforces:
//   * use as many dice as possible
//   * if exactly one die can be used and dice differ, must use the larger if possible
//   * keeps every distinct full sub-move sequence — no dedup by final position.
//     This is essential for the UI: when two orderings reach the same final
//     but differ at ANY sub-move position, the user must be able to play
//     either one (e.g., doubles 5-5 with [19→14, 19→14, …] vs [19→14, 14→9, …]
//     reaching the same final — the user wants to tap 19 twice). Identical
//     sub-move sequences are still deduped (can't happen for non-doubles
//     since each ordering yields a unique first-die-value).
// Returns [[]] (a single empty play) if no dice can be used at all.
export function generatePlays(p: Position, d1: number, d2: number): Play[] {
  const orderings: number[][] = d1 === d2 ? [[d1, d1, d1, d1]] : [[d1, d2], [d2, d1]];
  const found: Found[] = [];
  for (const order of orderings) explore(p, order, [], found);

  let maxLen = 0;
  for (const f of found) if (f.play.length > maxLen) maxLen = f.play.length;

  if (maxLen === 0) return [[]];

  let candidates = found.filter((f) => f.play.length === maxLen);

  if (maxLen === 1 && d1 !== d2) {
    const larger = Math.max(d1, d2);
    const usingLarger = candidates.filter((c) => c.play[0].die === larger);
    if (usingLarger.length > 0) candidates = usingLarger;
  }

  // Dedup only by full sub-move sequence (identical plays). Two plays that
  // share the same final but reach it via different sub-move orderings are
  // both kept so the UI can let the user tap any legal continuation.
  const seen = new Map<string, Play>();
  for (const c of candidates) {
    const key = c.play.map((s) => `${s.from},${s.to},${s.die}`).join("|");
    if (!seen.has(key)) seen.set(key, c.play);
  }
  return [...seen.values()];
}
