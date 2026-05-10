// Position representation.
//
// All engine code operates from the perspective of the player on roll ("us").
// `points[i]` is signed: positive = our checkers at point i, negative = opponent's.
// Index 0 is our 1-point (about to bear off), index 23 is our 24-point (farthest).
// Movement direction: us 23 -> 0 -> off; opponent 0 -> 23 (so we see them as -23..-1).
// Bar entry for us: die `d` enters at index 24 - d (in opponent's home, indices 18..23).
// Bear off: legal once all our checkers are at indices 0..5 or borne off.

export const POINTS = 24;
export const BAR = 24;   // virtual "from" index for bar entry
export const OFF = -1;   // virtual "to" index for bear-off
export const CHECKERS_PER_SIDE = 15;

export type Side = 0 | 1; // absolute side: 0 = white, 1 = black

export interface Position {
  // length 24, signed; +N = N our checkers, -N = N opponent checkers
  points: Int8Array;
  barUs: number;
  barThem: number;
  offUs: number;
  offThem: number;
  // The absolute side currently on roll.
  turn: Side;
  // Dice on roll (after roll, before fully consumed). Length 2 or 4 (doubles), or null.
  dice: number[] | null;
  // Doubling cube state.
  cube: { value: number; owner: Side | null };
  // Match score (points won so far this match).
  score: [number, number];
  matchLength: number; // 1 for money/single game; >1 for matches
  crawford: boolean;
  postCrawford: boolean;
}

export function startingPosition(opts: { matchLength?: number } = {}): Position {
  const points = new Int8Array(POINTS);
  // Our checkers (perspective of player on roll, who is white at game start):
  //   24-point (idx 23): 2, 13-point (idx 12): 5, 8-point (idx 7): 3, 6-point (idx 5): 5
  points[23] = 2;
  points[12] = 5;
  points[7] = 3;
  points[5] = 5;
  // Opponent's checkers (mirrored): at our 1, 12, 17, 19 points.
  points[0] = -2;
  points[11] = -5;
  points[16] = -3;
  points[18] = -5;
  return {
    points,
    barUs: 0,
    barThem: 0,
    offUs: 0,
    offThem: 0,
    turn: 0,
    dice: null,
    cube: { value: 1, owner: null },
    score: [0, 0],
    matchLength: opts.matchLength ?? 1,
    crawford: false,
    postCrawford: false,
  };
}

export function clonePosition(p: Position): Position {
  return {
    points: new Int8Array(p.points),
    barUs: p.barUs,
    barThem: p.barThem,
    offUs: p.offUs,
    offThem: p.offThem,
    turn: p.turn,
    dice: p.dice ? [...p.dice] : null,
    cube: { ...p.cube },
    score: [p.score[0], p.score[1]],
    matchLength: p.matchLength,
    crawford: p.crawford,
    postCrawford: p.postCrawford,
  };
}

// Flip perspective: swap us/them, mirror points 0..23 -> 23..0.
// Used at turn boundary so the new player sees themselves as "us".
export function mirror(p: Position): Position {
  const points = new Int8Array(POINTS);
  for (let i = 0; i < POINTS; i++) points[i] = -p.points[POINTS - 1 - i] as number;
  return {
    points,
    barUs: p.barThem,
    barThem: p.barUs,
    offUs: p.offThem,
    offThem: p.offUs,
    turn: (1 - p.turn) as Side,
    dice: p.dice ? [...p.dice] : null,
    cube: { ...p.cube },
    score: [p.score[0], p.score[1]],
    matchLength: p.matchLength,
    crawford: p.crawford,
    postCrawford: p.postCrawford,
  };
}

// True if all our checkers are in our home board (indices 0..5) or borne off.
export function allHome(p: Position): boolean {
  if (p.barUs > 0) return false;
  for (let i = 6; i < POINTS; i++) if (p.points[i] > 0) return false;
  return true;
}

// Pip count for "us" — sum of (point-index + 1) over our checkers, plus 25 per checker on bar.
export function pipCount(p: Position): number {
  let pips = p.barUs * 25;
  for (let i = 0; i < POINTS; i++) {
    if (p.points[i] > 0) pips += p.points[i] * (i + 1);
  }
  return pips;
}

// Pip count for "them".
export function pipCountThem(p: Position): number {
  let pips = p.barThem * 25;
  for (let i = 0; i < POINTS; i++) {
    if (p.points[i] < 0) pips += -p.points[i] * (POINTS - i);
  }
  return pips;
}

// Stable string hash of the position (us perspective).
export function hashPosition(p: Position): string {
  // Pack: 24 signed bytes as offset chars + bar/off counts. Match-state ignored on purpose.
  let s = "";
  for (let i = 0; i < POINTS; i++) s += String.fromCharCode((p.points[i] + 16) & 0xff);
  s += String.fromCharCode(p.barUs, p.barThem, p.offUs, p.offThem);
  return s;
}

// Fast equality on board state only (ignores dice, cube, score).
export function boardEquals(a: Position, b: Position): boolean {
  if (a.barUs !== b.barUs || a.barThem !== b.barThem) return false;
  if (a.offUs !== b.offUs || a.offThem !== b.offThem) return false;
  for (let i = 0; i < POINTS; i++) if (a.points[i] !== b.points[i]) return false;
  return true;
}
