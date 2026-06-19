// Opening / reply book.
//
// Opening backgammon plays differ by only ~0.002–0.02 equity — inside the
// noise of any 0/2-ply evaluator — so a neural net routinely mis-ranks them
// and the tutor flags the rollout-best play as an "error". The book sidesteps
// that entirely: for a booked (position, dice) it returns gnubg-derived
// equities for EVERY legal play, so both the move-picker and the tutor use
// rollout-quality numbers and never flag a book-acceptable play.
//
// The book is generated offline by training/build_opening_book.py and shipped
// as public/weights/opening_book.json. Entries are keyed by a board hash plus
// the dice; each play is keyed by its resulting-position board hash so a legal
// play maps to its book equity regardless of sub-move ordering.

import { POINTS, Position } from "../engine/position";
import { Play, applyPlay } from "../engine/moves";

export interface BookPlay {
  k: string; // hex board hash of the resulting position
  g: number; // game equity (pWin - pLoss), the tutor's grading scale
  p?: number; // points equity (gammon-aware); used to pick "best". Falls back to g.
  m?: string; // human-readable move, debug only
}

export interface BookEntry {
  complete: boolean; // true ⇒ every legal final for this (position,dice) is present
  plays: BookPlay[];
}

export interface OpeningBook {
  version: number;
  scale: string; // "cubeless-game-equity"
  source?: string;
  entries: Record<string, BookEntry>;
}

let cached: OpeningBook | null = null;
let cachedFetch: Promise<OpeningBook | null> | null = null;

// Hex of the 28 board bytes: 24 signed points (offset by 16) + bar/off counts.
// Byte-identical to training/engine.py `board_hash(p).hex()` and to the
// `hexBoardHash` reference in test/neural.test.ts. Pinned by a parity fixture.
export function boardKeyHex(p: Position): string {
  let s = "";
  for (let i = 0; i < POINTS; i++) {
    s += ((p.points[i] + 16) & 0xff).toString(16).padStart(2, "0");
  }
  s += (p.barUs & 0xff).toString(16).padStart(2, "0");
  s += (p.barThem & 0xff).toString(16).padStart(2, "0");
  s += (p.offUs & 0xff).toString(16).padStart(2, "0");
  s += (p.offThem & 0xff).toString(16).padStart(2, "0");
  return s;
}

// Entry key: `${boardHash}:${hi}${lo}` with the dice sorted descending so the
// rolled order doesn't matter. Doubles (length-4 dice) collapse to `${d}${d}`.
export function bookEntryKey(p: Position, dice: number[]): string {
  const a = dice[0];
  const b = dice.length > 1 ? dice[1] : dice[0];
  const hi = Math.max(a, b);
  const lo = Math.min(a, b);
  return `${boardKeyHex(p)}:${hi}${lo}`;
}

export async function loadBook(url = "/weights/opening_book.json"): Promise<OpeningBook | null> {
  if (cached) return cached;
  if (cachedFetch) return cachedFetch;
  cachedFetch = (async () => {
    try {
      const res = await fetch(url);
      if (!res.ok) return null;
      const book = (await res.json()) as OpeningBook;
      if (!book || typeof book.entries !== "object") return null;
      cached = book;
      return book;
    } catch {
      return null;
    }
  })();
  return cachedFetch;
}

export function setBook(b: OpeningBook | null): void {
  cached = b;
}

export function getBook(): OpeningBook | null {
  return cached;
}

export interface BookLookup {
  gameEq: number[]; // per legal play, game equity (pWin - pLoss)
  pointsEq: number[]; // per legal play, points equity (gammon-aware)
  covered: boolean[]; // per legal play, whether the book had it
  complete: boolean; // entry is complete AND every legal play is covered
}

// Look up book equities for every legal play of a booked (position, dice).
// Returns null on a miss. When `complete` is true the caller can grade against
// these equities directly; otherwise it should fall back to search.
export function bookEquities(
  p: Position,
  legalPlays: Play[],
  dice: number[] | null,
): BookLookup | null {
  const book = cached;
  if (!book || !dice || dice.length === 0) return null;
  const entry = book.entries[bookEntryKey(p, dice)];
  if (!entry) return null;

  const byFinal = new Map<string, BookPlay>();
  for (const bp of entry.plays) byFinal.set(bp.k, bp);

  const gameEq: number[] = new Array(legalPlays.length);
  const pointsEq: number[] = new Array(legalPlays.length);
  const covered: boolean[] = new Array(legalPlays.length);
  let anyUncovered = false;
  for (let i = 0; i < legalPlays.length; i++) {
    const h = boardKeyHex(applyPlay(p, legalPlays[i]));
    const bp = byFinal.get(h);
    if (bp) {
      gameEq[i] = bp.g;
      pointsEq[i] = bp.p ?? bp.g;
      covered[i] = true;
    } else {
      gameEq[i] = -Infinity;
      pointsEq[i] = -Infinity;
      covered[i] = false;
      anyUncovered = true;
    }
  }
  return { gameEq, pointsEq, covered, complete: entry.complete && !anyUncovered };
}
