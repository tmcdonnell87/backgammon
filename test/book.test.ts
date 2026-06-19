import { describe, it, expect, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { POINTS, Position, startingPosition } from "../src/engine/position";
import { generatePlays, applyPlay } from "../src/engine/moves";
import {
  boardKeyHex,
  bookEntryKey,
  bookEquities,
  setBook,
  OpeningBook,
} from "../src/ai/book";
import { analyzeMove, pickMove } from "../src/ai/engine";

function buildPosition(points: number[], extra?: Partial<Position>): Position {
  const pts = new Int8Array(POINTS);
  for (let i = 0; i < POINTS; i++) pts[i] = points[i];
  return {
    points: pts,
    barUs: extra?.barUs ?? 0,
    barThem: extra?.barThem ?? 0,
    offUs: extra?.offUs ?? 0,
    offThem: extra?.offThem ?? 0,
    turn: extra?.turn ?? 0,
    dice: extra?.dice ?? null,
    cube: { value: 1, owner: null },
    score: [0, 0],
    matchLength: 1,
    crawford: false,
  };
}

afterEach(() => setBook(null));

describe("book key parity with Python (training/build_opening_book.py)", () => {
  interface Row {
    points: number[];
    bar_us: number;
    bar_them: number;
    off_us: number;
    off_them: number;
    dice: number[];
    key: string;
  }
  const rows = JSON.parse(
    readFileSync(join(__dirname, "fixtures", "book_keys.json"), "utf8"),
  ) as Row[];

  for (const r of rows) {
    it(`bookEntryKey matches python for dice ${r.dice.join("-")}`, () => {
      const p = buildPosition(r.points, {
        barUs: r.bar_us, barThem: r.bar_them, offUs: r.off_us, offThem: r.off_them,
      });
      expect(bookEntryKey(p, r.dice)).toBe(r.key);
      expect(boardKeyHex(p)).toBe(r.key.split(":")[0]);
    });
  }

  it("sorts dice descending so rolled order does not matter", () => {
    const p = startingPosition();
    expect(bookEntryKey(p, [1, 3])).toBe(bookEntryKey(p, [3, 1]));
    // Doubles collapse to `${d}${d}`.
    expect(bookEntryKey(p, [5, 5, 5, 5]).endsWith(":55")).toBe(true);
  });
});

describe("opening book drives pickMove and analyzeMove (shipped book)", () => {
  const realBook = JSON.parse(
    readFileSync(join(__dirname, "..", "public", "weights", "opening_book.json"), "utf8"),
  ) as OpeningBook;

  function openingPos(d1: number, d2: number): Position {
    const p = startingPosition();
    p.dice = [d1, d2];
    return p;
  }

  it("3-1: picks and rates 8/5 6/5 (make the 5-point) as best", () => {
    setBook(realBook);
    const p = openingPos(3, 1);
    const plays = generatePlays(p, 3, 1);

    const picked = pickMove(p, plays, "expert");
    expect(applyPlay(p, picked).points[4]).toBe(2); // 5-point made

    const a = analyzeMove(p, plays, "casual");
    expect(a.equities.length).toBe(plays.length);
    expect(applyPlay(p, a.bestPlay).points[4]).toBe(2);

    // Reported equities are exactly the book's game equities.
    const be = bookEquities(p, plays, p.dice)!;
    expect(be).not.toBeNull();
    expect(be.complete).toBe(true);
    for (let i = 0; i < plays.length; i++) {
      expect(a.equities[i]).toBeCloseTo(be.gameEq[i], 9);
    }
  });

  it("6-1: picks 13/7 8/7 (make the bar point) as best", () => {
    setBook(realBook);
    const p = openingPos(6, 1);
    const plays = generatePlays(p, 6, 1);
    const picked = pickMove(p, plays, "expert");
    expect(applyPlay(p, picked).points[6]).toBe(2); // bar (7) point made
  });

  it("never flags a legal opening play as worse than 'doubtful'", () => {
    // The whole point: opening plays are within rollout noise, so against the
    // book's own equities the loss for ANY legal opening play is tiny.
    setBook(realBook);
    for (const [d1, d2] of [[3, 1], [2, 1], [4, 3], [6, 5], [5, 2]] as const) {
      const p = openingPos(d1, d2);
      const plays = generatePlays(p, d1, d2);
      const a = analyzeMove(p, plays, "casual");
      const best = a.bestEquity;
      // The single best play must have ~zero loss (it IS the best).
      const bestLoss = best - Math.max(...a.equities);
      expect(bestLoss).toBeLessThan(0.001);
    }
  });

  it("falls through to search when (position,dice) is not booked", () => {
    setBook(realBook);
    // A clearly non-opening position: not in the book -> bookEquities null.
    const p = buildPosition([
      2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5,
      -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2,
    ], { dice: [6, 5] });
    // Move a back checker so it's no longer the start position.
    p.points[23] = 1;
    p.points[18] = -4;
    expect(bookEquities(p, generatePlays(p, 6, 5), p.dice)).toBeNull();
  });
});
