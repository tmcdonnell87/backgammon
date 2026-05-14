import { describe, it, expect } from "vitest";
import { BAR, OFF, Position, startingPosition } from "../src/engine/position";
import { applyPlay, generatePlays, legalSubMoves } from "../src/engine/moves";

function emptyBoard(): Position {
  const p = startingPosition();
  p.points.fill(0);
  p.barUs = 0;
  p.barThem = 0;
  p.offUs = 0;
  p.offThem = 0;
  return p;
}

describe("legalSubMoves", () => {
  it("from starting position with die 6: 24/18 is legal", () => {
    const p = startingPosition();
    const subs = legalSubMoves(p, 6);
    // 24-pt is index 23, dest 17. We have 2 checkers there. Index 17 is empty.
    expect(subs.some((s) => s.from === 23 && s.to === 17)).toBe(true);
  });

  it("cannot land on opponent's made point", () => {
    const p = startingPosition();
    // Opponent has 5 on our 19-point (idx 18). 24/19 with die 5 should be blocked.
    const subs = legalSubMoves(p, 5);
    expect(subs.some((s) => s.from === 23 && s.to === 18)).toBe(false);
  });

  it("must enter from bar before any other move", () => {
    const p = startingPosition();
    p.barUs = 1;
    const subs = legalSubMoves(p, 3);
    // Only entries from BAR allowed. Die 3 enters at idx 21 (24-3).
    expect(subs.length).toBe(1);
    expect(subs[0].from).toBe(BAR);
    expect(subs[0].to).toBe(21);
  });

  it("cannot enter from bar onto opponent's made point", () => {
    const p = startingPosition();
    p.barUs = 1;
    // Opponent has 5 at idx 18 (our 19-point). Die 6 would enter at idx 18 — blocked.
    const subs = legalSubMoves(p, 6);
    expect(subs.length).toBe(0);
  });
});

describe("bear-off", () => {
  it("exact die bears off", () => {
    const p = emptyBoard();
    p.points[2] = 1; // 3-point
    p.offUs = 14;
    const subs = legalSubMoves(p, 3);
    expect(subs.some((s) => s.from === 2 && s.to === OFF)).toBe(true);
  });

  it("overshoot bears off only from highest occupied point", () => {
    const p = emptyBoard();
    p.points[1] = 1;
    p.points[3] = 1;
    p.offUs = 13;
    // Die 6: only 4-point (idx 3) can bear off; idx 1 cannot since idx 3 is occupied
    const subs = legalSubMoves(p, 6);
    expect(subs.length).toBe(1);
    expect(subs[0].from).toBe(3);
    expect(subs[0].to).toBe(OFF);
  });

  it("cannot bear off if a checker is outside home", () => {
    const p = emptyBoard();
    p.points[2] = 1;
    p.points[10] = 1;
    p.offUs = 13;
    const subs = legalSubMoves(p, 3);
    expect(subs.some((s) => s.to === OFF)).toBe(false);
  });
});

describe("hits", () => {
  it("landing on a blot sends opponent to bar", () => {
    const p = emptyBoard();
    p.points[7] = 1; // our 8-point
    p.points[3] = -1; // opponent blot at our 4-point
    const subs = legalSubMoves(p, 4);
    const hit = subs.find((s) => s.from === 7 && s.to === 3)!;
    expect(hit).toBeDefined();
    const np = applyPlay(p, [hit]);
    expect(np.points[3]).toBe(1);
    expect(np.barThem).toBe(1);
  });
});

describe("generatePlays", () => {
  it("doubles produce up to 4 sub-moves", () => {
    const p = emptyBoard();
    p.points[20] = 4;
    const plays = generatePlays(p, 6, 6);
    const lens = new Set(plays.map((pl) => pl.length));
    expect(Math.max(...lens)).toBe(4);
  });

  it("keeps every distinct sub-move sequence (no final-position dedup)", () => {
    // 5-3 with checkers on the 8-point only. Plays like (8/3, 8/5) and
    // (8/5, 8/3) reach the same final but differ in sub-move ordering. Both
    // must be kept so the UI can play either order.
    const p = emptyBoard();
    p.points[7] = 4; // our 8-point
    const plays = generatePlays(p, 5, 3);
    const keys = new Set(
      plays.map((pl) => pl.map((s) => `${s.from},${s.to},${s.die}`).join("|")),
    );
    expect(keys.size).toBe(plays.length);
    // And both orderings should be present.
    const hasFirstThree = plays.some((pl) => pl[0]?.die === 3);
    const hasFirstFive = plays.some((pl) => pl[0]?.die === 5);
    expect(hasFirstThree).toBe(true);
    expect(hasFirstFive).toBe(true);
  });

  it("must use larger die when both can be used singly but not together", () => {
    // Bar=1, dice (1,4). Either die can enter and that's all (followup blocked).
    const p = emptyBoard();
    p.barUs = 1;
    p.points[19] = -2; // blocks both followups: 23->19 (die4) and 20->19 (die1)
    const plays = generatePlays(p, 1, 4);
    expect(plays.length).toBeGreaterThan(0);
    for (const pl of plays) {
      expect(pl.length).toBe(1);
      expect(pl[0].die).toBe(4);
    }
  });

  it("returns [[]] when no dice can be used", () => {
    const p = emptyBoard();
    p.barUs = 1;
    // Block all entries
    for (let i = 18; i < 24; i++) p.points[i] = -2;
    const plays = generatePlays(p, 3, 5);
    expect(plays).toEqual([[]]);
  });
});
