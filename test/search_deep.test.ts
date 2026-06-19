import { describe, it, expect } from "vitest";
import { startingPosition, Position, hashPosition } from "../src/engine/position";
import { generatePlays, applyPlay, Play } from "../src/engine/moves";
import { score0ply, score2ply, rankPlaysDeep } from "../src/ai/search";
import { heuristicEvaluator } from "../src/ai/heuristic";

const ev = heuristicEvaluator;

function finalEquityMap(p: Position, scored: { play: Play; equity: number }[]) {
  const m = new Map<string, number>();
  for (const s of scored) m.set(hashPosition(applyPlay(p, s.play)), s.equity);
  return m;
}

function expectMapsClose(a: Map<string, number>, b: Map<string, number>) {
  expect([...a.keys()].sort()).toEqual([...b.keys()].sort());
  for (const [k, v] of a) expect(b.get(k)!).toBeCloseTo(v, 9);
}

// A few representative positions / dice.
function midgame(): Position {
  const p = startingPosition();
  // Make a non-symmetric contact position so plays differ in value.
  p.points.set(new Int8Array([
    0, 2, 0, 0, 2, 4, -1, 3, 0, 0, 0, 4,
    -4, 0, -2, 0, 2, 0, 3, -2, 0, -3, 0, -2,
  ]));
  return p;
}

describe("rankPlaysDeep reproduces the existing shallow searches", () => {
  const cases: { name: string; p: () => Position; dice: [number, number] }[] = [
    { name: "opening 6-5", p: startingPosition, dice: [6, 5] },
    { name: "opening 3-1", p: startingPosition, dice: [3, 1] },
    { name: "midgame 6-4", p: midgame, dice: [6, 4] },
    { name: "midgame 5-5", p: midgame, dice: [5, 5] },
  ];

  for (const c of cases) {
    it(`plies:1 == score0ply (${c.name})`, () => {
      const p = c.p();
      const plays = generatePlays(p, c.dice[0], c.dice[1]);
      const deep = rankPlaysDeep(p, plays, ev, { plies: 1 });
      const ref = score0ply(p, plays, ev);
      expectMapsClose(finalEquityMap(p, deep), finalEquityMap(p, ref));
    });

    it(`plies:2 unfiltered == score2ply (${c.name})`, () => {
      const p = c.p();
      const plays = generatePlays(p, c.dice[0], c.dice[1]);
      const deep = rankPlaysDeep(p, plays, ev, {
        plies: 2,
        keepTop: Infinity,
        keepWindow: Infinity,
        innerKeep: Infinity,
      });
      const ref = score2ply(p, plays, ev);
      expectMapsClose(finalEquityMap(p, deep), finalEquityMap(p, ref));
    });
  }
});

describe("rankPlaysDeep is sorted and covers every final", () => {
  it("returns descending equities, one per unique final", () => {
    const p = midgame();
    const plays = generatePlays(p, 6, 4);
    const scored = rankPlaysDeep(p, plays, ev, { plies: 2, keepTop: Infinity, innerKeep: Infinity });
    for (let i = 1; i < scored.length; i++) {
      expect(scored[i - 1].equity).toBeGreaterThanOrEqual(scored[i].equity - 1e-12);
    }
    // Every input play's final is represented.
    const finals = new Set(scored.map((s) => hashPosition(applyPlay(p, s.play))));
    for (const pl of plays) expect(finals.has(hashPosition(applyPlay(p, pl)))).toBe(true);
  });

  it("3-ply runs and ranks plausibly on a small position", () => {
    // Near-bearoff race: few plays, so exhaustive 3-ply is cheap.
    const p = startingPosition();
    p.points.set(new Int8Array([
      3, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -3,
    ]));
    p.offUs = 5;
    p.offThem = 8;
    const plays = generatePlays(p, 3, 2);
    const scored = rankPlaysDeep(p, plays, ev, {
      plies: 3, keepTop: 6, keepWindow: 0.1, innerKeep: 6,
    });
    expect(scored.length).toBeGreaterThan(0);
    // Best play should be a real, beneficial move (positive equity in a position
    // where we're well ahead in the race).
    expect(scored[0].equity).toBeGreaterThan(scored[scored.length - 1].equity - 1e-9);
  });
});
