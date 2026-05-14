import { describe, it, expect } from "vitest";
import { startingPosition } from "../src/engine/position";
import { generatePlays, applyPlay } from "../src/engine/moves";
import { analyzeMove, pickMove } from "../src/ai/engine";
import { evaluateHeuristic } from "../src/ai/heuristic";
import { mirror } from "../src/engine/position";

describe("heuristic evaluator", () => {
  it("rates starting position near zero", () => {
    const eq = evaluateHeuristic(startingPosition());
    expect(Math.abs(eq)).toBeLessThan(0.05);
  });

  it("rewards being ahead in pip count", () => {
    const p = startingPosition();
    // Move our 24-pt checkers to 13 (free pip count gain): not realistic but tests the gradient
    const ahead = startingPosition();
    ahead.points[23] = 0;
    ahead.points[12] = 7; // dump them on the 13-pt
    expect(evaluateHeuristic(ahead)).toBeGreaterThan(evaluateHeuristic(p));
  });

  it("symmetric: equity(p) ≈ -equity(mirror(p)) for typical positions", () => {
    const p = startingPosition();
    const e1 = evaluateHeuristic(p);
    const e2 = evaluateHeuristic(mirror(p));
    expect(Math.abs(e1 + e2)).toBeLessThan(0.05);
  });
});

describe("pickMove", () => {
  it("returns a legal play from the candidate set", () => {
    const p = startingPosition();
    const plays = generatePlays(p, 3, 1);
    const chosen = pickMove(p, plays, "casual");
    expect(plays).toContain(chosen);
  });

  it("opening 3-1 prefers building the 5-pt", () => {
    // The standard "best" 3-1 play is 8/5, 6/5, making the 5-point.
    const p = startingPosition();
    const plays = generatePlays(p, 3, 1);
    const chosen = pickMove(p, plays, "expert");
    const after = applyPlay(p, chosen);
    expect(after.points[4]).toBe(2); // 5-point is made
  });

  it("opening 4-2 prefers making the 4-point", () => {
    // The standard "best" 4-2 play is 8/4, 6/4, making the 4-point.
    const p = startingPosition();
    const plays = generatePlays(p, 4, 2);
    const chosen = pickMove(p, plays, "expert");
    const after = applyPlay(p, chosen);
    expect(after.points[3]).toBe(2); // 4-point is made
  });

  it("end-game race: bear off when all checkers are home", () => {
    // Construct a pure race position: 2 checkers on each of points 1..5,
    // 5 already off, opponent similarly. Dice 6-5 should bear off both
    // back checkers (since 6 = bear-off from any point in home, 5 = bear-off
    // from the 5-point).
    const p = startingPosition();
    for (let i = 0; i < 24; i++) p.points[i] = 0;
    p.points[0] = 2;
    p.points[1] = 2;
    p.points[2] = 2;
    p.points[3] = 2;
    p.points[4] = 2;
    p.offUs = 5;
    p.points[23] = -2;
    p.points[22] = -2;
    p.points[21] = -2;
    p.points[20] = -2;
    p.points[19] = -2;
    p.offThem = 5;
    const plays = generatePlays(p, 6, 5);
    const chosen = pickMove(p, plays, "expert");
    const after = applyPlay(p, chosen);
    // Both dice should be used to bear off (offUs increases by 2).
    expect(after.offUs).toBe(7);
  });
});

describe("analyzeMove", () => {
  it("returns equities for every candidate", () => {
    const p = startingPosition();
    const plays = generatePlays(p, 6, 5);
    const r = analyzeMove(p, plays, "strong");
    expect(r.equities.length).toBe(plays.length);
    expect(plays).toContain(r.bestPlay);
  });
});
