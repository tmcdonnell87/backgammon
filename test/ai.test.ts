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
