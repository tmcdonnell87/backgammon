import { describe, it, expect } from "vitest";
import {
  CHECKERS_PER_SIDE,
  POINTS,
  allHome,
  clonePosition,
  hashPosition,
  mirror,
  pipCount,
  pipCountThem,
  startingPosition,
} from "../src/engine/position";

describe("starting position", () => {
  it("has 15 checkers per side", () => {
    const p = startingPosition();
    let us = p.barUs + p.offUs;
    let them = p.barThem + p.offThem;
    for (let i = 0; i < POINTS; i++) {
      if (p.points[i] > 0) us += p.points[i];
      else if (p.points[i] < 0) them += -p.points[i];
    }
    expect(us).toBe(CHECKERS_PER_SIDE);
    expect(them).toBe(CHECKERS_PER_SIDE);
  });

  it("has equal pip counts (167) for both sides", () => {
    const p = startingPosition();
    expect(pipCount(p)).toBe(167);
    expect(pipCountThem(p)).toBe(167);
  });

  it("is symmetric under mirror", () => {
    const p = startingPosition();
    const m = mirror(p);
    expect(pipCount(m)).toBe(pipCount(p));
    expect(pipCountThem(m)).toBe(pipCountThem(p));
  });
});

describe("position helpers", () => {
  it("clone is independent", () => {
    const p = startingPosition();
    const c = clonePosition(p);
    c.points[0] = 99;
    expect(p.points[0]).toBe(-2);
  });

  it("hash differs for different positions", () => {
    const p = startingPosition();
    const c = clonePosition(p);
    c.points[0] = 0;
    expect(hashPosition(p)).not.toBe(hashPosition(c));
  });

  it("allHome detects when all our checkers are in 0..5", () => {
    const p = startingPosition();
    expect(allHome(p)).toBe(false);
    // Construct a fake all-home position
    const c = clonePosition(p);
    c.points.fill(0);
    c.points[0] = 5;
    c.points[3] = 10;
    expect(allHome(c)).toBe(true);
    c.points[8] = 1;
    expect(allHome(c)).toBe(false);
  });
});
