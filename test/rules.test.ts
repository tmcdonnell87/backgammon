import { describe, it, expect } from "vitest";
import { startingPosition } from "../src/engine/position";
import { checkWin } from "../src/engine/rules";

describe("checkWin", () => {
  it("returns null at game start", () => {
    expect(checkWin(startingPosition())).toBeNull();
  });

  it("single win when both sides have borne off", () => {
    const p = startingPosition();
    p.offUs = 15;
    p.offThem = 5;
    const r = checkWin(p)!;
    expect(r.kind).toBe("single");
    expect(r.basePoints).toBe(1);
  });

  it("gammon when opponent has borne off none and not in our home or bar", () => {
    const p = startingPosition();
    p.points.fill(0);
    p.points[12] = -15; // opponent has all checkers in their outer board
    p.offUs = 15;
    p.offThem = 0;
    p.barThem = 0;
    const r = checkWin(p)!;
    expect(r.kind).toBe("gammon");
    expect(r.basePoints).toBe(2);
  });

  it("backgammon when opponent has a checker in our home board", () => {
    const p = startingPosition();
    p.points.fill(0);
    p.points[3] = -1; // opponent checker in our home
    p.points[12] = -14;
    p.offUs = 15;
    p.offThem = 0;
    const r = checkWin(p)!;
    expect(r.kind).toBe("backgammon");
    expect(r.basePoints).toBe(3);
  });

  it("backgammon when opponent has a checker on the bar", () => {
    const p = startingPosition();
    p.points.fill(0);
    p.points[12] = -14;
    p.barThem = 1;
    p.offUs = 15;
    p.offThem = 0;
    const r = checkWin(p)!;
    expect(r.kind).toBe("backgammon");
  });
});
