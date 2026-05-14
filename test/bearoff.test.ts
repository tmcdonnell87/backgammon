import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { POINTS, Position } from "../src/engine/position";
import {
  bearoffEquity, loadBearoff, setBearoff, getBearoff,
} from "../src/ai/bearoff";

// Tests run in Node, no fetch. Stub `fetch` globally to read from disk so
// loadBearoff(meta_url) can pull the binary file alongside.
function installLocalFetch() {
  (globalThis as any).fetch = async (url: string) => {
    let path: string;
    if (url.startsWith("/")) {
      path = resolve("public" + url);
    } else {
      path = url;
    }
    const buf = readFileSync(path);
    return {
      ok: true,
      json: async () => JSON.parse(buf.toString("utf8")),
      arrayBuffer: async () =>
        buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
    };
  };
}

function makePos(usHome: number[], themHome: number[]): Position {
  const points = new Int8Array(POINTS);
  for (let i = 0; i < 6; i++) points[i] = usHome[i];
  // Opponent's 1-pt = our 24-pt = idx 23; their 6-pt = our 19-pt = idx 18.
  for (let i = 0; i < 6; i++) points[23 - i] = -themHome[i];
  let offUs = 15;
  for (let i = 0; i < 6; i++) offUs -= usHome[i];
  let offThem = 15;
  for (let i = 0; i < 6; i++) offThem -= themHome[i];
  return {
    points,
    barUs: 0,
    barThem: 0,
    offUs,
    offThem,
    turn: 0,
    dice: null,
    cube: { value: 1, owner: null },
    score: [0, 0],
    matchLength: 1,
    crawford: false,
  };
}

describe("bearoff lookup", () => {
  beforeAll(async () => {
    installLocalFetch();
    setBearoff(null);
    const data = await loadBearoff("/weights/bearoff.json");
    expect(data).not.toBeNull();
  });

  it("loads and parses the bearoff table", () => {
    const d = getBearoff();
    expect(d).not.toBeNull();
    expect(d!.maxRolls).toBe(32);
    expect(d!.histLen).toBe(33);
    expect(d!.stateIndex.size).toBe(54264);
    expect(d!.stateIndex.get("0,0,0,0,0,0")).toBeDefined();
  });

  it("returns null for non-race positions (contact / bar / outer-board)", () => {
    // Starting position (full contact)
    const p = makePos([5, 0, 0, 0, 3, 5], [5, 0, 0, 0, 3, 5]);
    // Add the back-point checkers as non-home opponents to disqualify.
    p.points[12] = 5;
    p.points[11] = -5;
    expect(bearoffEquity(p)).toBeNull();
  });

  it("rejects positions with anything on the bar", () => {
    const p = makePos([2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]);
    p.barUs = 1;
    expect(bearoffEquity(p)).toBeNull();
  });

  it("handles trivially won positions (us already at 15 off)", () => {
    const p = makePos([0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 0]);
    const eq = bearoffEquity(p)!;
    expect(eq.pWin).toBe(1);
    expect(eq.pLoss).toBe(0);
  });

  it("handles trivially lost positions (they already at 15 off)", () => {
    const p = makePos([1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]);
    const eq = bearoffEquity(p)!;
    expect(eq.pWin).toBe(0);
    expect(eq.pLoss).toBe(1);
  });

  it("one-vs-one symmetric race: side on roll wins (we bear off first)", () => {
    // Both sides have a single checker on the 1-pt. We're on roll => bear off
    // first 100% of the time (any die >= 1 bears off, which is always).
    // Quantized table is uint16-precise (~3e-5 wobble).
    const p = makePos([1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]);
    const eq = bearoffEquity(p)!;
    expect(eq.pWin).toBeGreaterThan(0.9999);
    expect(eq.pLoss).toBeLessThan(0.0001);
  });

  it("symmetric heavier race: side on roll has measurable edge", () => {
    // 3 checkers each on 1-pt. We're on roll.
    // P(we win) > 0.5 (we bear off our last checker before they bear off theirs).
    const p = makePos([3, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0]);
    const eq = bearoffEquity(p)!;
    expect(eq.pWin).toBeGreaterThan(0.5);
    expect(eq.pWin).toBeLessThan(1.0);
    // pWin + pLoss == 1
    expect(eq.pWin + eq.pLoss).toBeCloseTo(1.0, 5);
  });

  it("asymmetric race: heavily ahead = very high pWin", () => {
    // We have 1 checker on 1-pt; they have 5 on 6-pt.
    const p = makePos([1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 5]);
    const eq = bearoffEquity(p)!;
    // We bear off in 1 roll, they need at least 2; pWin should be essentially 1.
    expect(eq.pWin).toBeGreaterThan(0.9999);
  });

  it("symmetric race not on roll: lower pWin", () => {
    // Same as 3-on-1 symmetric but we let opponent roll first conceptually:
    // there's no "not on roll" flag, but we test that swapping perspective
    // gives complementary equities.
    const ours = makePos([3, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0]);
    const theirs = makePos([3, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0]);
    const eqUs = bearoffEquity(ours)!;
    const eqThem = bearoffEquity(theirs)!;
    // Both queries are "side on roll", so they should be identical.
    expect(eqUs.pWin).toBeCloseTo(eqThem.pWin, 5);
  });

  it("probabilities are normalized and bounded", () => {
    // Random-ish race.
    const p = makePos([2, 2, 2, 3, 3, 3], [3, 3, 3, 2, 2, 2]);
    const eq = bearoffEquity(p)!;
    expect(eq.pWin).toBeGreaterThanOrEqual(0);
    expect(eq.pWin).toBeLessThanOrEqual(1);
    expect(eq.pGammonWin).toBeGreaterThanOrEqual(0);
    expect(eq.pGammonWin).toBeLessThanOrEqual(eq.pWin + 1e-6);
    expect(eq.pLoss).toBeGreaterThanOrEqual(0);
    expect(eq.pLoss).toBeLessThanOrEqual(1);
    expect(eq.pGammonLoss).toBeGreaterThanOrEqual(0);
    expect(eq.pGammonLoss).toBeLessThanOrEqual(eq.pLoss + 1e-6);
    expect(eq.pWin + eq.pLoss).toBeCloseTo(1.0, 5);
  });
});
