import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { parseMet, metEntry, Met } from "../src/ai/met";

function loadMet(): Met {
  const raw = readFileSync(
    join(__dirname, "..", "public", "weights", "met.json"),
    "utf8",
  );
  return parseMet(JSON.parse(raw));
}

describe("Match Equity Table", () => {
  const met = loadMet();

  it("MET[a][a] === 0.5 for all in-range a (symmetric play)", () => {
    for (let a = 1; a <= met.matches; a++) {
      expect(metEntry(met, a, a)).toBeCloseTo(0.5, 6);
    }
  });

  it("MET[a][b] + MET[b][a] === 1 (anti-symmetric)", () => {
    for (let a = 1; a <= met.matches; a++) {
      for (let b = 1; b <= met.matches; b++) {
        const sum = metEntry(met, a, b) + metEntry(met, b, a);
        expect(sum).toBeCloseTo(1.0, 6);
      }
    }
  });

  it("boundary: a=0 (we won) -> 1, b=0 (they won) -> 0", () => {
    expect(metEntry(met, 0, 5)).toBe(1.0);
    expect(metEntry(met, 0, 1)).toBe(1.0);
    expect(metEntry(met, 5, 0)).toBe(0.0);
    expect(metEntry(met, 1, 0)).toBe(0.0);
  });

  it("monotone: more lead -> higher MWC (fix b, decrease a)", () => {
    for (let b = 1; b <= met.matches; b++) {
      let prev = -Infinity;
      for (let a = met.matches; a >= 1; a--) {
        const mwc = metEntry(met, a, b);
        expect(mwc).toBeGreaterThanOrEqual(prev);
        prev = mwc;
      }
    }
  });

  it("monotone: more lead -> higher MWC (fix a, increase b)", () => {
    for (let a = 1; a <= met.matches; a++) {
      let prev = -Infinity;
      for (let b = 1; b <= met.matches; b++) {
        const mwc = metEntry(met, a, b);
        expect(mwc).toBeGreaterThanOrEqual(prev);
        prev = mwc;
      }
    }
  });

  it("MET[1][1] === 0.5 exactly (DMP)", () => {
    expect(metEntry(met, 1, 1)).toBeCloseTo(0.5, 9);
  });

  it("MET clamps scores outside [0, matches]", () => {
    // Beyond-table us-away clamps to met.matches; the value at the corner is small but positive.
    const big = metEntry(met, met.matches + 5, 1);
    expect(big).toBe(metEntry(met, met.matches, 1));
    const huge = metEntry(met, 1, met.matches + 5);
    expect(huge).toBe(metEntry(met, 1, met.matches));
  });

  // The dead-cube model overshoots live-cube published values by a few pp
  // at large leads, because in real play the trailer doubles aggressively
  // to escape gammons. We assert the rough shape against Janowski's 1993
  // published live-cube MET with a generous tolerance — this is a
  // structural sanity check (should the recursion ever be implemented
  // wrong, these are off by 20+ pp), not a precision claim.
  it("rough agreement with Janowski 1993 published 7-pt MET (±15pp)", () => {
    // Published MET (us-away vs 7-away, live-cube, gnubg-derived):
    //   2/7 = 0.85, 3/7 = 0.79, 4/7 = 0.74, 5/7 = 0.66, 6/7 = 0.58.
    const refs: Array<[number, number, number]> = [
      [2, 7, 0.85],
      [3, 7, 0.79],
      [4, 7, 0.74],
      [5, 7, 0.66],
      [6, 7, 0.58],
    ];
    for (const [a, b, ref] of refs) {
      const mwc = metEntry(met, a, b);
      expect(Math.abs(mwc - ref)).toBeLessThan(0.15);
    }
  });
});
