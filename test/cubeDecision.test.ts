import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { POINTS, Position, Side } from "../src/engine/position";
import {
  decideCubeAction,
  decideTakeDrop,
  Outcomes4,
} from "../src/ai/cubeDecision";
import { parseMet, Met } from "../src/ai/met";

function loadMet(): Met {
  const raw = readFileSync(
    join(__dirname, "..", "public", "weights", "met.json"),
    "utf8",
  );
  return parseMet(JSON.parse(raw));
}

function makePosition(opts: {
  matchLength: number;
  scoreUs?: number;
  scoreThem?: number;
  cubeValue?: number;
  cubeOwner?: Side | null;
  crawford?: boolean;
  turn?: Side;
}): Position {
  const turn: Side = opts.turn ?? 0;
  return {
    points: new Int8Array(POINTS),
    barUs: 0,
    barThem: 0,
    offUs: 0,
    offThem: 0,
    turn,
    dice: null,
    cube: { value: opts.cubeValue ?? 1, owner: opts.cubeOwner ?? null },
    score: [
      turn === 0 ? (opts.scoreUs ?? 0) : (opts.scoreThem ?? 0),
      turn === 0 ? (opts.scoreThem ?? 0) : (opts.scoreUs ?? 0),
    ],
    matchLength: opts.matchLength,
    crawford: opts.crawford ?? false,
  };
}

// Make a 4-outcome vector from a target cubeless equity (with optional
// gammon distribution). Useful for sweeping equity in tests.
function outcomes(pWin: number, gFraction: number = 0.0): Outcomes4 {
  // pGammonWin = gFraction * pWin, pGammonLoss = gFraction * (1 - pWin).
  return {
    pWin,
    pGammonWin: gFraction * pWin,
    pLoss: 1 - pWin,
    pGammonLoss: gFraction * (1 - pWin),
  };
}

describe("decideCubeAction", () => {
  const met = loadMet();

  it("returns no_double in money game (matchLength=1)", () => {
    const p = makePosition({ matchLength: 1 });
    expect(decideCubeAction(p, 0, outcomes(0.9, 0.3), met)).toBe("no_double");
  });

  it("returns no_double in Crawford", () => {
    const p = makePosition({
      matchLength: 7,
      scoreUs: 6,
      scoreThem: 3,
      crawford: true,
    });
    expect(decideCubeAction(p, 0, outcomes(0.9, 0.3), met)).toBe("no_double");
  });

  it("returns no_double when opponent owns the cube", () => {
    const p = makePosition({
      matchLength: 7,
      cubeValue: 2,
      cubeOwner: 1,
    });
    expect(decideCubeAction(p, 0, outcomes(0.9, 0.3), met)).toBe("no_double");
  });

  it("returns no_double at coin-flip equity (no value in doubling)", () => {
    const p = makePosition({ matchLength: 7 });
    expect(decideCubeAction(p, 0, outcomes(0.5, 0.0), met)).toBe("no_double");
  });

  it("equity sweep hits all three actions across 0.5..0.99 with gammons", () => {
    // With gammon potential, the action sequence as equity grows is
    // typically no_double -> double_take -> double_drop -> (back to)
    // no_double — the last "too good to double" regime happens when a
    // gammon at V=1 outscores the opponent's drop at the cube boundary.
    // We don't assert strict monotonicity; we do assert that every
    // action appears somewhere in the sweep.
    const p = makePosition({ matchLength: 7 });
    const saw: Record<string, boolean> = {};
    for (let i = 0; i <= 50; i++) {
      const pWin = 0.5 + 0.49 * (i / 50);
      const action = decideCubeAction(p, 0, outcomes(pWin, 0.1), met);
      saw[action] = true;
    }
    expect(saw.no_double).toBe(true);
    expect(saw.double_take).toBe(true);
    expect(saw.double_drop).toBe(true);
  });

  it("doubles when receiver's take would be dominated by the drop", () => {
    // 5-pt match, score 0-0, centered cube. High equity AND low gammon
    // potential — no "too good to double" pull, so we should cash. (With
    // significant gammons at this equity, the live-cube model can correctly
    // prefer playing on for the gammon at the current cube value.)
    const p = makePosition({ matchLength: 5 });
    const action = decideCubeAction(p, 0, outcomes(0.85, 0.0), met);
    expect(action).toBe("double_drop");
  });
});

describe("decideTakeDrop", () => {
  const met = loadMet();

  it("takes coin flips (cube=2): receiver has plenty of room", () => {
    const p = makePosition({
      matchLength: 7,
      cubeValue: 2,
      cubeOwner: 0,
    });
    expect(decideTakeDrop(p, 0, outcomes(0.5, 0.0), met)).toBe("take");
  });

  it("drops when our equity is poor enough vs match-equity loss", () => {
    const p = makePosition({
      matchLength: 7,
      cubeValue: 1,
      cubeOwner: null,
    });
    // pWin=0.20 means heavy underdog; drop V=1 vs taking a 2V=2 swing.
    expect(decideTakeDrop(p, 0, outcomes(0.2, 0.1), met)).toBe("drop");
  });

  it("monotone equity sweep: take iff equity high enough", () => {
    const p = makePosition({ matchLength: 7 });
    let crossings = 0;
    let prev = decideTakeDrop(p, 0, outcomes(0.05, 0.05), met);
    for (let i = 1; i <= 50; i++) {
      const pWin = 0.05 + 0.9 * (i / 50);
      const action = decideTakeDrop(p, 0, outcomes(pWin, 0.05), met);
      if (action !== prev) crossings++;
      prev = action;
    }
    // Exactly one transition from "drop" to "take" as equity increases.
    expect(crossings).toBe(1);
  });
});
