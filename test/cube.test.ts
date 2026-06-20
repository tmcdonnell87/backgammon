import { describe, it, expect } from "vitest";
import { Position, Side, startingPosition } from "../src/engine/position";
import { canDouble, applyDoubleAccepted, maxUsefulCube } from "../src/engine/cube";

function makeMatchPos(opts: {
  matchLength: number;
  whiteScore?: number;
  blackScore?: number;
  cubeValue?: number;
  cubeOwner?: Side | null;
  crawford?: boolean;
  turn?: Side;
}): Position {
  const p = startingPosition({ matchLength: opts.matchLength });
  if (opts.turn === 1) p.turn = 1;
  p.score = [opts.whiteScore ?? 0, opts.blackScore ?? 0];
  p.cube = { value: opts.cubeValue ?? 1, owner: opts.cubeOwner ?? null };
  p.crawford = opts.crawford ?? false;
  return p;
}

describe("engine/cube primitives", () => {
  it("canDouble for centered cube: either side may double", () => {
    const p = makeMatchPos({ matchLength: 7 });
    expect(canDouble(p, 0)).toBe(true);
    expect(canDouble(p, 1)).toBe(true);
  });

  it("canDouble respects ownership", () => {
    const p = makeMatchPos({ matchLength: 7, cubeValue: 2, cubeOwner: 0 });
    expect(canDouble(p, 0)).toBe(true);
    expect(canDouble(p, 1)).toBe(false);
  });

  it("canDouble = false in a Crawford game", () => {
    const p = makeMatchPos({ matchLength: 7, crawford: true });
    expect(canDouble(p, 0)).toBe(false);
    expect(canDouble(p, 1)).toBe(false);
  });

  it("canDouble in money game (matchLength=1) ignores Crawford flag", () => {
    const p = makeMatchPos({ matchLength: 1, crawford: true });
    // matchLength <= 1 means the Crawford gate is not active; both can double.
    expect(canDouble(p, 0)).toBe(true);
    expect(canDouble(p, 1)).toBe(true);
  });

  it("applyDoubleAccepted doubles value, transfers ownership to receiver", () => {
    const p = makeMatchPos({ matchLength: 7, cubeValue: 2, cubeOwner: 0 });
    const np = applyDoubleAccepted(p, 0);
    expect(np.cube.value).toBe(4);
    expect(np.cube.owner).toBe(1);
    // Original unchanged.
    expect(p.cube.value).toBe(2);
    expect(p.cube.owner).toBe(0);
  });

  it("maxUsefulCube caps at the smallest power of 2 ≥ remaining points to win", () => {
    const p = makeMatchPos({
      matchLength: 7, whiteScore: 0, blackScore: 0, cubeValue: 1,
    });
    // need = 7 for white; next power of two ≥ 7 is 8.
    expect(maxUsefulCube(p, 0)).toBe(8);
  });
});

// --- Crawford state machine on GameController ---

import { vi, beforeEach, afterEach } from "vitest";
import { GameController, GameSettings } from "../src/game/controller";
import { createMainThreadClient } from "../src/ai/client";

const CUBE_SETTINGS: GameSettings = {
  matchLength: 7,
  cubeEnabled: true,
  whitePlayer: "cpu",
  blackPlayer: "cpu",
  whiteName: "W",
  blackName: "B",
  cpuDifficulty: "casual",
  tutorMode: "off",
  showPipCount: false,
  showEquity: false,
};

describe("Crawford on GameController.startNewGame", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("does not set crawford when both scores are below matchLength-1", () => {
    const gc = new GameController(CUBE_SETTINGS, createMainThreadClient());
    gc.state.whiteScore = 3;
    gc.state.blackScore = 2;
    gc.startNewGame();
    expect(gc.state.position.crawford).toBe(false);
  });

  it("sets crawford when a side first reaches matchLength-1", () => {
    const gc = new GameController(CUBE_SETTINGS, createMainThreadClient());
    gc.state.whiteScore = 6; // matchLength=7, so 6 = matchLength-1
    gc.state.blackScore = 3;
    gc.startNewGame();
    expect(gc.state.position.crawford).toBe(true);
  });

  it("clears crawford in the game AFTER the Crawford game (post-Crawford)", () => {
    const gc = new GameController(CUBE_SETTINGS, createMainThreadClient());
    gc.state.whiteScore = 6;
    gc.state.blackScore = 3;
    gc.startNewGame();
    expect(gc.state.position.crawford).toBe(true);
    // Simulate the Crawford game ending without changing scores.
    gc.state.crawfordPlayed = true;
    gc.startNewGame();
    expect(gc.state.position.crawford).toBe(false);
  });

  it("does not set crawford in money game (matchLength=1)", () => {
    const gc = new GameController({ ...CUBE_SETTINGS, matchLength: 1 }, createMainThreadClient());
    gc.state.whiteScore = 0;
    gc.state.blackScore = 0;
    gc.startNewGame();
    expect(gc.state.position.crawford).toBe(false);
  });
});
