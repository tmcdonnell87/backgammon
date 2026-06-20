import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import {
  GameController,
  GameSettings,
  checkerInvariant,
} from "../src/game/controller";
import { createMainThreadClient } from "../src/ai/client";
import { pickMove } from "../src/ai/engine";

// Mulberry32 — seeded RNG so games are deterministic for the test.
function makeSeededRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
    t = (t ^ (t + Math.imul(t ^ (t >>> 7), t | 61))) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SETTINGS_2P: GameSettings = {
  matchLength: 1,
  cubeEnabled: false,
  whitePlayer: "human",
  blackPlayer: "human",
  whiteName: "White",
  blackName: "Black",
  cpuDifficulty: "casual",
  tutorMode: "off",
  showPipCount: false,
  showEquity: false,
};

const SETTINGS_VS_CPU: GameSettings = {
  ...SETTINGS_2P,
  blackPlayer: "cpu",
};

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

// Drive a controller until the game ends. For human sides, pick a move with the
// heuristic and commit it via simulated taps (pickFrom + pickTo) — that's what
// the real UI does, so this also exercises the tap-handling code path.
async function playToCompletion(
  controller: GameController,
  humanSides: Set<0 | 1>,
  maxIters = 4000,
): Promise<void> {
  for (let i = 0; i < maxIters; i++) {
    // Flush any pending setTimeouts (afterRoll delay, CPU thinking, forfeit
    // commit, post-commit auto-roll) and their resulting microtasks.
    await vi.runAllTimersAsync();
    const s = controller.state;
    const phase = s.phase;

    expect(checkerInvariant(s.position)).toBe(true);

    if (phase.kind === "won" || phase.kind === "match-won") return;

    if (phase.kind === "menu" || phase.kind === "opening" || phase.kind === "cpu-thinking") {
      continue;
    }

    if (phase.kind === "roll") {
      // Auto-roll fires via setTimeout(350) after every commit, so we just wait.
      continue;
    }

    if (phase.kind === "play") {
      if (!humanSides.has(s.position.turn as 0 | 1)) {
        // CPU side — controller drives commit on its own.
        continue;
      }
      // Forfeit — controller auto-commits after a delay.
      if (phase.legalPlays.length === 1 && phase.legalPlays[0].length === 0) {
        continue;
      }
      // Choose a sensible play with the heuristic so the game converges.
      const play = pickMove(s.position, phase.legalPlays, "casual");
      for (const sub of play) {
        controller.pickFrom(sub.from);
        controller.pickTo(sub.to);
      }
      // Tap-to-confirm replaces the old auto-commit on the last sub-move.
      controller.commitPlay();
    }
  }
  throw new Error(`Game did not finish within ${maxIters} iterations`);
}

describe("full game integration", () => {
  test("two human players complete a game via simulated taps", async () => {
    vi.spyOn(Math, "random").mockImplementation(makeSeededRng(42));
    const ai = createMainThreadClient();
    const controller = new GameController(SETTINGS_2P, ai);
    controller.startNewMatch();

    await playToCompletion(controller, new Set([0, 1]));

    const phase = controller.state.phase;
    expect(phase.kind === "won" || phase.kind === "match-won").toBe(true);
    // Verify someone bore off all 15
    const pos = controller.state.position;
    const someoneOff = pos.offUs >= 15 || pos.offThem >= 15;
    expect(someoneOff).toBe(true);
  }, 60_000);

  test("human vs CPU completes a game", async () => {
    vi.spyOn(Math, "random").mockImplementation(makeSeededRng(17));
    const ai = createMainThreadClient();
    const controller = new GameController(SETTINGS_VS_CPU, ai);
    controller.startNewMatch();

    await playToCompletion(controller, new Set([0]));

    const phase = controller.state.phase;
    expect(phase.kind === "won" || phase.kind === "match-won").toBe(true);
    const pos = controller.state.position;
    expect(pos.offUs >= 15 || pos.offThem >= 15).toBe(true);
  }, 60_000);

  test("can enter off the bar via simulated taps", async () => {
    vi.spyOn(Math, "random").mockImplementation(makeSeededRng(99));
    const ai = createMainThreadClient();
    const controller = new GameController(SETTINGS_2P, ai);
    controller.startNewMatch();
    // Run through opening delay
    await vi.runAllTimersAsync();

    // Hand-craft: stick the player-on-roll on the bar and force a roll.
    const s = controller.state;
    s.position.barUs = 1;
    s.position.points[23] = s.position.points[23] - 1; // remove one from our 24-pt to balance count
    if (s.position.points[23] < 0) {
      // Edge case if our seed put 23 already at 0 — restore and use a different point
      s.position.points[23] = s.position.points[23] + 1;
      s.position.points[12] -= 1;
    }
    // Force a specific dice roll that allows entry (5,3 → enters on point 21 or 23 minus die)
    s.position.dice = [5, 3];
    // Re-enter the play phase by calling afterRoll directly via internals: simplest
    // way is to ask the engine for legal plays and inject. Use private method via cast.
    type WithAfterRoll = { afterRoll: () => void };
    (controller as unknown as WithAfterRoll).afterRoll();
    await vi.runAllTimersAsync();

    const phase = controller.state.phase;
    expect(phase.kind).toBe("play");
    if (phase.kind !== "play") return;
    // The bar (idx 24) must be a legal source for at least one sub-move
    const fromSet = new Set(phase.legalPlays.flatMap((p) => p.map((sm) => sm.from)));
    expect(fromSet.has(24)).toBe(true);

    // Simulate the human entering: pick first play, run its sub-moves via taps.
    const play = phase.legalPlays[0];
    for (const sub of play) {
      controller.pickFrom(sub.from);
      controller.pickTo(sub.to);
    }
    controller.commitPlay();
    // After committing the entering play, bar should be zero (or fewer than before).
    await vi.runAllTimersAsync();
    expect(checkerInvariant(controller.state.position)).toBe(true);
  });
});
