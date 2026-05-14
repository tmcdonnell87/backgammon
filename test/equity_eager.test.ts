import { describe, it, expect, beforeEach, vi } from "vitest";
import { GameController, GameSettings, Phase } from "../src/game/controller";
import type { AIClient, MoveAnalysis } from "../src/ai/api";
import type { Play } from "../src/engine/moves";

function makeSettings(overrides: Partial<GameSettings> = {}): GameSettings {
  return {
    matchLength: 5,
    cubeEnabled: false,
    whitePlayer: "human",
    blackPlayer: "cpu",
    whiteName: "White",
    blackName: "Black",
    cpuDifficulty: "casual",
    tutorEnabled: false,
    showPipCount: false,
    showEquity: true,
    ...overrides,
  };
}

function makeMockAi(): AIClient {
  return {
    analyze: vi.fn<(p: unknown, legalPlays: Play[]) => Promise<MoveAnalysis>>(),
    pickMove: vi.fn(),
    decideCube: vi.fn(),
    decideTake: vi.fn(),
  } as unknown as AIClient;
}

describe("eager equity tracking", () => {
  let mockAi: AIClient;

  beforeEach(() => {
    mockAi = makeMockAi();
  });

  it("triggers analysis on roll and updates currentEquity on commit (White POV)", async () => {
    (mockAi.analyze as unknown as { mockImplementation: Function }).mockImplementation(
      (_p: unknown, legalPlays: Play[]) =>
        Promise.resolve({
          bestPlay: legalPlays[0],
          bestEquity: 0.123,
          equities: legalPlays.map(() => 0.123),
        }),
    );

    const controller = new GameController(makeSettings(), mockAi);
    controller.state.phase = { kind: "roll" };
    controller.rollDice();

    expect(mockAi.analyze).toHaveBeenCalledTimes(1);
    const analysisPromise = controller.getActiveAnalysis();
    expect(analysisPromise).not.toBeNull();
    await analysisPromise!;

    // Nothing committed yet — bar shows nothing.
    expect(controller.state.currentEquity).toBeNull();

    const phase = controller.state.phase as unknown as Extract<Phase, { kind: "play" }>;
    expect(phase.kind).toBe("play");
    const play = phase.legalPlays[0];
    controller.state.pendingPlay = [...play];
    controller.commitPlay();

    // Wait for the analysis-resolved equity write.
    await new Promise((r) => setTimeout(r, 10));

    expect(controller.state.currentEquity).toBe(0.123);
  });

  it("updates currentEquity for Black POV (sign-flips on-roll equity to white-absolute)", async () => {
    (mockAi.analyze as unknown as { mockImplementation: Function }).mockImplementation(
      (_p: unknown, legalPlays: Play[]) =>
        Promise.resolve({
          bestPlay: legalPlays[0],
          bestEquity: 0.5,
          equities: legalPlays.map(() => 0.5),
        }),
    );

    const settings = makeSettings({ blackPlayer: "human" });
    const controller = new GameController(settings, mockAi);
    // Pretend it's Black's turn (turn=1). The legalPlays generated from this
    // position will still be valid; only the POV / sign-flip changes.
    controller.state.position.turn = 1;
    controller.state.phase = { kind: "roll" };
    controller.rollDice();

    const phase = controller.state.phase as unknown as Extract<Phase, { kind: "play" }>;
    expect(phase.kind).toBe("play");
    controller.state.pendingPlay = [...phase.legalPlays[0]];
    controller.commitPlay();

    await new Promise((r) => setTimeout(r, 10));

    // analyze returned +0.5 in Black's POV; white-absolute is -0.5.
    expect(controller.state.currentEquity).toBe(-0.5);
  });

  it("currentEquity matches the row index of the play the user committed (bar↔hint parity)", async () => {
    // Each play gets a distinct equity so we can verify the row lookup is precise.
    (mockAi.analyze as unknown as { mockImplementation: Function }).mockImplementation(
      (_p: unknown, legalPlays: Play[]) =>
        Promise.resolve({
          bestPlay: legalPlays[0],
          bestEquity: 0.01 * 0,
          equities: legalPlays.map((_, i) => 0.01 * i),
        }),
    );

    const controller = new GameController(makeSettings(), mockAi);
    controller.state.phase = { kind: "roll" };
    controller.rollDice();
    await controller.getActiveAnalysis()!;

    const phase = controller.state.phase as unknown as Extract<Phase, { kind: "play" }>;
    expect(phase.kind).toBe("play");
    // Pick a non-best, non-first row so the assertion can't accidentally
    // succeed via the bestEquity fallback.
    const targetIdx = Math.min(3, phase.legalPlays.length - 1);
    expect(targetIdx).toBeGreaterThan(0);
    const expectedEquity = 0.01 * targetIdx;

    controller.state.pendingPlay = [...phase.legalPlays[targetIdx]];
    controller.commitPlay();
    await new Promise((r) => setTimeout(r, 10));

    expect(controller.state.currentEquity).toBe(expectedEquity);
  });

  it("swapDice re-triggers analysis with a fresh promise", async () => {
    (mockAi.analyze as unknown as { mockImplementation: Function }).mockImplementation(
      (_p: unknown, legalPlays: Play[]) =>
        Promise.resolve({
          bestPlay: legalPlays[0] ?? [],
          bestEquity: 0,
          equities: legalPlays.map(() => 0),
        }),
    );

    // Force non-double dice [5, 3] so swapDice's guard accepts the call.
    // rollDie does Math.floor(rng() * 6) + 1; 0.8 → 5, 0.4 → 3.
    const rand = vi.spyOn(Math, "random");
    rand.mockReturnValueOnce(0.8).mockReturnValueOnce(0.4);

    const controller = new GameController(makeSettings(), mockAi);
    controller.state.phase = { kind: "roll" };
    controller.rollDice();

    expect(controller.state.position.dice).toEqual([5, 3]);
    expect(mockAi.analyze).toHaveBeenCalledTimes(1);
    const first = controller.getActiveAnalysis();
    expect(first).not.toBeNull();

    controller.swapDice();

    expect(controller.state.position.dice).toEqual([3, 5]);
    expect(mockAi.analyze).toHaveBeenCalledTimes(2);
    const second = controller.getActiveAnalysis();
    expect(second).not.toBeNull();
    expect(second).not.toBe(first);

    rand.mockRestore();
  });
});
