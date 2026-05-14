import { describe, it, expect, vi } from "vitest";
import { GameController, GameSettings, Phase, TutorEntry } from "../src/game/controller";
import type { AIClient, MoveAnalysis } from "../src/ai/api";
import type { Play } from "../src/engine/moves";

function makeSettings(): GameSettings {
  return {
    matchLength: 1,
    cubeEnabled: false,
    whitePlayer: "human",
    blackPlayer: "cpu",
    whiteName: "White",
    blackName: "Black",
    cpuDifficulty: "casual",
    tutorEnabled: true,
    showPipCount: false,
    showEquity: false,
  };
}

describe("tutor entry startPos", () => {
  it("survives the commit/mirror cycle with checkers in the pre-move locations", async () => {
    const mockAi = {
      analyze: vi.fn<(p: unknown, legalPlays: Play[]) => Promise<MoveAnalysis>>().mockImplementation(
        (_p, legalPlays) =>
          Promise.resolve({
            // Return the FIRST legal play as best with high equity, so any
            // other play registers as a big enough loss to be a blunder.
            bestPlay: legalPlays[0] ?? [],
            bestEquity: 0.5,
            equities: legalPlays.map((_, i) => (i === 0 ? 0.5 : -0.5)),
          }),
      ),
      pickMove: vi.fn(),
      decideCube: vi.fn(),
      decideTake: vi.fn(),
    } as unknown as AIClient;

    const controller = new GameController(makeSettings(), mockAi);
    // Capture the pre-move points snapshot.
    controller.state.phase = { kind: "roll" };
    controller.rollDice();
    await controller.getActiveAnalysis();

    const phase = controller.state.phase as unknown as Extract<Phase, { kind: "play" }>;
    expect(phase.kind).toBe("play");

    // Snapshot the pre-move points (deep copy) before any mutation.
    const prePoints = new Int8Array(controller.state.position.points);
    const preTurn = controller.state.position.turn;
    // Pick a non-best play so it registers as a blunder.
    const target = phase.legalPlays.length > 1 ? phase.legalPlays[1] : phase.legalPlays[0];
    controller.state.pendingPlay = [...target];
    controller.commitPlay();

    // commitPlay triggers runTutorAnalysis async — wait for it to push an entry.
    for (let i = 0; i < 20 && controller.state.tutor.history.length === 0; i++) {
      await new Promise((r) => setTimeout(r, 5));
    }
    expect(controller.state.tutor.history.length).toBeGreaterThan(0);
    const entry: TutorEntry = controller.state.tutor.history[0];

    // The startPos must be the pre-move position (same turn, same points).
    expect(entry.side).toBe(preTurn);
    expect(Array.from(entry.startPos.points)).toEqual(Array.from(prePoints));
    expect(entry.startPos.turn).toBe(preTurn);

    // And state.position should now be the MIRRORED post-move position — turn flipped.
    expect(controller.state.position.turn).toBe((1 - preTurn) as 0 | 1);

    // Sanity: the entry.startPos object should NOT be the same reference as
    // controller.state.position (which has been reassigned to the mirrored pos).
    expect(entry.startPos).not.toBe(controller.state.position);
  });
});
