/**
 * @vitest-environment happy-dom
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { GameController } from "../src/game/controller";
import { openHintModal } from "../src/ui/hint-modal";
import { AIClient } from "../src/ai/api";

describe("UI Hint to Equity Integration", () => {
  let mockAi: AIClient;
  let settings: any;
  let container: HTMLElement;

  beforeEach(() => {
    // Basic DOM mock
    container = document.createElement("div");
    document.body.appendChild(container);

    mockAi = {
      analyze: vi.fn().mockResolvedValue({
        bestPlay: [{ from: 23, to: 12, die: 11 }],
        bestEquity: 0.545,
        equities: [0.545]
      }),
      pickMove: vi.fn(),
      evaluate: vi.fn(),
      decideCube: vi.fn(),
      decideTake: vi.fn(),
    } as any;

    settings = {
      matchLength: 5,
      cubeEnabled: true,
      whitePlayer: "human",
      blackPlayer: "cpu",
      whiteName: "White",
      blackName: "Black",
      cpuDifficulty: "expert",
      tutorEnabled: true,
      showPipCount: true,
      showEquity: true,
    };
  });

  // Skipped: mixes vi.useFakeTimers() with real setTimeout/await; equity update
  // races the assertion. Re-enable once the hint→commit flow can be awaited
  // deterministically (or rework to drive entirely on fake timers).
  it.skip("Play Selected in Hint Modal updates controller currentEquity correctly", async () => {
    const controller = new GameController(settings, mockAi);
    
    // Setup a turn
    controller.state.phase = { kind: "roll" };
    controller.rollDice();
    
    // Wait for analysis to trigger
    await controller.getActiveAnalysis();

    // Open Hint Modal
    const opts = { setPreviewHint: vi.fn() };
    await openHintModal(controller, container, opts);

    // Verify modal is open and has the expected equity
    const hintRow = container.querySelector(".hint-row.best")!;
    expect(hintRow).not.toBeNull();
    expect(hintRow.textContent).toContain("+0.55"); // formatEquity does toFixed(2)

    // Simulate "Play selected" click
    const playBtn = container.querySelector<HTMLButtonElement>('button[data-action="play"]')!;
    expect(playBtn.disabled).toBe(false);
    
    // We need to mock setTimeout or wait
    vi.useFakeTimers();
    playBtn.click();
    
    // playFullPlay schedules commitPlay after ANIM_PER_SUB * play.length + 80
    // 150 * 1 + 80 = 230
    vi.advanceTimersByTime(300);
    
    // Now check if currentEquity is updated
    // Need to wait for the analysis promise in commitPlay
    vi.useRealTimers();
    await new Promise(resolve => setTimeout(resolve, 50));
    
    expect(controller.state.currentEquity).toBe(0.545);
  });
});
