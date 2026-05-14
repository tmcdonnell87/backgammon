import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import { GameController, GameSettings } from "../src/game/controller";
import { createMainThreadClient } from "../src/ai/client";
import { handleTap } from "../src/ui/interaction";
import { startingPosition } from "../src/engine/position";

const SETTINGS_2P: GameSettings = {
  matchLength: 1,
  cubeEnabled: false,
  whitePlayer: "human",
  blackPlayer: "human",
  whiteName: "White",
  blackName: "Black",
  cpuDifficulty: "casual",
  tutorEnabled: false,
  showPipCount: false,
  showEquity: false,
};

// Build a controller with a hand-crafted position and dice, ready in "play" phase.
function makePlayingController(args: {
  setupPos: (pos: ReturnType<typeof startingPosition>) => void;
  dice: [number, number];
}): GameController {
  const ai = createMainThreadClient();
  const c = new GameController(SETTINGS_2P, ai);
  const pos = startingPosition();
  args.setupPos(pos);
  pos.dice = args.dice;
  c.state.position = pos;
  type WithAfterRoll = { afterRoll: () => void };
  (c as unknown as WithAfterRoll).afterRoll();
  return c;
}

describe("handleTap", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  test("single-tap auto-move: tapping a legal source moves the checker immediately when destination is unique", () => {
    // Standard opening, dice 6-5. Every source has a unique destination
    // (the engine dedupes alternate orderings). So tapping any legal source
    // should commit a move without needing a second tap.
    const c = makePlayingController({
      setupPos: () => {},
      dice: [6, 5],
    });
    expect(c.state.phase.kind).toBe("play");
    if (c.state.phase.kind !== "play") return;
    // Source idx 12 is the 13-point; idx 7 is the 8-point.
    handleTap(c, 12);
    expect(c.state.pendingPlay.length).toBeGreaterThan(0);
    expect(c.state.pendingPlay[0].from).toBe(12);
  });

  test("two sequential taps build a full pending play, ready to confirm via dice tap", () => {
    const c = makePlayingController({
      setupPos: () => {},
      dice: [6, 5],
    });
    if (c.state.phase.kind !== "play") return;
    const startTurn = c.state.position.turn;
    handleTap(c, 12); // 12/6 (d=6) via auto-exec
    expect(c.state.pendingPlay.length).toBe(1);
    handleTap(c, 7); // 7/2 (d=5) via auto-exec
    // Pending should be complete but not yet committed — turn hasn't changed.
    expect(c.state.pendingPlay.length).toBe(2);
    expect(c.state.position.turn).toBe(startTurn);
    // User confirms via commitPlay (the dice tap)
    c.commitPlay();
    expect(c.state.position.turn).not.toBe(startTurn);
  });

  test("tapping an illegal source is a no-op", () => {
    const c = makePlayingController({
      setupPos: () => {},
      dice: [6, 5],
    });
    handleTap(c, 0); // 1-pt — opponent's checkers there, not a legal source
    expect(c.state.pendingPlay.length).toBe(0);
    expect(c.state.selectedFrom).toBeNull();
  });

  test("entering from bar: a single tap auto-enters using the bigger die (default order)", () => {
    const c = makePlayingController({
      setupPos: (pos) => {
        pos.barUs = 1;
        pos.points[23] = pos.points[23] - 1;
      },
      dice: [3, 5],
    });
    if (c.state.phase.kind !== "play") return;
    // After normalization dice display [5, 3]; autoExec uses leftmost = 5.
    expect(c.state.position.dice).toEqual([5, 3]);
    handleTap(c, 24);
    expect(c.state.pendingPlay.some((m) => m.from === 24 && m.to === 19 && m.die === 5)).toBe(true);
  });

  test("after swapDice, entering from bar uses the smaller die", () => {
    const c = makePlayingController({
      setupPos: (pos) => {
        pos.barUs = 1;
        pos.points[23] = pos.points[23] - 1;
      },
      dice: [3, 5],
    });
    if (c.state.phase.kind !== "play") return;
    // After normalization: [5, 3]. Swap inverts to [3, 5]. Leftmost = 3.
    c.swapDice();
    handleTap(c, 24);
    expect(c.state.pendingPlay.some((m) => m.from === 24 && m.to === 21 && m.die === 3)).toBe(true);
  });

  test("swapDice reverses dice order when nothing has been played; no-op on doubles or partial", () => {
    const c = makePlayingController({
      setupPos: () => {},
      dice: [6, 5],
    });
    if (c.state.phase.kind !== "play") return;
    expect(c.state.position.dice).toEqual([6, 5]);
    c.swapDice();
    expect(c.state.position.dice).toEqual([5, 6]);
    // After making a move, swap is a no-op
    handleTap(c, 12);
    expect(c.state.pendingPlay.length).toBe(1);
    const beforeSwap = [...(c.state.position.dice ?? [])];
    c.swapDice();
    expect(c.state.position.dice).toEqual(beforeSwap);

    // Doubles: no swap
    const c2 = makePlayingController({ setupPos: () => {}, dice: [3, 3] });
    if (c2.state.phase.kind !== "play") return;
    c2.swapDice();
    expect(c2.state.position.dice).toEqual([3, 3]);
  });


  test("two sources reaching the same final are both startable (either-order play)", () => {
    // Position: 1 white at idx 12 (13-pt), 1 at idx 7 (8-pt). Dice [3, 5].
    // Orderings [12→9 (3), 9→4 (5)] and [7→4 (3), 12→7 (5)] reach the same
    // final (12:0, 7:1, 4:1). Historically only the 12-first ordering was
    // kept by dedup, so tapping 7 silently no-op'd. Relaxed dedup keeps both
    // so the user can begin from either point.
    const c = makePlayingController({
      setupPos: (pos) => {
        for (let i = 0; i < 24; i++) pos.points[i] = 0;
        pos.points[12] = 1;
        pos.points[7] = 1;
        pos.barUs = 0;
        pos.barThem = 0;
      },
      dice: [3, 5],
    });
    if (c.state.phase.kind !== "play") return;
    handleTap(c, 7);
    expect(c.state.pendingPlay.length).toBeGreaterThanOrEqual(1);
    expect(c.state.pendingPlay[0].from).toBe(7);
  });

  // ── Contract tests for silent-failure paths ──
  // These tests assert the CURRENT behavior of paths that silently no-op. If we
  // later add user feedback (flash, toast, etc.) these become regression checks
  // for that contract. If we decide to FIX one of them, the corresponding test
  // is the place that needs updating.

  test("tap during cpu-thinking phase is a silent no-op", () => {
    const c = makePlayingController({ setupPos: () => {}, dice: [6, 5] });
    if (c.state.phase.kind !== "play") return;
    // Force the phase as if a CPU were thinking.
    c.state.phase = { kind: "cpu-thinking" };
    const before = c.state.pendingPlay.length;
    handleTap(c, 12);
    expect(c.state.phase.kind).toBe("cpu-thinking");
    expect(c.state.pendingPlay.length).toBe(before);
    expect(c.state.selectedFrom).toBeNull();
  });

  test("with barUs > 0, tapping a non-bar source is a silent no-op (must enter first)", () => {
    const c = makePlayingController({
      setupPos: (pos) => {
        pos.barUs = 1;
        pos.points[23] = pos.points[23] - 1;
      },
      dice: [3, 5],
    });
    if (c.state.phase.kind !== "play") return;
    // Tap an off-bar source (idx 12, the 13-pt has 5 white in starting position).
    // legalNextTargets here only contains {24}; idx 12 is not a legal source.
    handleTap(c, 12);
    expect(c.state.pendingPlay.length).toBe(0);
    expect(c.state.selectedFrom).toBeNull();
  });

  test("tap on an empty point with no selection is a silent no-op", () => {
    const c = makePlayingController({ setupPos: () => {}, dice: [6, 5] });
    if (c.state.phase.kind !== "play") return;
    // idx 9 (10-pt) is empty in the starting position. Tapping it with nothing
    // selected falls through to pickTo, which requires a unique source — there
    // is none for 10-pt with [6,5], so it no-ops.
    handleTap(c, 9);
    expect(c.state.pendingPlay.length).toBe(0);
    expect(c.state.selectedFrom).toBeNull();
  });

  test("after autoExec consumes one die, tap on a source with no continuation is a no-op", () => {
    // Build a position where the only legal play uses two specific sources
    // (e.g., 13→10 (die 3), 10→5 (die 5)). After the first sub-move, the only
    // legal continuation source is 10 (the freshly-moved checker). Tapping the
    // ORIGINAL source 13 again — even though it still has checkers — is not a
    // valid continuation, and there's no alternative full play starting from 13.
    const c = makePlayingController({
      setupPos: (pos) => {
        for (let i = 0; i < 24; i++) pos.points[i] = 0;
        pos.points[12] = 1; // 1 white at 13-pt
        // Block die-5 from any other source so the only 2-die play is 13→10→5.
        // Specifically block all squares reachable by die 5 from anywhere else.
        for (let i = 0; i < 12; i++) {
          if (i !== 9 && i !== 4 && i !== 7) pos.points[i] = -2;
        }
        pos.barUs = 0;
        pos.barThem = 0;
      },
      dice: [3, 5],
    });
    if (c.state.phase.kind !== "play") return;
    // Tap the 13-pt — autoExec uses leftmost die. Position normalized: [5,3].
    // 13→8 (die 5) - is 8 (idx 7) open? Yes per setup. Then die 3 from 8 → 5.
    handleTap(c, 12);
    expect(c.state.pendingPlay.length).toBe(1);
    const afterFirst = c.state.pendingPlay[0];
    // Tap 12 again — 12 has 0 checkers now, not a valid continuation.
    handleTap(c, 12);
    // No additional sub-move; state unchanged from after the first move.
    expect(c.state.pendingPlay.length).toBe(1);
    expect(c.state.pendingPlay[0]).toEqual(afterFirst);
  });

  test("tap clears stale selection if dest is not legal (cleanup contract)", () => {
    const c = makePlayingController({ setupPos: () => {}, dice: [6, 5] });
    if (c.state.phase.kind !== "play") return;
    // Manually set a selection — then tap an illegal destination. Selection
    // should clear (not stay stuck).
    c.pickFrom(12); // 13-pt is a legal source
    expect(c.state.selectedFrom).toBe(12);
    // idx 9 (10-pt) is NOT a legal destination of 12 with [6,5] — destinations
    // would be idx 6 (die 6) or idx 7 (die 5). Tap 9 → falls through, clears.
    handleTap(c, 9);
    expect(c.state.selectedFrom).toBeNull();
    expect(c.state.pendingPlay.length).toBe(0);
  });

  test("commitPlay rejects an incomplete pending play (guards against premature dice-tap)", () => {
    const c = makePlayingController({ setupPos: () => {}, dice: [6, 5] });
    if (c.state.phase.kind !== "play") return;
    const startTurn = c.state.position.turn;
    handleTap(c, 12); // one sub-move
    expect(c.state.pendingPlay.length).toBe(1);
    c.commitPlay(); // not yet complete (5 still unused) — should no-op
    expect(c.state.position.turn).toBe(startTurn);
    expect(c.state.pendingPlay.length).toBe(1);
  });
});
