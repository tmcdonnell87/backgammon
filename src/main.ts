import { GameController, Phase } from "./game/controller";
import { loadSettings } from "./game/persistence";
import { createWorkerClient } from "./ai/client";
import { renderBoard, animateSubMove } from "./ui/board.svg";
import { BoardLayout, DESIGN_W, VIEW_H, makeLayout } from "./ui/layout";
import {
  legalNextTargets,
  consumedDice,
  isPendingComplete,
} from "./game/controller";
import { Position } from "./engine/position";
import { Play, applyPlay } from "./engine/moves";
import { showMenu } from "./ui/menu";
import { renderPostGameReport } from "./ui/tutor-panel";
import { handleTap } from "./ui/interaction";
import { openHintModal } from "./ui/hint-modal";
import { openTutorModal } from "./ui/tutor-modal";
import { openSettingsModal } from "./ui/settings-modal";

async function main(): Promise<void> {
  registerServiceWorker();
  const settings = await loadSettings();
  const ai = createWorkerClient();
  const controller = new GameController(settings, ai);

  const root = document.getElementById("app")!;
  buildShell(root);

  const svg = root.querySelector<SVGSVGElement>("svg.board-svg")!;
  const boardWrap = root.querySelector<HTMLElement>(".board-wrap")!;
  const overlayContainer = root.querySelector<HTMLElement>(".overlay")!;

  // Responsive layout — viewW matches the container's aspect ratio so that
  // circles always render as perfect circles (no preserveAspectRatio stretch).
  // We clamp viewW so triangles don't get absurdly wide on ultrawide displays.
  let currentLayout: BoardLayout = makeLayout(DESIGN_W, VIEW_H);
  function recomputeLayout(): boolean {
    const rect = boardWrap.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    const aspect = rect.width / rect.height;
    const viewW = Math.max(DESIGN_W, Math.min(2000, Math.round(VIEW_H * aspect)));
    if (viewW === currentLayout.viewW) return false;
    currentLayout = makeLayout(viewW, VIEW_H);
    return true;
  }
  recomputeLayout();
  const tutorEl = root.querySelector<HTMLElement>(".tutor-slot")!;
  const scoreEl = root.querySelector<HTMLElement>(".score")!;
  const undoBtn = root.querySelector<HTMLButtonElement>('[data-action="undo"]')!;
  const newBtn = root.querySelector<HTMLButtonElement>('[data-action="new"]')!;
  const hintBtn = root.querySelector<HTMLButtonElement>('[data-action="hint"]')!;
  const settingsBtn = root.querySelector<HTMLButtonElement>('[data-action="settings"]')!;
  const fullscreenBtn = root.querySelector<HTMLButtonElement>('[data-action="fullscreen"]')!;

  // Diagnostic tap log (dev only). Reproduces a "tap did nothing" failure by
  // reading window.__taplog after the user hits a silent failure. Each entry
  // carries enough state to identify which exit path fired.
  type TapEntry = {
    t: number;
    event: string;
    reason: string;
    clientX?: number;
    clientY?: number;
    vbX?: number;
    vbY?: number;
    targetTag?: string;
    onDice?: boolean;
    hitIdx?: number | null;
    phase: string;
    pendingLen: number;
    selectedFrom: number | null;
    dice: number[] | null;
    processing?: boolean;
    pendingLenAfter?: number;
    selectedFromAfter?: number | null;
  };
  const taplog: TapEntry[] = [];
  const TAPLOG_MAX = 50;
  function recordTap(entry: Partial<TapEntry> & { event: string; reason: string }): void {
    if (!import.meta.env?.DEV) return;
    const s = controller.state;
    const full: TapEntry = {
      t: Date.now(),
      event: entry.event,
      reason: entry.reason,
      clientX: entry.clientX,
      clientY: entry.clientY,
      vbX: entry.vbX,
      vbY: entry.vbY,
      targetTag: entry.targetTag,
      onDice: entry.onDice,
      hitIdx: entry.hitIdx,
      phase: s.phase.kind,
      pendingLen: s.pendingPlay.length,
      selectedFrom: s.selectedFrom,
      dice: s.position.dice ? [...s.position.dice] : null,
      processing: entry.processing,
      pendingLenAfter: entry.pendingLenAfter,
      selectedFromAfter: entry.selectedFromAfter,
    };
    taplog.push(full);
    if (taplog.length > TAPLOG_MAX) taplog.shift();
  }
  if (import.meta.env?.DEV) {
    (window as unknown as { __bgctrl: GameController; __taplog: TapEntry[] }).__bgctrl = controller;
    (window as unknown as { __bgctrl: GameController; __taplog: TapEntry[] }).__taplog = taplog;
  }

  // Pointer-based interaction — supports tap-to-move AND drag-to-move.
  // <6px movement after pointerdown = tap; >=6px = drag.
  type DragState = {
    pointerId: number;
    startClientX: number;
    startClientY: number;
    startIdx: number;
    startIsSource: boolean;
    isDiceTap: boolean;
    ghost: SVGCircleElement | null;
    dragging: boolean;
  };
  let drag: DragState | null = null;
  // Drag-vs-tap threshold in client pixels. Touchscreens have natural finger
  // slop on a tap; too small a threshold turned slight slides into "cancelled
  // drag" with the user seeing nothing happen. 12px feels right on phones.
  const DRAG_THRESHOLD = 12;

  svg.addEventListener("pointerdown", (ev) => {
    if (controller.state.phase.kind !== "play") {
      recordTap({ event: "pointerdown", reason: "wrong-phase", clientX: ev.clientX, clientY: ev.clientY, targetTag: (ev.target as Element | null)?.tagName });
      return;
    }
    const target = ev.target as Element | null;
    const onDice = !!(target && target.closest('[data-dice-zone]'));
    const pt = svgPoint(svg, ev.clientX, ev.clientY);
    if (!pt) {
      recordTap({ event: "pointerdown", reason: "svgPoint-null", clientX: ev.clientX, clientY: ev.clientY, targetTag: target?.tagName });
      return;
    }
    const idx = onDice ? null : currentLayout.hitTest(pt.x, pt.y, controller.shouldFlipDisplay());
    if (!onDice && idx === null) {
      recordTap({ event: "pointerdown", reason: "hittest-null", clientX: ev.clientX, clientY: ev.clientY, vbX: pt.x, vbY: pt.y, targetTag: target?.tagName, onDice, hitIdx: idx });
      return;
    }
    const phase = controller.state.phase as Extract<Phase, { kind: "play" }>;
    const targets = legalNextTargets(phase.legalPlays, controller.state.pendingPlay);
    try {
      svg.setPointerCapture(ev.pointerId);
    } catch {
      // pointer capture not supported — fall back to delegated events
    }
    drag = {
      pointerId: ev.pointerId,
      startClientX: ev.clientX,
      startClientY: ev.clientY,
      startIdx: idx ?? -2,
      startIsSource: idx !== null && targets.has(idx),
      isDiceTap: onDice,
      ghost: null,
      dragging: false,
    };
    recordTap({ event: "pointerdown", reason: onDice ? "captured-dice" : drag.startIsSource ? "captured-source" : "captured-non-source", clientX: ev.clientX, clientY: ev.clientY, vbX: pt.x, vbY: pt.y, targetTag: target?.tagName, onDice, hitIdx: idx });
  });

  svg.addEventListener("pointermove", (ev) => {
    if (!drag || ev.pointerId !== drag.pointerId) return;
    if (!drag.startIsSource) return;
    const dx = ev.clientX - drag.startClientX;
    const dy = ev.clientY - drag.startClientY;
    if (!drag.dragging) {
      if (Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
      drag.dragging = true;
      controller.pickFrom(drag.startIdx);
      const ourColor: "white" | "black" = controller.state.position.turn === 0 ? "white" : "black";
      const pt = svgPoint(svg, ev.clientX, ev.clientY);
      if (pt) drag.ghost = createDragGhost(svg, pt.x, pt.y, ourColor);
    } else if (drag.ghost) {
      const pt = svgPoint(svg, ev.clientX, ev.clientY);
      if (pt) {
        drag.ghost.setAttribute("cx", String(pt.x));
        drag.ghost.setAttribute("cy", String(pt.y));
      }
    }
  });

  function finishInteraction(ev: PointerEvent, cancelled: boolean): void {
    if (!drag || ev.pointerId !== drag.pointerId) return;
    const d = drag;
    drag = null;
    try {
      svg.releasePointerCapture(ev.pointerId);
    } catch {
      // ignore
    }
    if (d.ghost) d.ghost.remove();
    const pendingBefore = controller.state.pendingPlay.length;
    if (cancelled) {
      if (d.dragging) controller.clearSelection();
      recordTap({ event: "pointercancel", reason: d.dragging ? "drag-cancelled" : "tap-cancelled", clientX: ev.clientX, clientY: ev.clientY, hitIdx: d.startIdx, processing });
      return;
    }
    if (d.isDiceTap && !d.dragging) {
      handleDiceTap();
      const after = controller.state;
      recordTap({ event: "pointerup", reason: "dice-tap", clientX: ev.clientX, clientY: ev.clientY, onDice: true, processing, pendingLenAfter: after.pendingPlay.length, selectedFromAfter: after.selectedFrom });
      return;
    }
    const pt = svgPoint(svg, ev.clientX, ev.clientY);
    if (!d.dragging) {
      // Tap path
      const idx = pt ? currentLayout.hitTest(pt.x, pt.y, controller.shouldFlipDisplay()) : null;
      if (idx !== null) {
        handleTap(controller, idx);
        const after = controller.state;
        const moved = after.pendingPlay.length !== pendingBefore || after.selectedFrom !== null;
        recordTap({ event: "pointerup", reason: moved ? "tap-moved" : "tap-noop", clientX: ev.clientX, clientY: ev.clientY, vbX: pt?.x, vbY: pt?.y, hitIdx: idx, processing, pendingLenAfter: after.pendingPlay.length, selectedFromAfter: after.selectedFrom });
      } else {
        recordTap({ event: "pointerup", reason: "tap-hittest-null", clientX: ev.clientX, clientY: ev.clientY, vbX: pt?.x, vbY: pt?.y, hitIdx: idx, processing });
      }
      return;
    }
    // Drag path: drop only if drop target is a legal destination of the source.
    const dropIdx = pt ? currentLayout.hitTest(pt.x, pt.y, controller.shouldFlipDisplay()) : null;
    // If the drag ended on the same idx it started (finger slop on a tap that
    // happened to cross the threshold), treat it as a tap so the user gets a
    // result instead of a silent cancel.
    if (dropIdx === d.startIdx) {
      handleTap(controller, d.startIdx);
      const after = controller.state;
      const moved = after.pendingPlay.length !== pendingBefore || after.selectedFrom !== null;
      recordTap({ event: "pointerup", reason: moved ? "drag-tap-moved" : "drag-tap-noop", clientX: ev.clientX, clientY: ev.clientY, hitIdx: d.startIdx, processing, pendingLenAfter: after.pendingPlay.length });
      return;
    }
    if (dropIdx === null) {
      controller.clearSelection();
      recordTap({ event: "pointerup", reason: "drag-drop-off-board", clientX: ev.clientX, clientY: ev.clientY, hitIdx: dropIdx, processing });
      return;
    }
    const phase = controller.state.phase;
    if (phase.kind !== "play") {
      recordTap({ event: "pointerup", reason: "drag-wrong-phase", clientX: ev.clientX, clientY: ev.clientY, hitIdx: dropIdx, processing });
      return;
    }
    const targets = legalNextTargets(phase.legalPlays, controller.state.pendingPlay);
    const dests = targets.get(d.startIdx);
    if (dests && dests.has(dropIdx)) {
      controller.pickTo(dropIdx);
      recordTap({ event: "pointerup", reason: "drag-moved", clientX: ev.clientX, clientY: ev.clientY, hitIdx: dropIdx, processing, pendingLenAfter: controller.state.pendingPlay.length });
    } else {
      controller.clearSelection();
      recordTap({ event: "pointerup", reason: "drag-drop-not-dest", clientX: ev.clientX, clientY: ev.clientY, hitIdx: dropIdx, processing });
    }
  }

  function handleDiceTap(): void {
    const s = controller.state;
    if (s.phase.kind !== "play") return;
    // Pending complete -> confirm/commit.
    const complete = isPendingComplete(s.phase.legalPlays, s.pendingPlay);
    if (complete) {
      controller.commitPlay();
      return;
    }
    // Pending empty + two non-double dice -> swap order.
    if (s.pendingPlay.length === 0 && s.position.dice && s.position.dice.length === 2 && s.position.dice[0] !== s.position.dice[1]) {
      controller.swapDice();
    }
  }

  svg.addEventListener("pointerup", (ev) => finishInteraction(ev, false));
  svg.addEventListener("pointercancel", (ev) => finishInteraction(ev, true));
  // Prevent the synthetic mouse-click that browsers fire after a touchend from
  // re-dispatching tap logic — our pointer handlers already covered it.
  svg.addEventListener("click", (ev) => ev.preventDefault());

  // Ephemeral hint preview: set by the hint modal to overlay arrows on the
  // board, cleared when the modal closes (or the user dismisses it). Cleared
  // on every controller emit too — moving the pieces invalidates the preview.
  let previewHintMoves: { from: number; to: number }[] | null = null;
  // Tutor preview: while the error/blunder analysis modal is open we paint
  // the board in its pre-move state so the user can see arrows referencing
  // the position they actually played from. Cleared on modal close.
  let tutorPreview: { startPos: Position; flipped: boolean; ourColor: "white" | "black" } | null = null;

  function repaintWithOverrides(): void {
    if (tutorPreview) {
      const snap: BoardSnapshot = {
        position: tutorPreview.startPos,
        pendingPlay: [],
        selectedFrom: null,
        phase: { kind: "play", legalPlays: [] },
        flipped: tutorPreview.flipped,
        ourColor: tutorPreview.ourColor,
      };
      paintBoardFrame(tutorPreview.startPos, snap, { hideTopFrom: null, pendingPlayLen: 0 });
      return;
    }
    if (lastShown) {
      paintBoardFrame(workingFromSnap(lastShown), lastShown, {
        hideTopFrom: null,
        pendingPlayLen: lastShown.pendingPlay.length,
      });
    }
  }

  function setPreviewHint(moves: { from: number; to: number }[] | null): void {
    previewHintMoves = moves;
    repaintWithOverrides();
  }

  function computeFlippedFor(turn: 0 | 1): boolean {
    const s = controller.state.settings;
    if (s.whitePlayer === "cpu" && s.blackPlayer === "cpu") return false;
    if (s.whitePlayer === "human" && s.blackPlayer === "human") {
      return turn === 1;
    }
    const humanSide = s.whitePlayer === "human" ? 0 : 1;
    return turn !== humanSide;
  }

  function setTutorPreview(
    pre: { startPos: Position; side: 0 | 1; hintMoves: { from: number; to: number }[] | null } | null,
  ): void {
    if (pre === null) {
      tutorPreview = null;
      previewHintMoves = null;
      repaintWithOverrides();
      return;
    }
    tutorPreview = {
      startPos: pre.startPos,
      flipped: computeFlippedFor(pre.side),
      ourColor: pre.side === 0 ? "white" : "black",
    };
    previewHintMoves = pre.hintMoves;
    repaintWithOverrides();
  }

  undoBtn.addEventListener("click", () => controller.undo());
  newBtn.addEventListener("click", () => openMenu(controller, root));
  hintBtn.addEventListener("click", () => void openHintModal(controller, overlayContainer, { setPreviewHint }));
  settingsBtn.addEventListener("click", () => openSettingsModal(controller, overlayContainer));
  fullscreenBtn.addEventListener("click", () => {
    if (document.fullscreenElement) {
      void document.exitFullscreen().catch(() => {});
    } else {
      void document.documentElement.requestFullscreen().catch(() => {});
    }
  });
  const updateFullscreenIcon = (): void => {
    const icon = fullscreenBtn.querySelector<HTMLElement>(".icon");
    if (!icon) return;
    icon.textContent = document.fullscreenElement ? "⛶" : "⛶";
    fullscreenBtn.title = document.fullscreenElement ? "Exit fullscreen" : "Fullscreen";
    fullscreenBtn.setAttribute("aria-label", fullscreenBtn.title);
  };
  document.addEventListener("fullscreenchange", updateFullscreenIcon);

  // ---- Board animation pipeline ----
  // Chrome (topbar/buttons/overlays/tutor) updates synchronously on every emit.
  // The board renders through an async queue so we can play a ~150ms slide for
  // each sub-move (CPU plays multiple at once; humans appear one-by-one).

  type BoardSnapshot = {
    position: Position;
    pendingPlay: Play;
    selectedFrom: number | null;
    phase: Phase;
    flipped: boolean;
    ourColor: "white" | "black";
  };

  const snapshotQueue: BoardSnapshot[] = [];
  let lastShown: BoardSnapshot | null = null;
  let processing = false;

  function snapshotBoard(): BoardSnapshot {
    const s = controller.state;
    return {
      position: s.position,
      pendingPlay: [...s.pendingPlay],
      selectedFrom: s.selectedFrom,
      phase: s.phase,
      flipped: controller.shouldFlipDisplay(),
      ourColor: s.position.turn === 0 ? "white" : "black",
    };
  }

  // Tutor-modal de-dup: only pop a modal the first time a given tutor entry
  // is observed. We compare history length so emits that don't add a new
  // entry (board re-paints, etc.) don't re-open the modal.
  let lastTutorHistoryLen = 0;
  controller.subscribe(() => {
    // Any state change invalidates an in-flight hint preview — clear it so
    // arrows don't linger over a position they no longer describe.
    previewHintMoves = null;
    renderChrome();
    snapshotQueue.push(snapshotBoard());
    if (!processing) void processQueue();
    // If a new tutor entry was just recorded and it's an error/blunder,
    // surface the analysis modal so the user can learn from it.
    const s = controller.state;
    if (
      s.settings.tutorEnabled &&
      s.tutor.history.length > lastTutorHistoryLen
    ) {
      const newest = s.tutor.history[s.tutor.history.length - 1];
      lastTutorHistoryLen = s.tutor.history.length;
      if (newest.classification === "error" || newest.classification === "blunder") {
        openTutorModal(newest, overlayContainer, {
          setTutorPreview,
          onDismiss: () => controller.ackTutor(),
        });
      }
    }
  });

  async function processQueue(): Promise<void> {
    processing = true;
    try {
      while (snapshotQueue.length > 0) {
        const next = snapshotQueue.shift()!;
        await transitionBoard(next);
        lastShown = next;
      }
    } finally {
      processing = false;
    }
  }

  async function transitionBoard(next: BoardSnapshot): Promise<void> {
    // The tutor preview owns the canvas while it's showing (post-error/blunder
    // modal). Skip painting queued snapshots — otherwise the post-mirror
    // snapshot pushed by the commit's emit() lands on top of the pre-move
    // preview, leaving the arrows pointing at the opponent's pieces.
    if (tutorPreview) return;
    // Animate the new sub-moves if (a) we share a position with the last frame
    // and (b) `next.pendingPlay` is an extension of `lastShown.pendingPlay`.
    if (
      lastShown &&
      samePosition(lastShown.position, next.position) &&
      isAppend(lastShown.pendingPlay, next.pendingPlay)
    ) {
      for (let i = lastShown.pendingPlay.length; i < next.pendingPlay.length; i++) {
        const before = applyPlay(next.position, next.pendingPlay.slice(0, i));
        const sub = next.pendingPlay[i];
        paintBoardFrame(before, next, {
          hideTopFrom: sub.from,
          pendingPlayLen: i,
        });
        await animateSubMove(svg, before, sub, next.flipped, next.ourColor, currentLayout, 150);
      }
    }
    paintBoardFrame(workingFromSnap(next), next, {
      hideTopFrom: null,
      pendingPlayLen: next.pendingPlay.length,
    });
  }

  function workingFromSnap(s: BoardSnapshot): Position {
    if (s.pendingPlay.length === 0) return s.position;
    return applyPlay(s.position, s.pendingPlay);
  }

  function paintBoardFrame(
    pos: Position,
    s: BoardSnapshot,
    opts: { hideTopFrom: number | null; pendingPlayLen: number },
  ): void {
    const phase = s.phase;
    const dice = s.position.dice ?? [];
    // Build the dice display array: rolled order is preserved across the turn,
    // each slot independently marked used/unused. This keeps the left die in
    // the left position even after it's consumed (it fades in place).
    let diceDisplay: { d: number; used: boolean }[] = [];
    if (dice.length === 2) {
      const rolled = dice[0] === dice[1] ? [dice[0], dice[0], dice[0], dice[0]] : [dice[0], dice[1]];
      if (phase.kind === "play") {
        const consumedPrefix = s.pendingPlay.slice(0, opts.pendingPlayLen);
        const consumed = consumedDice(consumedPrefix);
        const counts: Record<number, number> = {};
        for (const c of consumed) counts[c] = (counts[c] ?? 0) + 1;
        diceDisplay = rolled.map((d) => {
          if ((counts[d] ?? 0) > 0) {
            counts[d] = (counts[d] as number) - 1;
            return { d, used: true };
          }
          return { d, used: false };
        });
        // If no further sub-move is possible from this prefix (i.e., the
        // turn is over even though some dice were never consumed — common
        // when only one die can legally be played), mark those leftover
        // dice as "used" too. Visually signals "your turn is done".
        if (isPendingComplete(phase.legalPlays, consumedPrefix)) {
          diceDisplay = diceDisplay.map((d) => (d.used ? d : { d: d.d, used: true }));
        }
      } else if (phase.kind === "opening" || phase.kind === "cpu-thinking" || phase.kind === "roll") {
        diceDisplay = rolled.map((d) => ({ d, used: false }));
      }
    }
    // The source wash (faint highlight on tappable points) is driven by the
    // KEYS of legalDestsFrom — we always compute it from the CURRENT pending
    // prefix so it matches the visible board state. Suppressing it during
    // intermediate animation frames would make the wash blink off and on for
    // ~150 ms each sub-move, which reads as point-color flashing.
    const targets =
      phase.kind === "play"
        ? legalNextTargets(phase.legalPlays, s.pendingPlay.slice(0, opts.pendingPlayLen))
        : new Map<number, Set<number>>();
    // The diceCue (commit/swap tap-zones) is only meaningful on the FINAL
    // settled frame — never during animation — so we gate it by checking
    // whether we're rendering the full pending play.
    const isFinalFrame = opts.pendingPlayLen === s.pendingPlay.length;
    let diceCue: "swap" | "confirm" | null = null;
    if (phase.kind === "play" && isFinalFrame) {
      if (isPendingComplete(phase.legalPlays, s.pendingPlay)) diceCue = "confirm";
      else if (
        s.pendingPlay.length === 0 &&
        s.position.dice &&
        s.position.dice.length === 2 &&
        s.position.dice[0] !== s.position.dice[1]
      ) {
        diceCue = "swap";
      }
    }
    // Pip counts from the canonical (us-perspective) position currently
    // being rendered. Only emitted when the user opted in via settings.
    const pipCount = controller.state.settings.showPipCount ? computePipsUs(pos) : null;
    // Equity (absolute white POV) representing the game state after the LAST
    // committed move. Shown in the wood divider bar between the trays.
    const equity = controller.state.settings.showEquity ? controller.state.currentEquity : null;
    renderBoard(svg, pos, {
      layout: currentLayout,
      flipped: s.flipped,
      // Selection ring is hidden during animation because the source's top
      // checker is itself hidden (hideTopFrom). On the final frame, selectedFrom
      // is naturally null (pickTo cleared it) so this just shows nothing.
      selectedFrom: isFinalFrame ? s.selectedFrom : null,
      legalDestsFrom: targets,
      dice: diceDisplay,
      ourColor: s.ourColor,
      hideTopFrom: opts.hideTopFrom,
      diceCue,
      cpuThinking: phase.kind === "cpu-thinking",
      hintMoves: previewHintMoves ?? undefined,
      pipCount,
      equity,
    });
  }

  function renderChrome(): void {
    const s = controller.state;
    const phase = s.phase;
    const wName = s.settings.whiteName;
    const bName = s.settings.blackName;
    scoreEl.textContent = `${wName} ${s.whiteScore} — ${s.blackScore} ${bName}`;
    undoBtn.disabled = !(phase.kind === "play" && s.pendingPlay.length > 0);
    hintBtn.disabled = !(phase.kind === "play");

    // New-game button doubles as surrender mid-game and "start a game" when
    // no game is in progress. Swap the icon/tooltip so the affordance matches.
    const inGame = phase.kind === "play"
      || phase.kind === "roll"
      || phase.kind === "opening"
      || phase.kind === "cpu-thinking"
      || phase.kind === "cube-decision";
    const newIcon = newBtn.querySelector<HTMLElement>(".icon")!;
    if (inGame) {
      newIcon.textContent = "🏳️";
      newBtn.title = "Surrender · new game";
      newBtn.setAttribute("aria-label", "Surrender · new game");
    } else {
      newIcon.textContent = "🎲";
      newBtn.title = "New game";
      newBtn.setAttribute("aria-label", "New game");
    }

    // Clear only phase-driven overlays (pass-overlay, won, match-won). Leave
    // user-triggered modals (hint, tutor analysis) alone — they own their own
    // lifecycle and must not be wiped by an unrelated state emit.
    for (const node of Array.from(overlayContainer.querySelectorAll(".phase-overlay"))) {
      node.remove();
    }
    if (phase.kind === "won") {
      const div = document.createElement("div");
      div.className = "modal-backdrop phase-overlay";
      const m = document.createElement("div");
      m.className = "modal";
      const winnerName = phase.result.winner === 0 ? wName : bName;
      const verb = winnerIsSecondPerson(winnerName) ? "win" : "wins";
      const kind =
        phase.result.kind === "single"
          ? verb
          : phase.result.kind === "gammon"
            ? `${verb} a gammon`
            : `${verb} a backgammon`;
      m.innerHTML = `<h2>${escapeHtml(winnerName)} ${kind}!</h2>`;
      if (s.settings.tutorEnabled && s.tutor.history.length > 0) {
        const reportSlot = document.createElement("div");
        renderPostGameReport(reportSlot, s.tutor.history, wName, bName);
        m.appendChild(reportSlot);
      }
      const actions = document.createElement("div");
      actions.className = "actions";
      const matchOver =
        s.settings.matchLength > 1 && (s.whiteScore >= s.settings.matchLength || s.blackScore >= s.settings.matchLength);
      if (!matchOver) {
        const next = document.createElement("button");
        next.className = "primary";
        next.textContent = "Next game";
        next.addEventListener("click", () => controller.startNewGame());
        actions.appendChild(next);
      }
      const menu = document.createElement("button");
      menu.textContent = "Main menu";
      menu.addEventListener("click", () => openMenu(controller, root));
      actions.appendChild(menu);
      m.appendChild(actions);
      div.appendChild(m);
      overlayContainer.appendChild(div);
    } else if (phase.kind === "match-won") {
      const div = document.createElement("div");
      div.className = "modal-backdrop phase-overlay";
      const m = document.createElement("div");
      m.className = "modal";
      const winnerName = phase.winner === 0 ? wName : bName;
      const matchVerb = winnerIsSecondPerson(winnerName) ? "win" : "wins";
      m.innerHTML = `<h2>${escapeHtml(winnerName)} ${matchVerb} the match!</h2>
        <p>${escapeHtml(wName)} ${s.whiteScore} — ${s.blackScore} ${escapeHtml(bName)}</p>`;
      if (s.settings.tutorEnabled && s.tutor.history.length > 0) {
        const reportSlot = document.createElement("div");
        renderPostGameReport(reportSlot, s.tutor.history, wName, bName);
        m.appendChild(reportSlot);
      }
      const actions = document.createElement("div");
      actions.className = "actions";
      const menu = document.createElement("button");
      menu.className = "primary";
      menu.textContent = "Main menu";
      menu.addEventListener("click", () => openMenu(controller, root));
      actions.appendChild(menu);
      m.appendChild(actions);
      div.appendChild(m);
      overlayContainer.appendChild(div);
    }

    tutorEl.innerHTML = "";
  }

  // Initial paint
  recomputeLayout();
  const initSnap = snapshotBoard();
  renderChrome();
  paintBoardFrame(workingFromSnap(initSnap), initSnap, {
    hideTopFrom: null,
    pendingPlayLen: initSnap.pendingPlay.length,
  });
  lastShown = initSnap;

  // Re-render on container resize so the viewBox tracks the window aspect.
  // We only re-render when the computed viewW actually changes (round-trip
  // through `recomputeLayout`'s no-op guard) to avoid wasted work.
  const ro = new ResizeObserver(() => {
    if (!recomputeLayout()) return;
    if (!lastShown) return;
    paintBoardFrame(workingFromSnap(lastShown), lastShown, {
      hideTopFrom: null,
      pendingPlayLen: lastShown.pendingPlay.length,
    });
  });
  ro.observe(boardWrap);
}

// Pip count from a canonical (us-coords) position. Renderer uses these
// directly with the {us, them} convention.
function computePipsUs(p: Position): { us: number; them: number } {
  let us = p.barUs * 25;
  let them = p.barThem * 25;
  for (let i = 0; i < 24; i++) {
    const v = p.points[i];
    if (v > 0) us += v * (i + 1);
    else if (v < 0) them += -v * (24 - i);
  }
  return { us, them };
}

function samePosition(a: Position, b: Position): boolean {
  if (a.turn !== b.turn) return false;
  if (a.barUs !== b.barUs || a.barThem !== b.barThem) return false;
  if (a.offUs !== b.offUs || a.offThem !== b.offThem) return false;
  for (let i = 0; i < 24; i++) if (a.points[i] !== b.points[i]) return false;
  return true;
}

function isAppend(prev: Play, next: Play): boolean {
  if (next.length <= prev.length) return false;
  for (let i = 0; i < prev.length; i++) {
    if (prev[i].from !== next[i].from || prev[i].to !== next[i].to || prev[i].die !== next[i].die) return false;
  }
  return true;
}

function buildShell(root: HTMLElement): void {
  root.innerHTML = `
    <div class="rotate-prompt">
      <div class="rotate-icon">↻</div>
      <div>Rotate your device to landscape</div>
    </div>
    <div class="topbar">
      <div class="title">Backgammon</div>
      <div class="score">—</div>
    </div>
    <div class="main">
      <div class="board-wrap">
        <svg class="board-svg" xmlns="http://www.w3.org/2000/svg"></svg>
      </div>
      <aside class="sidebar">
        <div class="actions actions-top">
          <button data-action="undo" title="Undo" aria-label="Undo"><span class="label">Undo</span><span class="icon" aria-hidden="true">↶</span></button>
          <button data-action="hint" title="Hint" aria-label="Hint"><span class="label">Hint</span><span class="icon" aria-hidden="true">💡</span></button>
          <button data-action="new" title="New game (surrender)" aria-label="New game"><span class="label">New game</span><span class="icon" aria-hidden="true">🏳️</span></button>
        </div>
        <div class="tutor-slot"></div>
        <div class="actions actions-bottom">
          <button data-action="fullscreen" title="Fullscreen" aria-label="Fullscreen"><span class="label">Fullscreen</span><span class="icon" aria-hidden="true">⛶</span></button>
          <button data-action="settings" title="Settings" aria-label="Settings"><span class="label">Settings</span><span class="icon" aria-hidden="true">⚙</span></button>
        </div>
      </aside>
    </div>
    <div class="overlay"></div>
  `;
}

function attemptOrientationLock(): void {
  type LockableOrientation = ScreenOrientation & { lock?: (o: string) => Promise<void> };
  const so = screen.orientation as LockableOrientation | undefined;
  if (so && typeof so.lock === "function") {
    so.lock("landscape").catch(() => {
      // Lock often requires fullscreen; silently ignore failures.
    });
  }
}

function createDragGhost(svg: SVGSVGElement, cx: number, cy: number, ourColor: "white" | "black"): SVGCircleElement {
  const fill = ourColor === "white" ? "var(--checker-w)" : "var(--checker-b)";
  const stroke = ourColor === "white" ? "var(--checker-w-edge)" : "var(--checker-b-edge)";
  const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  c.setAttribute("cx", String(cx));
  c.setAttribute("cy", String(cy));
  c.setAttribute("r", "24");
  c.setAttribute("fill", fill);
  c.setAttribute("stroke", stroke);
  c.setAttribute("stroke-width", "2");
  c.setAttribute("opacity", "0.85");
  c.setAttribute("pointer-events", "none");
  c.setAttribute("filter", "drop-shadow(0 2px 4px rgba(0,0,0,0.5))");
  svg.appendChild(c);
  return c;
}

function svgPoint(svg: SVGSVGElement, clientX: number, clientY: number): { x: number; y: number } | null {
  // Use the browser's screen CTM so we respect preserveAspectRatio (which
  // letterboxes/pillarboxes the viewBox inside the SVG element when their
  // aspects differ). A naive clientX/width * VIEW_W mapping ignores those
  // bars and pushes taps near point boundaries onto the wrong column.
  const ctm = svg.getScreenCTM();
  if (!ctm) return null;
  const pt = svg.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  const local = pt.matrixTransform(ctm.inverse());
  return { x: local.x, y: local.y };
}

function openMenu(controller: GameController, root: HTMLElement): void {
  const node = showMenu(controller.state.settings, {
    onStart: (settings) => {
      controller.setSettings(settings);
      controller.startNewMatch();
      attemptOrientationLock();
    },
    onClose: () => {
      // If we were already mid-game, stay; else show a 'tap new game' state
    },
  });
  root.appendChild(node);
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
}

// "You wins" → "You win". Second-person pronoun names take a plural verb.
function winnerIsSecondPerson(name: string): boolean {
  return /^you$/i.test(name.trim());
}

function registerServiceWorker(): void {
  if (!("serviceWorker" in navigator)) return;
  // In dev, never register — and proactively unregister any stale SW from a
  // prior prod-build visit on the same origin, plus wipe its caches. Without
  // this, the SW keeps serving the old built /index.html (which references
  // hashed bundles the dev server doesn't have), so refreshes return blank.
  if (import.meta.env?.DEV) {
    navigator.serviceWorker.getRegistrations().then((regs) => {
      for (const r of regs) void r.unregister();
    }).catch(() => undefined);
    if ("caches" in globalThis) {
      caches.keys().then((keys) => Promise.all(keys.map((k) => caches.delete(k)))).catch(() => undefined);
    }
    return;
  }
  if (location.protocol !== "https:" && location.hostname !== "localhost") return;
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => undefined);
  });
}

void main();
