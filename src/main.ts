import { GameController, GameSettings, Phase } from "./game/controller";
import { loadSettings } from "./game/persistence";
import { createWorkerClient } from "./ai/client";
import { renderBoard } from "./ui/board.svg";
import { hitTest, VIEW_W, VIEW_H } from "./ui/layout";
import {
  legalNextTargets,
  consumedDice,
  remainingDice,
  workingPosition,
} from "./game/controller";
import { showMenu } from "./ui/menu";
import { renderPostGameReport, renderTutorCard } from "./ui/tutor-panel";

async function main(): Promise<void> {
  registerServiceWorker();
  const settings = await loadSettings();
  const ai = createWorkerClient();
  const controller = new GameController(settings, ai);

  const root = document.getElementById("app")!;
  buildShell(root);

  const svg = root.querySelector<SVGSVGElement>("svg.board-svg")!;
  const overlayContainer = root.querySelector<HTMLElement>(".overlay")!;
  const tutorEl = root.querySelector<HTMLElement>(".tutor-slot")!;
  const turnLabel = root.querySelector<HTMLElement>(".turn-label")!;
  const scoreEl = root.querySelector<HTMLElement>(".score")!;
  const rollBtn = root.querySelector<HTMLButtonElement>('[data-action="roll"]')!;
  const undoBtn = root.querySelector<HTMLButtonElement>('[data-action="undo"]')!;
  const newBtn = root.querySelector<HTMLButtonElement>('[data-action="new"]')!;

  // SVG click handler — click-to-pickup, click-to-move
  svg.addEventListener("click", (ev) => {
    if (controller.state.phase.kind !== "play") return;
    const pt = svgPoint(svg, ev.clientX, ev.clientY);
    if (!pt) return;
    const idx = hitTest(pt.x, pt.y, controller.shouldFlipDisplay());
    if (idx === null) return;
    const targets = legalNextTargets(
      (controller.state.phase as Extract<Phase, { kind: "play" }>).legalPlays,
      controller.state.pendingPlay,
    );
    // If user has nothing selected and clicks a legal source -> pickup
    if (controller.state.selectedFrom === null) {
      if (targets.has(idx)) {
        controller.pickFrom(idx);
      } else {
        // Could be a destination with unique source — try pickTo
        controller.pickTo(idx);
      }
      return;
    }
    // Something is picked up
    if (targets.has(idx)) {
      // Switch selection to a different source
      controller.pickFrom(idx);
      return;
    }
    // Otherwise treat as destination
    controller.pickTo(idx);
  });

  rollBtn.addEventListener("click", () => controller.rollDice());
  undoBtn.addEventListener("click", () => controller.undo());
  newBtn.addEventListener("click", () => openMenu(controller, root));

  // Initial: open menu
  openMenu(controller, root);

  controller.subscribe(() => render());

  function render(): void {
    const s = controller.state;
    const phase = s.phase;
    const flipped = controller.shouldFlipDisplay();
    const ourColor: "white" | "black" = s.position.turn === 0 ? "white" : "black";
    // Render the working position (= position with pendingPlay applied)
    const pos = workingPosition(s);
    const dice = s.position.dice ?? [];
    let used: number[] = [];
    let remaining: number[] = [];
    if (phase.kind === "play") {
      const rolledAll = dice[0] === dice[1] ? [dice[0], dice[0], dice[0], dice[0]] : [dice[0], dice[1]];
      used = consumedDice(s.pendingPlay);
      remaining = remainingDice(rolledAll, s.pendingPlay);
    } else if (phase.kind === "opening" || phase.kind === "cpu-thinking") {
      const rolledAll = dice[0] === dice[1] ? [dice[0], dice[0], dice[0], dice[0]] : [dice[0], dice[1]];
      remaining = rolledAll;
    }

    const targets =
      phase.kind === "play" ? legalNextTargets(phase.legalPlays, s.pendingPlay) : new Map<number, Set<number>>();

    renderBoard(svg, pos, {
      flipped,
      selectedFrom: s.selectedFrom,
      legalDestsFrom: targets,
      diceRemaining: remaining,
      diceUsed: used,
      ourColor,
    });

    // Top bar: score / turn
    const wName = s.settings.whiteName;
    const bName = s.settings.blackName;
    scoreEl.textContent = `${wName} ${s.whiteScore} — ${s.blackScore} ${bName}`;
    turnLabel.innerHTML = turnLabelText(s);

    // Buttons
    rollBtn.disabled = !(phase.kind === "roll");
    rollBtn.textContent = phase.kind === "roll" ? "Roll" : "Roll";
    undoBtn.disabled = !(phase.kind === "play" && s.pendingPlay.length > 0);

    // Overlay (pass-screen, win, opening message)
    overlayContainer.innerHTML = "";
    if (phase.kind === "pass-overlay") {
      const div = document.createElement("div");
      div.className = "modal-backdrop";
      const m = document.createElement("div");
      m.className = "modal pass-screen";
      const name = phase.forSide === 0 ? wName : bName;
      m.innerHTML = `<h2>${escapeHtml(name)}'s turn</h2><p>Pass the device, then tap to begin.</p>`;
      const btn = document.createElement("button");
      btn.className = "primary";
      btn.textContent = "I'm ready";
      btn.addEventListener("click", () => controller.ackPassOverlay());
      m.appendChild(btn);
      div.appendChild(m);
      overlayContainer.appendChild(div);
    } else if (phase.kind === "won") {
      const div = document.createElement("div");
      div.className = "modal-backdrop";
      const m = document.createElement("div");
      m.className = "modal";
      const winnerName = phase.result.winner === 0 ? wName : bName;
      const kind = phase.result.kind === "single" ? "wins" : phase.result.kind === "gammon" ? "wins a gammon" : "wins a backgammon";
      m.innerHTML = `<h2>${escapeHtml(winnerName)} ${kind}!</h2>`;
      if (s.settings.tutorEnabled && s.tutor.history.length > 0) {
        const reportSlot = document.createElement("div");
        renderPostGameReport(reportSlot, s.tutor.history, wName, bName);
        m.appendChild(reportSlot);
      }
      const actions = document.createElement("div");
      actions.className = "actions";
      const matchOver = s.settings.matchLength > 1 && (s.whiteScore >= s.settings.matchLength || s.blackScore >= s.settings.matchLength);
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
      div.className = "modal-backdrop";
      const m = document.createElement("div");
      m.className = "modal";
      const winnerName = phase.winner === 0 ? wName : bName;
      m.innerHTML = `<h2>${escapeHtml(winnerName)} wins the match!</h2>
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

    // Tutor card
    if (s.settings.tutorEnabled && (phase.kind === "play" || phase.kind === "roll" || phase.kind === "cpu-thinking")) {
      renderTutorCard(tutorEl, s.tutor.lastEntry);
    } else {
      tutorEl.innerHTML = "";
    }
  }

  render();
}

function buildShell(root: HTMLElement): void {
  root.innerHTML = `
    <div class="topbar">
      <div class="title">Backgammon</div>
      <div class="score">—</div>
    </div>
    <div class="board-wrap">
      <svg class="board-svg" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
    <div class="controls">
      <div class="left">
        <button data-action="new">New game</button>
        <button data-action="undo">Undo</button>
      </div>
      <div class="turn-label">—</div>
      <div class="right">
        <div class="tutor-slot"></div>
        <button class="primary" data-action="roll">Roll</button>
      </div>
    </div>
    <div class="overlay"></div>
  `;
}

function turnLabelText(s: { phase: Phase; settings: GameSettings; position: { turn: 0 | 1 } }): string {
  const phase = s.phase;
  const name = s.position.turn === 0 ? s.settings.whiteName : s.settings.blackName;
  switch (phase.kind) {
    case "menu":
      return "—";
    case "opening":
      return `Opening roll: ${s.settings.whiteName} ${phase.whiteDie}, ${s.settings.blackName} ${phase.blackDie} → <strong>${escapeHtml(s.position.turn === 0 ? s.settings.whiteName : s.settings.blackName)}</strong> starts`;
    case "pass-overlay":
      return "—";
    case "roll":
      return `<strong>${escapeHtml(name)}</strong> to roll`;
    case "play":
      return `<strong>${escapeHtml(name)}</strong> to play`;
    case "cpu-thinking":
      return `<strong>${escapeHtml(name)}</strong> thinking…`;
    case "won":
      return "Game over";
    case "match-won":
      return "Match over";
  }
}

function svgPoint(svg: SVGSVGElement, clientX: number, clientY: number): { x: number; y: number } | null {
  const rect = svg.getBoundingClientRect();
  // Map client coords to viewBox coords
  const x = ((clientX - rect.left) / rect.width) * VIEW_W;
  const y = ((clientY - rect.top) / rect.height) * VIEW_H;
  return { x, y };
}

function openMenu(controller: GameController, root: HTMLElement): void {
  const node = showMenu(controller.state.settings, {
    onStart: (settings) => {
      controller.setSettings(settings);
      controller.startNewMatch();
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

function registerServiceWorker(): void {
  if (!("serviceWorker" in navigator)) return;
  if (location.protocol !== "https:" && location.hostname !== "localhost") return;
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => undefined);
  });
}

void main();
