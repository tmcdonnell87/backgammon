import { GameController } from "../game/controller";
import { Play, applyPlay } from "../engine/moves";
import { hashPosition } from "../engine/position";
import { buildHintTable, HintRow } from "./hint-table";

export interface HintModalOpts {
  /** Push hint arrows onto the board. Pass null to clear. */
  setPreviewHint: (moves: { from: number; to: number }[] | null) => void;
}

export async function openHintModal(
  controller: GameController,
  overlayContainer: HTMLElement,
  opts: HintModalOpts,
): Promise<void> {
  const s = controller.state;
  if (s.phase.kind !== "play") return;
  const legalPlays = s.phase.legalPlays;
  if (legalPlays.length === 0 || (legalPlays.length === 1 && legalPlays[0].length === 0)) {
    return;
  }

  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop hint-modal-backdrop";
  const modal = document.createElement("div");
  modal.className = "modal hint-modal";
  modal.innerHTML = `
    <h2>Best moves</h2>
    <div class="hint-body">Analyzing…</div>
    <div class="actions">
      <button data-action="play" class="primary" disabled>Play selected</button>
      <button data-action="close">Close</button>
    </div>
  `;
  backdrop.appendChild(modal);
  overlayContainer.appendChild(backdrop);

  let selectedPlay: Play | null = null;
  const playBtn = modal.querySelector<HTMLButtonElement>('[data-action="play"]')!;
  const closeBtn = modal.querySelector<HTMLButtonElement>('[data-action="close"]')!;

  const close = (): void => {
    opts.setPreviewHint(null);
    backdrop.remove();
  };
  closeBtn.addEventListener("click", close);
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) close();
  });
  playBtn.addEventListener("click", () => {
    if (!selectedPlay) return;
    const p = selectedPlay;
    close();
    controller.playFullPlay(p);
  });

  try {
    const analysisPromise = controller.getActiveAnalysis();
    if (!analysisPromise) throw new Error("No active analysis for this turn");
    const analysis = await analysisPromise;

    // Sort legal plays by equity desc, then deduplicate by final position.
    // Two plays like "7/3 6/3" and "6/3 7/3" reach the same final position
    // via different sub-move orderings and would otherwise show as two rows.
    const indices = legalPlays.map((_, i) => i);
    indices.sort((a, b) => analysis.equities[b] - analysis.equities[a]);
    const seen = new Set<string>();
    const dedupedIndices: number[] = [];
    for (const i of indices) {
      const h = hashPosition(applyPlay(s.position, legalPlays[i]));
      if (seen.has(h)) continue;
      seen.add(h);
      dedupedIndices.push(i);
    }
    const top = dedupedIndices.slice(0, 5);
    const bestEquity = analysis.bestEquity;

    const rows: HintRow[] = top.map((idx, i) => ({
      play: legalPlays[idx],
      equity: analysis.equities[idx],
      rank: i + 1,
      isBest: i === 0,
    }));

    const body = modal.querySelector<HTMLElement>(".hint-body")!;
    body.innerHTML = "";
    const table = buildHintTable({
      rows,
      bestEquity,
      onSelect: (row) => {
        selectedPlay = row.play;
        opts.setPreviewHint(row.play.map((sm) => ({ from: sm.from, to: sm.to })));
        playBtn.disabled = false;
      },
      defaultSelectIdx: 0,
    });
    body.appendChild(table);
  } catch (err) {
    const body = modal.querySelector<HTMLElement>(".hint-body")!;
    body.textContent = `Couldn't analyze: ${err instanceof Error ? err.message : String(err)}`;
  }
}
