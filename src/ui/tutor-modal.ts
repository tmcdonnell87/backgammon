import { TutorEntry } from "../game/controller";
import { Play, applyPlay } from "../engine/moves";
import { Position, hashPosition } from "../engine/position";
import { buildHintTable, HintRow } from "./hint-table";

export interface TutorModalOpts {
  setTutorPreview: (
    pre: { startPos: Position; side: 0 | 1; hintMoves: { from: number; to: number }[] | null } | null,
  ) => void;
  onDismiss: () => void;
}

export function openTutorModal(
  entry: TutorEntry,
  overlayContainer: HTMLElement,
  opts: TutorModalOpts,
): void {
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop tutor-modal-backdrop";
  const modal = document.createElement("div");
  modal.className = `modal tutor-analysis-modal ${entry.classification}`;
  const head = entry.classification === "blunder" ? "Blunder" : "Error";
  modal.innerHTML = `
    <h2 class="tutor-head">${head}</h2>
    <div class="hint-body"></div>
    <div class="actions">
      <button data-action="ok" class="primary">Got it</button>
    </div>
  `;
  backdrop.appendChild(modal);
  overlayContainer.appendChild(backdrop);

  // Rank plays by equity desc and dedup by final position. Numbering follows
  // the sorted order — Best = #1, second-best = #2, etc. If the user's move
  // is best, their row appears at #1 with "you · best" tags. If the user's
  // move is suboptimal, hoist a copy of their row above the numbered list
  // (with no rank number) so the modal opens on the move the user actually
  // played, with numbered alternatives below.
  const yourFinalHash = hashPosition(applyPlay(entry.startPos, entry.yourPlay));
  const indices = entry.legalPlays.map((_, i) => i);
  indices.sort((a, b) => entry.equities[b] - entry.equities[a]);
  const seen = new Set<string>();
  type SortedRow = { play: Play; equity: number; isYours: boolean; rank: number };
  const sortedAll: SortedRow[] = [];
  let nextRank = 1;
  for (const i of indices) {
    const play = entry.legalPlays[i];
    const h = hashPosition(applyPlay(entry.startPos, play));
    if (seen.has(h)) continue;
    seen.add(h);
    const isYours = h === yourFinalHash;
    sortedAll.push({
      play: isYours ? entry.yourPlay : play,
      equity: entry.equities[i],
      isYours,
      rank: nextRank++,
    });
  }
  // Top 5 plus a guaranteed user row if theirs fell outside.
  const top: SortedRow[] = sortedAll.slice(0, 5);
  if (!top.some((r) => r.isYours)) {
    const userRow = sortedAll.find((r) => r.isYours);
    if (userRow) top.push(userRow);
    else {
      top.push({ play: entry.yourPlay, equity: -Infinity, isYours: true, rank: -1 });
    }
  }
  const bestEquity = sortedAll.length > 0 ? sortedAll[0].equity : 0;
  const userRow = top.find((r) => r.isYours);
  const userIsBest = userRow !== undefined && userRow.rank === 1;

  // If user isn't already at the top (because they're not the best), hoist
  // a copy of their row to the top with showRank=false; the original
  // numbered row is filtered out so the user's move shows exactly once.
  const renderedRows: HintRow[] = [];
  if (userRow && !userIsBest) {
    renderedRows.push({
      play: userRow.play,
      equity: userRow.equity,
      rank: 0, // suppresses the rank cell
      isBest: false,
      isYours: true,
    });
    for (const r of top) {
      if (r === userRow) continue;
      renderedRows.push({
        play: r.play,
        equity: r.equity,
        rank: r.rank,
        isBest: r.rank === 1,
        isYours: r.isYours,
      });
    }
  } else {
    for (const r of top) {
      renderedRows.push({
        play: r.play,
        equity: r.equity,
        rank: r.rank,
        isBest: r.rank === 1,
        isYours: r.isYours,
      });
    }
  }

  const body = modal.querySelector<HTMLElement>(".hint-body")!;
  const table = buildHintTable({
    rows: renderedRows,
    bestEquity,
    onSelect: (row) => {
      opts.setTutorPreview({
        startPos: entry.startPos,
        side: entry.side as 0 | 1,
        hintMoves: row.play.map((sm) => ({ from: sm.from, to: sm.to })),
      });
    },
    // Default to the user's row — that's row 0 since we hoisted it (or
    // it was already at rank 1).
    defaultSelectIdx: 0,
  });
  body.appendChild(table);

  let dismissed = false;
  const close = (): void => {
    if (dismissed) return;
    dismissed = true;
    opts.setTutorPreview(null);
    backdrop.remove();
    opts.onDismiss();
  };
  modal.querySelector<HTMLButtonElement>('[data-action="ok"]')!.addEventListener("click", close);
}
