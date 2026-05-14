// Shared "ranked plays" table used by the hint modal (pre-commit) and the
// tutor analysis modal (post-commit error/blunder). Both render the same
// rank · moves · equity · loss columns; only the row interactions differ.

import { Play } from "../engine/moves";
import { BAR } from "../engine/position";

export interface HintRow {
  play: Play;
  equity: number;
  /** 1-based rank shown in the first column. Use <= 0 to render the cell blank. */
  rank: number;
  isBest: boolean;
  isYours?: boolean;
}

export interface HintTableOpts {
  rows: HintRow[];
  /** Best equity among all rows — used to compute each row's loss. */
  bestEquity: number;
  /** Fires when a row is clicked (after the "selected" highlight has been moved). */
  onSelect?: (row: HintRow, index: number) => void;
  /** If set, this row index is clicked once after the table is built. */
  defaultSelectIdx?: number;
}

export function buildHintTable(opts: HintTableOpts): HTMLTableElement {
  const table = document.createElement("table");
  table.className = "hint-table";
  const trs: HTMLTableRowElement[] = [];

  opts.rows.forEach((r, i) => {
    const tr = document.createElement("tr");
    tr.className = ["hint-row", r.isBest ? "best" : "", r.isYours ? "yours" : ""]
      .filter(Boolean)
      .join(" ");
    tr.title = "Tap to preview on the board";

    const rank = document.createElement("td");
    rank.className = "rank";
    rank.textContent = r.rank > 0 ? String(r.rank) : "";

    const moves = document.createElement("td");
    moves.className = "moves";
    const tags: string[] = [];
    if (r.isYours) tags.push("you");
    if (r.isBest) tags.push("best");
    moves.textContent = tags.length > 0 ? `${formatPlay(r.play)}  · ${tags.join(" · ")}` : formatPlay(r.play);

    const equity = document.createElement("td");
    equity.className = "equity";
    equity.textContent = Number.isFinite(r.equity) ? formatEquity(r.equity) : "—";

    const loss = document.createElement("td");
    loss.className = "loss";
    const eqLoss = opts.bestEquity - r.equity;
    loss.textContent = r.isBest
      ? "—"
      : !Number.isFinite(r.equity) || eqLoss < 0.001
        ? "—"
        : `−${eqLoss.toFixed(3)}`;

    tr.appendChild(rank);
    tr.appendChild(moves);
    tr.appendChild(equity);
    tr.appendChild(loss);

    tr.addEventListener("click", () => {
      for (const other of trs) other.classList.remove("selected");
      tr.classList.add("selected");
      opts.onSelect?.(r, i);
    });

    trs.push(tr);
    table.appendChild(tr);
  });

  if (
    opts.defaultSelectIdx !== undefined
    && opts.defaultSelectIdx >= 0
    && opts.defaultSelectIdx < trs.length
  ) {
    trs[opts.defaultSelectIdx].click();
  }

  return table;
}

function formatPlay(play: Play): string {
  if (play.length === 0) return "(no move)";
  return play.map((sm) => `${formatPoint(sm.from)}/${formatPoint(sm.to)}`).join(" ");
}

function formatPoint(idx: number): string {
  if (idx === BAR) return "bar";
  if (idx === -1) return "off";
  return String(idx + 1);
}

function formatEquity(eq: number): string {
  const v = eq.toFixed(2);
  return eq >= 0 ? `+${v}` : v;
}
