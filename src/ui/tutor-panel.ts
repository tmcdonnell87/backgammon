import { TutorEntry } from "../game/controller";
import { Play } from "../engine/moves";
import { OFF, BAR } from "../engine/position";

export function renderTutorCard(parent: HTMLElement, entry: TutorEntry | undefined): void {
  parent.innerHTML = "";
  if (!entry) return;
  const card = document.createElement("div");
  card.className = `tutor-card ${entry.classification}`;
  const label = document.createElement("div");
  label.className = "label";
  label.textContent = labelFor(entry.classification, entry.equityLoss);
  card.appendChild(label);
  if (entry.classification !== "good") {
    const detail = document.createElement("div");
    detail.className = "detail";
    detail.textContent = `Best: ${describePlay(entry.bestPlay)}`;
    card.appendChild(detail);
  }
  parent.appendChild(card);
}

function labelFor(c: TutorEntry["classification"], loss: number): string {
  const lossStr = loss < 0.001 ? "" : ` · ${loss.toFixed(3)}`;
  switch (c) {
    case "good":
      return `Good${lossStr}`;
    case "doubtful":
      return `Doubtful${lossStr}`;
    case "error":
      return `Error${lossStr}`;
    case "blunder":
      return `Blunder${lossStr}`;
  }
}

export function describePlay(play: Play): string {
  if (play.length === 0) return "no move";
  return play.map(describeSub).join(", ");
}

function describeSub(sub: { from: number; to: number; die: number }): string {
  const from = sub.from === BAR ? "bar" : String(sub.from + 1);
  const to = sub.to === OFF ? "off" : String(sub.to + 1);
  return `${from}/${to}`;
}

export function renderPostGameReport(parent: HTMLElement, history: TutorEntry[], whiteName: string, blackName: string): void {
  parent.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "report";
  const totals: Record<0 | 1, { moves: number; loss: number; counts: Record<string, number> }> = {
    0: { moves: 0, loss: 0, counts: { good: 0, doubtful: 0, error: 0, blunder: 0 } },
    1: { moves: 0, loss: 0, counts: { good: 0, doubtful: 0, error: 0, blunder: 0 } },
  };
  for (const e of history) {
    const t = totals[e.side];
    t.moves++;
    t.loss += e.equityLoss;
    t.counts[e.classification]++;
  }
  for (const side of [0, 1] as const) {
    const t = totals[side];
    if (t.moves === 0) continue;
    const name = side === 0 ? whiteName : blackName;
    const pr = t.moves > 0 ? (t.loss / t.moves) * 500 : 0;
    const h = document.createElement("h3");
    h.textContent = `${name} — PR ${pr.toFixed(1)}`;
    wrap.appendChild(h);
    const tbl = document.createElement("table");
    tbl.innerHTML = `
      <tbody>
        <tr><td>Moves</td><td>${t.moves}</td></tr>
        <tr><td>Total equity lost</td><td>${t.loss.toFixed(3)}</td></tr>
        <tr><td>Good</td><td>${t.counts.good}</td></tr>
        <tr><td>Doubtful</td><td>${t.counts.doubtful}</td></tr>
        <tr><td>Errors</td><td>${t.counts.error}</td></tr>
        <tr><td>Blunders</td><td>${t.counts.blunder}</td></tr>
      </tbody>
    `;
    wrap.appendChild(tbl);
  }
  parent.appendChild(wrap);
}
