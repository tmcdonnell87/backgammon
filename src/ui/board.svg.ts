import { POINTS, Position } from "../engine/position";
import {
  BAR_END,
  BAR_START,
  BAR_TOP,
  BAR_BOTTOM,
  BOT_POINT_APEX,
  CHECKER_R,
  CHECKER_D,
  LEFT_HALF_START,
  RIGHT_HALF_END,
  RIGHT_HALF_START,
  TOP_POINT_APEX,
  TRAY_END,
  TRAY_START,
  VIEW_H,
  VIEW_W,
  barGeometry,
  checkerCenter,
  pointGeometry,
  trayGeometry,
} from "./layout";

const SVG_NS = "http://www.w3.org/2000/svg";

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  return e;
}

export interface RenderOpts {
  flipped: boolean; // if true, render mirrored (so the absolute opponent of "us" is at the bottom)
  selectedFrom: number | null; // currently picked-up source point (in us-coords)
  legalDestsFrom: Map<number, Set<number>>; // from -> Set<to> (us-coords)
  diceRemaining: number[]; // unused dice values
  diceUsed: number[]; // used dice values
  ourColor: "white" | "black"; // absolute color of "us" (player on roll)
}

const COLOR_WHITE = "var(--checker-w)";
const COLOR_WHITE_EDGE = "var(--checker-w-edge)";
const COLOR_BLACK = "var(--checker-b)";
const COLOR_BLACK_EDGE = "var(--checker-b-edge)";

function checkerColor(absSide: "white" | "black"): { fill: string; stroke: string; textFill: string } {
  return absSide === "white"
    ? { fill: COLOR_WHITE, stroke: COLOR_WHITE_EDGE, textFill: "#1a1d24" }
    : { fill: COLOR_BLACK, stroke: COLOR_BLACK_EDGE, textFill: "#f5f1e6" };
}

export function renderBoard(svg: SVGSVGElement, p: Position, opts: RenderOpts): void {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  svg.setAttribute("viewBox", `0 0 ${VIEW_W} ${VIEW_H}`);
  svg.setAttribute("class", "board-svg");
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  // Outer wood frame
  const frame = el("rect", { x: 0, y: 0, width: VIEW_W, height: VIEW_H, fill: "var(--board-edge)", rx: 12 });
  svg.appendChild(frame);
  const inner = el("rect", {
    x: 6,
    y: 6,
    width: VIEW_W - 12,
    height: VIEW_H - 12,
    fill: "var(--board)",
    rx: 8,
  });
  svg.appendChild(inner);

  // Bar
  const bar = el("rect", {
    x: BAR_START,
    y: 6,
    width: BAR_END - BAR_START,
    height: VIEW_H - 12,
    fill: "var(--bar)",
  });
  svg.appendChild(bar);

  // Tray
  const tray = el("rect", {
    x: TRAY_START,
    y: 6,
    width: TRAY_END - TRAY_START,
    height: VIEW_H - 12,
    fill: "var(--bar)",
    rx: 4,
  });
  svg.appendChild(tray);

  // Points (triangles)
  for (let idx = 0; idx < POINTS; idx++) {
    const g = pointGeometry(idx, opts.flipped);
    const left = g.cx - 28;
    const right = g.cx + 28;
    // Alternate light/dark by visual column (so adjacent points differ)
    const visualCol = g.isTop ? Math.round((g.cx - LEFT_HALF_START) / 64) : Math.round((g.cx - LEFT_HALF_START) / 64);
    const isLight = (visualCol + (g.isTop ? 0 : 1)) % 2 === 0;
    const fill = isLight ? "var(--point-light)" : "var(--point-dark)";
    const tri = el("polygon", {
      points: `${left},${g.baseY} ${right},${g.baseY} ${g.cx},${g.apexY}`,
      fill,
      stroke: "rgba(0,0,0,0.25)",
      "stroke-width": 1,
    });
    tri.setAttribute("data-point", String(idx));
    svg.appendChild(tri);
  }

  // Highlight: selected source
  if (opts.selectedFrom !== null && opts.selectedFrom >= 0 && opts.selectedFrom < 24) {
    const g = pointGeometry(opts.selectedFrom, opts.flipped);
    svg.appendChild(
      el("polygon", {
        points: `${g.cx - 28},${g.baseY} ${g.cx + 28},${g.baseY} ${g.cx},${g.apexY}`,
        fill: "none",
        stroke: "var(--highlight)",
        "stroke-width": 4,
        "pointer-events": "none",
      }),
    );
  }
  // Highlight: legal destinations (or all legal sources when no selection)
  const showAsTargets =
    opts.selectedFrom !== null
      ? opts.legalDestsFrom.get(opts.selectedFrom) ?? new Set<number>()
      : new Set<number>();
  for (const t of showAsTargets) {
    if (t === -1) {
      // OFF tray
      const g = trayGeometry("us", opts.flipped);
      svg.appendChild(
        el("rect", {
          x: TRAY_START,
          y: g.topY - 6,
          width: TRAY_END - TRAY_START,
          height: g.bottomY - g.topY + 12,
          fill: "none",
          stroke: "var(--legal)",
          "stroke-width": 4,
          rx: 6,
          "pointer-events": "none",
        }),
      );
    } else {
      const g = pointGeometry(t, opts.flipped);
      svg.appendChild(
        el("circle", {
          cx: g.cx,
          cy: (g.baseY + g.apexY) / 2,
          r: 12,
          fill: "var(--legal)",
          opacity: 0.7,
          "pointer-events": "none",
        }),
      );
    }
  }
  // Highlight: legal sources when nothing picked up
  if (opts.selectedFrom === null) {
    for (const from of opts.legalDestsFrom.keys()) {
      if (from >= 0 && from < 24) {
        const g = pointGeometry(from, opts.flipped);
        svg.appendChild(
          el("polygon", {
            points: `${g.cx - 28},${g.baseY} ${g.cx + 28},${g.baseY} ${g.cx},${g.apexY}`,
            fill: "var(--highlight)",
            opacity: 0.15,
            "pointer-events": "none",
          }),
        );
      } else if (from === 24) {
        // Bar
        const cx = (BAR_START + BAR_END) / 2;
        const y0 = !opts.flipped ? BAR_TOP : (BAR_TOP + BAR_BOTTOM) / 2;
        const y1 = !opts.flipped ? (BAR_TOP + BAR_BOTTOM) / 2 : BAR_BOTTOM;
        svg.appendChild(
          el("rect", {
            x: cx - CHECKER_R - 4,
            y: y0 - 4,
            width: 2 * (CHECKER_R + 4),
            height: y1 - y0 + 8,
            fill: "var(--highlight)",
            opacity: 0.18,
            "pointer-events": "none",
          }),
        );
      }
    }
  }

  // Checkers on points
  const ourC = checkerColor(opts.ourColor);
  const theirC = checkerColor(opts.ourColor === "white" ? "black" : "white");
  for (let idx = 0; idx < POINTS; idx++) {
    const v = p.points[idx];
    if (v === 0) continue;
    const count = Math.abs(v);
    const c = v > 0 ? ourC : theirC;
    const visible = Math.min(count, 5);
    for (let i = 0; i < visible; i++) {
      const { cx, cy } = checkerCenter(idx, i, opts.flipped);
      svg.appendChild(
        el("circle", {
          cx,
          cy,
          r: CHECKER_R,
          fill: c.fill,
          stroke: c.stroke,
          "stroke-width": 1.5,
        }),
      );
    }
    if (count > 5) {
      const { cx, cy } = checkerCenter(idx, 4, opts.flipped);
      const text = el("text", {
        x: cx,
        y: cy + 5,
        "text-anchor": "middle",
        "font-size": 18,
        "font-weight": 700,
        fill: c.textFill,
        "pointer-events": "none",
      });
      text.textContent = String(count);
      svg.appendChild(text);
    }
  }

  // Bar checkers
  const renderBarStack = (count: number, side: "us" | "them"): void => {
    if (count <= 0) return;
    const g = barGeometry(side, opts.flipped);
    const c = side === "us" ? ourC : theirC;
    const visible = Math.min(count, 5);
    const startY = g.topY;
    for (let i = 0; i < visible; i++) {
      svg.appendChild(
        el("circle", {
          cx: g.cx,
          cy: startY + i * CHECKER_D,
          r: CHECKER_R,
          fill: c.fill,
          stroke: c.stroke,
          "stroke-width": 1.5,
        }),
      );
    }
    if (count > 5) {
      const t = el("text", {
        x: g.cx,
        y: startY + 4 * CHECKER_D + 5,
        "text-anchor": "middle",
        "font-size": 18,
        "font-weight": 700,
        fill: c.textFill,
        "pointer-events": "none",
      });
      t.textContent = String(count);
      svg.appendChild(t);
    }
  };
  renderBarStack(p.barUs, "us");
  renderBarStack(p.barThem, "them");

  // Tray (borne off) — small horizontal slabs accumulating from outside in
  const renderTray = (count: number, side: "us" | "them"): void => {
    if (count <= 0) return;
    const g = trayGeometry(side, opts.flipped);
    const c = side === "us" ? ourC : theirC;
    const slotH = (g.bottomY - g.topY) / 15;
    const atBottom = g.topY > VIEW_H / 2;
    for (let i = 0; i < count; i++) {
      const y = atBottom ? g.bottomY - (i + 1) * slotH : g.topY + i * slotH;
      svg.appendChild(
        el("rect", {
          x: TRAY_START + 6,
          y,
          width: TRAY_END - TRAY_START - 12,
          height: slotH - 2,
          fill: c.fill,
          stroke: c.stroke,
          rx: 3,
        }),
      );
    }
  };
  renderTray(p.offUs, "us");
  renderTray(p.offThem, "them");

  // Dice — render in the center of the active half (right half, since we move toward our 1-pt)
  // For visibility, place dice in the center of the bottom half if not flipped, top half if flipped.
  if (opts.diceRemaining.length > 0 || opts.diceUsed.length > 0) {
    const allDice = [...opts.diceRemaining.map((d) => ({ d, used: false })), ...opts.diceUsed.map((d) => ({ d, used: true }))];
    const diceY = !opts.flipped ? (BAR_BOTTOM + BOT_POINT_APEX) / 2 + 30 : (TOP_POINT_APEX + BAR_TOP) / 2 - 30;
    const diceCenterX = (RIGHT_HALF_START + RIGHT_HALF_END) / 2;
    const dieSize = 40;
    const dieGap = 12;
    const total = allDice.length * dieSize + (allDice.length - 1) * dieGap;
    let x = diceCenterX - total / 2;
    for (const die of allDice) {
      svg.appendChild(renderDie(die.d, x, diceY - dieSize / 2, dieSize, die.used));
      x += dieSize + dieGap;
    }
  }
}

function renderDie(value: number, x: number, y: number, size: number, used: boolean): SVGGElement {
  const g = el("g", {});
  const fill = used ? "rgba(245, 241, 230, 0.35)" : "var(--checker-w)";
  const stroke = used ? "rgba(0,0,0,0.3)" : "var(--checker-w-edge)";
  const rect = el("rect", {
    x,
    y,
    width: size,
    height: size,
    rx: 6,
    fill,
    stroke,
    "stroke-width": 1.5,
  });
  g.appendChild(rect);
  // Pip positions in a 3x3 grid
  const pad = size * 0.22;
  const r = size * 0.07;
  const cx = (col: number) => x + pad + ((size - 2 * pad) * col) / 2;
  const cy = (row: number) => y + pad + ((size - 2 * pad) * row) / 2;
  const pipMap: Record<number, [number, number][]> = {
    1: [[1, 1]],
    2: [
      [0, 0],
      [2, 2],
    ],
    3: [
      [0, 0],
      [1, 1],
      [2, 2],
    ],
    4: [
      [0, 0],
      [2, 0],
      [0, 2],
      [2, 2],
    ],
    5: [
      [0, 0],
      [2, 0],
      [1, 1],
      [0, 2],
      [2, 2],
    ],
    6: [
      [0, 0],
      [2, 0],
      [0, 1],
      [2, 1],
      [0, 2],
      [2, 2],
    ],
  };
  const pipFill = used ? "rgba(0,0,0,0.4)" : "#1a1d24";
  for (const [c, r2] of pipMap[value] ?? []) {
    g.appendChild(el("circle", { cx: cx(c), cy: cy(r2), r, fill: pipFill }));
  }
  return g;
}
