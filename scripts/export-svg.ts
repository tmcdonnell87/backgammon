// Export faithful, standalone .svg snapshots of the in-game board art so they
// can be dropped straight into Claude Design (claude.ai/design) for critique
// and redesign. Everything here renders through the REAL renderer in
// src/ui/board.svg.ts — no shapes/colors are re-implemented, so the exported
// files match what ships in-game pixel-for-pixel.
//
// How it works: board.svg.ts draws via the GLOBAL `document` (its el() helper
// calls document.createElementNS). We give it a DOM by booting a happy-dom
// Window (already a devDependency — no new installs) and binding it onto
// globalThis before any render call. The board's gradients/filters live in an
// inline <defs> that the renderer injects itself, so a single synchronous
// renderBoard() call yields a fully self-contained SVG. The only non-defs
// dependency is four CSS custom properties from src/style.css :root, which we
// hard-substitute to hex on the way out so the files render identically in any
// viewer (Claude Design, a browser, an SVG->PNG converter).
//
// Regenerate with:  pnpm export:svg   (output: design/assets/*.svg)

import { writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { Window } from "happy-dom";
import {
  renderBoard,
  makeChecker,
  renderDie,
  buildDefs,
  type RenderOpts,
} from "../src/ui/board.svg";
import { makeLayout } from "../src/ui/layout";
import {
  startingPosition,
  pipCount,
  pipCountThem,
  type Position,
} from "../src/engine/position";

const SVG_NS = "http://www.w3.org/2000/svg";

// Boot a DOM and bind it globally so board.svg.ts's el() (global document) works.
const win = new Window();
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).window = win;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).document = win.document;

const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "design", "assets");
mkdirSync(OUT, { recursive: true });

// src/style.css :root — the four custom properties the SVG references inline.
// Hard-substituted to hex so the standalone files are viewer-independent.
const CSS_VARS: Array<[string, string]> = [
  ["--board", "#5a3a2a"],
  ["--highlight", "#ffd24a"],
  ["--legal", "#6dd47e"],
  ["--hint", "#6ec9ff"],
];

function serialize(svg: SVGSVGElement): string {
  svg.setAttribute("xmlns", SVG_NS);
  let s = svg.outerHTML;
  for (const [name, hex] of CSS_VARS) s = s.split(`var(${name})`).join(hex);
  return `<?xml version="1.0" encoding="UTF-8"?>\n${s}\n`;
}

function write(name: string, svg: SVGSVGElement): void {
  writeFileSync(join(OUT, name), serialize(svg));
  // eslint-disable-next-line no-console
  console.log("wrote design/assets/" + name);
}

function newSvg(): SVGSVGElement {
  return document.createElementNS(SVG_NS, "svg") as unknown as SVGSVGElement;
}

// Build a Position literal from a sparse points map (+N = our/white, -N = opp/black).
function mkPos(spec: {
  pts: Record<number, number>;
  barUs?: number;
  barThem?: number;
  offUs?: number;
  offThem?: number;
}): Position {
  const points = new Int8Array(24);
  for (const k of Object.keys(spec.pts)) points[Number(k)] = spec.pts[Number(k)];
  return {
    points,
    barUs: spec.barUs ?? 0,
    barThem: spec.barThem ?? 0,
    offUs: spec.offUs ?? 0,
    offThem: spec.offThem ?? 0,
    turn: 0,
    dice: null,
    cube: { value: 1, owner: null },
    score: [0, 0],
    matchLength: 1,
    crawford: false,
  };
}

// Render a full board with sensible defaults; `opts` overrides any field.
function board(name: string, pos: Position, opts: Partial<RenderOpts> = {}): void {
  const svg = newSvg();
  renderBoard(svg, pos, {
    layout: makeLayout(),
    flipped: false,
    selectedFrom: null,
    legalDestsFrom: new Map(),
    dice: [],
    ourColor: "white",
    ...opts,
  });
  write(name, svg);
}

// Isolated single-asset snapshot: gradients/filters from the real buildDefs(),
// a felt backdrop so both light and dark pieces read on the design canvas, then
// the caller-supplied element(s).
function isolated(name: string, size: number, build: (svg: SVGSVGElement) => void): void {
  const svg = newSvg();
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
  svg.setAttribute("class", "board-svg");
  svg.appendChild(buildDefs());
  const bg = document.createElementNS(SVG_NS, "rect");
  bg.setAttribute("x", "0");
  bg.setAttribute("y", "0");
  bg.setAttribute("width", String(size));
  bg.setAttribute("height", String(size));
  bg.setAttribute("rx", "8");
  bg.setAttribute("fill", "var(--board)");
  svg.appendChild(bg);
  build(svg);
  write(name, svg);
}

// ---- Full-board scenes -----------------------------------------------------

// 1. Opening position — clean board, both checker colors, points, empty trays.
board("board-start.svg", startingPosition());

// 2. Active mid-game — exercises the widest set of elements at once: a stack
//    >5 (count badge), a checker on the bar, borne-off tray slabs (both sides),
//    an opponent blot, a selected source with legal-destination markers, used +
//    unused dice, live equity plaque, and pip-count labels.
const active = mkPos({
  pts: { 5: 7, 7: 3, 12: 2, 0: -2, 11: -4, 16: -3, 18: -3, 19: -1 },
  barUs: 1,
  offUs: 2,
  offThem: 2,
});
board("board-active.svg", active, {
  selectedFrom: 7,
  legalDestsFrom: new Map([[7, new Set([1, 4])]]),
  dice: [
    { d: 6, used: false },
    { d: 3, used: true },
  ],
  pipCount: { us: pipCount(active), them: pipCountThem(active) },
  equity: 0.42,
});

// 3. Hint preview — opening 3-1 making the 5-point (8/5 6/5): dashed hint arrows
//    + endpoint dots, the "Thinking…" indicator, dice, equity.
const hint = startingPosition();
board("board-hint.svg", hint, {
  dice: [
    { d: 3, used: false },
    { d: 1, used: false },
  ],
  hintMoves: [
    { from: 7, to: 4 },
    { from: 5, to: 4 },
  ],
  cpuThinking: true,
  equity: 0.01,
  pipCount: { us: pipCount(hint), them: pipCountThem(hint) },
});

// 4. Bear-off endgame — heavily populated trays (slabs, both colors) + low home points.
const bearoff = mkPos({
  pts: { 0: 2, 1: 2, 2: 1, 22: -2, 23: -1 },
  offUs: 10,
  offThem: 12,
});
board("board-bearoff.svg", bearoff, {
  dice: [
    { d: 5, used: false },
    { d: 4, used: false },
  ],
  equity: 0.85,
  pipCount: { us: pipCount(bearoff), them: pipCountThem(bearoff) },
});

// ---- Isolated single-asset scenes -----------------------------------------

isolated("checker-white.svg", 64, (svg) => svg.appendChild(makeChecker(32, 32, "white")));
isolated("checker-black.svg", 64, (svg) => svg.appendChild(makeChecker(32, 32, "black")));
isolated("die-white-5.svg", 72, (svg) => svg.appendChild(renderDie(5, 8, 8, 56, false, "white")));
isolated("die-black-3.svg", 72, (svg) => svg.appendChild(renderDie(3, 8, 8, 56, false, "black")));
isolated("die-white-used-2.svg", 72, (svg) => svg.appendChild(renderDie(2, 8, 8, 56, true, "white")));

// eslint-disable-next-line no-console
console.log(`\nDone. ${OUT}`);
