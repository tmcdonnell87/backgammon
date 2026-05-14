import { POINTS, Position } from "../engine/position";
import { SubMove } from "../engine/moves";
import {
  BoardLayout,
  CHECKER_R,
  CHECKER_PITCH,
  DEFAULT_LAYOUT,
} from "./layout";

const SVG_NS = "http://www.w3.org/2000/svg";

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  return e;
}

export interface RenderOpts {
  layout: BoardLayout;
  flipped: boolean;
  selectedFrom: number | null;
  legalDestsFrom: Map<number, Set<number>>;
  dice: { d: number; used: boolean }[];
  ourColor: "white" | "black";
  hideTopFrom?: number | null;
  diceCue?: "swap" | "confirm" | null;
  hintMoves?: { from: number; to: number }[];
  // True while the CPU is computing its play. Draws a soft pulsing "Thinking…"
  // text under the dice so the user knows the app hasn't frozen.
  cpuThinking?: boolean;
  // Live pip counts in us-perspective ({us, them}). Rendered as small labels
  // inside each player's tray near the equity bar when showPipCount is on.
  pipCount?: { us: number; them: number } | null;
  // Live equity in absolute white POV (positive = white is ahead, negative =
  // black is ahead). Rendered in the wood bar between the trays as the
  // absolute value in the leader's checker color.
  equity?: number | null;
}

function buildDefs(): SVGDefsElement {
  const defs = el("defs");
  defs.innerHTML = `
    <linearGradient id="lg-point-light" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#e6c89a"/>
      <stop offset="55%" stop-color="#c39862"/>
      <stop offset="100%" stop-color="#8d6234"/>
    </linearGradient>
    <linearGradient id="lg-point-dark" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#5a2f15"/>
      <stop offset="55%" stop-color="#361a08"/>
      <stop offset="100%" stop-color="#1a0c04"/>
    </linearGradient>
    <!-- Checker gradients: gradient circle centered (cx/cy=0.5) so the
         shading wraps the full disk evenly, but focal point at (0.32,0.30)
         places the brightest spot upper-left. Two stops, low contrast —
         reads as a flat playing piece with a soft directional light cue,
         not a 3D marble. -->
    <radialGradient id="rg-checker-white" cx="0.5" cy="0.5" r="0.7" fx="0.32" fy="0.30">
      <stop offset="0%" stop-color="#ebe2c5"/>
      <stop offset="100%" stop-color="#a8987a"/>
    </radialGradient>
    <radialGradient id="rg-checker-black" cx="0.5" cy="0.5" r="0.7" fx="0.32" fy="0.30">
      <stop offset="0%" stop-color="#42414b"/>
      <stop offset="100%" stop-color="#12111a"/>
    </radialGradient>
    <radialGradient id="rg-checker-white-used" cx="0.5" cy="0.5" r="0.7" fx="0.32" fy="0.30">
      <stop offset="0%" stop-color="#c5bca0"/>
      <stop offset="100%" stop-color="#7d7256"/>
    </radialGradient>
    <radialGradient id="rg-checker-black-used" cx="0.5" cy="0.5" r="0.7" fx="0.32" fy="0.30">
      <stop offset="0%" stop-color="#2e2d36"/>
      <stop offset="100%" stop-color="#0a090f"/>
    </radialGradient>
    <linearGradient id="lg-bar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#150a05"/>
      <stop offset="20%" stop-color="#3a2418"/>
      <stop offset="50%" stop-color="#4a2e1c"/>
      <stop offset="80%" stop-color="#3a2418"/>
      <stop offset="100%" stop-color="#150a05"/>
    </linearGradient>
    <linearGradient id="lg-bar-wood" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#2a1a0e"/>
      <stop offset="50%" stop-color="#4a2e1c"/>
      <stop offset="100%" stop-color="#2a1a0e"/>
    </linearGradient>
    <linearGradient id="lg-tray-felt" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0e0703"/>
      <stop offset="50%" stop-color="#1c1108"/>
      <stop offset="100%" stop-color="#0e0703"/>
    </linearGradient>
    <linearGradient id="lg-tray" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#150a05"/>
      <stop offset="50%" stop-color="#3a2418"/>
      <stop offset="100%" stop-color="#150a05"/>
    </linearGradient>
    <linearGradient id="lg-wood" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4a2614"/>
      <stop offset="35%" stop-color="#6a3a22"/>
      <stop offset="65%" stop-color="#5a3019"/>
      <stop offset="100%" stop-color="#321609"/>
    </linearGradient>
    <linearGradient id="lg-die-white" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#e2d8b8"/>
      <stop offset="100%" stop-color="#b3a884"/>
    </linearGradient>
    <linearGradient id="lg-die-black" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#34343c"/>
      <stop offset="100%" stop-color="#16161c"/>
    </linearGradient>
    <linearGradient id="lg-die-white-used" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#bbb59a"/>
      <stop offset="100%" stop-color="#807760"/>
    </linearGradient>
    <linearGradient id="lg-die-black-used" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#2a2a32"/>
      <stop offset="100%" stop-color="#0c0a0e"/>
    </linearGradient>
    <radialGradient id="rg-pip-dark" cx="0.3" cy="0.3" r="0.7">
      <stop offset="0%" stop-color="#3a3528"/>
      <stop offset="100%" stop-color="#0c0a06"/>
    </radialGradient>
    <radialGradient id="rg-pip-light" cx="0.3" cy="0.3" r="0.7">
      <stop offset="0%" stop-color="#fffdf3"/>
      <stop offset="100%" stop-color="#bbb59a"/>
    </radialGradient>
    <filter id="f-shadow" x="-50%" y="-50%" width="200%" height="200%">
      <feDropShadow dx="0.5" dy="1" stdDeviation="1" flood-color="#000" flood-opacity="0.40"/>
    </filter>
    <filter id="f-die-shadow" x="-50%" y="-50%" width="200%" height="200%">
      <feDropShadow dx="2" dy="3" stdDeviation="2" flood-color="#000" flood-opacity="0.55"/>
    </filter>
    <filter id="f-soft-shadow" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="1.5" stdDeviation="1" flood-color="#000" flood-opacity="0.4"/>
    </filter>
    <!-- Wood grain — wide, smoothly-varying bands rather than tight noise.
         Two layers: a slow horizontal-banding turbulence for the "rings",
         displaced gently to suggest organic flow. Output color is dark
         walnut at low alpha so the underlying lg-wood gradient dominates. -->
    <filter id="f-wood-grain" x="0" y="0" width="100%" height="100%">
      <feTurbulence type="turbulence" baseFrequency="0.025 0.6" numOctaves="3" seed="9" result="noise"/>
      <feColorMatrix in="noise" type="matrix" values="0 0 0 0 0.12
                                                       0 0 0 0 0.05
                                                       0 0 0 0 0.02
                                                       0 0 0 0.35 0"/>
    </filter>
    <!-- Secondary fine-grain pass for subtle high-frequency texture
         (knot-like flecks) at very low opacity. -->
    <filter id="f-wood-fleck" x="0" y="0" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="0.85 0.04" numOctaves="2" seed="3" result="noise"/>
      <feColorMatrix in="noise" type="matrix" values="0 0 0 0 0
                                                       0 0 0 0 0
                                                       0 0 0 0 0
                                                       0 0 0 0.18 0"/>
    </filter>
    <filter id="f-felt-grain" x="0" y="0" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="2.2" numOctaves="1" seed="11" result="noise"/>
      <feColorMatrix in="noise" type="matrix" values="0 0 0 0 0
                                                       0 0 0 0 0
                                                       0 0 0 0 0
                                                       0 0 0 0.12 0"/>
    </filter>
  `;
  return defs as SVGDefsElement;
}

export function renderBoard(svg: SVGSVGElement, p: Position, opts: RenderOpts): void {
  const L = opts.layout;
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  svg.setAttribute("viewBox", `0 0 ${L.viewW} ${L.viewH}`);
  svg.setAttribute("class", "board-svg");
  // viewBox aspect now matches the container aspect (the container is measured
  // and viewW computed accordingly), so meet/slice/none all collapse to the
  // same thing. We keep the standard "meet" so circles stay perfectly round
  // even if the container observer fires slightly behind a resize.
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  svg.setAttribute("shape-rendering", "geometricPrecision");
  svg.setAttribute("text-rendering", "geometricPrecision");

  svg.appendChild(buildDefs());

  // Outer wood frame — square corners (traditional rectangular wood box),
  // base gradient + two grain layers: wide banding (rings) and very subtle
  // fleck. Both at low alpha so the gradient color dominates.
  svg.appendChild(
    el("rect", { x: 0, y: 0, width: L.viewW, height: L.viewH, fill: "url(#lg-wood)" }),
  );
  svg.appendChild(
    el("rect", {
      x: 0,
      y: 0,
      width: L.viewW,
      height: L.viewH,
      fill: "#6a3a22",
      filter: "url(#f-wood-grain)",
      "pointer-events": "none",
      opacity: 0.45,
    }),
  );
  svg.appendChild(
    el("rect", {
      x: 0,
      y: 0,
      width: L.viewW,
      height: L.viewH,
      fill: "#5a3019",
      filter: "url(#f-wood-fleck)",
      "pointer-events": "none",
      opacity: 0.55,
    }),
  );
  // Inner-edge bevel — thin dark line just inside the frame so the wood
  // reads as having a carved edge rather than a flat painted rectangle.
  svg.appendChild(
    el("rect", {
      x: 3,
      y: 3,
      width: L.viewW - 6,
      height: L.viewH - 6,
      fill: "none",
      stroke: "rgba(0, 0, 0, 0.55)",
      "stroke-width": 1,
      "pointer-events": "none",
    }),
  );

  // Felt (interior playing surface) — slight chamfer (rx:2) to suggest the
  // felt is set into the frame, but not so much that it rounds visibly.
  const innerX = 6;
  const innerY = 6;
  const innerW = L.viewW - 12;
  const innerH = L.viewH - 12;
  svg.appendChild(
    el("rect", {
      x: innerX,
      y: innerY,
      width: innerW,
      height: innerH,
      fill: "var(--board)",
      rx: 2,
    }),
  );
  // Felt micro-grain
  svg.appendChild(
    el("rect", {
      x: innerX,
      y: innerY,
      width: innerW,
      height: innerH,
      fill: "#6b4226",
      filter: "url(#f-felt-grain)",
      rx: 2,
      "pointer-events": "none",
    }),
  );

  // Bar — wooden divider between the two play halves. Layered like the outer
  // frame: gradient base, soft wood-grain at low opacity, fine fleck for
  // high-frequency detail, plus top-highlight and bottom-shadow edge strips
  // that read as a "raised" carved divider. Keeping the grain subtle avoids
  // the previous coarse banding that made the bar look crudely textured.
  // The visual rect is inset by BAR_PAD on each side so felt shows between
  // the bar and the adjacent inner points (6/13/18/19 columns). The bar's
  // layout zone (BAR_START..BAR_END) stays full-width for hit-testing and
  // for centering checkers placed on the bar.
  const barH = L.BOT_POINT_BASE - L.TOP_POINT_BASE;
  const BAR_PAD = 3;
  const barVX = L.BAR_START + BAR_PAD;
  const barVW = (L.BAR_END - L.BAR_START) - 2 * BAR_PAD;
  svg.appendChild(
    el("rect", {
      x: barVX,
      y: L.TOP_POINT_BASE,
      width: barVW,
      height: barH,
      fill: "url(#lg-bar-wood)",
      rx: 4,
    }),
  );
  svg.appendChild(
    el("rect", {
      x: barVX,
      y: L.TOP_POINT_BASE,
      width: barVW,
      height: barH,
      fill: "#3a2014",
      filter: "url(#f-wood-grain)",
      rx: 4,
      opacity: 0.22,
      "pointer-events": "none",
    }),
  );
  svg.appendChild(
    el("rect", {
      x: barVX,
      y: L.TOP_POINT_BASE,
      width: barVW,
      height: barH,
      fill: "#3a2014",
      filter: "url(#f-wood-fleck)",
      rx: 4,
      opacity: 0.4,
      "pointer-events": "none",
    }),
  );
  // Top highlight stripe (catches light → raised feel).
  svg.appendChild(
    el("rect", {
      x: barVX,
      y: L.TOP_POINT_BASE,
      width: barVW,
      height: 2,
      fill: "rgba(255, 220, 180, 0.18)",
      rx: 4,
      "pointer-events": "none",
    }),
  );
  // Bottom shadow stripe.
  svg.appendChild(
    el("rect", {
      x: barVX,
      y: L.BOT_POINT_BASE - 2.5,
      width: barVW,
      height: 2.5,
      fill: "rgba(0, 0, 0, 0.45)",
      rx: 4,
      "pointer-events": "none",
    }),
  );

  // Tray — recessed compartment for borne-off stones. To read as a distinct
  // sunken box (not just darker felt), we stack:
  //   1. A wood-colored frame slightly larger than the tray (its "walls").
  //   2. A dark interior fill (much darker than the surrounding felt).
  //   3. A faint grain overlay (kept very subtle so it doesn't lift the dark).
  //   4. Inset shadow strips on all four sides — top + left dark, right +
  //      bottom highlight — for the recessed-compartment 3D effect.
  //   5. A horizontal mid-divider separating each player's bear-off region.
  const trayH = L.BOT_POINT_BASE - L.TOP_POINT_BASE;
  const trayW = L.TRAY_END - L.TRAY_START;
  // 1. Wood frame around the tray (the compartment's walls). Sits in the same
  // plane as the surrounding wood — no soft shadow — so it reads as carved
  // into the frame, not floating above it. rx:2 matches the felt's chamfer.
  svg.appendChild(
    el("rect", {
      x: L.TRAY_START - 3,
      y: L.TOP_POINT_BASE - 3,
      width: trayW + 6,
      height: trayH + 6,
      fill: "url(#lg-wood)",
      rx: 2,
      "pointer-events": "none",
    }),
  );
  // 2. Dark interior fill.
  svg.appendChild(
    el("rect", {
      x: L.TRAY_START,
      y: L.TOP_POINT_BASE,
      width: trayW,
      height: trayH,
      fill: "url(#lg-tray-felt)",
      rx: 2,
    }),
  );
  // 3. Subtle grain overlay — dimmed so it doesn't wash out the darkness.
  svg.appendChild(
    el("rect", {
      x: L.TRAY_START,
      y: L.TOP_POINT_BASE,
      width: trayW,
      height: trayH,
      fill: "#3a2014",
      filter: "url(#f-felt-grain)",
      rx: 2,
      opacity: 0.4,
      "pointer-events": "none",
    }),
  );
  // 4a. Top edge — strong recessed shadow.
  svg.appendChild(
    el("rect", {
      x: L.TRAY_START,
      y: L.TOP_POINT_BASE,
      width: trayW,
      height: 4,
      fill: "rgba(0, 0, 0, 0.75)",
      rx: 2,
      "pointer-events": "none",
    }),
  );
  // 4b. Left edge — recessed shadow.
  svg.appendChild(
    el("rect", {
      x: L.TRAY_START,
      y: L.TOP_POINT_BASE,
      width: 3,
      height: trayH,
      fill: "rgba(0, 0, 0, 0.55)",
      rx: 2,
      "pointer-events": "none",
    }),
  );
  // 4c. Bottom edge — highlight (catches reflected light off the floor of the compartment).
  svg.appendChild(
    el("rect", {
      x: L.TRAY_START,
      y: L.BOT_POINT_BASE - 2.5,
      width: trayW,
      height: 2.5,
      fill: "rgba(255, 220, 180, 0.18)",
      rx: 2,
      "pointer-events": "none",
    }),
  );
  // 4d. Right edge — highlight.
  svg.appendChild(
    el("rect", {
      x: L.TRAY_END - 2,
      y: L.TOP_POINT_BASE,
      width: 2,
      height: trayH,
      fill: "rgba(255, 220, 180, 0.12)",
      rx: 2,
      "pointer-events": "none",
    }),
  );
  // 5. Equity bar — wood-styled divider between the two bear-off
  // compartments. Carries the live equity readout when showEquity is on.
  // Geometry: matches the BAR_HALF gap exposed by trayGeometry — the bar
  // fills exactly the y range that the trays' felt rects leave open.
  const midY = (L.TOP_POINT_APEX + L.BOT_POINT_APEX) / 2;
  const barH2 = 36; // 2 * BAR_HALF
  const barTop = midY - barH2 / 2;
  svg.appendChild(
    el("rect", {
      x: L.TRAY_START,
      y: barTop,
      width: trayW,
      height: barH2,
      fill: "url(#lg-wood)",
      rx: 4,
    }),
  );
  svg.appendChild(
    el("rect", {
      x: L.TRAY_START,
      y: barTop,
      width: trayW,
      height: barH2,
      fill: "#5a3019",
      filter: "url(#f-wood-fleck)",
      opacity: 0.55,
      rx: 4,
      "pointer-events": "none",
    }),
  );
  // Top highlight + bottom shadow give the bar a raised feel.
  svg.appendChild(
    el("rect", {
      x: L.TRAY_START,
      y: barTop,
      width: trayW,
      height: 1.5,
      fill: "rgba(255, 220, 180, 0.22)",
      "pointer-events": "none",
    }),
  );
  svg.appendChild(
    el("rect", {
      x: L.TRAY_START,
      y: barTop + barH2 - 1.5,
      width: trayW,
      height: 1.5,
      fill: "rgba(0, 0, 0, 0.45)",
      "pointer-events": "none",
    }),
  );
  if (opts.equity !== null && opts.equity !== undefined) {
    const eq = opts.equity;
    // eq is in absolute white POV: positive = white leads, negative = black
    // leads. Display the magnitude in the leader's checker color.
    const leaderColor = eq >= 0 ? "white" : "black";
    const fill = leaderColor === "white" ? "#f5f1e6" : "#2c2c33";
    // Inset plaque behind the equity readout. The dark wood divider alone
    // gives strong contrast for white text but almost none for black; a
    // warm mid-tone plaque keeps the wood aesthetic while supporting both.
    // Luma ~125 (relL ~0.19) yields ~3.5:1 against both #f5f1e6 and #2c2c33,
    // clearing WCAG large-text (15px / 700-weight qualifies).
    const plaqueW = 56;
    const plaqueH = 22;
    const plaqueX = L.TRAY_START + trayW / 2 - plaqueW / 2;
    // Center plaque exactly on the divider midline so it visually aligns
    // with the centered digits (cap-height midpoint of 15px digits at y=midY+5
    // sits at midY, matching the plaque center).
    const plaqueY = midY - plaqueH / 2;
    // Subtle inset gradient — darker rim on top (recess shadow), lighter
    // middle, slightly lifted lower edge (raised feel).
    svg.appendChild(
      el("rect", {
        x: plaqueX,
        y: plaqueY,
        width: plaqueW,
        height: plaqueH,
        rx: 4,
        fill: "#7e6b56",
        "pointer-events": "none",
      }),
    );
    // Top inner shadow (1px) + bottom inner highlight (1px) for the inset
    // look. These also reinforce contrast at the text's top/bottom edges.
    svg.appendChild(
      el("rect", {
        x: plaqueX,
        y: plaqueY,
        width: plaqueW,
        height: 1,
        rx: 4,
        fill: "rgba(0,0,0,0.45)",
        "pointer-events": "none",
      }),
    );
    svg.appendChild(
      el("rect", {
        x: plaqueX,
        y: plaqueY + plaqueH - 1,
        width: plaqueW,
        height: 1,
        rx: 4,
        fill: "rgba(255,225,190,0.30)",
        "pointer-events": "none",
      }),
    );
    const equityText = el("text", {
      x: L.TRAY_START + trayW / 2,
      y: midY + 5,
      "text-anchor": "middle",
      "font-size": 15,
      "font-weight": 700,
      fill,
      "pointer-events": "none",
      "letter-spacing": "0.04em",
    });
    equityText.textContent = `+${Math.abs(eq).toFixed(2)}`;
    svg.appendChild(equityText);
  }

  // Points (triangles with vertical gradient + diamond apex inlay)
  for (let idx = 0; idx < POINTS; idx++) {
    const g = L.pointGeometry(idx, opts.flipped);
    const left = g.cx - g.halfW;
    const right = g.cx + g.halfW;
    const isLight = (g.col + (g.isTop ? 0 : 1)) % 2 === 0;
    const fill = isLight ? "url(#lg-point-light)" : "url(#lg-point-dark)";
    const tri = el("polygon", {
      points: `${left},${g.baseY} ${right},${g.baseY} ${g.cx},${g.apexY}`,
      fill,
      stroke: "rgba(0,0,0,0.35)",
      "stroke-width": 0.75,
      filter: "url(#f-soft-shadow)",
    });
    tri.setAttribute("data-point", String(idx));
    svg.appendChild(tri);
    // Diamond inlay near the apex. The diamond is wider than the triangle
    // at this distance from the apex (~3.6 units triangle width vs 7-unit
    // diamond), so its top/bottom tips overhang onto the felt. For the
    // overhang to be visible — making the diamond read as full-sized rather
    // than clipped by the triangle silhouette — its color must contrast
    // strongly with the felt (#5a3a2a). The light diamond (#e6c89a) on
    // dark points already contrasts; the dark diamond on light points
    // needs to be much darker than the felt, otherwise the overhang
    // blends into the background and the diamond appears constrained.
    const diamondCY = g.isTop ? g.apexY - 14 : g.apexY + 14;
    const diamondFill = isLight ? "#1a0a02" : "#e6c89a";
    // The diamond is wider (7) than the triangle at this distance from the
    // apex (~3.6 units), so its left/right tips overhang onto the felt
    // (#5a3a2a). The light diamond's overhang reads naturally against the
    // dark felt (high luminance delta). The dark diamond's overhang has
    // too little luminance delta to read at a 1.7-unit slice, so we add a
    // light stroke that silhouettes the overhang against the felt and
    // makes the dark diamond appear full-sized like its lighter sibling.
    const attrs: Record<string, string | number> = {
      points: `${g.cx},${diamondCY - 6} ${g.cx + 3.5},${diamondCY} ${g.cx},${diamondCY + 6} ${g.cx - 3.5},${diamondCY}`,
      fill: diamondFill,
      "pointer-events": "none",
    };
    if (isLight) {
      attrs.stroke = "rgba(230, 200, 154, 0.55)";
      attrs["stroke-width"] = 0.6;
    }
    svg.appendChild(el("polygon", attrs));
  }

  // Highlight: selected source
  if (opts.selectedFrom !== null && opts.selectedFrom >= 0 && opts.selectedFrom < 24) {
    const g = L.pointGeometry(opts.selectedFrom, opts.flipped);
    svg.appendChild(
      el("polygon", {
        points: `${g.cx - g.halfW},${g.baseY} ${g.cx + g.halfW},${g.baseY} ${g.cx},${g.apexY}`,
        fill: "none",
        stroke: "var(--highlight)",
        "stroke-width": 4,
        "pointer-events": "none",
      }),
    );
  } else if (opts.selectedFrom === 24) {
    const cx = (L.BAR_START + L.BAR_END) / 2;
    const y0 = !opts.flipped ? L.BAR_TOP : (L.BAR_TOP + L.BAR_BOTTOM) / 2;
    const y1 = !opts.flipped ? (L.BAR_TOP + L.BAR_BOTTOM) / 2 : L.BAR_BOTTOM;
    svg.appendChild(
      el("rect", {
        x: cx - CHECKER_R - 4,
        y: y0 - 4,
        width: 2 * (CHECKER_R + 4),
        height: y1 - y0 + 8,
        fill: "none",
        stroke: "var(--highlight)",
        "stroke-width": 4,
        rx: 6,
        "pointer-events": "none",
      }),
    );
  }
  // Highlight: legal destinations
  const showAsTargets =
    opts.selectedFrom !== null
      ? opts.legalDestsFrom.get(opts.selectedFrom) ?? new Set<number>()
      : new Set<number>();
  for (const t of showAsTargets) {
    if (t === -1) {
      const g = L.trayGeometry("us", opts.flipped);
      svg.appendChild(
        el("rect", {
          x: L.TRAY_START,
          y: g.topY - 6,
          width: L.TRAY_END - L.TRAY_START,
          height: g.bottomY - g.topY + 12,
          fill: "none",
          stroke: "var(--legal)",
          "stroke-width": 4,
          rx: 6,
          "pointer-events": "none",
        }),
      );
    } else {
      const g = L.pointGeometry(t, opts.flipped);
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
  // Note: previously a subtle yellow wash painted every point that had at
  // least one legal source. It flickered off the moment the last die was
  // consumed (legalDestsFrom empties before the turn commits), which read
  // as a visual hiccup, so the wash was removed. Legal destinations
  // (green rings) for a SELECTED source are still rendered above.

  // Checkers on points
  for (let idx = 0; idx < POINTS; idx++) {
    const v = p.points[idx];
    if (v === 0) continue;
    let count = Math.abs(v);
    if (opts.hideTopFrom === idx && v > 0) count = Math.max(0, count - 1);
    if (count === 0) continue;
    const absSide: "white" | "black" =
      v > 0 ? opts.ourColor : opts.ourColor === "white" ? "black" : "white";
    const visible = Math.min(count, 5);
    for (let i = 0; i < visible; i++) {
      const { cx, cy } = L.checkerCenter(idx, i, opts.flipped);
      svg.appendChild(makeChecker(cx, cy, absSide));
    }
    if (count > 5) {
      const { cx, cy } = L.checkerCenter(idx, 4, opts.flipped);
      svg.appendChild(makeCountBadge(cx, cy, count, absSide));
    }
  }

  // Bar checkers
  const renderBarStack = (rawCount: number, side: "us" | "them"): void => {
    let count = rawCount;
    if (side === "us" && opts.hideTopFrom === 24) count = Math.max(0, count - 1);
    if (count <= 0) return;
    const g = L.barGeometry(side, opts.flipped);
    const absSide: "white" | "black" =
      side === "us" ? opts.ourColor : opts.ourColor === "white" ? "black" : "white";
    const visible = Math.min(count, 5);
    const startY = g.topY;
    for (let i = 0; i < visible; i++) {
      svg.appendChild(makeChecker(g.cx, startY + i * CHECKER_PITCH, absSide));
    }
    if (count > 5) {
      svg.appendChild(makeCountBadge(g.cx, startY + 4 * CHECKER_PITCH, count, absSide));
    }
  };
  renderBarStack(p.barUs, "us");
  renderBarStack(p.barThem, "them");

  // Tray (borne off) — horizontal slabs with gradient fill
  const renderTray = (count: number, side: "us" | "them"): void => {
    if (count <= 0) return;
    const g = L.trayGeometry(side, opts.flipped);
    const absSide: "white" | "black" =
      side === "us" ? opts.ourColor : opts.ourColor === "white" ? "black" : "white";
    const slabFill = absSide === "white" ? "url(#rg-checker-white)" : "url(#rg-checker-black)";
    const slotH = (g.bottomY - g.topY) / 15;
    const atBottom = g.topY > L.viewH / 2;
    for (let i = 0; i < count; i++) {
      const y = atBottom ? g.bottomY - (i + 1) * slotH : g.topY + i * slotH;
      svg.appendChild(
        el("rect", {
          x: L.TRAY_START + 6,
          y,
          width: L.TRAY_END - L.TRAY_START - 12,
          height: slotH - 2,
          fill: slabFill,
          stroke: "rgba(0,0,0,0.35)",
          "stroke-width": 0.5,
          rx: 3,
          filter: "url(#f-soft-shadow)",
        }),
      );
    }
  };
  renderTray(p.offUs, "us");
  renderTray(p.offThem, "them");

  // Pip-count labels float inside each compartment near the equity bar — the
  // "open" end of the tray, where bear-off slabs DON'T accumulate.
  if (opts.pipCount) {
    const trayCx = (L.TRAY_START + L.TRAY_END) / 2;
    const labelFor = (count: number, side: "us" | "them"): void => {
      // Once a side has borne off every checker, the pip count is 0 and the
      // overlay just clutters the empty tray — hide it.
      if (count <= 0) return;
      const g = L.trayGeometry(side, opts.flipped);
      const atBottom = g.topY > L.viewH / 2;
      // For "us" at bottom: slabs grow upward from g.bottomY, so the open
      // end is the top of the compartment (g.topY). For "them" at top:
      // slabs grow downward from g.topY, so the open end is the bottom.
      const cy = atBottom ? g.topY + 16 : g.bottomY - 6;
      const text = el("text", {
        x: trayCx,
        y: cy,
        "text-anchor": "middle",
        "font-size": 13,
        "font-weight": 700,
        fill: "rgba(240, 230, 208, 0.85)",
        "pointer-events": "none",
      });
      text.textContent = String(count);
      svg.appendChild(text);
    };
    labelFor(opts.pipCount.us, "us");
    labelFor(opts.pipCount.them, "them");
  }

  // Dice. White rolls on the right half (player's right); Black rolls on the
  // left half (their right when sitting on the other side of the board).
  if (opts.dice.length > 0) {
    const allDice = opts.dice;
    const dieSize = 56;
    const dieGap = 14;
    const diceY = (L.TOP_POINT_APEX + L.BOT_POINT_APEX) / 2;
    const onLeft = opts.ourColor === "black";
    const halfStart = onLeft ? L.LEFT_HALF_START : L.RIGHT_HALF_START;
    const halfEnd = onLeft ? L.LEFT_HALF_END : L.RIGHT_HALF_END;
    const diceCenterX = (halfStart + halfEnd) / 2;
    const total = allDice.length * dieSize + (allDice.length - 1) * dieGap;
    let x = diceCenterX - total / 2;

    const group = el("g", {});
    group.setAttribute("data-dice-zone", "true");
    if (opts.diceCue === "confirm") {
      group.appendChild(
        el("rect", {
          x: halfStart,
          y: L.TOP_POINT_APEX,
          width: halfEnd - halfStart,
          height: L.BOT_POINT_APEX - L.TOP_POINT_APEX,
          fill: "none",
          "pointer-events": "all",
        }),
      );
    }
    for (const die of allDice) {
      group.appendChild(renderDie(die.d, x, diceY - dieSize / 2, dieSize, die.used, opts.ourColor));
      x += dieSize + dieGap;
    }
    if (opts.cpuThinking) {
      const label = el("text", {
        x: diceCenterX,
        y: diceY + dieSize / 2 + 22,
        "text-anchor": "middle",
        "font-size": 16,
        "font-weight": 600,
        fill: "#f0e6d0",
        "pointer-events": "none",
        "letter-spacing": "0.06em",
      });
      label.textContent = "Thinking…";
      // Pulse opacity via SMIL — well-supported and self-contained, no JS loop.
      const animate = document.createElementNS(SVG_NS, "animate");
      animate.setAttribute("attributeName", "opacity");
      animate.setAttribute("values", "0.35;1;0.35");
      animate.setAttribute("dur", "1.2s");
      animate.setAttribute("repeatCount", "indefinite");
      label.appendChild(animate);
      group.appendChild(label);
    }
    svg.appendChild(group);
  }

  // Hint outlines. Arrows originate from the top of the source stack and end
  // stacked on the destination — for doubles (e.g. 8/2 8/2 8/2 8/2), arrow N
  // starts at the N-th stone from the top and lands at the N-th unoccupied
  // slot on the destination. When a sub-move's source matches the prior
  // sub-move's destination (multi-hop on the same checker), the new arrow
  // chains visually onto the previous endpoint.
  if (opts.hintMoves && opts.hintMoves.length > 0) {
    // Walking counts: how many stones remain on each source, how many have
    // landed on each destination. Initialized from the painted position.
    const srcRemaining = new Map<number, number>();
    const dstLanded = new Map<number, number>();
    const sourceOf = (idx: number): number => {
      if (srcRemaining.has(idx)) return srcRemaining.get(idx)!;
      let v = 0;
      if (idx === 24) v = p.barUs;
      else if (idx >= 0 && idx < 24) v = Math.max(0, p.points[idx]);
      srcRemaining.set(idx, v);
      return v;
    };
    const destBaseOf = (idx: number): number => {
      if (dstLanded.has(idx)) return dstLanded.get(idx)!;
      let v = 0;
      if (idx === -1) v = p.offUs;
      else if (idx >= 0 && idx < 24) {
        // If the point already holds our checkers, stack on top of them.
        // Opp blots (value < 0) are hit and replaced — start at slot 0.
        v = p.points[idx] > 0 ? p.points[idx] : 0;
      }
      dstLanded.set(idx, v);
      return v;
    };
    let prevTo: number | null = null;
    let prevEnd: { cx: number; cy: number } | null = null;
    for (const m of opts.hintMoves) {
      const chained = prevTo !== null && m.from === prevTo;
      let start: { cx: number; cy: number };
      if (chained && prevEnd) {
        start = prevEnd;
      } else {
        const remaining = sourceOf(m.from);
        const srcSlot = Math.max(0, remaining - 1);
        start = pointSlotPos(L, m.from, srcSlot, opts.flipped);
        srcRemaining.set(m.from, remaining - 1);
      }
      const dstSlot = destBaseOf(m.to);
      dstLanded.set(m.to, dstSlot + 1);
      const end = pointSlotPos(L, m.to, dstSlot, opts.flipped);
      drawHintArrowAt(svg, start, end);
      prevTo = m.to;
      prevEnd = end;
    }
  }
}

// Pixel anchor for a slot at a given point/bar/tray index.
function pointSlotPos(L: BoardLayout, idx: number, slot: number, flipped: boolean): { cx: number; cy: number } {
  if (idx === 24) {
    const g = L.barGeometry("us", flipped);
    return { cx: g.cx, cy: g.topY + slot * CHECKER_PITCH };
  }
  if (idx === -1) {
    const g = L.trayGeometry("us", flipped);
    const slotH = (g.bottomY - g.topY) / 15;
    const atBottom = g.topY > L.viewH / 2;
    const cy = atBottom ? g.bottomY - (slot + 0.5) * slotH : g.topY + (slot + 0.5) * slotH;
    return { cx: (L.TRAY_START + L.TRAY_END) / 2, cy };
  }
  return L.checkerCenter(idx, Math.min(slot, 4), flipped);
}

function makeChecker(cx: number, cy: number, absSide: "white" | "black"): SVGGElement {
  const g = el("g", { "pointer-events": "none" });
  const fillId = absSide === "white" ? "rg-checker-white" : "rg-checker-black";
  // Single disc with a substantive dark edge stroke — reads as a flat
  // playing piece with a beveled rim rather than a 3D sphere.
  g.appendChild(
    el("circle", {
      cx,
      cy,
      r: CHECKER_R - 0.75,
      fill: `url(#${fillId})`,
      stroke: absSide === "white" ? "#3a2614" : "#000000",
      "stroke-width": 1.5,
      filter: "url(#f-shadow)",
    }),
  );
  return g as SVGGElement;
}

function makeCountBadge(cx: number, cy: number, count: number, absSide: "white" | "black"): SVGTextElement {
  const t = el("text", {
    x: cx,
    y: cy + 5,
    "text-anchor": "middle",
    "font-size": 18,
    "font-weight": 700,
    fill: absSide === "white" ? "#1a1d24" : "#f5f1e6",
    "pointer-events": "none",
  });
  t.textContent = String(count);
  return t as SVGTextElement;
}

function drawHintArrowAt(
  svg: SVGSVGElement,
  start: { cx: number; cy: number },
  end: { cx: number; cy: number },
): void {
  const stroke = "var(--hint)";
  svg.appendChild(
    el("line", {
      x1: start.cx,
      y1: start.cy,
      x2: end.cx,
      y2: end.cy,
      stroke,
      "stroke-width": 3,
      "stroke-linecap": "round",
      "stroke-dasharray": "6 4",
      opacity: 0.85,
      "pointer-events": "none",
    }),
  );
  svg.appendChild(
    el("circle", {
      cx: end.cx,
      cy: end.cy,
      r: 8,
      fill: stroke,
      opacity: 0.9,
      "pointer-events": "none",
    }),
  );
}

export function checkerAnchor(
  p: Position,
  idx: number,
  role: "from" | "to",
  flipped: boolean,
  layout: BoardLayout = DEFAULT_LAYOUT,
): { cx: number; cy: number } {
  if (idx === 24) {
    const g = layout.barGeometry("us", flipped);
    const slot = Math.max(0, p.barUs - 1);
    return { cx: g.cx, cy: g.topY + slot * CHECKER_PITCH };
  }
  if (idx === -1) {
    const g = layout.trayGeometry("us", flipped);
    const slotH = (g.bottomY - g.topY) / 15;
    const atBottom = g.topY > layout.viewH / 2;
    const slotIdx = p.offUs;
    const y = atBottom
      ? g.bottomY - (slotIdx + 1) * slotH + slotH / 2
      : g.topY + slotIdx * slotH + slotH / 2;
    return { cx: (layout.TRAY_START + layout.TRAY_END) / 2, cy: y };
  }
  const v = p.points[idx];
  let slot: number;
  if (role === "from") {
    slot = Math.max(0, v - 1);
  } else {
    slot = v >= 0 ? v : 0;
  }
  return layout.checkerCenter(idx, Math.min(slot, 4), flipped);
}

export async function animateSubMove(
  svg: SVGSVGElement,
  before: Position,
  sub: SubMove,
  flipped: boolean,
  ourColor: "white" | "black",
  layout: BoardLayout,
  duration = 150,
): Promise<void> {
  const start = checkerAnchor(before, sub.from, "from", flipped, layout);
  const end = checkerAnchor(before, sub.to, "to", flipped, layout);
  const ghost = makeChecker(start.cx, start.cy, ourColor);
  svg.appendChild(ghost);
  const dx0 = start.cx;
  const dy0 = start.cy;
  const t0 = performance.now();
  await new Promise<void>((resolve) => {
    const step = (now: number): void => {
      const t = Math.min(1, (now - t0) / duration);
      const eased = 1 - (1 - t) ** 2;
      const cx = start.cx + (end.cx - start.cx) * eased;
      const cy = start.cy + (end.cy - start.cy) * eased;
      ghost.setAttribute("transform", `translate(${cx - dx0} ${cy - dy0})`);
      if (t >= 1) {
        ghost.remove();
        resolve();
      } else {
        requestAnimationFrame(step);
      }
    };
    requestAnimationFrame(step);
  });
}

function renderDie(value: number, x: number, y: number, size: number, used: boolean, ourColor: "white" | "black"): SVGGElement {
  const g = el("g", {});
  const isWhite = ourColor === "white";
  const faceId = used
    ? isWhite
      ? "lg-die-white-used"
      : "lg-die-black-used"
    : isWhite
      ? "lg-die-white"
      : "lg-die-black";
  const pipId = isWhite ? "rg-pip-dark" : "rg-pip-light";
  const stroke = isWhite ? "rgba(80, 60, 30, 0.55)" : "rgba(220, 220, 235, 0.20)";
  // Die face with gradient fill, rounded corners, drop shadow.
  g.appendChild(
    el("rect", {
      x,
      y,
      width: size,
      height: size,
      rx: 8,
      fill: `url(#${faceId})`,
      stroke,
      "stroke-width": 0.75,
      filter: used ? "url(#f-soft-shadow)" : "url(#f-die-shadow)",
      opacity: used ? 0.6 : 1,
    }),
  );
  // Bevel highlight (subtle inset stripe).
  g.appendChild(
    el("rect", {
      x: x + 2,
      y: y + 2,
      width: size - 4,
      height: size - 4,
      rx: 6,
      fill: "none",
      stroke: isWhite ? "rgba(255,250,235,0.18)" : "rgba(255,255,255,0.05)",
      "stroke-width": 0.5,
      "pointer-events": "none",
    }),
  );
  // Pip positions — strict 25%/50%/75% grid relative to the die bounding
  // box. All pips share the same radius so they look uniform across faces.
  const pad = size * 0.25;
  const r = size * 0.085;
  const cxFn = (col: number) => x + pad + ((size - 2 * pad) * col) / 2;
  const cyFn = (row: number) => y + pad + ((size - 2 * pad) * row) / 2;
  const pipMap: Record<number, [number, number][]> = {
    1: [[1, 1]],
    2: [[0, 0], [2, 2]],
    3: [[0, 0], [1, 1], [2, 2]],
    4: [[0, 0], [2, 0], [0, 2], [2, 2]],
    5: [[0, 0], [2, 0], [1, 1], [0, 2], [2, 2]],
    6: [[0, 0], [2, 0], [0, 1], [2, 1], [0, 2], [2, 2]],
  };
  for (const [c, row] of pipMap[value] ?? []) {
    g.appendChild(
      el("circle", {
        cx: cxFn(c),
        cy: cyFn(row),
        r,
        fill: `url(#${pipId})`,
        opacity: used ? 0.55 : 1,
      }),
    );
  }
  return g;
}
