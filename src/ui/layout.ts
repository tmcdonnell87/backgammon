// Pure layout math for the SVG board. No DOM. No state.

export const VIEW_W = 960;
export const VIEW_H = 660;

const LEFT_MARGIN = 20;
const POINT_W = 64;
const POINT_H = 240;
const BAR_W = 60;
const BAR_H = 140;
const HALF_W = 6 * POINT_W; // 384
const TOP_MARGIN = 10;
const TRAY_GAP = 30;
const TRAY_W = 60;

export const LEFT_HALF_START = LEFT_MARGIN; // 20
export const LEFT_HALF_END = LEFT_HALF_START + HALF_W; // 404
export const BAR_START = LEFT_HALF_END; // 404
export const BAR_END = BAR_START + BAR_W; // 464
export const RIGHT_HALF_START = BAR_END; // 464
export const RIGHT_HALF_END = RIGHT_HALF_START + HALF_W; // 848
export const TRAY_START = RIGHT_HALF_END + TRAY_GAP; // 878
export const TRAY_END = TRAY_START + TRAY_W; // 938

export const TOP_POINT_BASE = TOP_MARGIN; // y of triangle base (wide top edge)
export const TOP_POINT_APEX = TOP_MARGIN + POINT_H; // y of triangle apex (250)
export const BAR_TOP = TOP_POINT_APEX; // 250
export const BAR_BOTTOM = BAR_TOP + BAR_H; // 390
export const BOT_POINT_APEX = BAR_BOTTOM; // 390
export const BOT_POINT_BASE = BOT_POINT_APEX + POINT_H; // 630

export const CHECKER_R = 24;
export const CHECKER_D = 2 * CHECKER_R;

export interface PointGeom {
  cx: number;
  baseY: number; // y of triangle base (wide edge)
  apexY: number; // y of triangle apex (point)
  isTop: boolean;
}

// Logical idx 0..23 -> screen geometry, when board is rendered from "us" perspective.
// idx 0 = our 1-pt at bottom-right; idx 23 = our 24-pt at top-right.
export function pointGeometry(idx: number, flipped = false): PointGeom {
  const fidx = flipped ? 23 - idx : idx;
  const isTop = fidx >= 12;
  let col: number;
  if (isTop) {
    col = fidx - 12; // 0..11 left-to-right (12,13,...23)
  } else {
    col = 11 - fidx; // 0..11 left-to-right (11,10,...0)
  }
  let cx: number;
  if (col < 6) {
    cx = LEFT_HALF_START + (col + 0.5) * POINT_W;
  } else {
    cx = RIGHT_HALF_START + (col - 6 + 0.5) * POINT_W;
  }
  return {
    cx,
    baseY: isTop ? TOP_POINT_BASE : BOT_POINT_BASE,
    apexY: isTop ? TOP_POINT_APEX : BOT_POINT_APEX,
    isTop,
  };
}

// Position of i-th checker (0-indexed from base toward center) on a point.
export function checkerCenter(idx: number, slot: number, flipped = false): { cx: number; cy: number } {
  const g = pointGeometry(idx, flipped);
  const offset = CHECKER_R + slot * CHECKER_D;
  const cy = g.isTop ? g.baseY + offset : g.baseY - offset;
  return { cx: g.cx, cy };
}

export interface BarGeom {
  cx: number;
  topY: number;
  bottomY: number;
}

// Bar geometry. side === 'us' means our checkers (entering on opp's home, top); 'them' the opposite.
// When flipped (rendering opp perspective), swap.
export function barGeometry(side: "us" | "them", flipped = false): BarGeom {
  const cx = (BAR_START + BAR_END) / 2;
  const usOnTop = !flipped; // by default, our bar checkers sit on the upper bar (closer to our 24-pt)
  const onTop = side === "us" ? usOnTop : !usOnTop;
  return {
    cx,
    topY: onTop ? BAR_TOP + CHECKER_R : (BAR_TOP + BAR_BOTTOM) / 2 + CHECKER_R / 2,
    bottomY: onTop ? (BAR_TOP + BAR_BOTTOM) / 2 - CHECKER_R / 2 : BAR_BOTTOM - CHECKER_R,
  };
}

export interface TrayGeom {
  cx: number;
  topY: number;
  bottomY: number;
}

export function trayGeometry(side: "us" | "them", flipped = false): TrayGeom {
  const cx = (TRAY_START + TRAY_END) / 2;
  const usAtBottom = !flipped;
  const atBottom = side === "us" ? usAtBottom : !usAtBottom;
  return {
    cx,
    topY: atBottom ? (TOP_POINT_APEX + BOT_POINT_APEX) / 2 + 5 : TOP_POINT_BASE,
    bottomY: atBottom ? BOT_POINT_BASE : (TOP_POINT_APEX + BOT_POINT_APEX) / 2 - 5,
  };
}

// Hit-test: which point (0..23) is at (x, y) within the SVG viewBox? Or BAR (24) or OFF (-1)?
// Returns null if outside any interactive zone.
export function hitTest(x: number, y: number, flipped = false): number | null {
  // Bar?
  if (x >= BAR_START && x <= BAR_END && y >= BAR_TOP && y <= BAR_BOTTOM) {
    return 24; // BAR
  }
  // Tray?
  if (x >= TRAY_START && x <= TRAY_END && y >= TOP_POINT_BASE && y <= BOT_POINT_BASE) {
    return -1; // OFF
  }
  // Points
  const inTopRow = y >= TOP_POINT_BASE && y <= TOP_POINT_APEX;
  const inBotRow = y >= BOT_POINT_APEX && y <= BOT_POINT_BASE;
  if (!inTopRow && !inBotRow) return null;
  let col: number;
  if (x >= LEFT_HALF_START && x < LEFT_HALF_END) {
    col = Math.floor((x - LEFT_HALF_START) / POINT_W);
  } else if (x >= RIGHT_HALF_START && x < RIGHT_HALF_END) {
    col = 6 + Math.floor((x - RIGHT_HALF_START) / POINT_W);
  } else {
    return null;
  }
  let fidx: number;
  if (inTopRow) fidx = 12 + col;
  else fidx = 11 - col;
  return flipped ? 23 - fidx : fidx;
}
