// Pure layout math for the SVG board. No DOM. No state.
//
// The layout is *responsive in the horizontal axis*. The vertical axis is
// fixed at 660 units; the horizontal viewBox width is computed at render time
// from the container's aspect ratio (so a 16:9 window gets a wider viewBox).
// All horizontal segments (margins, points, bar, tray) scale uniformly with
// viewW. Checkers stay at a fixed radius (24 units), so since the viewBox
// aspect always matches the container, circles render as perfect circles.

export const VIEW_H = 606;
// Sum of base widths laid out left→right:
//   LEFT_MARGIN (20) + 2×HALF (768) + BAR (40) + TRAY_GAP (20)
//   + TRAY (60) + RIGHT_MARGIN (6) = 914
// The bar is 40 units — a narrow wood divider, with a small visual inset
// rendered inside it (~3 units each side) so felt shows between the bar
// and the adjacent points. Checkers placed on the bar (CHECKER_D=48)
// overflow slightly on each side, which reads as resting on the bar.
// The right wood-wall after the tray is 6 units — matching the felt's
// 6-unit top/bottom/left wood-wall thickness — so the tray reads as a
// compartment carved into the same wood frame, not a panel inlaid in
// extra-thick wood. The 20-unit wood corridor on the left (LEFT_MARGIN)
// and between the rightmost point and the tray (TRAY_GAP) are the
// "play-area framing" strips, distinct from the felt-frame thickness.
export const DESIGN_W = 914;

// Checker geometry (fixed; in viewBox units).
export const CHECKER_R = 24;
export const CHECKER_D = 2 * CHECKER_R; // 48 — checkers-touching pitch
// Visible stack pitch: 50 leaves a 2-unit gap between adjacent stones. With
// 5 stones in a column the bottom of the topmost extends 8 units past the
// triangle apex into the dice strip, which reads as natural physical stacking
// on a real board.
export const CHECKER_PITCH = 50;

// Bar tap-zone padding (horizontal slop on each side of the bar rect).
export const BAR_HIT_PAD = 4;

export interface PointGeom {
  cx: number;
  baseY: number;
  apexY: number;
  isTop: boolean;
  col: number; // 0..11 left-to-right column index (used by renderer for alternating colors)
  halfW: number; // triangle half-width at the base
}

export interface BarGeom {
  cx: number;
  topY: number;
  bottomY: number;
}

export interface TrayGeom {
  cx: number;
  topY: number;
  bottomY: number;
}

export interface BoardLayout {
  viewW: number;
  viewH: number;
  // Horizontal segments (all scaled with viewW)
  POINT_W: number;
  POINT_HALF: number; // triangle half-width at base
  BAR_W: number;
  TRAY_W: number;
  LEFT_HALF_START: number;
  LEFT_HALF_END: number;
  BAR_START: number;
  BAR_END: number;
  RIGHT_HALF_START: number;
  RIGHT_HALF_END: number;
  TRAY_START: number;
  TRAY_END: number;
  // Vertical segments (fixed)
  POINT_H: number;
  BAR_H: number;
  TOP_POINT_BASE: number;
  TOP_POINT_APEX: number;
  BAR_TOP: number;
  BAR_BOTTOM: number;
  BOT_POINT_APEX: number;
  BOT_POINT_BASE: number;
  // Geometry functions (closures over this layout)
  pointGeometry(idx: number, flipped?: boolean): PointGeom;
  checkerCenter(idx: number, slot: number, flipped?: boolean): { cx: number; cy: number };
  barGeometry(side: "us" | "them", flipped?: boolean): BarGeom;
  trayGeometry(side: "us" | "them", flipped?: boolean): TrayGeom;
  hitTest(x: number, y: number, flipped?: boolean): number | null;
}

export function makeLayout(viewW: number = DESIGN_W, viewH: number = VIEW_H): BoardLayout {
  const s = viewW / DESIGN_W;
  // Horizontal segments
  const LEFT_MARGIN = 20 * s;
  const POINT_W = 64 * s;
  // Triangle bases are flush — full POINT_W wide — so the felt doesn't peek
  // through between adjacent points. cx is exactly (col + 0.5) * POINT_W
  // within each half (pointGeometry below), so every checker sits on the
  // arithmetic midpoint of its triangle.
  const POINT_HALF = POINT_W / 2;
  const BAR_W = 40 * s;
  // Gap between rightmost point and tray, sized to mirror LEFT_MARGIN so the
  // wood strip on the right side of the play area matches the wood strip on
  // the left side of the leftmost point.
  const TRAY_GAP = 20 * s;
  const TRAY_W = 60 * s;
  const HALF_W = 6 * POINT_W;

  const LEFT_HALF_START = LEFT_MARGIN;
  const LEFT_HALF_END = LEFT_HALF_START + HALF_W;
  const BAR_START = LEFT_HALF_END;
  const BAR_END = BAR_START + BAR_W;
  const RIGHT_HALF_START = BAR_END;
  const RIGHT_HALF_END = RIGHT_HALF_START + HALF_W;
  const TRAY_START = RIGHT_HALF_END + TRAY_GAP;
  // Tray's right edge sits flush at viewW — see DESIGN_W note: the sum of
  // base widths (LEFT_MARGIN + 2×HALF + BAR + TRAY_GAP + TRAY_W) is exactly
  // DESIGN_W, so TRAY_START + TRAY_W == viewW for every viewW.
  const TRAY_END = TRAY_START + TRAY_W;

  // Vertical segments (fixed). Symmetric top/bottom margins:
  // 8 + 250 + 90 + 250 + 8 = 606. POINT_H=250 gives 5 stones at pitch 50
  // exactly enough room (24+4·50+24 = 248) with 2 units of headroom below
  // the apex. Tight margins keep more board visible on phones in landscape
  // where vertical space is the binding constraint; the slimmer bar brings
  // the top and bottom point rows visually closer too.
  const TOP_MARGIN = 8;
  const POINT_H = 250;
  const BAR_H = 90;
  const TOP_POINT_BASE = TOP_MARGIN;
  const TOP_POINT_APEX = TOP_MARGIN + POINT_H;
  const BAR_TOP = TOP_POINT_APEX;
  const BAR_BOTTOM = BAR_TOP + BAR_H;
  const BOT_POINT_APEX = BAR_BOTTOM;
  const BOT_POINT_BASE = BOT_POINT_APEX + POINT_H;

  function pointGeometry(idx: number, flipped = false): PointGeom {
    const fidx = flipped ? 23 - idx : idx;
    const isTop = fidx >= 12;
    let col: number;
    if (isTop) {
      col = fidx - 12;
    } else {
      col = 11 - fidx;
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
      col,
      halfW: POINT_HALF,
    };
  }

  function checkerCenter(idx: number, slot: number, flipped = false): { cx: number; cy: number } {
    const g = pointGeometry(idx, flipped);
    const offset = CHECKER_R + slot * CHECKER_PITCH;
    const cy = g.isTop ? g.baseY + offset : g.baseY - offset;
    return { cx: g.cx, cy };
  }

  function barGeometry(side: "us" | "them", flipped = false): BarGeom {
    const cx = (BAR_START + BAR_END) / 2;
    const usOnTop = !flipped;
    const onTop = side === "us" ? usOnTop : !usOnTop;
    return {
      cx,
      topY: onTop ? BAR_TOP + CHECKER_R : (BAR_TOP + BAR_BOTTOM) / 2 + CHECKER_R / 2,
      bottomY: onTop ? (BAR_TOP + BAR_BOTTOM) / 2 - CHECKER_R / 2 : BAR_BOTTOM - CHECKER_R,
    };
  }

  function trayGeometry(side: "us" | "them", flipped = false): TrayGeom {
    const cx = (TRAY_START + TRAY_END) / 2;
    const usAtBottom = !flipped;
    const atBottom = side === "us" ? usAtBottom : !usAtBottom;
    // Half-height of the equity-bar gap that splits us/them compartments.
    // Wider than a divider line so the bar can carry the equity readout.
    const BAR_HALF = 18;
    return {
      cx,
      topY: atBottom ? (TOP_POINT_APEX + BOT_POINT_APEX) / 2 + BAR_HALF : TOP_POINT_BASE,
      bottomY: atBottom ? BOT_POINT_BASE : (TOP_POINT_APEX + BOT_POINT_APEX) / 2 - BAR_HALF,
    };
  }

  function hitTest(x: number, y: number, flipped = false): number | null {
    if (
      x >= BAR_START - BAR_HIT_PAD &&
      x <= BAR_END + BAR_HIT_PAD &&
      y >= TOP_POINT_BASE &&
      y <= BOT_POINT_BASE
    ) {
      return 24; // BAR
    }
    if (x >= TRAY_START && x <= TRAY_END && y >= TOP_POINT_BASE && y <= BOT_POINT_BASE) {
      return -1; // OFF
    }
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

  return {
    viewW,
    viewH,
    POINT_W,
    POINT_HALF,
    BAR_W,
    TRAY_W,
    LEFT_HALF_START,
    LEFT_HALF_END,
    BAR_START,
    BAR_END,
    RIGHT_HALF_START,
    RIGHT_HALF_END,
    TRAY_START,
    TRAY_END,
    POINT_H,
    BAR_H,
    TOP_POINT_BASE,
    TOP_POINT_APEX,
    BAR_TOP,
    BAR_BOTTOM,
    BOT_POINT_APEX,
    BOT_POINT_BASE,
    pointGeometry,
    checkerCenter,
    barGeometry,
    trayGeometry,
    hitTest,
  };
}

// Backwards-compatible default layout (design dimensions). Used by tests and
// any caller that doesn't have a live container to measure.
export const DEFAULT_LAYOUT = makeLayout();
