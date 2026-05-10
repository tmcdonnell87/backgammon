"""TD-Gammon style 198-input encoding of a Position from "us" perspective.

Per side, per point (24 points): 4 features
  f0 = 1 if n >= 1
  f1 = 1 if n >= 2
  f2 = 1 if n >= 3
  f3 = (n - 3) / 2 if n >= 4 else 0
=> 24 * 4 = 96 features per side, 192 total.

Then:
  + bar count us / 2, bar count them / 2          (2)
  + off us / 15, off them / 15                    (2)
  + turn one-hot (always [1, 0] from "us" view,
    but kept for compat with TD-Gammon)            (2)
Total: 198.
"""
import numpy as np
from engine import Position, POINTS

INPUT_SIZE = 198


def _encode_side(out: np.ndarray, off: int, count_iter):
    """Write 4*24 features into out[off:off+96]."""
    for i, n in enumerate(count_iter):
        base = off + i * 4
        if n >= 1:
            out[base] = 1.0
        if n >= 2:
            out[base + 1] = 1.0
        if n >= 3:
            out[base + 2] = 1.0
        if n >= 4:
            out[base + 3] = (n - 3) / 2.0


def encode(p: Position) -> np.ndarray:
    x = np.zeros(INPUT_SIZE, dtype=np.float32)
    # Us: count at point i is max(p.points[i], 0).
    _encode_side(x, 0, (max(p.points[i], 0) for i in range(POINTS)))
    # Them: count at point i is max(-p.points[i], 0).
    # We list "them" points in their forward direction (mirror), so feature[POINTS-1-i].
    _encode_side(x, 96, (max(-p.points[POINTS - 1 - i], 0) for i in range(POINTS)))
    x[192] = p.bar_us / 2.0
    x[193] = p.bar_them / 2.0
    x[194] = p.off_us / 15.0
    x[195] = p.off_them / 15.0
    x[196] = 1.0  # always us-on-roll in "us" perspective
    x[197] = 0.0
    return x
