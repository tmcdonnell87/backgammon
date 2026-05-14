"""Outcome classification for the 4-output network.

Maps a terminal Position to a 4-vector (p_win, p_gammon_win, p_loss, p_gammon_loss)
from the requested perspective. Backgammons collapse into the gammon class
(documented Phase A choice).
"""
from __future__ import annotations
from typing import Tuple

from engine import Position, check_win


def classify_outcome(p_terminal: Position,
                     perspective_abs_side: int) -> Tuple[float, float, float, float]:
    r = check_win(p_terminal)
    if r is None:
        raise ValueError("classify_outcome called on a non-terminal position")
    winner_abs, base_points = r
    won = winner_abs == perspective_abs_side
    is_at_least_gammon = base_points >= 2
    if won:
        return (1.0, 1.0 if is_at_least_gammon else 0.0, 0.0, 0.0)
    return (0.0, 0.0, 1.0, 1.0 if is_at_least_gammon else 0.0)
