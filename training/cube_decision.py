"""Python port of src/ai/cubeDecision.ts.

Used by training/bench_match.py to drive cubeful AI-vs-AI matches. Parity
with the TS implementation is asserted by the matching unit test on the TS
side; this file should track src/ai/cubeDecision.ts line-for-line.

The Python engine (training/engine.py) does not model the cube or match
state — those live on a per-match wrapper in bench_match.py. We accept the
relevant fields directly here.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Met:
    matches: int
    p_w: float
    gammon_rate: float
    backgammon_rate: float
    cube_efficiency: float
    table: List[List[float]]  # table[a][b]


def load_met(path: str) -> Met:
    with open(path) as f:
        raw = json.load(f)
    return Met(
        matches=int(raw["matches"]),
        p_w=float(raw["p_w"]),
        gammon_rate=float(raw["gammon_rate"]),
        backgammon_rate=float(raw.get("backgammon_rate", 0.0)),
        cube_efficiency=float(raw.get("cube_efficiency", 0.7)),
        table=raw["met"],
    )


def met_entry(met: Met, away_us: int, away_them: int) -> float:
    """MWC for us at (away_us, away_them). Boundary: a=0 → 1, b=0 → 0."""
    N = met.matches
    a = max(0, min(N, int(round(away_us))))
    b = max(0, min(N, int(round(away_them))))
    if a == 0:
        return 1.0
    if b == 0:
        return 0.0
    return float(met.table[a][b])


def _mwc_after_game(met: Met,
                    away_us: int, away_them: int, V: int,
                    p_win: float, p_gammon_win: float,
                    p_loss: float, p_gammon_loss: float) -> float:
    p_sw = max(0.0, p_win - p_gammon_win)
    p_gw = max(0.0, p_gammon_win)
    p_sl = max(0.0, p_loss - p_gammon_loss)
    p_gl = max(0.0, p_gammon_loss)
    return (
        p_sw * met_entry(met, away_us - V, away_them)
        + p_gw * met_entry(met, away_us - 2 * V, away_them)
        + p_sl * met_entry(met, away_us, away_them - V)
        + p_gl * met_entry(met, away_us, away_them - 2 * V)
    )


def _can_double(cube_value: int, cube_owner: Optional[int],
                side: int, match_length: int, crawford: bool) -> bool:
    if match_length > 1 and crawford:
        return False
    if cube_owner is None:
        return True
    return cube_owner == side


def decide_cube_action(*,
                       side: int,
                       score: Tuple[int, int],
                       match_length: int,
                       crawford: bool,
                       cube_value: int,
                       cube_owner: Optional[int],
                       outcomes: Tuple[float, float, float, float],
                       met: Met) -> str:
    """Returns 'no_double' | 'double_take' | 'double_drop'.

    outcomes is (p_win, p_gammon_win, p_loss, p_gammon_loss) from `side`'s
    perspective.
    """
    if match_length <= 1:
        return "no_double"
    if crawford:
        return "no_double"
    if not _can_double(cube_value, cube_owner, side, match_length, crawford):
        return "no_double"

    p_win, p_gw, p_loss, p_gl = outcomes
    away_us = match_length - score[side]
    away_them = match_length - score[1 - side]
    V = cube_value

    mwc_no_double = _mwc_after_game(met, away_us, away_them, V,
                                    p_win, p_gw, p_loss, p_gl)
    mwc_take = _mwc_after_game(met, away_us, away_them, 2 * V,
                               p_win, p_gw, p_loss, p_gl)
    mwc_drop = met_entry(met, away_us - V, away_them)

    mwc_doubled = min(mwc_take, mwc_drop)

    # Live-cube waiting value (Janowski). Mirror of src/ai/cubeDecision.ts.
    cube_room = max(0.0, (away_us - V)) / max(1, met.matches)
    live_cube_margin = met.cube_efficiency * 0.06 * cube_room

    if mwc_doubled <= mwc_no_double + live_cube_margin:
        return "no_double"
    return "double_drop" if mwc_drop <= mwc_take else "double_take"


def decide_take_drop(*,
                     side: int,
                     score: Tuple[int, int],
                     match_length: int,
                     crawford: bool,
                     cube_value: int,
                     outcomes: Tuple[float, float, float, float],
                     met: Met) -> str:
    """Returns 'take' | 'drop'."""
    if match_length <= 1:
        return "take"
    if crawford:
        return "take"

    p_win, p_gw, p_loss, p_gl = outcomes
    away_us = match_length - score[side]
    away_them = match_length - score[1 - side]
    V = cube_value

    mwc_take = _mwc_after_game(met, away_us, away_them, 2 * V,
                               p_win, p_gw, p_loss, p_gl)
    mwc_drop = met_entry(met, away_us, away_them - V)
    return "take" if mwc_take >= mwc_drop else "drop"


if __name__ == "__main__":
    # Smoke test: parity with the TS test fixtures.
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    met = load_met(os.path.join(here, "..", "public", "weights", "met.json"))
    # Money game -> no_double regardless of equity.
    a = decide_cube_action(side=0, score=(0, 0), match_length=1, crawford=False,
                           cube_value=1, cube_owner=None,
                           outcomes=(0.9, 0.27, 0.1, 0.01), met=met)
    assert a == "no_double", f"money game should not double, got {a}"
    # 7-pt match at 0-0, high equity -> double_drop.
    a = decide_cube_action(side=0, score=(0, 0), match_length=7, crawford=False,
                           cube_value=1, cube_owner=None,
                           outcomes=(0.9, 0.18, 0.1, 0.02), met=met)
    assert a == "double_drop", f"high-eq 7pt should double_drop, got {a}"
    # 7-pt match at 0-0, coin-flip -> no_double.
    a = decide_cube_action(side=0, score=(0, 0), match_length=7, crawford=False,
                           cube_value=1, cube_owner=None,
                           outcomes=(0.5, 0.0, 0.5, 0.0), met=met)
    assert a == "no_double", f"coin-flip 7pt should not double, got {a}"
    # Crawford -> no_double.
    a = decide_cube_action(side=0, score=(6, 3), match_length=7, crawford=True,
                           cube_value=1, cube_owner=None,
                           outcomes=(0.9, 0.18, 0.1, 0.02), met=met)
    assert a == "no_double", f"crawford should not double, got {a}"
    # Receiver: drop a hopeless take.
    t = decide_take_drop(side=0, score=(0, 0), match_length=7, crawford=False,
                         cube_value=1,
                         outcomes=(0.2, 0.02, 0.8, 0.16), met=met)
    assert t == "drop", f"hopeless take should drop, got {t}"
    # Receiver: take a coin flip.
    t = decide_take_drop(side=0, score=(0, 0), match_length=7, crawford=False,
                         cube_value=2,
                         outcomes=(0.5, 0.0, 0.5, 0.0), met=met)
    assert t == "take", f"coin-flip take should take, got {t}"
    print("cube_decision.py smoke: OK")
