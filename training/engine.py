"""Backgammon engine port - kept faithful to src/engine/{position,moves,rules}.ts.

Convention: every function operates from "us" (player on roll) perspective.
points[i] > 0 = our checkers, < 0 = opponent's.
Index 0 = our 1-pt (about to bear off), index 23 = our 24-pt.
Bar entry for us with die d goes to index 24-d.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple, Set

POINTS = 24
BAR = 24
OFF = -1
CHECKERS_PER_SIDE = 15


@dataclass
class Position:
    points: List[int]                      # length 24, signed
    bar_us: int = 0
    bar_them: int = 0
    off_us: int = 0
    off_them: int = 0
    turn: int = 0                          # absolute side currently on roll

    def clone(self) -> "Position":
        return Position(list(self.points), self.bar_us, self.bar_them,
                        self.off_us, self.off_them, self.turn)


def starting_position() -> Position:
    pts = [0] * POINTS
    pts[23] = 2
    pts[12] = 5
    pts[7] = 3
    pts[5] = 5
    pts[0] = -2
    pts[11] = -5
    pts[16] = -3
    pts[18] = -5
    return Position(pts)


def mirror(p: Position) -> Position:
    pts = [-p.points[POINTS - 1 - i] for i in range(POINTS)]
    return Position(pts, p.bar_them, p.bar_us, p.off_them, p.off_us, 1 - p.turn)


def all_home(p: Position) -> bool:
    if p.bar_us > 0:
        return False
    for i in range(6, POINTS):
        if p.points[i] > 0:
            return False
    return True


def pip_count(p: Position) -> int:
    pips = p.bar_us * 25
    for i in range(POINTS):
        if p.points[i] > 0:
            pips += p.points[i] * (i + 1)
    return pips


def pip_count_them(p: Position) -> int:
    pips = p.bar_them * 25
    for i in range(POINTS):
        if p.points[i] < 0:
            pips += -p.points[i] * (POINTS - i)
    return pips


def board_hash(p: Position) -> bytes:
    # board-only hash: 24 points + bar/off
    return bytes([(v + 16) & 0xff for v in p.points] +
                 [p.bar_us, p.bar_them, p.off_us, p.off_them])


# --- moves ---------------------------------------------------------------

def _can_bear_off_from(p: Position, frm: int, die: int) -> bool:
    if die == frm + 1:
        return True
    if die < frm + 1:
        return False
    for j in range(frm + 1, 6):
        if p.points[j] > 0:
            return False
    return True


def legal_sub_moves(p: Position, die: int) -> List[Tuple[int, int, int]]:
    """Returns list of (from, to, die)."""
    out: List[Tuple[int, int, int]] = []
    if p.bar_us > 0:
        dest = BAR - die  # 18..23
        if p.points[dest] >= -1:
            out.append((BAR, dest, die))
        return out
    for frm in range(POINTS):
        if p.points[frm] <= 0:
            continue
        dest = frm - die
        if dest >= 0:
            if p.points[dest] >= -1:
                out.append((frm, dest, die))
        else:
            if all_home(p) and _can_bear_off_from(p, frm, die):
                out.append((frm, OFF, die))
    return out


def apply_sub_move(p: Position, sub: Tuple[int, int, int]) -> Position:
    frm, to, _die = sub
    np_ = p.clone()
    if frm == BAR:
        np_.bar_us -= 1
    else:
        np_.points[frm] -= 1
    if to == OFF:
        np_.off_us += 1
    elif np_.points[to] == -1:
        np_.points[to] = 1
        np_.bar_them += 1
    else:
        np_.points[to] += 1
    return np_


def apply_play(p: Position, play: List[Tuple[int, int, int]]) -> Position:
    cur = p
    for sub in play:
        cur = apply_sub_move(cur, sub)
    return cur


def _explore(p: Position, remaining: List[int], partial: List, out: List):
    if not remaining:
        out.append((tuple(partial), p))
        return
    die = remaining[0]
    rest = remaining[1:]
    subs = legal_sub_moves(p, die)
    if not subs:
        out.append((tuple(partial), p))
        return
    for sub in subs:
        np_ = apply_sub_move(p, sub)
        _explore(np_, rest, partial + [sub], out)


def generate_plays(p: Position, d1: int, d2: int) -> List[Tuple[Tuple, Position]]:
    """Returns list of (play_tuple, resulting_position) deduped by board state."""
    orderings = [[d1, d1, d1, d1]] if d1 == d2 else [[d1, d2], [d2, d1]]
    found: List = []
    for order in orderings:
        _explore(p, order, [], found)

    max_len = 0
    for play, _pos in found:
        if len(play) > max_len:
            max_len = len(play)

    if max_len == 0:
        return [((), p)]

    candidates = [(pl, ps) for pl, ps in found if len(pl) == max_len]

    if max_len == 1 and d1 != d2:
        larger = max(d1, d2)
        using_larger = [c for c in candidates if c[0][0][2] == larger]
        if using_larger:
            candidates = using_larger

    seen = {}
    for play, pos in candidates:
        h = board_hash(pos)
        if h not in seen:
            seen[h] = (play, pos)
    return list(seen.values())


# --- rules ---------------------------------------------------------------

def check_win(p: Position) -> Optional[Tuple[int, int]]:
    """Return (winner_abs_side, base_points 1/2/3) or None."""
    if p.off_us >= 15:
        kind = 1
        if p.off_them == 0:
            bg = p.bar_them > 0
            if not bg:
                for i in range(6):
                    if p.points[i] < 0:
                        bg = True
                        break
            kind = 3 if bg else 2
        return (p.turn, kind)
    if p.off_them >= 15:
        kind = 1
        if p.off_us == 0:
            bg = p.bar_us > 0
            if not bg:
                for i in range(18, POINTS):
                    if p.points[i] > 0:
                        bg = True
                        break
            kind = 3 if bg else 2
        return (1 - p.turn, kind)
    return None
