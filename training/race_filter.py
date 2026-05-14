"""Race vs contact classification (Python port of stillInContact).

Used by Phase C to dispatch between a race-specific net and a contact-specific
net at evaluation time, and to filter self-play training data by phase.
"""
from __future__ import annotations
from engine import Position, POINTS


def still_in_contact(p: Position) -> bool:
    """True if the two sides' checkers can still hit each other.

    Convention: we move from high index toward 0 (index 0 = our 1-pt, bear-off
    direction). Opponent moves from low index toward 23 (their bear-off
    direction in our coords). Contact requires *our highest piece* to be at or
    above *their lowest piece*: from there we could move backward across them,
    or they could move forward into us. Otherwise we're in race phase.

    (Note: the pre-Phase-C version of this check used `ours_min <= opps_max`,
    which is the wrong direction and classified almost every position as
    contact, making the heuristic's race branch effectively dead code. Fixed
    here so Phase C dispatch and the race heuristic both work correctly.)
    """
    if p.bar_us > 0 or p.bar_them > 0:
        return True
    ours_max = -1
    opps_min = POINTS
    for i in range(POINTS):
        if p.points[i] > 0 and i > ours_max:
            ours_max = i
        if p.points[i] < 0 and i < opps_min:
            opps_min = i
    if ours_max == -1 or opps_min == POINTS:
        return False
    return ours_max >= opps_min


def is_race(p: Position) -> bool:
    return not still_in_contact(p)
