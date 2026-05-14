"""2-ply expectimax rollout target for TD training (Phase D).

Replaces the standard TD bootstrap V(s_{t+1}) with a higher-signal target:
   R(s) = E_dice[ min_over_opp_plays of V(mirror(s_after_opp_play)) ]
where the expectation is over all 21 distinct dice combos (1/36 for doubles,
2/36 for non-doubles).

Convention (must match training/encoding.py): encode() reads "positive
checkers = player on roll." So whenever we want the net to evaluate a state
in a particular player's frame, we must mirror the state so that *that
player's* checkers are positive before encoding. After we play, opp is on
roll, so opp_view = mirror(after) is opp's frame. After opp plays in
opp_view, *we* are on roll again, so we must mirror opp_after back before
encoding (us_again = mirror(opp_after)).

Equivalent to src/ai/search.ts:score2ply — by design, since the deployment
search runs that exact algorithm and we want to train against the same
target the net will be ranked by at inference time.

For batched per-roll candidate evaluation, all candidates within a roll go
through a single net.equity_batched() forward (the BLAS-friendly operation
we want; per-candidate scalar forwards are ~20x slower for 2-ply).
"""
from __future__ import annotations
from typing import List

import numpy as np
import torch

from engine import (
    Position, generate_plays, apply_play, mirror, check_win,
)
from encoding import encode
from net_torch import Net, P_W, P_GW, P_L, P_GL
from outcome import classify_outcome


# (d1, d2, prob) — 21 distinct dice combos, matches TS ALL_ROLLS exactly.
ALL_ROLLS: List = []
for _i in range(1, 7):
    for _j in range(_i, 7):
        _p = 1.0 / 36.0 if _i == _j else 2.0 / 36.0
        ALL_ROLLS.append((_i, _j, _p))


def _terminal_equity_after_play(p: Position) -> float:
    """Returns signed equity in mover's frame if p is terminal, else None.
    Mirrors src/ai/search.ts:terminalEquityAfterMyPlay (which returns in the
    mover's frame, with magnitude 1/2/3 for single/gammon/backgammon)."""
    r = check_win(p)
    if r is None:
        return None
    winner_abs, base_pts = r
    sign = 1.0 if winner_abs == p.turn else -1.0
    return sign * float(base_pts)


def _swap_wl_4vec(v):
    """Convert a 4-vector between mover-frames (swap W/GW with L/GL)."""
    # v can be a Python list/tuple/torch.Tensor.
    if isinstance(v, torch.Tensor):
        return v[[P_L, P_GL, P_W, P_GW]]
    return [v[P_L], v[P_GL], v[P_W], v[P_GW]]


def rollout_target_4vec_us_frame(net: Net, after: Position) -> torch.Tensor:
    """2-ply expectimax 4-vector target in the *post-play mover's* (us) frame.

    Used as a higher-signal TD target in place of swap_wl(y_opp). For each
    dice roll, opp picks the candidate that minimizes our cubeless equity;
    we accumulate that candidate's 4-vector (us-frame). Average over dice.

    Returns a torch.Tensor of shape (4,) in (p_w, p_gw, p_l, p_gl) order,
    in the frame of the player who just played `after`.
    """
    term_eq = _terminal_equity_after_play(after)
    if term_eq is not None:
        # Terminal in mover frame (mover.turn = after.turn).
        return torch.tensor(classify_outcome(after, after.turn),
                            dtype=torch.float32)

    opp_view = mirror(after)
    accum = torch.zeros(4)
    for d1, d2, prob in ALL_ROLLS:
        opp_plays = generate_plays(opp_view, d1, d2)
        # Collect per-candidate (4-vector in us-frame, scalar us-equity).
        candidate_4vecs: List = []
        candidate_us_eqs: List = []
        batched_positions: List = []
        batched_indices: List[int] = []
        for ci, (_pl, opp_after) in enumerate(opp_plays):
            oterm = _terminal_equity_after_play(opp_after)
            if oterm is not None:
                # Terminal at opp's turn. classify_outcome returns opp_after's
                # mover frame 4-vec (opp_after.turn = opp). Swap to us-frame.
                opp_4 = classify_outcome(opp_after, opp_after.turn)
                us_4 = _swap_wl_4vec(list(opp_4))
                us_4_t = torch.tensor(us_4, dtype=torch.float32)
                candidate_4vecs.append(us_4_t)
                candidate_us_eqs.append(
                    us_4[P_W] + us_4[P_GW] - us_4[P_L] - us_4[P_GL]
                )
            else:
                candidate_4vecs.append(None)
                candidate_us_eqs.append(None)
                us_again = mirror(opp_after)
                batched_positions.append(encode(us_again))
                batched_indices.append(ci)
        if batched_positions:
            X = torch.from_numpy(np.asarray(batched_positions,
                                            dtype=np.float32))
            probs = net.value_batched(X)  # (B, 4), us-frame
            eqs = probs[:, P_W] + probs[:, P_GW] - probs[:, P_L] - probs[:, P_GL]
            for bi, ci in enumerate(batched_indices):
                candidate_4vecs[ci] = probs[bi]
                candidate_us_eqs[ci] = float(eqs[bi].item())
        if not candidate_4vecs:
            continue
        min_idx = 0
        for i in range(1, len(candidate_us_eqs)):
            if candidate_us_eqs[i] < candidate_us_eqs[min_idx]:
                min_idx = i
        accum += prob * candidate_4vecs[min_idx]
    return accum


def rollout_target_us_frame(net: Net, after: Position) -> float:
    """Return the 2-ply expectimax equity in the *us* (post-play mover's)
    frame, treating the position immediately after we played `after`.

    Matches src/ai/search.ts:score2ply per-play `total` value.
    """
    term = _terminal_equity_after_play(after)
    if term is not None:
        return term  # already in us-frame

    opp_view = mirror(after)
    total = 0.0
    for d1, d2, prob in ALL_ROLLS:
        opp_plays = generate_plays(opp_view, d1, d2)
        candidate_us_eqs: List = []
        batched_positions: List = []
        batched_indices: List[int] = []
        for ci, (_pl, opp_after) in enumerate(opp_plays):
            oterm = _terminal_equity_after_play(opp_after)
            if oterm is not None:
                # _terminal_equity_after_play returns equity in opp_after's
                # mover frame (opp_after.turn == opp). us-frame = -oterm.
                candidate_us_eqs.append(-oterm)
            else:
                candidate_us_eqs.append(None)
                # We need us-frame equity at opp_after; mirror so we = positive.
                us_again = mirror(opp_after)
                batched_positions.append(encode(us_again))
                batched_indices.append(ci)
        if batched_positions:
            X = torch.from_numpy(np.asarray(batched_positions,
                                            dtype=np.float32))
            eqs = net.equity_batched(X)  # us_eq from us_frame
            for bi, ci in enumerate(batched_indices):
                candidate_us_eqs[ci] = float(eqs[bi].item())
        # Opp picks the play that minimizes our equity.
        best_for_opp = min(candidate_us_eqs) if candidate_us_eqs else 0.0
        total += prob * best_for_opp
    return total
