"""Play two nets head-to-head over N games. Reports A's winrate.

Auto-detects net format: 1-output (legacy numpy net.py) or 4-output (net_torch.py),
so a new 4-output candidate can be compared against the current published
expert.json (which may still be 1-output).
"""
import argparse
import json
import math
import os
import random

from engine import starting_position, mirror, generate_plays, check_win
from encoding import encode
from race_filter import still_in_contact


def load_net_auto(path):
    """Returns (net_obj, equity_fn). equity_fn(pos) -> cubeless equity from
    pos's perspective (the player on roll in `pos`).

    Auto-detects three weight schemas:
      * Phased manifest {"version":2, "contact":..., "race":...}
      * 1-output legacy numpy net (output:1)
      * 4-output torch net (output:4, with hidden as int or list)
    """
    with open(path) as f:
        d = json.load(f)
    # Phased manifest
    if d.get("version") == 2 and "contact" in d and "race" in d:
        from bench import heuristic_value
        base = os.path.dirname(path)

        def _resolve(slot: str):
            """A sub-slot value can be:
              * "heuristic" — dispatch to the hand-tuned heuristic evaluator
                (lets us bench a single specialist with the other half held
                constant against the heuristic baseline).
              * any other string — relative or absolute path to weights JSON.
            Returns (loaded_obj, equity_fn). loaded_obj is None for heuristic.
            """
            if slot == "heuristic":
                return None, heuristic_value
            spath = slot if os.path.isabs(slot) else os.path.join(base, slot)
            _n, eq = load_net_auto(spath)
            return _n, eq

        _c, eq_c = _resolve(d["contact"])
        _r, eq_r = _resolve(d["race"])

        def eq(pos, _ec=eq_c, _er=eq_r):
            return _ec(pos) if still_in_contact(pos) else _er(pos)
        return (_c, _r), eq

    out_dim = d.get("output", 1)
    if out_dim == 4:
        from net_torch import Net as TorchNet
        net = TorchNet.from_dict(d)
        def eq(pos, _net=net):
            return _net.equity(encode(pos))
        return net, eq
    if out_dim == 1:
        from net import Net as NumpyNet
        net = NumpyNet.from_dict(d)
        def eq(pos, _net=net):
            # value returns P(we win) in [0,1]; cubeless equity = 2y - 1.
            return 2.0 * _net.value(encode(pos)) - 1.0
        return net, eq
    raise ValueError(f"Unknown net output dim: {out_dim}")


def pick_with_equity(p, plays, equity_in_opp_frame):
    """Pick play index maximizing our (us-frame) equity."""
    if len(plays) == 1:
        return 0
    best_i, best = 0, -math.inf
    for i, (_pl, after) in enumerate(plays):
        win = check_win(after)
        if win is not None:
            _winner_abs, base_pts = win
            us_eq = float(base_pts)
        else:
            opp = mirror(after)
            us_eq = -equity_in_opp_frame(opp)
        if us_eq > best:
            best = us_eq
            best_i = i
    return best_i


def play_one(a_side, eq_A, eq_B, rng):
    p = starting_position()
    d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
    while d1 == d2:
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
    a_to_move = (a_side == 0)
    while True:
        plays = generate_plays(p, d1, d2)
        if len(plays) == 1 and len(plays[0][0]) == 0:
            after = p
        else:
            eq_fn = eq_A if a_to_move else eq_B
            idx = pick_with_equity(p, plays, eq_fn)
            _, after = plays[idx]
        win = check_win(after)
        if win is not None:
            winner_abs, _ = win
            a_won = (winner_abs == 0 and a_side == 0) or (winner_abs == 1 and a_side == 1)
            return 1 if a_won else 0
        p = mirror(after)
        a_to_move = not a_to_move
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True)
    ap.add_argument("--B", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    _netA, eq_A = load_net_auto(args.A)
    _netB, eq_B = load_net_auto(args.B)
    rng = random.Random(args.seed)
    a_wins = 0
    for g in range(args.games):
        a_wins += play_one(g & 1, eq_A, eq_B, rng)
    wr = a_wins / args.games
    se = math.sqrt(wr * (1 - wr) / args.games) if 0 < wr < 1 else 0
    print(f"A wins {a_wins}/{args.games} = {wr:.3f}  (CI +/-{1.96*se:.3f})")


if __name__ == "__main__":
    main()
