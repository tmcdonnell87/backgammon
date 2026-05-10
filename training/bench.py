"""Play games of (Net vs Heuristic) at 0-ply move selection and report
winrate of the net side. Useful to decide when to publish a checkpoint.

Usage:
   python3 bench.py --weights runs/expert/weights-latest.json --games 200
"""
from __future__ import annotations
import argparse
import math
import random
import time
from typing import Callable

import numpy as np

from engine import (
    Position, starting_position, mirror, generate_plays, check_win, POINTS,
    pip_count, pip_count_them,
)
from encoding import encode
from net import Net


# --- Heuristic port (kept faithful to src/ai/heuristic.ts) ---------------

W_PIP = 0.0035
W_OFF = 0.07
W_BAR = 0.18
W_POINT_MID = 0.018
W_POINT_GOLDEN = 0.045
W_POINT_BAR = 0.040
W_POINT_HOME = 0.030
W_HOME_BUILDERS = 0.008
W_BLOT_BASE = 0.012
W_OUR_HOME_BLOT = 0.018
W_BLOT_SAFE = 0.004
W_OPP_HOME_ANCHOR = 0.06
W_PRIME_PER = 0.08
W_FULL_PRIME = 0.18
W_CLOSEOUT = 0.30
W_BACK_LONE = 0.025
W_HIT_OPP_BLOT = 0.05


def _still_in_contact(p: Position) -> bool:
    if p.bar_us > 0 or p.bar_them > 0:
        return True
    ours_min = POINTS
    opps_max = -1
    for i in range(POINTS):
        if p.points[i] > 0 and i < ours_min:
            ours_min = i
        if p.points[i] < 0 and i > opps_max:
            opps_max = i
    if ours_min == POINTS or opps_max == -1:
        return False
    return ours_min <= opps_max


def _shots_to_prob(needed):
    if not needed:
        return 0
    count = 0
    for i in range(1, 7):
        for j in range(1, 7):
            sums = {i, j}
            if i == j:
                sums |= {2 * i, 3 * i, 4 * i}
            else:
                sums.add(i + j)
            if any(n in sums for n in needed):
                count += 1
    return count / 36.0


def _blot_shot_prob(p: Position, i: int) -> float:
    distances = set()
    if p.bar_them > 0:
        if 0 <= i <= 5:
            distances.add(i + 1)
    else:
        for j in range(POINTS):
            if p.points[j] >= 0:
                continue
            d = i - j
            if 0 < d <= 24:
                distances.add(d)
    return _shots_to_prob(distances)


def _sided_score(p: Position) -> float:
    s = -W_PIP * pip_count(p)
    s += W_OFF * p.off_us
    s += -W_BAR * p.bar_us
    for i in range(POINTS):
        v = p.points[i]
        if v >= 2:
            w = W_POINT_MID
            if i <= 5:
                w = W_POINT_HOME
            if i == 4:
                w = W_POINT_GOLDEN
            if i == 6:
                w = W_POINT_BAR
            s += w
            if v > 2:
                s += W_HOME_BUILDERS * (v - 2)
        elif v == 1:
            shot = _blot_shot_prob(p, i)
            w = W_BLOT_BASE
            if i <= 5:
                w = W_OUR_HOME_BLOT
            if shot < 0.05:
                w = W_BLOT_SAFE
            s -= w * (1 + shot * 6)
    for i in range(18, POINTS):
        if p.points[i] >= 2:
            s += W_OPP_HOME_ANCHOR
    prime_run = best_prime = 0
    for i in range(0, 8):
        if p.points[i] >= 2:
            prime_run += 1
            if prime_run > best_prime:
                best_prime = prime_run
        else:
            prime_run = 0
    if best_prime >= 2:
        s += W_PRIME_PER * (best_prime - 1)
    if best_prime >= 6:
        s += W_FULL_PRIME
    closed = all(p.points[i] >= 2 for i in range(0, 6))
    if closed and p.bar_them > 0:
        s += W_CLOSEOUT
    for i in range(18, POINTS):
        if p.points[i] == 1:
            s -= W_BACK_LONE
    for i in range(POINTS):
        if p.points[i] != -1:
            continue
        can_hit = False
        if p.bar_us > 0:
            ed = 24 - i
            if 1 <= ed <= 6:
                can_hit = True
        else:
            for d in range(1, 7):
                src = i + d
                if 0 <= src < POINTS and p.points[src] >= 1:
                    can_hit = True
                    break
        if can_hit:
            s += W_HIT_OPP_BLOT
    return s


def heuristic_value(p: Position) -> float:
    """Returns equity in [-1, 1] from us perspective."""
    if not _still_in_contact(p):
        pip_diff = pip_count_them(p) - pip_count(p)
        return math.tanh(0.045 * pip_diff + 0.6 * (p.off_us - p.off_them) / 15)
    return math.tanh(_sided_score(p) - _sided_score(mirror(p)))


# --- Match harness -------------------------------------------------------

def pick_with(p: Position, plays, score_fn_after_play) -> int:
    """Pick play index minimizing opponent score on resulting position."""
    if len(plays) == 1:
        return 0
    best_i, best = 0, -math.inf
    for i, (_pl, after) in enumerate(plays):
        win = check_win(after)
        if win is not None:
            us = 1.0
        else:
            opp = mirror(after)
            opp_score = score_fn_after_play(opp)
            us = -opp_score  # they're maximizing their value, so our value is negated
        if us > best:
            best = us
            best_i = i
    return best_i


def play_one_game(net_side: int, net: Net, rng: random.Random) -> int:
    """Returns 1 if the side that started as net_side won, else 0."""
    p = starting_position()
    d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
    while d1 == d2:
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)

    # turn 0 means net_side is on roll
    turn_is_net = (net_side == 0)
    while True:
        plays = generate_plays(p, d1, d2)
        if len(plays) == 1 and len(plays[0][0]) == 0:
            after = p
        else:
            if turn_is_net:
                # net evaluates from "us" perspective via 1-2y mapping.
                idx = pick_with(p, plays,
                                lambda pos: 2 * net.value(encode(pos)) - 1)
            else:
                idx = pick_with(p, plays, heuristic_value)
            _play, after = plays[idx]
        win = check_win(after)
        if win is not None:
            winner_abs, base = win
            net_won = (winner_abs == 0 and net_side == 0) or \
                      (winner_abs == 1 and net_side == 1)
            return 1 if net_won else 0
        p = mirror(after)
        turn_is_net = not turn_is_net
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    net = Net.load_json(args.weights)
    rng = random.Random(args.seed)
    n = args.games
    wins = 0
    t0 = time.time()
    for g in range(n):
        # Alternate which side is the net to remove first-mover bias.
        net_side = g & 1
        wins += play_one_game(net_side, net, rng)
        if (g + 1) % 50 == 0:
            wr = wins / (g + 1)
            print(f"  game {g+1}/{n} net_winrate={wr:.3f}", flush=True)
    elapsed = time.time() - t0
    wr = wins / n
    # Wald 95% CI
    se = math.sqrt(wr * (1 - wr) / n) if 0 < wr < 1 else 0
    print(f"NET vs HEURISTIC: {wins}/{n} = {wr:.3f}  "
          f"(95% CI ±{1.96 * se:.3f})  elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
