"""Play two nets head-to-head over N games. Reports A's winrate."""
import argparse
import math
import random
import sys
from net import Net
from engine import starting_position, mirror, generate_plays, check_win
from encoding import encode


def pick_with_net(p, plays, net):
    if len(plays) == 1:
        return 0
    best_i, best = 0, -math.inf
    for i, (_pl, after) in enumerate(plays):
        win = check_win(after)
        if win is not None:
            us = 1.0
        else:
            opp = mirror(after)
            opp_y = net.value(encode(opp))
            us = 1.0 - opp_y
        if us > best:
            best = us
            best_i = i
    return best_i


def play_one(a_side, A, B, rng):
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
            net = A if a_to_move else B
            idx = pick_with_net(p, plays, net)
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
    A = Net.load_json(args.A)
    B = Net.load_json(args.B)
    rng = random.Random(args.seed)
    a_wins = 0
    for g in range(args.games):
        a_wins += play_one(g & 1, A, B, rng)
    wr = a_wins / args.games
    se = math.sqrt(wr * (1 - wr) / args.games) if 0 < wr < 1 else 0
    print(f"A wins {a_wins}/{args.games} = {wr:.3f}  (CI +/-{1.96*se:.3f})")


if __name__ == "__main__":
    main()
