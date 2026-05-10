"""Bench a checkpoint vs heuristic; publish to public/weights/expert.json if
the net wins at least `--min-winrate` of N games.

Used by the operator to decide which checkpoint becomes Expert.
"""
from __future__ import annotations
import argparse
import math
import os
import random
import shutil
import sys

from net import Net
from bench import play_one_game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--min-winrate", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--dst", default="../public/weights/expert.json")
    args = ap.parse_args()

    net = Net.load_json(args.weights)
    rng = random.Random(args.seed)
    wins = 0
    n = args.games
    for g in range(n):
        net_side = g & 1
        wins += play_one_game(net_side, net, rng)
        if (g + 1) % 50 == 0:
            print(f"  game {g+1}/{n} winrate={wins/(g+1):.3f}", flush=True)
    wr = wins / n
    se = math.sqrt(wr * (1 - wr) / n) if 0 < wr < 1 else 0
    print(f"NET vs HEURISTIC: {wins}/{n} = {wr:.3f} (CI ±{1.96*se:.3f})")

    if wr >= args.min_winrate:
        os.makedirs(os.path.dirname(args.dst), exist_ok=True)
        shutil.copy(args.weights, args.dst)
        sz = os.path.getsize(args.dst)
        print(f"PUBLISHED -> {args.dst} ({sz} bytes)")
        return 0
    print(f"NOT publishing (winrate {wr:.3f} < {args.min_winrate})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
