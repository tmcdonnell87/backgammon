"""Race-only distillation: warm-start for the race specialist net.

Collects positions from heuristic self-play, accepting only those where
`is_race(p)` (no contact possible — opp's lowest checker is past our
highest). Trains the standard 4-output net against the same soft heuristic
targets as distill.py.

Race is mostly a function of pip count and bear-off efficiency, so a small
net (`--hidden 40`) is sufficient. We size the warm-start at 100k positions
(vs 200k for the contact distill) since the data is more redundant.

Usage:
   cd training
   ../.venv/bin/python distill_race.py \\
       --positions 100000 --epochs 20 --hidden 40 \\
       --lr 0.15 --out runs/distill-race --seed 1
"""
from __future__ import annotations
import argparse
import random
import time

from engine import (
    starting_position, mirror, generate_plays, check_win,
)
from bench import heuristic_value, pick_with
from race_filter import is_race
from distill import train_distill


def collect_race_positions(target: int, rng: random.Random,
                           max_plies: int = 250) -> list:
    """Heuristic-vs-heuristic self-play; accept only race-phase positions.

    Each game is played fully (heuristic vs heuristic). We sample every
    visited position; race positions are kept. Most early-game plies are
    contact, so the sampler runs many games before saturating the target.
    """
    out = []
    games = 0
    while len(out) < target:
        games += 1
        p = starting_position()
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        while d1 == d2:
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        plies = 0
        while plies < max_plies:
            plies += 1
            plays = generate_plays(p, d1, d2)
            if len(plays) == 1 and len(plays[0][0]) == 0:
                after = p
            else:
                idx = pick_with(p, plays, heuristic_value)
                _pl, after = plays[idx]
            if is_race(after):
                out.append(after)
                if len(out) >= target:
                    return out
            win = check_win(after)
            if win is not None:
                break
            p = mirror(after)
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/distill-race")
    ap.add_argument("--positions", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", default="40",
                    help="int (one hidden layer) or comma list (default: 40)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init", default=None,
                    help="warm-start from existing weights JSON")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"collecting {args.positions} race-phase positions via heuristic self-play...",
          flush=True)
    t0 = time.time()
    positions = collect_race_positions(args.positions, rng)
    print(f"collected {len(positions)} in {time.time() - t0:.1f}s", flush=True)

    hidden_arg = args.hidden
    if "," in hidden_arg:
        hidden_layers = [int(s) for s in hidden_arg.split(",") if s]
    else:
        hidden_layers = int(hidden_arg)
    train_distill(positions, hidden_layers, args.epochs, args.lr,
                  args.batch, args.out, seed=args.seed, init=args.init)


if __name__ == "__main__":
    main()
