"""Benchmark our net against GNU Backgammon over a number of games.

gnubg is slow via subprocess (~5-10 moves/sec including parsing), so bench
runs are capped at a small number of games (~200) and used as a gold-standard
publishing gate, not for inner-loop training feedback.

Architecture: we play both sides programmatically. On the net's turn we use
our usual `pick_with` + neural evaluator. On gnubg's turn we send the current
position to gnubg via `set board`, ask for `hint`, parse gnubg's best play
suggestion, and apply it ourselves.

Usage (after installing gnubg via `sudo apt-get install gnubg`):
   ../.venv/bin/python bench_gnubg.py --weights public/weights/expert.json \\
       --games 200 --ply 0
"""
from __future__ import annotations
import argparse
import math
import random
import sys
import time

from engine import (
    starting_position, generate_plays, mirror, check_win,
)
from encoding import encode
from match_nets import load_net_auto
from bench import pick_with
from gnubg_client import GnubgClient, gnubg_installed


def _pick_with_gnubg(client: GnubgClient, p, plays):
    """Ask gnubg for its preferred play. We approximate by evaluating each
    candidate's resulting position and picking the one that maximizes our
    cubeless equity at gnubg's eval depth."""
    if len(plays) == 1:
        return 0
    best_i, best_eq = 0, -math.inf
    for i, (_pl, after) in enumerate(plays):
        win = check_win(after)
        if win is not None:
            _w, base = win
            us_eq = float(base)
        else:
            opp_view = mirror(after)
            r = client.evaluate_position(opp_view)
            if r is None:
                continue
            # gnubg returned opp_view's eval (player on roll = opp). Equity is
            # opp-frame; us-frame = -opp_equity.
            us_eq = -r.equity
        if us_eq > best_eq:
            best_eq = us_eq
            best_i = i
    return best_i


def play_one_game(net_side: int, net_eq_fn, gnubg_client: GnubgClient,
                  rng: random.Random) -> int:
    p = starting_position()
    d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
    while d1 == d2:
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
    turn_is_net = (net_side == 0)
    while True:
        plays = generate_plays(p, d1, d2)
        if len(plays) == 1 and len(plays[0][0]) == 0:
            after = p
        else:
            if turn_is_net:
                idx = pick_with(p, plays, net_eq_fn)
            else:
                idx = _pick_with_gnubg(gnubg_client, p, plays)
            _play, after = plays[idx]
        win = check_win(after)
        if win is not None:
            winner_abs, _ = win
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
    ap.add_argument("--ply", type=int, default=0,
                    help="gnubg eval ply (0 = fast, 2 = world-class but slow)")
    args = ap.parse_args()

    if not gnubg_installed():
        print("gnubg not in PATH. Install via: sudo apt-get install gnubg")
        sys.exit(1)

    _net, net_eq = load_net_auto(args.weights)
    client = GnubgClient(ply=args.ply)
    rng = random.Random(args.seed)
    wins = 0
    t0 = time.time()
    try:
        for g in range(args.games):
            net_side = g & 1
            wins += play_one_game(net_side, net_eq, client, rng)
            if (g + 1) % 10 == 0:
                wr = wins / (g + 1)
                elapsed = time.time() - t0
                print(f"  game {g+1}/{args.games} net_winrate={wr:.3f}  "
                      f"elapsed={elapsed:.0f}s ({(g+1)/elapsed:.1f}g/s)",
                      flush=True)
    finally:
        client.close()
    wr = wins / args.games
    se = math.sqrt(wr * (1 - wr) / args.games) if 0 < wr < 1 else 0
    print(f"NET vs GNUBG-{args.ply}ply: {wins}/{args.games} = {wr:.3f}  "
          f"(95% CI ±{1.96 * se:.3f})  elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
