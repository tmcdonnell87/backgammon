"""TD(λ) self-play trainer for backgammon.

Both players use the same network. Each move:
  1. Generate legal plays for the current dice.
  2. For each candidate, evaluate the resulting position from the *opponent's*
     perspective (mirror after applying play). The score "we keep this play if
     it minimizes their winning probability" -> pick play with lowest opp-y.
  3. Apply the chosen play, mirror, roll new dice. Repeat.
  4. After each move, run a TD(λ) update: previous y vs current y from the
     mover's perspective. Because perspective flips between moves, we negate
     across mirroring.

We run on a single thread; numpy keeps it fast enough (~10-20 games/sec
on CPU at H=80).

CLI:
   python3 train.py --games 10000 --out runs/run1
will write checkpoints to runs/run1/ckpt-{N}.npz and weights-{N}.json,
plus runs/run1/log.csv.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import signal
import sys
import time
from typing import Optional

import numpy as np

from engine import (
    Position, starting_position, generate_plays, apply_play, mirror,
    check_win,
)
from encoding import encode
from net import Net, save_npz, load_npz


# Hyperparameters. With our 1-output sigmoid + eligibility traces, alpha=0.1
# diverges (weight norms blow up after ~5k games of self-play). 0.03 with
# lambda=0.5 trains stably; the effective per-trace step is alpha/(1-lambda)
# which we want < ~0.1.
ALPHA = 0.03
LAMBDA = 0.5
GAMMA = 1.0           # episodic, no discount inside a game


def _argmax_play(p: Position, plays, net: Net) -> int:
    """Return index of play that minimizes opponent's winning probability
    (i.e. best for us)."""
    if len(plays) == 1:
        return 0
    best_i = 0
    best_score = -1.0
    for i, (_play, after) in enumerate(plays):
        win = check_win(after)
        if win is not None:
            # Terminal after our play. Winner is p.turn (us). Score = 1 (we win).
            us_y = 1.0
        else:
            opp_view = mirror(after)
            opp_y = net.value(encode(opp_view))
            us_y = 1.0 - opp_y
        if us_y > best_score:
            best_score = us_y
            best_i = i
    return best_i


def _roll(rng: random.Random):
    return rng.randint(1, 6), rng.randint(1, 6)


def play_game(net: Net, alpha: float, lam: float, rng: random.Random,
              learn: bool = True) -> dict:
    """Play one self-play game, doing TD(λ) updates if learn=True.

    Returns stats dict.
    """
    p = starting_position()
    # Opening roll: re-roll doubles.
    d1, d2 = _roll(rng)
    while d1 == d2:
        d1, d2 = _roll(rng)
    if d1 < d2:  # convention: bigger goes first; doesn't really matter, we just play
        d1, d2 = d2, d1

    if learn:
        net.reset_traces()

    # State_t for TD update: (x_t, h_t, y_t) of the mover whose move just produced it.
    # On first move there's no previous state.
    prev_x = None
    prev_h = None
    prev_y = None
    prev_sign = 1   # +1 if prev_y is from current mover's view; flips on mirror

    plies = 0
    winner = None  # absolute side winner
    base_pts = 1
    first_perspective = 0  # us at start = side 0

    while True:
        plies += 1
        # Generate plays for current dice.
        plays = generate_plays(p, d1, d2)
        # If only the no-op play (all dice unusable), skip move.
        if len(plays) == 1 and len(plays[0][0]) == 0:
            after = p
        else:
            idx = _argmax_play(p, plays, net)
            _play, after = plays[idx]

        # Check terminal.
        win = check_win(after)
        if win is not None:
            winner_abs, base_pts = win
            # Map to "first_perspective" frame: did first_perspective win?
            # Mover at this turn: depends on how many flips. We don't track
            # absolute turn; just use win_abs vs first_perspective.
            winner = winner_abs
            # TD update: target = 1 if mover wins, 0 if loses.
            # `after` is from mover's perspective (haven't mirrored yet).
            # winner_abs == p.turn means mover wins => target = 1 from mover view.
            mover_target = 1.0 if winner_abs == p.turn else 0.0
            if learn and prev_x is not None:
                # prev_y was from prev_mover view. Current mover view value = mover_target.
                # If perspective flipped (prev_sign = -1), then in prev_mover frame
                # current value = 1 - mover_target.
                cur_in_prev_frame = mover_target if prev_sign == 1 else 1.0 - mover_target
                td_err = cur_in_prev_frame - prev_y
                net.td_step(prev_x, prev_y, prev_h, td_err, alpha, lam)
            break

        # Non-terminal: get y_t for current mover at `after` BEFORE mirroring.
        # That value = "mover's win probability after their play" = 1 - opp_view_y.
        opp_view = mirror(after)
        x_opp, = (encode(opp_view),)
        y_opp, h_opp, _ = net.forward(x_opp)
        # "mover view value" = 1 - y_opp; but for TD, easier to use opp-view value
        # consistently. We use mover-view: y_t_mover = 1 - y_opp.
        # However eligibility traces need grad of y_t we used. We'll do TD in
        # opponent-frame because that's what we have h for: define value as
        # P(current-to-move wins) and compute traces at the *new* mover (the opp).
        # Simpler: we run TD in the about-to-move perspective each step.

        # Switch to next mover frame.
        if learn and prev_x is not None:
            # cur value in prev mover frame:
            #   y_opp is value for the NEW mover (opp) of "opp wins"
            #   In prev mover frame (which IS the opp's opp), the prev mover's
            #   win prob equals 1 - y_opp.
            cur_in_prev_frame = 1.0 - y_opp
            td_err = cur_in_prev_frame - prev_y
            net.td_step(prev_x, prev_y, prev_h, td_err, alpha, lam)

        # Set prev to (x_opp, h_opp, y_opp) — this is the new mover's frame.
        prev_x = x_opp
        prev_h = h_opp
        prev_y = y_opp
        # Flip perspective.
        p = opp_view
        d1, d2 = _roll(rng)

    return {
        "plies": plies,
        "winner": winner,
        "base_pts": base_pts,
    }


def run_training(out_dir: str, games: int, ckpt_every: int,
                 alpha: float = ALPHA, lam: float = LAMBDA,
                 hidden: int = 80, seed: int = 1, resume: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    if resume and os.path.exists(resume):
        net = load_npz(resume)
        print(f"resumed from {resume} (hidden={net.hidden})", flush=True)
    else:
        net = Net(hidden=hidden, seed=seed)
        print(f"fresh net hidden={hidden}", flush=True)

    rng = random.Random(seed)

    log_path = os.path.join(out_dir, "log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", buffering=1)
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["games", "plies_avg", "side0_winrate", "elapsed_s"])

    # Counters since last checkpoint.
    plies_sum = 0
    side0_wins = 0
    window = max(50, ckpt_every // 10)
    win_window = []

    stop = {"requested": False}

    def _sigterm(_signo, _frame):
        stop["requested"] = True
        print("stop requested", flush=True)

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    t_start = time.time()

    for g in range(1, games + 1):
        stats = play_game(net, alpha, lam, rng, learn=True)
        plies_sum += stats["plies"]
        # winner field is absolute side index. side 0 in our self-play is the
        # first mover at game start, but we don't track absolute alignment;
        # instead just note 1 if winner == 0 else 0 to gauge symmetry / drift.
        if stats["winner"] == 0:
            side0_wins += 1
        win_window.append(1 if stats["winner"] == 0 else 0)
        if len(win_window) > window:
            win_window.pop(0)

        if g % 50 == 0:
            elapsed = time.time() - t_start
            avg_plies = plies_sum / g
            wr = sum(win_window) / len(win_window)
            print(f"games={g} avg_plies={avg_plies:.1f} side0_wr={wr:.2f} "
                  f"elapsed={elapsed:.0f}s rate={g/elapsed:.1f}g/s",
                  flush=True)
            log.writerow([g, f"{avg_plies:.2f}", f"{wr:.3f}", f"{elapsed:.1f}"])

        if g % ckpt_every == 0:
            ckpt = os.path.join(out_dir, f"ckpt-{g}.npz")
            jpath = os.path.join(out_dir, f"weights-{g}.json")
            save_npz(net, ckpt)
            net.save_json(jpath)
            # also publish as latest
            net.save_json(os.path.join(out_dir, "weights-latest.json"))
            print(f"checkpoint games={g} -> {ckpt}, {jpath}", flush=True)

        if stop["requested"]:
            break

    # Final checkpoint.
    ckpt = os.path.join(out_dir, f"ckpt-final.npz")
    jpath = os.path.join(out_dir, f"weights-final.json")
    save_npz(net, ckpt)
    net.save_json(jpath)
    net.save_json(os.path.join(out_dir, "weights-latest.json"))
    print(f"FINAL games={g} -> {ckpt}, {jpath}", flush=True)
    log_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--lambda", dest="lam", type=float, default=LAMBDA)
    ap.add_argument("--hidden", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    run_training(args.out, args.games, args.ckpt_every,
                 alpha=args.alpha, lam=args.lam, hidden=args.hidden,
                 seed=args.seed, resume=args.resume)


if __name__ == "__main__":
    main()
