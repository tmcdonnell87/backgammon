"""TD(λ) self-play trainer for backgammon, 4-output edition.

Both players use the same network. Each move:
  1. Generate legal plays for the current dice.
  2. For each candidate, evaluate the resulting position from the *opponent's*
     perspective (mirror after applying play). Pick the play that maximizes
     our cubeless equity (= -opp equity).
  3. Apply the chosen play, mirror, roll new dice. Repeat.
  4. After each move, run a TD(λ) update with a 4-head sigmoid target
     (p_win, p_gammon_win, p_loss, p_gammon_loss). Perspective flips between
     plies; the head vector permutes by SWAP_WL when crossing a mirror.

Single-thread torch CPU here; Phase B introduces multiprocessing in
parallel_train.py. CLI:

   python3 train.py --games 10000 --out runs/run1

writes runs/run1/weights-{N}.json + weights-latest.json + log.csv.
"""
from __future__ import annotations
import argparse
import csv
import os
import random
import signal
import time
from typing import Optional

import torch

from engine import (
    Position, starting_position, generate_plays, mirror, check_win,
)
from encoding import encode
from net_torch import Net, equity_from_probs, P_W, P_GW, P_L, P_GL
from outcome import classify_outcome
from race_filter import still_in_contact, is_race
from rollout import rollout_target_4vec_us_frame


# Hyperparameters. The 4-output head shares the trunk (W1, b1) across all
# heads, so the W1 update is the sum of 4 per-head gradients each weighted by
# its own TD error. Effective step on shared params is ~4× the scalar case;
# 0.03 (which worked for the legacy single-output net) diverges here. 0.01
# with λ=0.5 trains stably. Per-trace effective step α/(1−λ) stays < ~0.05.
ALPHA = 0.01
LAMBDA = 0.5
GAMMA = 1.0           # episodic, no discount inside a game

# Permutation that converts a 4-head value vector from one perspective to the
# opposite perspective: swap W with L heads. (p_w, p_gw, p_l, p_gl) viewed
# from the other player becomes (p_l, p_gl, p_w, p_gw).
SWAP_WL = [P_L, P_GL, P_W, P_GW]


def _swap_wl(v: torch.Tensor) -> torch.Tensor:
    return v[SWAP_WL]


def _argmax_play(p: Position, plays, net: Net) -> int:
    """Return index of the play that maximizes our cubeless equity."""
    if len(plays) == 1:
        return 0
    best_i = 0
    best_score = -float("inf")
    for i, (_play, after) in enumerate(plays):
        win = check_win(after)
        if win is not None:
            # Terminal after our play => we won (we can only push our own
            # checkers off). base_points = 1/2/3 already encodes gammon.
            _winner_abs, base_pts = win
            us_eq = float(base_pts)
        else:
            opp_view = mirror(after)
            y_opp, _h, _x = net.forward(encode(opp_view))
            # Equity in opp frame is opp's equity; ours is the negation.
            us_eq = -float(equity_from_probs(y_opp).item())
        if us_eq > best_score:
            best_score = us_eq
            best_i = i
    return best_i


def _roll(rng: random.Random):
    return rng.randint(1, 6), rng.randint(1, 6)


def _phase_matches_input(prev_was_contact: Optional[bool], mode: str) -> bool:
    """Should this ply's TD update fire?

    Phase is the phase of the INPUT to the model (= prev_x's underlying state
    = the position at the start of this ply). In contact mode we only update
    the contact net on contact inputs; in race mode only on race inputs.
    For boundary plies where the input was contact but `after` is race, the
    contact net still updates — its target uses the frozen race net's value
    (the cross-net bootstrap) when one is provided.
    """
    if prev_was_contact is None:
        return False
    if mode == "all":
        return True
    if mode == "contact":
        return prev_was_contact
    if mode == "race":
        return not prev_was_contact
    raise ValueError(f"unknown mode: {mode}")


def play_game(net: Net, alpha: float, lam: float, rng: random.Random,
              learn: bool = True, mode: str = "all",
              rollout_fraction: float = 0.0,
              bootstrap_net: Optional[Net] = None) -> dict:
    """Play one self-play game, doing TD(λ) updates if learn=True.

    All TD bookkeeping uses 4-head value vectors. `prev_y` is always stored
    in the *new mover's* frame so that subsequent steps' values (also in
    new mover frame) can be subtracted directly. Crossing a mirror swaps
    W↔L heads (SWAP_WL).

    `mode` selects which plies trigger TD updates:
      * "all"     — update every ply (Phase A/B default).
      * "contact" — update only on plies whose INPUT (prev_x) is contact.
                    Targets at the contact↔race boundary come from
                    `bootstrap_net` when one is provided (Phase F cross-net
                    bootstrap); otherwise from `net` itself.
      * "race"    — update only on plies whose INPUT (prev_x) is race.

    `bootstrap_net`: optional frozen forward-only Net (no traces, no
    gradient). When provided AND mode=='contact' AND `after` is a race
    position, the TD target for that ply uses `bootstrap_net`'s value
    instead of `net`'s. Used by Phase F to train the contact net against
    the race specialist net's evaluation at boundary states.

    `rollout_fraction` ∈ [0, 1]: per ply, with this probability replace the
    0-ply TD bootstrap target (swap_wl(y_opp)) with a 2-ply expectimax
    rollout target. Cost ~20x per affected ply; keep small (0.05–0.10).
    Phase D.
    """
    p = starting_position()
    # Opening roll: re-roll doubles.
    d1, d2 = _roll(rng)
    while d1 == d2:
        d1, d2 = _roll(rng)
    if d1 < d2:  # convention: bigger goes first; doesn't really matter
        d1, d2 = d2, d1

    if learn:
        net.reset_traces()

    # State_t for TD update: previous mover's frame value cache.
    prev_x = None
    prev_h = None
    prev_y = None  # 4-vector in (about-to-move) mover's frame.
    # Phase of the position at the start of the CURRENT ply (= prev_x's
    # underlying state). Used to decide whether to fire the TD update under
    # `mode`. Stays in sync with `p`: still_in_contact(p) at any iteration's
    # head, since mirroring doesn't change phase.
    prev_was_contact: Optional[bool] = None

    plies = 0
    winner = None  # absolute side winner
    base_pts = 1

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
            winner = winner_abs
            # `after` is from p.turn's frame; p.turn just played and (since we
            # can only push our own checkers off) is the winner. prev_y was
            # cached in the new-mover frame, which is exactly p.turn (we
            # haven't mirrored yet this iter). So the terminal target is in
            # the same frame as prev_y — no swap.
            target = torch.tensor(
                classify_outcome(after, p.turn), dtype=torch.float32
            )
            if (learn and prev_x is not None
                    and _phase_matches_input(prev_was_contact, mode)):
                td_err = target - prev_y
                net.td_step(prev_x, prev_y, prev_h, td_err, alpha, lam)
            break

        # Non-terminal: evaluate at opp view (i.e. from the next mover's
        # perspective). The cached prev_y is in *this* iteration's mover
        # frame (i.e. p.turn). Crossing the mirror swaps W↔L heads, so
        # cur-in-prev-frame = swap_wl(y_opp).
        opp_view = mirror(after)
        x_opp = encode(opp_view)
        y_opp, h_opp, x_opp_t = net.forward(x_opp)

        if (learn and prev_x is not None
                and _phase_matches_input(prev_was_contact, mode)):
            # Cross-net bootstrap: when training the contact net AND the
            # position after our play has crossed into the race phase, use
            # the frozen race net's evaluation as the target (boundary ply).
            use_bootstrap = (
                bootstrap_net is not None
                and mode == "contact"
                and is_race(after)
            )
            if use_bootstrap:
                # Forward-only; no traces, no gradient.
                with torch.no_grad():
                    y_boot, _h_b, _x_b = bootstrap_net.forward(x_opp)
                cur_in_prev_frame = _swap_wl(y_boot)
            elif rollout_fraction > 0.0 and rng.random() < rollout_fraction:
                # 2-ply expectimax target in prev-mover (= mover_k) frame.
                cur_in_prev_frame = rollout_target_4vec_us_frame(net, after)
            else:
                cur_in_prev_frame = _swap_wl(y_opp)
            td_err = cur_in_prev_frame - prev_y
            net.td_step(prev_x, prev_y, prev_h, td_err, alpha, lam)

        # Set prev to (x_opp, h_opp, y_opp) — this is the new mover's frame.
        prev_x = x_opp_t
        prev_h = h_opp
        prev_y = y_opp
        # `after` and `opp_view` share phase (mirror invariant); both are
        # the start of the NEXT ply. Cache that phase for the next iter's
        # TD-update gate.
        prev_was_contact = still_in_contact(after)
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
                 hidden=120, seed: int = 1, resume: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    if resume and os.path.exists(resume):
        net = Net.load_json(resume)
        print(f"resumed from {resume} (hidden_layers={net.hidden_layers})",
              flush=True)
    else:
        net = Net(hidden=hidden, seed=seed)
        print(f"fresh net hidden_layers={net.hidden_layers}", flush=True)

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
            jpath = os.path.join(out_dir, f"weights-{g}.json")
            net.save_json(jpath)
            net.save_json(os.path.join(out_dir, "weights-latest.json"))
            print(f"checkpoint games={g} -> {jpath}", flush=True)

        if stop["requested"]:
            break

    # Final checkpoint.
    jpath = os.path.join(out_dir, f"weights-final.json")
    net.save_json(jpath)
    net.save_json(os.path.join(out_dir, "weights-latest.json"))
    print(f"FINAL games={g} -> {jpath}", flush=True)
    log_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--lambda", dest="lam", type=float, default=LAMBDA)
    ap.add_argument("--hidden", default="120",
                    help="int or comma list like '200,200' (multi-layer)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    if "," in args.hidden:
        hidden = [int(s) for s in args.hidden.split(",") if s]
    else:
        hidden = int(args.hidden)
    run_training(args.out, args.games, args.ckpt_every,
                 alpha=args.alpha, lam=args.lam, hidden=hidden,
                 seed=args.seed, resume=args.resume)


if __name__ == "__main__":
    main()
