"""Move-selection accuracy bench: net vs gnubg.

For each sampled (position, dice) pair, generates legal plays, evaluates
every play via both the net and gnubg, and reports:

  * top-1 agreement: how often the net's best play matches gnubg's best.
  * top-3 agreement: how often the net's best is in gnubg's top-3.
  * equity loss: gnubg's equity of its pick minus gnubg's equity of
    the net's pick (≥ 0, in gnubg's equity scale ~[-3, +3]). The mean
    across positions answers "by how much do we lose, on average, by
    picking with the net instead of gnubg?".
  * Pearson correlation and signed bias of net vs gnubg equity
    across all (position, play) pairs.

`--ply` controls gnubg's eval depth (default 2). `--net-ply` controls the
net's eval depth on the comparison side (default 0; set to 2 to match the
in-app hint engine's 2-ply expectimax via search.ts:score2ply).

Speed: gnubg is the bottleneck (~5-15 evals/sec via subprocess). 200
positions × ~5 candidate plays each ≈ 1000 gnubg evals ≈ 1-3 min wall.
The net's 2-ply path costs ~21× a 0-ply forward; still tractable in
single-digit minutes for 200 positions.

Usage:
    cd training
    ../.venv/bin/python bench_move_accuracy.py \\
        --weights ../public/weights/expert.json \\
        --positions 200 --seed 1 --net-ply 2
"""
from __future__ import annotations
import argparse
import math
import random
import sys
import time
from typing import List, Tuple

from engine import (
    starting_position, mirror, generate_plays, check_win, Position,
    board_hash,
)
from bench import heuristic_value, pick_with
from match_nets import load_net_auto
from gnubg_client import GnubgClient, gnubg_installed
from rollout import ALL_ROLLS
from bearoff_verify import load_bearoff_bin, bearoff_equity, is_pure_race


def collect_mid_game_positions(target: int, rng: random.Random,
                               max_plies_per_game: int = 200) -> List[Position]:
    """Play heuristic self-play; save every visited position. We accept
    positions where the game is ongoing and at least one player has moved
    a checker off the back point (i.e. skip pure opening positions).
    """
    out: List[Position] = []
    while len(out) < target:
        p = starting_position()
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        while d1 == d2:
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        plies = 0
        while plies < max_plies_per_game:
            plies += 1
            plays = generate_plays(p, d1, d2)
            if len(plays) == 1 and len(plays[0][0]) == 0:
                after = p
            else:
                idx = pick_with(p, plays, heuristic_value)
                _pl, after = plays[idx]
            if plies > 3:  # skip very early plies
                out.append(after)
                if len(out) >= target:
                    return out
            win = check_win(after)
            if win is not None:
                break
            p = mirror(after)
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
    return out


def gnubg_equity_for_play(client: GnubgClient, p: Position,
                          after: Position) -> float:
    """Return gnubg's equity for OUR side after we play `play` from `p`."""
    win = check_win(after)
    if win is not None:
        _winner_abs, base_pts = win
        # Terminal: we won (we can only push our own checkers off).
        return float(base_pts)
    opp_view = mirror(after)
    r = client.evaluate_position(opp_view)
    if r is None:
        return float("nan")
    return -r.equity  # opp_view's equity is opp's; ours is the negation


def net_equity_for_play(net_eq_fn, p: Position, after: Position) -> float:
    win = check_win(after)
    if win is not None:
        _winner_abs, base_pts = win
        return float(base_pts)
    opp_view = mirror(after)
    return -net_eq_fn(opp_view)


def net_equity_for_play_2ply(net_eq_fn, p: Position, after: Position) -> float:
    """2-ply expectimax in the post-play (us) frame. Mirrors search.ts:score2ply.

    For each of 21 dice rolls, opp generates plays; opp picks the play that
    maximizes opp's equity (= minimizes ours). We expectation over rolls.
    Inner leaf calls `net_eq_fn` (0-ply us-frame equity).
    """
    win = check_win(after)
    if win is not None:
        _winner_abs, base_pts = win
        return float(base_pts)
    opp_view = mirror(after)
    total = 0.0
    for d1, d2, prob in ALL_ROLLS:
        opp_plays = generate_plays(opp_view, d1, d2)
        # Dedupe by final position to avoid duplicate work for equivalent
        # orderings (same final state, different sub-move sequence).
        seen = set()
        unique = []
        for _pl, opp_after in opp_plays:
            h = bytes(board_hash(opp_after))
            if h not in seen:
                seen.add(h)
                unique.append(opp_after)
        # Opp picks play maximizing opp-frame equity.
        best_opp_eq = -math.inf
        for opp_after in unique:
            owin = check_win(opp_after)
            if owin is not None:
                _wa, obp = owin
                # Terminal at opp's turn => opp wins => opp-frame equity = +obp.
                opp_eq = float(obp)
            else:
                us_again = mirror(opp_after)
                us_eq = net_eq_fn(us_again)
                opp_eq = -us_eq
            if opp_eq > best_opp_eq:
                best_opp_eq = opp_eq
        total += prob * (-best_opp_eq)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--ply", type=int, default=2,
                    help="gnubg eval ply (default: 2)")
    ap.add_argument("--net-ply", type=int, default=0, choices=[0, 2],
                    help="net eval ply on the comparison side (default: 0). "
                    "Set to 2 to match the in-app hint engine.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-candidates", type=int, default=12,
                    help="cap on candidate plays per position to limit gnubg cost")
    ap.add_argument("--bearoff", default=None,
                    help="path to bearoff.bin to enable exact race-endgame lookup "
                    "(use ../public/weights/bearoff.bin)")
    ap.add_argument("--bearoff-json", default=None,
                    help="path to bearoff.json metadata (defaults next to bin)")
    args = ap.parse_args()

    if not gnubg_installed():
        print("gnubg not in PATH; install with `sudo apt-get install gnubg`")
        sys.exit(1)

    _net, raw_net_eq = load_net_auto(args.weights)

    # Optionally wrap the equity function with bearoff lookup. When eligible,
    # the table's exact 4-vec replaces the net's static output. The wrapper
    # accepts a Position (us-frame) and returns scalar us-frame equity.
    bearoff_active = False
    if args.bearoff:
        import json as _json
        bo_json = args.bearoff_json or args.bearoff.replace(".bin", ".json")
        with open(bo_json) as _f:
            _meta = _json.load(_f)
        _idx_of, _finish, _first_off = load_bearoff_bin(
            args.bearoff, _meta["n_states"], _meta["max_rolls"])

        def net_eq(pos: Position) -> float:
            if is_pure_race(pos):
                bo = bearoff_equity(_idx_of, _finish, _first_off, pos)
                return (bo["pWin"] + bo["pGammonWin"]
                        - bo["pLoss"] - bo["pGammonLoss"])
            return raw_net_eq(pos)
        bearoff_active = True
    else:
        net_eq = raw_net_eq

    rng = random.Random(args.seed)
    print(f"sampling {args.positions} mid-game positions...", flush=True)
    positions = collect_mid_game_positions(args.positions, rng)
    print(f"  collected {len(positions)} positions", flush=True)
    print(f"  comparing net at ply={args.net_ply} vs gnubg at ply={args.ply}"
          f"  bearoff={'on' if bearoff_active else 'off'}",
          flush=True)

    if args.net_ply == 2:
        net_score = lambda p, after: net_equity_for_play_2ply(net_eq, p, after)
    else:
        net_score = lambda p, after: net_equity_for_play(net_eq, p, after)

    client = GnubgClient(ply=args.ply, timeout=5.0)

    top1 = 0
    top3 = 0
    equity_losses: List[float] = []
    net_eqs: List[float] = []
    gnubg_eqs: List[float] = []
    skipped = 0
    queried = 0
    t0 = time.time()

    try:
        for pi, p in enumerate(positions):
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            plays = generate_plays(p, d1, d2)
            # Skip forfeits and trivial-single-play positions.
            if len(plays) <= 1:
                skipped += 1
                continue
            # Cap candidate count to keep gnubg queries bounded. When over,
            # keep the top-K by Phase A equity (still favors the play we'd
            # actually pick, so disagreements remain detectable).
            if len(plays) > args.max_candidates:
                # Filter using cheap 0-ply scoring so we don't pay 2-ply cost
                # twice; the kept candidates are then scored with the configured
                # ply for the actual comparison.
                with_eq = [(net_equity_for_play(net_eq, p, after), i, after)
                           for i, (_pl, after) in enumerate(plays)]
                with_eq.sort(reverse=True)
                kept = with_eq[:args.max_candidates]
                idx_play_after = [(i, plays[i][0], plays[i][1]) for _, i, _ in kept]
            else:
                idx_play_after = [(i, plays[i][0], plays[i][1])
                                  for i in range(len(plays))]

            net_scores: List[float] = []
            gnubg_scores: List[float] = []
            for _i, _pl, after in idx_play_after:
                ne = net_score(p, after)
                ge = gnubg_equity_for_play(client, p, after)
                if math.isnan(ge):
                    ne = float("nan")  # so it's dropped consistently below
                net_scores.append(ne)
                gnubg_scores.append(ge)
                queried += 1

            # Drop nan rows for this position.
            valid = [j for j in range(len(net_scores))
                     if not (math.isnan(net_scores[j]) or math.isnan(gnubg_scores[j]))]
            if len(valid) < 2:
                skipped += 1
                continue

            net_valid = [net_scores[j] for j in valid]
            gnubg_valid = [gnubg_scores[j] for j in valid]

            net_best = max(range(len(net_valid)), key=lambda j: net_valid[j])
            gnubg_order = sorted(range(len(gnubg_valid)),
                                 key=lambda j: gnubg_valid[j], reverse=True)
            gnubg_best = gnubg_order[0]
            if net_best == gnubg_best:
                top1 += 1
            if net_best in gnubg_order[:3]:
                top3 += 1
            # Equity loss: how much gnubg-equity do we forfeit by picking
            # Phase A's choice instead of gnubg's?
            equity_losses.append(gnubg_valid[gnubg_best] - gnubg_valid[net_best])
            net_eqs.extend(net_valid)
            gnubg_eqs.extend(gnubg_valid)

            if (pi + 1) % 25 == 0:
                elapsed = time.time() - t0
                n_eval = len(equity_losses)
                if n_eval > 0:
                    print(f"  pos {pi+1}/{len(positions)}  "
                          f"top1={top1/n_eval:.3f} top3={top3/n_eval:.3f} "
                          f"mean_eq_loss={sum(equity_losses)/n_eval:.4f}  "
                          f"queried={queried} ({queried/elapsed:.1f}/s)  "
                          f"elapsed={elapsed:.0f}s",
                          flush=True)
    finally:
        client.close()

    n = len(equity_losses)
    if n == 0:
        print("no valid positions sampled; nothing to report")
        return

    top1_rate = top1 / n
    top3_rate = top3 / n
    mean_loss = sum(equity_losses) / n
    median_loss = sorted(equity_losses)[n // 2]
    max_loss = max(equity_losses)
    p90_loss = sorted(equity_losses)[max(0, int(n * 0.9) - 1)]

    # Pearson correlation and signed bias (mean of net - gnubg).
    m_net = sum(net_eqs) / len(net_eqs)
    m_g = sum(gnubg_eqs) / len(gnubg_eqs)
    cov = sum((n_ - m_net) * (g - m_g) for n_, g in zip(net_eqs, gnubg_eqs))
    var_net = sum((n_ - m_net) ** 2 for n_ in net_eqs)
    var_g = sum((g - m_g) ** 2 for g in gnubg_eqs)
    pearson = cov / math.sqrt(var_net * var_g) if var_net > 0 and var_g > 0 else float("nan")
    bias = m_net - m_g
    rmse = math.sqrt(sum((n_ - g) ** 2 for n_, g in zip(net_eqs, gnubg_eqs))
                     / len(net_eqs))

    print()
    print(f"=== Move-selection accuracy: net-{args.net_ply}ply vs gnubg-{args.ply}ply ===")
    print(f"positions evaluated: {n}  (skipped {skipped} forfeit/trivial)")
    print(f"  top-1 agreement:     {top1}/{n} = {top1_rate:.3f}")
    print(f"  top-3 agreement:     {top3}/{n} = {top3_rate:.3f}")
    print(f"  mean equity loss:    {mean_loss:.4f}  (gnubg's units)")
    print(f"  median equity loss:  {median_loss:.4f}")
    print(f"  p90 equity loss:     {p90_loss:.4f}")
    print(f"  max equity loss:     {max_loss:.4f}")
    print()
    print(f"=== Equity calibration (over {len(net_eqs)} (pos, play) pairs) ===")
    print(f"  Pearson correlation: {pearson:.4f}")
    print(f"  RMSE (net - gnubg):  {rmse:.4f}")
    print(f"  signed bias:         {bias:+.4f}  "
          f"({'net > gnubg' if bias > 0 else 'net < gnubg'})")


if __name__ == "__main__":
    main()
