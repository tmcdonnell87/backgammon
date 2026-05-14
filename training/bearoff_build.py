"""Build a one-sided race bear-off database.

For each home-board distribution (n_0, n_1, ..., n_5) with sum <= 15 (where
n_i = checkers on engine point i, i.e. the (i+1)-pt for the player on roll),
compute two probability histograms over the number of rolls needed to:

  1. finish[s, k]    = P(all 15 borne off in exactly k more rolls)
  2. first_off[s, k] = P(at least one bear-off has happened by end of roll k)

Computed by backward induction in increasing pip-count order. Each (state,
dice) pair picks the play that minimizes E[rolls remaining]; both finish
and first_off are accumulated against that same optimal play.

The first_off histogram is only meaningful when the side hasn't yet borne
off any checker (sum(state) == 15); for any state with sum < 15, the side
has already borne off, so first_off is identically 1.0 — we don't materialize
it for those states (saves ~7x of the storage).

From two such tables (us, them), a TS-side join function computes:
  P(we win) = sum_k p_us_finish[k] * sum_{j >= k} p_them_finish[j]
  P(we gammon) = sum_k p_us_finish[k] * (1 - p_them_first_off[k-1])
(where p_them_first_off[k-1] = 0 for k=1)

States: C(21, 6) = 54264. Wall time: ~5-15 min on a single core in pure Python;
~30-60s with the inner loop vectorized via numpy.

Output format: JSON with int16-quantized histograms (×10000) packed as a
single base64 string. ~1-2 MB on disk.
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from engine import (
    Position, POINTS, generate_plays, BAR, OFF, all_home,
)


MAX_ROLLS = 32  # P(finish > 32 rolls | home-board only) is negligible
HOME_POINTS = 6


# 21 distinct dice rolls with probabilities (matching ALL_ROLLS elsewhere).
DICE: List[Tuple[int, int, float]] = []
for _i in range(1, 7):
    for _j in range(_i, 7):
        DICE.append((_i, _j, 1.0 / 36.0 if _i == _j else 2.0 / 36.0))


def all_states() -> List[Tuple[int, ...]]:
    """Enumerate (n_0..n_5) with each n_i in [0..15], sum <= 15."""
    out: List[Tuple[int, ...]] = []
    # Stars and bars enumeration: place k checkers into 6 boxes.
    for total in range(0, 16):
        for combo in _multichoose(HOME_POINTS, total):
            out.append(combo)
    return out


def _multichoose(boxes: int, balls: int):
    """Yield all (n_0..n_{boxes-1}) with each in [0..], sum == balls.
    Order matches stars-and-bars (n_0 is innermost)."""
    if boxes == 1:
        yield (balls,)
        return
    for k in range(balls + 1):
        for rest in _multichoose(boxes - 1, balls - k):
            yield (k,) + rest


def pip_of(state: Tuple[int, ...]) -> int:
    return sum((i + 1) * n for i, n in enumerate(state))


def make_position(state: Tuple[int, ...]) -> Position:
    pts = [0] * POINTS
    for i in range(HOME_POINTS):
        pts[i] = state[i]
    return Position(
        points=pts,
        bar_us=0,
        bar_them=0,
        off_us=15 - sum(state),
        off_them=0,
        turn=0,
    )


def state_from_pos(p: Position) -> Tuple[int, ...]:
    return tuple(p.points[i] for i in range(HOME_POINTS))


def build_tables(verbose: bool = True):
    states = all_states()
    n_states = len(states)
    if verbose:
        print(f"enumerating {n_states} states", flush=True)
    states.sort(key=pip_of)
    idx_of: Dict[Tuple[int, ...], int] = {s: i for i, s in enumerate(states)}

    finish = np.zeros((n_states, MAX_ROLLS + 1), dtype=np.float64)
    first_off = np.zeros((n_states, MAX_ROLLS + 1), dtype=np.float64)

    # Terminal: empty home, all 15 off. Finishing took 0 *more* rolls.
    empty_idx = idx_of[(0,) * HOME_POINTS]
    finish[empty_idx, 0] = 1.0
    # first_off is meaningless for sum<15 states (treat as 1.0).
    first_off[empty_idx, :] = 1.0

    t0 = time.time()
    last_print = t0
    expected_rolls_cache = np.zeros(n_states, dtype=np.float64)
    expected_rolls_cache[empty_idx] = 0.0
    k_idx = np.arange(MAX_ROLLS + 1, dtype=np.float64)

    for processed, state in enumerate(states):
        si = idx_of[state]
        if state == (0,) * HOME_POINTS:
            continue
        pos = make_position(state)
        any_off = sum(state) < 15

        finish_hist = np.zeros(MAX_ROLLS + 1, dtype=np.float64)
        first_off_hist = np.zeros(MAX_ROLLS + 1, dtype=np.float64) if not any_off else None

        for d1, d2, prob in DICE:
            plays = generate_plays(pos, d1, d2)
            # Pick play minimizing E[rolls remaining].
            best_ns_idx = -1
            best_er = float("inf")
            for _pl, after in plays:
                after_state = state_from_pos(after)
                ns_idx = idx_of[after_state]
                er = expected_rolls_cache[ns_idx]
                if er < best_er:
                    best_er = er
                    best_ns_idx = ns_idx
            # Edge case: if `plays` is empty (no legal moves), generate_plays
            # returns [((), pos)] which keeps state — engine guarantees this.
            ns_state = states[best_ns_idx]
            ns_finish = finish[best_ns_idx]
            # finish_hist[k] += prob * ns_finish[k-1]  (shift by 1)
            finish_hist[1:] += prob * ns_finish[:-1]

            if not any_off:
                # The play either bore off (sum drops) or didn't.
                if sum(ns_state) < 15:
                    # Bear-off occurred on this very roll; first_off becomes
                    # true at roll k=1 and stays true. Contribute prob to all
                    # k >= 1.
                    first_off_hist[1:] += prob
                else:
                    # No bear-off this roll; recurse.
                    first_off_hist[1:] += prob * first_off[best_ns_idx, :-1]

        finish[si] = finish_hist
        if not any_off:
            first_off[si] = first_off_hist
        else:
            first_off[si, :] = 1.0
        expected_rolls_cache[si] = float((finish_hist * k_idx).sum())

        if verbose and (processed + 1) % 5000 == 0:
            now = time.time()
            rate = (processed + 1) / (now - t0)
            eta = (n_states - processed - 1) / max(rate, 1e-6)
            print(f"  {processed + 1}/{n_states}  ({rate:.0f}/s)  "
                  f"eta={eta:.0f}s", flush=True)
            last_print = now

    if verbose:
        print(f"done in {time.time() - t0:.1f}s", flush=True)
    return states, idx_of, finish, first_off


def quantize(arr: np.ndarray) -> np.ndarray:
    """Quantize probability histograms to uint16 (×65535). Sufficient
    precision (~1.5e-5) for the long tail; saves 50% vs float32 and 4x vs
    base64 of int16. The TS side divides by 65535.0 on load."""
    return np.clip(np.round(arr * 65535.0), 0, 65535).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin", default="../public/weights/bearoff.bin")
    ap.add_argument("--out-json", default="../public/weights/bearoff.json")
    args = ap.parse_args()

    states, _idx, finish, first_off = build_tables(verbose=True)

    # Self-consistency: every finish histogram should sum to 1.0.
    sums = finish.sum(axis=1)
    bad = (np.abs(sums - 1.0) > 1e-6).sum()
    print(f"self-consistency: max |sum - 1| = {float(np.max(np.abs(sums - 1.0))):.2e} "
          f"(bad rows: {bad})", flush=True)
    if bad > 0:
        i = int(np.argmax(np.abs(sums - 1.0)))
        print(f"  bad row example: state={states[i]} sum={sums[i]}", flush=True)

    # Pack as binary: [states (N_STATES × 6 × uint8) | finish (N_STATES × (MAX_ROLLS+1) × uint16) | first_off (same)]
    # The states block makes TS look-up trivial (no need to reproduce Python's
    # enumeration order). TS builds a Map<state_key, index> by iterating.
    states_q = np.array(states, dtype=np.uint8)  # (N_STATES, 6)
    finish_q = quantize(finish)
    first_off_q = quantize(first_off)
    payload = states_q.tobytes() + finish_q.tobytes() + first_off_q.tobytes()
    os.makedirs(os.path.dirname(args.out_bin) or ".", exist_ok=True)
    with open(args.out_bin, "wb") as f:
        f.write(payload)
    bin_size_kb = os.path.getsize(args.out_bin) / 1024

    # Small JSON sidecar with metadata, fixtures, and the state-index
    # convention so TS can reproduce the lookup. State enumeration:
    # stars-and-bars over (n_0..n_5), then sorted by pip count. TS side
    # rebuilds the same ordering deterministically (see bearoff.ts).
    out = {
        "version": 1,
        "max_rolls": MAX_ROLLS,
        "n_states": len(states),
        "home_points": HOME_POINTS,
        "checkers_per_side": 15,
        "bin_file": os.path.basename(args.out_bin),
        "bin_layout": [
            {"name": "states", "shape": [len(states), HOME_POINTS],
             "dtype": "uint8", "scale": 1.0},
            {"name": "finish", "shape": [len(states), MAX_ROLLS + 1],
             "dtype": "uint16", "scale": 65535.0},
            {"name": "first_off", "shape": [len(states), MAX_ROLLS + 1],
             "dtype": "uint16", "scale": 65535.0},
        ],
        "fixtures": [
            {
                "state": list(s),
                "index": _idx[s],
                "finish_first5": [float(round(x, 4)) for x in finish[_idx[s], :5]],
                "mean_rolls_to_finish": float(round(
                    (finish[_idx[s]] * np.arange(MAX_ROLLS + 1)).sum(), 4)),
            }
            for s in [
                (0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 1),
                (3, 3, 3, 3, 3, 0),
                (2, 2, 2, 3, 3, 3),
                (15, 0, 0, 0, 0, 0),
            ]
        ],
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    json_size_kb = os.path.getsize(args.out_json) / 1024
    print(f"wrote {args.out_bin}  ({bin_size_kb:.1f} KB)", flush=True)
    print(f"wrote {args.out_json}  ({json_size_kb:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()
