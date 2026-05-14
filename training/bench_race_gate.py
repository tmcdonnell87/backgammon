"""Race-net gate: phased(heuristic-contact, race-net) vs heuristic-only.

Tests the race net's marginal contribution. Both sides play the heuristic
during contact phase; only race phase differs. A winrate > 0.50 means the
race net plays race better than the heuristic baseline; the plan target is
≥ 0.52 (race phase makes up roughly 60-70% of plies, so the per-ply
advantage is amplified at the game level but still bounded — 0.52 game
winrate ≈ a real race-phase edge).

Usage:
   cd training
   ../.venv/bin/python bench_race_gate.py \\
       --race-weights runs/race-td/weights-final.json --games 1000
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import tempfile

from match_nets import load_net_auto, play_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race-weights", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    race_abs = os.path.abspath(args.race_weights)

    # Manifest A: heuristic contact, race-net race.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"version": 2, "contact": "heuristic", "race": race_abs}, f)
        manifest_A = f.name
    # Manifest B: heuristic for both phases (= pure heuristic baseline).
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"version": 2, "contact": "heuristic", "race": "heuristic"}, f)
        manifest_B = f.name

    try:
        _A, eq_A = load_net_auto(manifest_A)
        _B, eq_B = load_net_auto(manifest_B)
        rng = random.Random(args.seed)
        wins = 0
        n = args.games
        for g in range(n):
            wins += play_one(g & 1, eq_A, eq_B, rng)
            if (g + 1) % 100 == 0:
                wr = wins / (g + 1)
                print(f"  game {g+1}/{n}  phased winrate={wr:.3f}",
                      flush=True)
    finally:
        os.unlink(manifest_A)
        os.unlink(manifest_B)

    wr = wins / n
    se = math.sqrt(wr * (1 - wr) / n) if 0 < wr < 1 else 0
    print(f"\nphased(heur-contact, race-net) vs heur-only: "
          f"{wins}/{n} = {wr:.3f}  (CI ±{1.96 * se:.3f})")


if __name__ == "__main__":
    main()
