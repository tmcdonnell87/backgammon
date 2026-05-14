"""Bench checkpoints vs heuristic; publish to public/weights/expert.json if
the net wins at least `--min-winrate` of N games.

Two modes:
  Single-net (default):
      publish_if_good.py --weights runs/.../weights-final.json
    The checkpoint is benched directly; if it clears the gate, it's copied
    over public/weights/expert.json.

  Paired phased manifest (Phase F):
      publish_if_good.py --contact runs/contact/weights-final.json \\
                         --race    runs/race/weights-final.json
    A {version: 2, contact: ..., race: ...} manifest is constructed and
    benched. If it clears the gate, the manifest + both sub-weights are
    copied into public/weights/, and the prior expert.json is saved to
    public/weights/expert.<prev-tag>.json (default: "phaseA").
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile

from match_nets import load_net_auto
from bench import play_one_game


def _bench_winrate(eq_fn, n: int, seed: int) -> float:
    rng = random.Random(seed)
    wins = 0
    for g in range(n):
        net_side = g & 1
        wins += play_one_game(net_side, eq_fn, rng)
        if (g + 1) % 50 == 0:
            print(f"  game {g+1}/{n} winrate={wins/(g+1):.3f}", flush=True)
    return wins / n


def _publish_single(weights: str, dst: str, min_wr: float, games: int,
                    seed: int) -> int:
    _net, net_eq = load_net_auto(weights)
    wr = _bench_winrate(net_eq, games, seed)
    se = math.sqrt(wr * (1 - wr) / games) if 0 < wr < 1 else 0
    print(f"NET vs HEURISTIC: {games} games, winrate={wr:.3f} "
          f"(CI ±{1.96 * se:.3f})")
    if wr >= min_wr:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(weights, dst)
        print(f"PUBLISHED -> {dst} ({os.path.getsize(dst)} bytes)")
        return 0
    print(f"NOT publishing (winrate {wr:.3f} < {min_wr})")
    return 1


def _publish_paired(contact: str, race: str, dst: str, min_wr: float,
                    games: int, seed: int,
                    contact_filename: str = "expert.contact.json",
                    race_filename: str = "expert.race.json",
                    rollback_tag: str = "phaseA") -> int:
    """Build a phased manifest from the two sub-weights, bench it via
    load_net_auto, and publish if it clears the gate."""
    # Build a temp manifest pointing at the original weight paths so
    # load_net_auto can resolve them. After the bench passes, we copy the
    # sub-weights into the dest dir and write a manifest with relative paths.
    dst_dir = os.path.dirname(os.path.abspath(dst))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     dir=dst_dir, delete=False) as f:
        json.dump({
            "version": 2,
            "contact": os.path.abspath(contact),
            "race": os.path.abspath(race),
        }, f)
        tmp_manifest = f.name
    try:
        _net, eq_fn = load_net_auto(tmp_manifest)
        wr = _bench_winrate(eq_fn, games, seed)
    finally:
        os.unlink(tmp_manifest)
    se = math.sqrt(wr * (1 - wr) / games) if 0 < wr < 1 else 0
    print(f"PHASED MANIFEST vs HEURISTIC: {games} games, winrate={wr:.3f} "
          f"(CI ±{1.96 * se:.3f})")
    if wr < min_wr:
        print(f"NOT publishing (winrate {wr:.3f} < {min_wr})")
        return 1

    # Roll back previous expert.json if present.
    if os.path.exists(dst):
        rollback_path = os.path.join(dst_dir, f"expert.{rollback_tag}.json")
        shutil.copy(dst, rollback_path)
        print(f"rolled back prior {dst} -> {rollback_path}")

    contact_dst = os.path.join(dst_dir, contact_filename)
    race_dst = os.path.join(dst_dir, race_filename)
    shutil.copy(contact, contact_dst)
    shutil.copy(race, race_dst)
    manifest = {
        "version": 2,
        "contact": contact_filename,
        "race": race_filename,
    }
    with open(dst, "w") as f:
        json.dump(manifest, f, separators=(",", ":"))
    print(f"PUBLISHED phased manifest -> {dst}")
    print(f"  contact: {contact_dst} ({os.path.getsize(contact_dst)} bytes)")
    print(f"  race:    {race_dst} ({os.path.getsize(race_dst)} bytes)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="single-net path (mutually exclusive with --contact/--race)")
    ap.add_argument("--contact", default=None,
                    help="phased manifest mode: contact net weights")
    ap.add_argument("--race", default=None,
                    help="phased manifest mode: race net weights")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--min-winrate", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--dst", default="../public/weights/expert.json")
    ap.add_argument("--contact-filename", default="expert.contact.json",
                    help="output filename for the contact sub-weights in --dst's directory")
    ap.add_argument("--race-filename", default="expert.race.json")
    ap.add_argument("--rollback-tag", default="phaseA",
                    help="tag for the rollback copy (expert.<tag>.json)")
    args = ap.parse_args()

    paired = (args.contact is not None) or (args.race is not None)
    if paired:
        if not (args.contact and args.race):
            ap.error("paired mode requires both --contact and --race")
        if args.weights:
            ap.error("--weights cannot be combined with --contact/--race")
        return _publish_paired(
            args.contact, args.race, args.dst, args.min_winrate,
            args.games, args.seed,
            contact_filename=args.contact_filename,
            race_filename=args.race_filename,
            rollback_tag=args.rollback_tag,
        )
    if not args.weights:
        ap.error("either --weights or (--contact and --race) is required")
    return _publish_single(args.weights, args.dst, args.min_winrate,
                           args.games, args.seed)


if __name__ == "__main__":
    sys.exit(main())
