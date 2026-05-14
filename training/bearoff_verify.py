"""Cross-check the bearoff table against gnubg-0ply on pure-race positions.

Loads bearoff.bin (or rebuilds), then for a list of pure-race positions
computes our exact equity and gnubg's, reports max/mean absolute diff
across the four heads.
"""
from __future__ import annotations
import argparse
import os
import random
import sys
from typing import List, Tuple

import numpy as np

from engine import Position, POINTS, mirror, generate_plays, check_win
from gnubg_client import GnubgClient, gnubg_installed


def load_bearoff_bin(bin_path: str, n_states: int, max_rolls: int):
    histLen = max_rolls + 1
    with open(bin_path, "rb") as f:
        data = f.read()
    states_bytes = n_states * 6
    finish_bytes = n_states * histLen * 2
    expected = states_bytes + 2 * finish_bytes
    assert len(data) == expected, (
        f"bin size {len(data)} != expected {expected}")
    states = np.frombuffer(data[:states_bytes], dtype=np.uint8).reshape(
        (n_states, 6))
    finish = np.frombuffer(
        data[states_bytes:states_bytes + finish_bytes],
        dtype=np.uint16).reshape((n_states, histLen)).astype(np.float64) / 65535.0
    first_off = np.frombuffer(
        data[states_bytes + finish_bytes:],
        dtype=np.uint16).reshape((n_states, histLen)).astype(np.float64) / 65535.0
    idx_of = {tuple(int(v) for v in s): i for i, s in enumerate(states)}
    return idx_of, finish, first_off


def is_pure_race(p: Position) -> bool:
    if p.bar_us != 0 or p.bar_them != 0:
        return False
    for i in range(6, 18):
        if p.points[i] != 0:
            return False
    for i in range(6):
        if p.points[i] < 0:
            return False
    for i in range(18, 24):
        if p.points[i] > 0:
            return False
    return True


def us_state(p: Position):
    return tuple(int(p.points[i]) for i in range(6))


def them_state(p: Position):
    return tuple(int(-p.points[23 - i]) for i in range(6))


def bearoff_equity(idx_of, finish, first_off, p: Position):
    us = us_state(p)
    them = them_state(p)
    sus = sum(us)
    sthem = sum(them)
    if sus == 0:
        # already won
        return dict(pWin=1.0, pGammonWin=1.0 if sthem == 15 else 0.0,
                    pLoss=0.0, pGammonLoss=0.0)
    if sthem == 0:
        return dict(pWin=0.0, pGammonWin=0.0, pLoss=1.0,
                    pGammonLoss=1.0 if sus == 15 else 0.0)
    ui = idx_of[us]
    ti = idx_of[them]
    K = finish.shape[1]
    p_us = finish[ui]
    p_them = finish[ti]
    fo_us = first_off[ui]
    fo_them = first_off[ti]
    suffix_them = np.zeros(K + 1)
    for j in range(K - 1, -1, -1):
        suffix_them[j] = suffix_them[j + 1] + p_them[j]
    pWin = 0.0
    pGammonWin = 0.0
    for k in range(K):
        pk = p_us[k]
        if pk == 0:
            continue
        pWin += pk * suffix_them[k]
        fo_at_km1 = 0.0 if k == 0 else fo_them[k - 1]
        pGammonWin += pk * (1.0 - fo_at_km1)
    pGammonLoss = 0.0
    for j in range(K):
        pj = p_them[j]
        if pj == 0:
            continue
        fo_at_j = 0.0 if j == 0 else fo_us[j]
        pGammonLoss += pj * (1.0 - fo_at_j)
    return dict(pWin=min(max(pWin, 0), 1),
                pGammonWin=min(max(pGammonWin, 0), 1),
                pLoss=min(max(1 - pWin, 0), 1),
                pGammonLoss=min(max(pGammonLoss, 0), 1))


def sample_pure_race_positions(n: int, rng: random.Random) -> List[Position]:
    """Random valid pure-home-race positions with both sides having checkers
    distributed in home boards (some may already be off)."""
    out = []
    while len(out) < n:
        us = [0] * 6
        rem = 15
        # mix: some prob of borne-off, then distribute the rest randomly
        off_us = rng.randint(0, 12)
        rem -= off_us
        for _ in range(rem):
            us[rng.randint(0, 5)] += 1
        them = [0] * 6
        rem = 15
        off_them = rng.randint(0, 12)
        rem -= off_them
        for _ in range(rem):
            them[rng.randint(0, 5)] += 1
        pts = [0] * POINTS
        for i in range(6):
            pts[i] = us[i]
        for i in range(6):
            pts[23 - i] = -them[i]
        p = Position(points=pts, bar_us=0, bar_them=0,
                     off_us=off_us, off_them=off_them, turn=0)
        # check_win exits if either side is at 15 off; skip those.
        if check_win(p) is not None:
            continue
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="../public/weights/bearoff.bin")
    ap.add_argument("--json", default="../public/weights/bearoff.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import json
    with open(args.json) as f:
        meta = json.load(f)
    n_states = meta["n_states"]
    max_rolls = meta["max_rolls"]

    if not gnubg_installed():
        print("gnubg not in PATH", flush=True)
        sys.exit(1)

    idx_of, finish, first_off = load_bearoff_bin(args.bin, n_states, max_rolls)
    rng = random.Random(args.seed)
    positions = sample_pure_race_positions(args.n, rng)

    client = GnubgClient(ply=0, timeout=5.0)

    diffs = {"pWin": [], "pGammonWin": [], "pLoss": [], "pGammonLoss": [],
             "equity": []}
    try:
        for i, p in enumerate(positions):
            r = client.evaluate_position(p)
            if r is None:
                continue
            gb = dict(pWin=r.p_win, pGammonWin=r.p_gammon_win,
                      pLoss=r.p_loss, pGammonLoss=r.p_gammon_loss)
            bo = bearoff_equity(idx_of, finish, first_off, p)
            for k in ("pWin", "pGammonWin", "pLoss", "pGammonLoss"):
                diffs[k].append(bo[k] - gb[k])
            eq_bo = bo["pWin"] + bo["pGammonWin"] - bo["pLoss"] - bo["pGammonLoss"]
            eq_gb = gb["pWin"] + gb["pGammonWin"] - gb["pLoss"] - gb["pGammonLoss"]
            diffs["equity"].append(eq_bo - eq_gb)
            if i < 5:
                print(f"pos {i}: bo pWin={bo['pWin']:.4f} gb pWin={gb['pWin']:.4f}  "
                      f"bo pGW={bo['pGammonWin']:.4f} gb pGW={gb['pGammonWin']:.4f}  "
                      f"eq diff={eq_bo - eq_gb:+.4f}", flush=True)
    finally:
        client.close()

    for k, vs in diffs.items():
        if not vs:
            continue
        arr = np.array(vs)
        print(f"{k}: mean={arr.mean():+.4f}  abs_mean={np.abs(arr).mean():.4f}  "
              f"abs_max={np.abs(arr).max():.4f}  n={len(arr)}", flush=True)


if __name__ == "__main__":
    main()
