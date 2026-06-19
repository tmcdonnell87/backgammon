#!/usr/bin/env python3
"""Build the opening / reply book shipped at public/weights/opening_book.json.

For every legal play of each booked (position, dice) we store gnubg's cubeless
evaluation, so the web app's move-picker and tutor use authoritative equities
for openings and common replies — instead of the neural net, which mis-ranks
the near-tied opening plays and so wrongly flags the rollout-best play as an
"error".

Coverage:
  * all 15 non-double opening rolls (complete: every legal play labeled)
  * the opponent's reply to each opening's gnubg-best play, for all 21 reply
    rolls incl. doubles — the positions a human reaches when responding to the
    opening (complete). The opponent at runtime plays the book-best opening, so
    branching from the gnubg-best keeps the book self-consistent.

Equity conventions — these MUST match src/ai/engine.ts. For a play from a
position P (us on roll), let after = apply_play(P, play); evaluate gnubg on
mirror(after) (opponent on roll) -> GnubgEval e:

    g = e.p_loss - e.p_win  == 1 - 2*e.p_win   # game equity (pWin - pLoss), our POV
    p = -e.equity                              # gnubg cubeless points equity, our POV

This reproduces gameEquityForPostMove() (engine.ts) for `g`, and the
gammon-aware ranking scalar for `p`. `g` is the tutor's grading scale; `p`
ranks the "best" play.

Run (gnubg must be installed: sudo apt-get install gnubg):
  python3 build_opening_book.py \
      --out ../public/weights/opening_book.json \
      --fixture ../test/fixtures/book_keys.json --workers 6 --ply 2
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from typing import Dict, List, Tuple

from engine import (
    BAR, OFF, Position,
    starting_position, mirror, generate_plays, board_hash,
)
from gnubg_client import gnubg_installed

# 15 non-double opening rolls, hi > lo.
OPENINGS: List[Tuple[int, int]] = [
    (h, l) for h in range(6, 0, -1) for l in range(h - 1, 0, -1)
]
# 21 distinct reply rolls (incl. doubles), hi >= lo.
REPLY_ROLLS: List[Tuple[int, int]] = [
    (h, l) for h in range(6, 0, -1) for l in range(h, 0, -1)
]


def move_str(play) -> str:
    """Human-readable move, e.g. ((7,4,3),(5,4,1)) -> '8/5 6/5'. Debug only."""
    parts = []
    for (frm, to, _die) in play:
        f = "bar" if frm == BAR else str(frm + 1)
        t = "off" if to == OFF else str(to + 1)
        parts.append(f"{f}/{t}")
    return " ".join(parts)


def entry_key(p: Position, hi: int, lo: int) -> str:
    return f"{board_hash(p).hex()}:{hi}{lo}"


# -- multiprocessing gnubg eval -----------------------------------------------

_CLIENT = None
_PLY = 2


def _init_worker(ply: int):
    global _CLIENT, _PLY
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from gnubg_client import GnubgClient
    _PLY = ply
    _CLIENT = GnubgClient(timeout=8.0, ply=ply)


def _eval_one(arg):
    """arg = (hash_hex, Position). Returns (hash_hex, (p_win, equity)) or
    (hash_hex, None)."""
    global _CLIENT
    h, pos = arg
    e = _CLIENT.evaluate_position(pos)
    if e is None:
        # Recycle a wedged subprocess and retry once.
        from gnubg_client import GnubgClient
        try:
            _CLIENT.close()
        except Exception:
            pass
        _CLIENT = GnubgClient(timeout=8.0, ply=_PLY)
        e = _CLIENT.evaluate_position(pos)
    if e is None:
        return (h, None)
    return (h, (e.p_win, e.equity))


def mp_eval(positions: Dict[str, Position], ply: int, workers: int,
            label: str) -> Dict[str, Tuple[float, float]]:
    """Evaluate a dict of {hash: Position}. Returns {hash: (p_win, equity)}."""
    items = list(positions.items())
    print(f"[{label}] evaluating {len(items)} unique positions "
          f"on {workers} workers (ply {ply})...", flush=True)
    out: Dict[str, Tuple[float, float]] = {}
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(processes=workers, initializer=_init_worker,
                  initargs=(ply,)) as pool:
        done = 0
        for h, res in pool.imap_unordered(_eval_one, items, chunksize=4):
            done += 1
            if res is None:
                print(f"  WARN: no gnubg eval for {h[:12]}…", flush=True)
                continue
            out[h] = res
            if done % 200 == 0 or done == len(items):
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done}/{len(items)}  {rate:.1f}/s", flush=True)
    return out


# -- book assembly ------------------------------------------------------------

def collect_entry(P: Position, hi: int, lo: int):
    """Return (key, [(final_hash, move_str, eval_pos, after_pos), ...]) for the
    legal plays of (P, hi, lo). eval_pos = mirror(after) (opponent on roll)."""
    recs = []
    for play, after in generate_plays(P, hi, lo):
        if len(play) == 0:
            continue  # forfeit — nothing to book
        recs.append((board_hash(after).hex(), move_str(play), mirror(after), after))
    return entry_key(P, hi, lo), recs


def build_plays(recs, evals) -> List[dict]:
    """Turn records + eval map into a sorted list of book plays (by p desc)."""
    plays = []
    for fh, ms, eval_pos, _after in recs:
        eh = board_hash(eval_pos).hex()
        ev = evals.get(eh)
        if ev is None:
            continue
        p_win, equity = ev
        g = 1.0 - 2.0 * p_win          # e.p_loss - e.p_win
        pts = -equity                  # our-POV points equity
        plays.append({"k": fh, "g": round(g, 5), "p": round(pts, 5), "m": ms})
    plays.sort(key=lambda d: d["p"], reverse=True)
    return plays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../public/weights/opening_book.json")
    ap.add_argument("--fixture", default="../test/fixtures/book_keys.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ply", type=int, default=2)
    args = ap.parse_args()

    if not gnubg_installed():
        print("gnubg not in PATH; install with `sudo apt-get install gnubg`",
              flush=True)
        sys.exit(1)

    start = starting_position()

    # --- Pass A: openings -------------------------------------------------
    opening_recs = {}            # (hi,lo) -> recs
    eval_positions = {}          # hash -> Position (unique)
    for (hi, lo) in OPENINGS:
        key, recs = collect_entry(start, hi, lo)
        opening_recs[(hi, lo)] = (key, recs)
        for _fh, _ms, eval_pos, _after in recs:
            eval_positions.setdefault(board_hash(eval_pos).hex(), eval_pos)

    evals = mp_eval(eval_positions, args.ply, args.workers, "openings")

    entries = {}
    best_after = {}              # (hi,lo) -> after position of the gnubg-best play
    for (hi, lo), (key, recs) in opening_recs.items():
        plays = build_plays(recs, evals)
        if not plays:
            print(f"  WARN: no plays for opening {hi}{lo}", flush=True)
            continue
        entries[key] = {"complete": True, "plays": plays}
        # Best after-position (for reply branching): find the rec whose final
        # hash matches the top play.
        top_k = plays[0]["k"]
        for _fh, _ms, _eval_pos, after in recs:
            if board_hash(after).hex() == top_k:
                best_after[(hi, lo)] = after
                break

    _sanity_check_openings(entries, start)

    # --- Pass B: replies to each opening's best play ----------------------
    reply_recs = []              # list of (key, recs)
    reply_eval_positions = {}
    for (hi, lo), after in best_after.items():
        Q = mirror(after)        # opponent (the replier) on roll, "us" frame
        for (a, b) in REPLY_ROLLS:
            key, recs = collect_entry(Q, a, b)
            if not recs:
                continue
            reply_recs.append((key, recs))
            for _fh, _ms, eval_pos, _after in recs:
                reply_eval_positions.setdefault(
                    board_hash(eval_pos).hex(), eval_pos)

    revals = mp_eval(reply_eval_positions, args.ply, args.workers, "replies")
    for key, recs in reply_recs:
        plays = build_plays(recs, revals)
        if plays:
            entries.setdefault(key, {"complete": True, "plays": plays})

    book = {
        "version": 1,
        "scale": "cubeless-game-equity",
        "source": f"gnubg-{args.ply}ply",
        "entries": entries,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(book, f, separators=(",", ":"))
    n_plays = sum(len(e["plays"]) for e in entries.values())
    sz = os.path.getsize(args.out)
    print(f"wrote {args.out}: {len(entries)} entries, {n_plays} plays, "
          f"{sz / 1024:.0f} KB", flush=True)

    _write_fixture(args.fixture, start, best_after)


def _sanity_check_openings(entries, start):
    """Hard sign-convention check: the gnubg-best 3-1 must make the 5-point and
    the best 4-2 must make the 4-point. If signs were flipped, the 'best' play
    would be garbage and this catches it before we ship a broken book."""
    checks = [((3, 1), 4), ((4, 2), 3)]  # (roll), our-home point index to be made
    for (hi, lo), pt in checks:
        key = entry_key(start, hi, lo)
        e = entries.get(key)
        assert e, f"missing opening entry {hi}{lo}"
        top = e["plays"][0]
        # Re-derive the best play's board and assert the point is made.
        made = False
        for play, after in generate_plays(start, hi, lo):
            if board_hash(after).hex() == top["k"]:
                made = after.points[pt] == 2
                break
        assert made, (
            f"SIGN CHECK FAILED: gnubg-best {hi}-{lo} is '{top['m']}' "
            f"(g={top['g']}, p={top['p']}) which does NOT make point index {pt}. "
            f"Equity sign convention is wrong.")
    print("  sign check OK: 3-1 makes the 5-pt, 4-2 makes the 4-pt", flush=True)


def _write_fixture(path, start, best_after):
    """Emit a small TS<->Python key-parity fixture for test/book.test.ts."""
    def row(p: Position, hi: int, lo: int):
        return {
            "points": list(p.points),
            "bar_us": p.bar_us, "bar_them": p.bar_them,
            "off_us": p.off_us, "off_them": p.off_them,
            "dice": [hi, lo],
            "key": entry_key(p, hi, lo),
        }
    rows = [row(start, hi, lo) for (hi, lo) in [(3, 1), (6, 5), (2, 1)]]
    # A couple of reply positions (opponent on roll after the best opening).
    for (hi, lo) in [(3, 1), (6, 1)]:
        if (hi, lo) in best_after:
            Q = mirror(best_after[(hi, lo)])
            rows.append(row(Q, 6, 4))
            rows.append(row(Q, 5, 5))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote fixture {path}: {len(rows)} key rows", flush=True)


if __name__ == "__main__":
    main()
