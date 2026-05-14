"""Multi-process gnubg-2ply labeling for distillation.

Spawns N gnubg subprocess workers, fans out positions over a shared queue,
and persists labeled (X, Y, valid) tensors to a checkpoint file. Resumable:
if `--label-cache` points to an existing checkpoint, only un-labeled
positions are sent to the workers.

Each worker holds its own long-lived `GnubgClient` (~10 evals/sec). With 4-5
workers we aggregate ~40-50 labels/sec, so 2M positions takes ~12-14h wall.

Inputs:
  --positions-file <path.pt> — a torch.save'd list of Position objects
  (use `--collect` to build one from heuristic + Phase A self-play).
  --label-cache <path.pt> — checkpoint file. Created if missing, resumed if
  present.

Outputs (saved to --label-cache):
  X: (N, INPUT_SIZE) float32
  Y: (N, 4) float32 — (p_w, p_gw, p_l, p_gl), gnubg-2ply targets
  valid: (N,) bool — True where Y is filled
"""
from __future__ import annotations

# Pin BLAS to one thread before importing torch (so 4-5 worker procs don't
# stomp on each other's CPU cycles). Workers re-pin these on spawn.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import pickle
import queue
import random
import sys
import time
from typing import List, Optional

import torch

from engine import Position, starting_position, mirror, generate_plays, check_win
from encoding import encode, INPUT_SIZE
from gnubg_client import GnubgClient, gnubg_installed


# -- Position collection helpers -----------------------------------------

def _pick_heuristic(p: Position, plays):
    from bench import heuristic_value, pick_with
    return pick_with(p, plays, heuristic_value)


def _pick_phase_a(net_eq, p: Position, plays):
    """Pick using a Phase-A-style 0-ply equity function (cheaper than 2-ply
    for *collection*; the labels themselves use gnubg-2ply)."""
    import math as _m
    if len(plays) == 1:
        return 0
    best_i, best = 0, -_m.inf
    for i, (_pl, after) in enumerate(plays):
        win = check_win(after)
        if win is not None:
            _wa, base_pts = win
            us_eq = float(base_pts)
        else:
            opp = mirror(after)
            us_eq = -net_eq(opp)
        if us_eq > best:
            best = us_eq
            best_i = i
    return best_i


def collect_mixed(target: int, phase_a_path: str, mix_phase_a: float,
                  seed: int) -> List[Position]:
    """Self-play position collection mixing Phase A (mix_phase_a fraction)
    and heuristic (rest). For each completed game, decide policy by coin flip.
    Returns `target` mid-game positions (skips trivial single-play and pure
    opening positions).
    """
    from match_nets import load_net_auto
    _net, net_eq = load_net_auto(phase_a_path)
    rng = random.Random(seed)
    out: List[Position] = []
    games_played = 0
    while len(out) < target:
        use_phase_a = rng.random() < mix_phase_a
        p = starting_position()
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        while d1 == d2:
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        plies = 0
        while plies < 200:
            plies += 1
            plays = generate_plays(p, d1, d2)
            if len(plays) == 1 and len(plays[0][0]) == 0:
                after = p
            else:
                if use_phase_a:
                    idx = _pick_phase_a(net_eq, p, plays)
                else:
                    idx = _pick_heuristic(p, plays)
                _pl, after = plays[idx]
            # Skip pure-opening plies; keep everything from ply 4 onward.
            # Also skip trivial single-play positions (no move-selection
            # signal); the labeler is bounded by gnubg cost, so we want
            # only "interesting" positions in the dataset.
            if plies > 3 and not (len(plays) == 1 and len(plays[0][0]) == 0):
                out.append(after)
                if len(out) >= target:
                    games_played += 1
                    if games_played % 200 == 0:
                        print(f"  collected {len(out)}/{target} "
                              f"(games={games_played})", flush=True)
                    return out
            win = check_win(after)
            if win is not None:
                break
            p = mirror(after)
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        games_played += 1
        if games_played % 200 == 0:
            print(f"  collected {len(out)}/{target} (games={games_played})",
                  flush=True)
    return out


# -- Worker -------------------------------------------------------------

def _worker(worker_id: int, ply: int, in_q: mp.Queue, out_q: mp.Queue,
            stop_evt):
    # Reset BLAS pinning inside the spawned process.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    client = GnubgClient(timeout=5.0, ply=ply)
    consec_failures = 0
    try:
        while not stop_evt.is_set():
            try:
                task = in_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if task is None:  # poison
                break
            idx, p = task
            r = client.evaluate_position(p)
            if r is None:
                consec_failures += 1
                # If gnubg has gotten itself wedged, recycle the subprocess.
                if consec_failures >= 5:
                    try:
                        client.close()
                    except Exception:
                        pass
                    client = GnubgClient(timeout=5.0, ply=ply)
                    consec_failures = 0
                out_q.put((idx, None))
            else:
                consec_failures = 0
                # gnubg's W(g) already includes backgammons (W(bg) ⊂ W(g) ⊂ Win),
                # so the gammon-or-better prob for our net is just r.p_gammon_win.
                vec = (
                    r.p_win,
                    r.p_gammon_win,
                    r.p_loss,
                    r.p_gammon_loss,
                )
                out_q.put((idx, vec))
    finally:
        try:
            client.close()
        except Exception:
            pass


# -- Master --------------------------------------------------------------

def label_positions_mp(positions: List[Position], cache_path: str,
                       workers: int = 4, ply: int = 2,
                       checkpoint_every: int = 10000) -> None:
    N = len(positions)
    # Resume from cache if present.
    X = torch.zeros((N, INPUT_SIZE), dtype=torch.float32)
    Y = torch.zeros((N, 4), dtype=torch.float32)
    valid = torch.zeros(N, dtype=torch.bool)
    done_mask = torch.zeros(N, dtype=torch.bool)  # labeled OR confirmed-bad
    if cache_path and os.path.exists(cache_path):
        ck = torch.load(cache_path, weights_only=False)
        if int(ck.get("N", -1)) == N:
            X = ck["X"]
            Y = ck["Y"]
            valid = ck["valid"]
            done_mask = ck.get("done", valid.clone())
            print(f"resumed from {cache_path}: "
                  f"{int(done_mask.sum())} already labeled, "
                  f"{int(valid.sum())} valid",
                  flush=True)
        else:
            print(f"WARN: cache size mismatch (N={N}, cache N={ck.get('N')}); "
                  f"starting fresh", flush=True)
    # Pre-encode X for every position (skip the ones we already have).
    print("encoding inputs...", flush=True)
    for i, p in enumerate(positions):
        if not done_mask[i]:
            X[i] = torch.from_numpy(encode(p))

    ctx = mp.get_context("spawn")
    in_q: mp.Queue = ctx.Queue(maxsize=workers * 32)
    out_q: mp.Queue = ctx.Queue()
    stop_evt = ctx.Event()

    procs = []
    for wid in range(workers):
        pr = ctx.Process(target=_worker, args=(wid, ply, in_q, out_q, stop_evt),
                         daemon=True)
        pr.start()
        procs.append(pr)

    to_label = [i for i in range(N) if not done_mask[i]]
    print(f"feeding {len(to_label)} positions to {workers} workers...",
          flush=True)
    t0 = time.time()
    fed = 0
    received = 0
    target = len(to_label)
    feed_idx = 0
    last_ckpt = int(done_mask.sum())

    def _save_checkpoint(reason: str):
        if cache_path:
            torch.save({"X": X, "Y": Y, "valid": valid, "done": done_mask,
                        "N": N},
                       cache_path + ".tmp")
            os.replace(cache_path + ".tmp", cache_path)
            print(f"  [ckpt:{reason}] valid={int(valid.sum())}/{N} "
                  f"({100.0 * int(valid.sum()) / N:.1f}%)  "
                  f"saved={cache_path}",
                  flush=True)

    try:
        while received < target:
            # Top up the input queue without blocking.
            while feed_idx < target:
                try:
                    in_q.put_nowait((to_label[feed_idx], positions[to_label[feed_idx]]))
                except queue.Full:
                    break
                feed_idx += 1
                fed += 1
            try:
                idx, vec = out_q.get(timeout=2.0)
            except queue.Empty:
                # No result in 2s: workers may be slow but not stuck.
                continue
            done_mask[idx] = True
            if vec is not None:
                Y[idx, 0] = vec[0]
                Y[idx, 1] = vec[1]
                Y[idx, 2] = vec[2]
                Y[idx, 3] = vec[3]
                valid[idx] = True
            received += 1
            n_valid = int(valid.sum())
            if received % 1000 == 0 or received == target:
                elapsed = time.time() - t0
                rate = received / max(elapsed, 1e-6)
                eta = (target - received) / max(rate, 1e-6)
                print(f"  {received}/{target}  "
                      f"valid={n_valid}/{N}  rate={rate:.1f}/s  "
                      f"elapsed={elapsed / 60:.1f}m  eta={eta / 60:.1f}m",
                      flush=True)
            if n_valid - last_ckpt >= checkpoint_every:
                _save_checkpoint(f"+{n_valid - last_ckpt}")
                last_ckpt = n_valid
        _save_checkpoint("final")
    finally:
        # Signal workers to drain and exit.
        stop_evt.set()
        for _ in procs:
            try:
                in_q.put_nowait(None)
            except queue.Full:
                pass
        for pr in procs:
            pr.join(timeout=5.0)
            if pr.is_alive():
                pr.terminate()


# -- CLI -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions-file", default=None,
                    help="torch.save'd list of Position objects to label")
    ap.add_argument("--collect", type=int, default=0,
                    help="if > 0, collect this many positions via mixed-policy "
                    "self-play and save to --positions-file before labeling")
    ap.add_argument("--phase-a-weights", default="../public/weights/expert.json",
                    help="weights for Phase A side of mixed-policy collection")
    ap.add_argument("--mix-phase-a", type=float, default=0.7,
                    help="fraction of games played with Phase A (rest heuristic)")
    ap.add_argument("--label-cache", required=True,
                    help="path to checkpoint .pt file; created or resumed")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ply", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=10000)
    args = ap.parse_args()

    if not gnubg_installed():
        print("gnubg not in PATH; install with `sudo apt-get install gnubg`",
              flush=True)
        sys.exit(1)

    # Positions: from --collect, or load --positions-file.
    if args.collect > 0:
        if not args.positions_file:
            print("--collect requires --positions-file (where to save)",
                  flush=True)
            sys.exit(2)
        if os.path.exists(args.positions_file):
            print(f"positions file exists; loading {args.positions_file}",
                  flush=True)
            with open(args.positions_file, "rb") as f:
                positions = pickle.load(f)
        else:
            print(f"collecting {args.collect} mixed-policy positions "
                  f"(mix_phase_a={args.mix_phase_a})...", flush=True)
            t0 = time.time()
            positions = collect_mixed(args.collect, args.phase_a_weights,
                                      args.mix_phase_a, args.seed)
            print(f"  collected {len(positions)} in {time.time() - t0:.1f}s",
                  flush=True)
            os.makedirs(os.path.dirname(args.positions_file) or ".",
                        exist_ok=True)
            with open(args.positions_file, "wb") as f:
                pickle.dump(positions, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  saved {args.positions_file}", flush=True)
    else:
        if not args.positions_file or not os.path.exists(args.positions_file):
            print("must supply --positions-file (or use --collect)", flush=True)
            sys.exit(2)
        with open(args.positions_file, "rb") as f:
            positions = pickle.load(f)
        print(f"loaded {len(positions)} positions from {args.positions_file}",
              flush=True)

    os.makedirs(os.path.dirname(args.label_cache) or ".", exist_ok=True)
    label_positions_mp(positions, args.label_cache, workers=args.workers,
                       ply=args.ply, checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
