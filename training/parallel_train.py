"""Parallel TD(λ) trainer using a worker pool + parameter server.

Architecture (one process per role):
  * N workers each play self-play minibatches against a local snapshot of
    the global weights, doing TD(λ) updates locally. Every `games_per_batch`
    games they ship back a parameter delta.
  * 1 server applies incoming deltas to W_global with a Polyak-blend factor
    and periodically broadcasts the new W_global back to all workers.

We use this over a lock-step vectorized self-play because backgammon games
vary 40–120 plies; vec-env stalls on the longest game per step. Per-worker
candidate-move batching within a ply is fine future work; the current
single-game inner loop already keeps each core busy.

Hardware note: this is CPU-only by design (small net, simulation-bound).
Each worker pins itself to 1 BLAS thread (see worker_proc.py header).

Usage:
   python3 parallel_train.py --out runs/phaseB --games 5000000 \\
       --hidden 200,200 --workers 12 --batch 100 --alpha 0.02 --lambda 0.7
"""
from __future__ import annotations
import argparse
import csv
import multiprocessing as mp
import os
import random
import signal
import time
from typing import List, Optional

# Same BLAS-thread pinning for the server process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch  # noqa: E402
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from net_torch import Net  # noqa: E402
from worker_proc import worker_loop  # noqa: E402


def _broadcast(weight_qs: List, snapshot: List[torch.Tensor]):
    """Push snapshot to each worker's queue, dropping stale items."""
    for q in weight_qs:
        # Each worker queue has maxsize=2; if full, drop the oldest.
        while True:
            try:
                q.put_nowait(snapshot)
                break
            except Exception:
                try:
                    q.get_nowait()
                except Exception:
                    break


def _apply_delta(net: Net, delta: List[torch.Tensor], blend: float):
    """Apply delta in-place: net.params += blend * delta.
    delta layout matches net.snapshot(): [W_0, b_0, W_1, b_1, ...]."""
    for li in range(len(net.W)):
        net.W[li].add_(blend * delta[2 * li])
        net.b[li].add_(blend * delta[2 * li + 1])


def run_parallel(out_dir: str, target_games: int,
                 hidden, n_workers: int, games_per_batch: int,
                 alpha: float, lam: float, blend: float,
                 broadcast_every: int, ckpt_every: int,
                 seed: int = 1, resume: Optional[str] = None,
                 mode: str = "all", rollout_fraction: float = 0.0,
                 bootstrap_weights: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)

    if resume and os.path.exists(resume):
        net = Net.load_json(resume)
        print(f"resumed from {resume} (hidden_layers={net.hidden_layers})",
              flush=True)
    else:
        net = Net(hidden=hidden, seed=seed)
        print(f"fresh net hidden_layers={net.hidden_layers}", flush=True)
    # Workers must use the same shape as the actual loaded net, not the
    # --hidden CLI value (which may differ from the resumed file).
    worker_hidden = list(net.hidden_layers)

    # Frozen bootstrap net (Phase F cross-net bootstrap). Loaded once on the
    # server; each worker receives the parameter snapshot at spawn-time and
    # constructs its own frozen forward-only Net from it.
    bootstrap_snapshot = None
    bootstrap_hidden = None
    if bootstrap_weights:
        bnet = Net.load_json(bootstrap_weights)
        bootstrap_snapshot = bnet.snapshot()
        bootstrap_hidden = list(bnet.hidden_layers)
        print(f"bootstrap net loaded from {bootstrap_weights} "
              f"(hidden_layers={bootstrap_hidden})", flush=True)

    # multiprocessing setup. spawn = clean Python processes that honor the
    # OMP/MKL/OPENBLAS env vars set at the top of worker_proc.py.
    ctx = mp.get_context("spawn")
    weight_qs = [ctx.Queue(maxsize=2) for _ in range(n_workers)]
    delta_q = ctx.Queue()
    stop_event = ctx.Event()

    procs = []
    for i in range(n_workers):
        p = ctx.Process(
            target=worker_loop,
            args=(i, seed + i + 1, weight_qs[i], delta_q, stop_event,
                  worker_hidden, alpha, lam, games_per_batch, mode,
                  rollout_fraction, bootstrap_hidden, bootstrap_snapshot),
            name=f"worker-{i}",
        )
        p.start()
        procs.append(p)
    print(f"spawned {n_workers} workers, mode={mode}, "
          f"rollout_fraction={rollout_fraction}, "
          f"bootstrap={'on' if bootstrap_weights else 'off'}, "
          f"broadcast_every={broadcast_every}, "
          f"games_per_batch={games_per_batch}, blend={blend}", flush=True)

    # Initial broadcast.
    _broadcast(weight_qs, net.snapshot())

    # Logging.
    log_path = os.path.join(out_dir, "log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", buffering=1)
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["games", "plies_avg", "side0_winrate", "rate_gps",
                      "elapsed_s"])

    total_games = 0
    plies_window: list = []
    wins_window: list = []
    next_broadcast = broadcast_every
    next_ckpt = ckpt_every
    next_log = 1000
    t0 = time.time()

    stop_flag = {"requested": False}

    def _sigterm(_signo, _frame):
        stop_flag["requested"] = True
        print("stop requested", flush=True)

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    try:
        while total_games < target_games and not stop_flag["requested"]:
            try:
                msg = delta_q.get(timeout=1.0)
            except Exception:
                continue
            _apply_delta(net, msg["delta"], blend)
            total_games += msg["games"]
            plies_window.append(msg["plies"] / msg["games"])
            wins_window.append(msg["wins0"] / msg["games"])
            if len(plies_window) > 50:
                plies_window.pop(0)
                wins_window.pop(0)

            if total_games >= next_broadcast:
                _broadcast(weight_qs, net.snapshot())
                next_broadcast = total_games + broadcast_every

            if total_games >= next_log:
                elapsed = time.time() - t0
                avg_plies = sum(plies_window) / max(1, len(plies_window))
                wr = sum(wins_window) / max(1, len(wins_window))
                rate = total_games / elapsed
                print(f"games={total_games} avg_plies={avg_plies:.1f} "
                      f"side0_wr={wr:.3f} rate={rate:.1f}g/s "
                      f"elapsed={elapsed:.0f}s",
                      flush=True)
                log.writerow([total_games, f"{avg_plies:.2f}", f"{wr:.3f}",
                              f"{rate:.2f}", f"{elapsed:.1f}"])
                next_log = total_games + 1000

            if total_games >= next_ckpt:
                jpath = os.path.join(out_dir, f"weights-{total_games}.json")
                net.save_json(jpath)
                net.save_json(os.path.join(out_dir, "weights-latest.json"))
                print(f"checkpoint games={total_games} -> {jpath}", flush=True)
                next_ckpt = total_games + ckpt_every
    finally:
        # Final checkpoint and shutdown.
        try:
            net.save_json(os.path.join(out_dir, "weights-final.json"))
            net.save_json(os.path.join(out_dir, "weights-latest.json"))
            print(f"FINAL games={total_games}", flush=True)
        except Exception as e:
            print(f"final save failed: {e}", flush=True)
        log_f.close()

        # Send poison pills, then join.
        stop_event.set()
        for q in weight_qs:
            try:
                q.put(None, timeout=1.0)
            except Exception:
                pass
        for p in procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)


def _parse_hidden(s: str):
    if "," in s:
        return [int(x) for x in s.split(",") if x]
    return int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=5_000_000)
    ap.add_argument("--hidden", default="200,200")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=100,
                    help="games_per_batch each worker plays before reporting")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.7)
    ap.add_argument("--blend", type=float, default=0.5,
                    help="Polyak blend factor for incoming worker deltas")
    ap.add_argument("--broadcast-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--mode", choices=["all", "contact", "race"], default="all",
                    help="TD update phase filter; 'contact' or 'race' for "
                         "phase-specific training (Phase C)")
    ap.add_argument("--rollout-fraction", type=float, default=0.0,
                    help="Per-ply probability of replacing 0-ply TD target "
                         "with a 2-ply expectimax rollout target (Phase D). "
                         "Cost ~20x per affected ply; keep small (0.05-0.10).")
    ap.add_argument("--bootstrap-weights", default=None,
                    help="Frozen race net JSON for Phase F cross-net "
                         "bootstrap. When provided AND mode=contact, the "
                         "contact net's TD target at boundary plies (after "
                         "becomes race) is this net's evaluation instead of "
                         "self-bootstrap.")
    args = ap.parse_args()

    hidden = _parse_hidden(args.hidden)
    run_parallel(
        args.out, args.games, hidden, args.workers, args.batch,
        args.alpha, args.lam, args.blend, args.broadcast_every,
        args.ckpt_every, seed=args.seed, resume=args.resume,
        mode=args.mode, rollout_fraction=args.rollout_fraction,
        bootstrap_weights=args.bootstrap_weights,
    )


if __name__ == "__main__":
    main()
