"""Worker process for parallel_train.py.

Each worker:
 1. Receives a parameter snapshot from the server via `weight_q`.
 2. Loads it into its local Net and plays `games_per_batch` self-play games
    with TD(λ) updates against its local copy.
 3. Computes a delta = (local_params − snapshot_params) and sends it back via
    `delta_q`, along with simple stats.
 4. Repeats until `stop_event` is set or a poison-pill (None snapshot) is
    received.

Threading is pinned to 1 BLAS thread per worker (set via env vars at import
time and `torch.set_num_threads(1)`) so 12 workers don't trample each other.
"""
from __future__ import annotations
import os
# Pin BLAS to 1 thread per process. These env vars must be set BEFORE the
# torch import below (and BEFORE numpy/MKL/openblas init in case they're
# already loaded). With multiprocessing start_method='spawn', a fresh
# interpreter runs this module and sees these env vars in time.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import random
from typing import List

import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Already set (e.g. when this module is imported in the parent process
    # before the workers are spawned). Safe to ignore.
    pass

from net_torch import Net  # noqa: E402
from train import play_game  # noqa: E402


def _drain_latest(q, block_timeout: float = 0.5):
    """Take the most-recent item from a queue, discarding any older items.
    Returns (item, was_poison_pill). With block_timeout=0, returns immediately
    if the queue is empty (item=None, poison=False)."""
    item = None
    try:
        if block_timeout > 0:
            item = q.get(timeout=block_timeout)
        else:
            item = q.get_nowait()
    except Exception:
        return None, False
    if item is None:
        return None, True
    while True:
        try:
            newer = q.get_nowait()
        except Exception:
            break
        if newer is None:
            return None, True
        item = newer
    return item, False


def worker_loop(worker_id: int, seed: int,
                weight_q, delta_q, stop_event,
                hidden, alpha: float, lam: float,
                games_per_batch: int, mode: str = "all",
                rollout_fraction: float = 0.0,
                bootstrap_hidden=None, bootstrap_snapshot=None):
    rng = random.Random(seed * 7919 + 1)
    # Initial Net; will be overwritten by the first snapshot from the server.
    net = Net(hidden=hidden, seed=seed)
    # Frozen bootstrap net (race specialist). Shared snapshot loaded once at
    # worker spawn time; never updated.
    bootstrap_net = None
    if bootstrap_snapshot is not None:
        bootstrap_net = Net(hidden=bootstrap_hidden, seed=seed)
        bootstrap_net.load_snapshot(bootstrap_snapshot)
    # Have we received an initial snapshot from the server?
    have_snapshot = False
    while not stop_event.is_set():
        # On startup, block until we get the first snapshot. Afterwards,
        # poll non-blockingly for newer ones; if one arrives, adopt it as
        # the new baseline and continue training from there. Between
        # broadcasts, workers keep training against their local Net and
        # report per-batch *incremental* deltas to the server.
        block_timeout = 0.5 if not have_snapshot else 0.0
        new_snapshot, poison = _drain_latest(weight_q, block_timeout=block_timeout)
        if poison:
            break
        if new_snapshot is not None:
            net.load_snapshot(new_snapshot)
            have_snapshot = True
        if not have_snapshot:
            continue
        # Snapshot the pre-batch state so we can compute an incremental delta.
        pre_batch = net.snapshot()
        plies_in_batch = 0
        wins0 = 0
        for _ in range(games_per_batch):
            stats = play_game(net, alpha=alpha, lam=lam, rng=rng,
                              learn=True, mode=mode,
                              rollout_fraction=rollout_fraction,
                              bootstrap_net=bootstrap_net)
            plies_in_batch += stats["plies"]
            if stats["winner"] == 0:
                wins0 += 1
        post_batch = net.snapshot()
        delta = [post - pre for post, pre in zip(post_batch, pre_batch)]
        try:
            delta_q.put({
                "worker_id": worker_id,
                "delta": delta,
                "games": games_per_batch,
                "plies": plies_in_batch,
                "wins0": wins0,
            }, timeout=5.0)
        except Exception:
            # Server hung up or stop_event set during put.
            break
