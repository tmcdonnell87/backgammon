"""Distill training targets from GNU Backgammon's 2-ply equity.

Higher-quality teacher than the hand-tuned heuristic — gnubg-2ply is widely
considered world-class. This is what the plan calls Phase E distillation.

Pipeline:
  1. Generate positions via heuristic self-play (cheap; `distill.collect_positions`).
  2. Label each with gnubg's 2-ply eval via subprocess (~5-10 evals/sec).
  3. Convert gnubg's (p_w, p_gw, p_bg, p_l, p_gl, p_bgl) into our 4-output
     target: backgammons collapse into gammons for Phase A/B compat.
  4. Standard supervised SGD with per-head BCE (reusing `distill.train_distill`
     once we've built the (X, Y) tensors).

Run after installing gnubg:
   sudo apt-get install gnubg
   cd training
   ../.venv/bin/python distill_gnubg.py --positions 500000 --epochs 5 \\
       --hidden 200,200 --lr 0.02 --out runs/distill-gnubg
"""
from __future__ import annotations
import argparse
import math
import os
import random
import sys
import time
from typing import List, Tuple

import torch

from engine import Position
from encoding import encode, INPUT_SIZE
from net_torch import Net
from gnubg_client import GnubgClient, gnubg_installed, GnubgEval
from distill import collect_positions, train_distill
from outcome import classify_outcome


def gnubg_eval_to_4vector(e: GnubgEval) -> List[float]:
    """gnubg returns 6 probs; map to our 4-output target.

    gnubg's W(g) row value is "P(win is gammon or better)" and already
    nests W(bg) inside it (W(bg) ⊂ W(g) ⊂ Win). So our p_gw target is
    just e.p_gammon_win, no addition needed.
    """
    return [
        e.p_win,
        e.p_gammon_win,
        e.p_loss,
        e.p_gammon_loss,
    ]


def label_with_gnubg(positions: List[Position], save_path: str = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (X, Y) tensors by querying gnubg for each position's 2-ply eval.
    Slow (~5-10 evals/sec). Optionally save partial progress to save_path."""
    if not gnubg_installed():
        raise RuntimeError("gnubg not in PATH. Install: sudo apt-get install gnubg")
    N = len(positions)
    X = torch.zeros((N, INPUT_SIZE), dtype=torch.float32)
    Y = torch.zeros((N, 4), dtype=torch.float32)
    valid = torch.zeros(N, dtype=torch.bool)
    client = GnubgClient(timeout=5.0, ply=2)
    t0 = time.time()
    try:
        for i, p in enumerate(positions):
            X[i] = torch.from_numpy(encode(p))
            r = client.evaluate_position(p)
            if r is None:
                continue  # leave valid[i] = False
            Y[i] = torch.tensor(gnubg_eval_to_4vector(r), dtype=torch.float32)
            valid[i] = True
            if (i + 1) % 1000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  labeled {i+1}/{N} ({100.0*(i+1)/N:.1f}%)  "
                      f"rate={rate:.1f}/s  valid={int(valid[:i+1].sum())}",
                      flush=True)
            if save_path and (i + 1) % 10000 == 0:
                torch.save({"X": X[:i+1], "Y": Y[:i+1], "valid": valid[:i+1]},
                           save_path)
    finally:
        client.close()
    return X[valid], Y[valid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/distill-gnubg")
    ap.add_argument("--positions", type=int, default=500_000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", default="200,200")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init", default=None)
    ap.add_argument("--label-cache", default=None,
                    help="torch.save path to checkpoint partial labeling")
    args = ap.parse_args()

    if not gnubg_installed():
        print("gnubg not in PATH. Install via: sudo apt-get install gnubg")
        sys.exit(1)

    rng = random.Random(args.seed)
    print(f"collecting {args.positions} positions via heuristic self-play...",
          flush=True)
    positions = collect_positions(args.positions, rng)
    print(f"collected {len(positions)} positions; labeling with gnubg 2-ply...",
          flush=True)
    X, Y = label_with_gnubg(positions, save_path=args.label_cache)
    print(f"gnubg labeled {X.shape[0]}/{args.positions} positions (others skipped)",
          flush=True)

    # train_distill expects a list of Position objects, but we've already
    # encoded into X,Y. Inline a tiny training loop that reuses the same math.
    from distill import train_distill as _td
    # Build a wrapper: write X, Y to a torch.save and reload via a small adapter.
    # Simpler: just inline an SGD here since we already have (X, Y).
    hidden = args.hidden
    if "," in hidden:
        hidden = [int(s) for s in hidden.split(",") if s]
    else:
        hidden = int(hidden)
    if args.init:
        net = Net.load_json(args.init)
    else:
        net = Net(hidden=hidden, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)

    rng_t = torch.Generator().manual_seed(args.seed)
    Y.clamp_(1e-4, 1.0 - 1e-4)
    t0 = time.time()
    N = X.shape[0]
    for epoch in range(args.epochs):
        progress = epoch / max(1, args.epochs - 1)
        cur_lr = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        perm = torch.randperm(N, generator=rng_t)
        loss_sum = 0.0
        for s in range(0, N, args.batch):
            bi = perm[s:s + args.batch]
            xb = X[bi]; yb = Y[bi]
            B = xb.shape[0]
            inputs = [xb]
            for li in range(net._L):
                z = inputs[-1] @ net.W[li].T + net.b[li]
                inputs.append(torch.tanh(z))
            z_out = inputs[-1] @ net.W[net._L].T + net.b[net._L]
            yhat = torch.sigmoid(z_out)
            loss = (-yb * torch.log(yhat + 1e-9)
                    - (1 - yb) * torch.log(1 - yhat + 1e-9)).mean()
            loss_sum += float(loss.item()) * B
            dZ = (yhat - yb) / 4.0
            dW_out = dZ.T @ inputs[-1] / B
            db_out = dZ.mean(dim=0)
            dH = dZ @ net.W[net._L]
            # Collect all per-layer gradients first; apply updates after the
            # full backward pass so dH propagates against the *original*
            # weights, not just-updated ones.
            hgrads_W = [None] * net._L
            hgrads_b = [None] * net._L
            for li in range(net._L - 1, -1, -1):
                h_i = inputs[li + 1]; x_i = inputs[li]
                dZ_h = dH * (1.0 - h_i * h_i)
                hgrads_W[li] = dZ_h.T @ x_i / B
                hgrads_b[li] = dZ_h.mean(dim=0)
                if li > 0:
                    dH = dZ_h @ net.W[li]
            for li in range(net._L):
                net.W[li] -= cur_lr * hgrads_W[li]
                net.b[li] -= cur_lr * hgrads_b[li]
            net.W[net._L] -= cur_lr * dW_out
            net.b[net._L] -= cur_lr * db_out
        print(f"epoch {epoch + 1}/{args.epochs}  ce={loss_sum / N:.5f}  "
              f"lr={cur_lr:.4f}  elapsed={time.time() - t0:.1f}s", flush=True)

    net.save_json(os.path.join(args.out, "weights-distill-gnubg.json"))
    net.save_json(os.path.join(args.out, "weights-latest.json"))
    print(f"saved {args.out}/weights-distill-gnubg.json", flush=True)


if __name__ == "__main__":
    main()
