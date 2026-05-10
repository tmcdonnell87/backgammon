"""Supervised pretrainer: regress the net's output toward the heuristic's
equity on a large set of positions sampled from self-play.

Why: pure TD(λ) self-play takes a very long time to surpass the hand-tuned
heuristic, and can stall early. A supervised pretrain gives the net a
heuristic-strength starting point. From there, TD fine-tuning can lift it
further (or at minimum, the net evaluates roughly as well as the heuristic
while running ~5x faster than the heuristic in the worker).

Pipeline:
  1. Generate N positions by playing self-play games with the heuristic and
     sampling the resulting positions.
  2. Label each position with `heuristic_value(p)` (mapped to [0, 1]).
  3. SGD with MSE on (encode(p), label).

Usage:
  python3 distill.py --positions 200000 --epochs 3 --out runs/distill
"""
from __future__ import annotations
import argparse
import math
import os
import random
import time
import numpy as np

from engine import (
    starting_position, mirror, generate_plays, check_win,
)
from encoding import encode, INPUT_SIZE
from net import Net, save_npz
from bench import heuristic_value, pick_with


def collect_positions(target: int, rng: random.Random) -> list:
    """Self-play with heuristic on both sides; return a list of Positions."""
    out = []
    while len(out) < target:
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
                idx = pick_with(p, plays, heuristic_value)
                _pl, after = plays[idx]
            # Sample positions: every position visited (after each move).
            out.append(after)
            if len(out) >= target:
                return out
            win = check_win(after)
            if win is not None:
                break
            p = mirror(after)
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
    return out


def train_distill(positions: list, hidden: int, epochs: int,
                  lr: float, batch: int, out_dir: str, seed: int = 1,
                  init: str | None = None):
    rng = np.random.default_rng(seed)

    # Build dataset: features X (N, 198), labels y in [0, 1].
    N = len(positions)
    X = np.zeros((N, INPUT_SIZE), dtype=np.float32)
    Y = np.zeros(N, dtype=np.float32)
    print(f"encoding {N} positions...", flush=True)
    for i, p in enumerate(positions):
        X[i] = encode(p)
        Y[i] = (heuristic_value(p) + 1.0) / 2.0
    # Clamp targets away from 0/1 a hair so log() stays bounded for tanh-saturated
    # heuristic outputs.
    Y = np.clip(Y, 1e-4, 1 - 1e-4)
    print(f"  Y stats: mean={Y.mean():.3f} std={Y.std():.3f} "
          f"min={Y.min():.3f} max={Y.max():.3f}", flush=True)

    if init:
        net = Net.load_json(init)
        if net.hidden != hidden:
            print(f"  WARN: --init hidden={net.hidden} != --hidden {hidden}; "
                  f"using init hidden", flush=True)
            hidden = net.hidden
        print(f"  initialized from {init}", flush=True)
    else:
        net = Net(hidden=hidden, seed=seed)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    for epoch in range(epochs):
        # Cosine LR schedule from `lr` down to `lr/10` over total epochs.
        progress = epoch / max(1, epochs - 1)
        cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))
        idx = rng.permutation(N)
        loss_sum = 0.0
        for s in range(0, N, batch):
            bi = idx[s:s + batch]
            xb = X[bi]                       # (B, 198)
            yb = Y[bi]                       # (B,)
            # Forward
            z1 = xb @ net.W1.T + net.b1      # (B, H)
            h = np.tanh(z1)                  # (B, H)
            z2 = h @ net.W2 + net.b2         # (B,)
            yhat = 1.0 / (1.0 + np.exp(-z2))
            # Cross-entropy loss: -y log yhat - (1-y) log (1-yhat).
            loss = float((-yb * np.log(yhat + 1e-9)
                          - (1 - yb) * np.log(1 - yhat + 1e-9)).mean())
            loss_sum += loss * len(bi)
            # Backward: dCE/dZ2 = yhat - y (no vanishing gradient).
            dZ2 = (yhat - yb)                # (B,)
            dW2 = h.T @ dZ2 / len(bi)        # (H,)
            db2 = float(dZ2.mean())
            dH = np.outer(dZ2, net.W2)       # (B, H)
            dZ1 = dH * (1 - h * h)
            dW1 = dZ1.T @ xb / len(bi)       # (H, 198)
            db1 = dZ1.mean(axis=0)
            net.W1 -= cur_lr * dW1
            net.b1 -= cur_lr * db1
            net.W2 -= cur_lr * dW2
            net.b2 = np.float32(net.b2 - cur_lr * db2)
        avg_loss = loss_sum / N
        # Report MSE alongside CE for comparability across versions.
        with np.errstate(over='ignore'):
            zfull1 = X @ net.W1.T + net.b1
            hfull = np.tanh(zfull1)
            zfull2 = hfull @ net.W2 + net.b2
            yfull = 1.0 / (1.0 + np.exp(-zfull2))
        mse = float(((yfull - Y) ** 2).mean())
        print(f"epoch {epoch + 1}/{epochs}  ce={avg_loss:.5f}  mse={mse:.5f}  "
              f"lr={cur_lr:.4f}  elapsed={time.time() - t0:.1f}s",
              flush=True)

    save_npz(net, os.path.join(out_dir, "ckpt-distill.npz"))
    net.save_json(os.path.join(out_dir, "weights-distill.json"))
    net.save_json(os.path.join(out_dir, "weights-latest.json"))
    print(f"saved {out_dir}/weights-distill.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/distill")
    ap.add_argument("--positions", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init", default=None,
                    help="warm-start from existing weights JSON")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"collecting {args.positions} positions via heuristic self-play...",
          flush=True)
    t0 = time.time()
    positions = collect_positions(args.positions, rng)
    print(f"collected {len(positions)} in {time.time() - t0:.1f}s", flush=True)

    train_distill(positions, args.hidden, args.epochs, args.lr,
                  args.batch, args.out, seed=args.seed, init=args.init)


if __name__ == "__main__":
    main()
