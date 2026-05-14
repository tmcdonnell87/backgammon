"""Supervised pretrainer for the 4-output net: regress each head toward a
soft 4-vector target derived from the heuristic's scalar equity.

Phase A keeps the distillation source as the heuristic (cheap, no external
deps); Phase E will replace it with gnubg's 2-ply equity. The soft target
shape is:
    p_w  = sigmoid(2 * heuristic_equity)
    p_l  = 1 - p_w
    p_gw = 0.18 * p_w    (empirical gammon-among-wins rate)
    p_gl = 0.18 * p_l
The 0.18 is rough; TD reshapes the gammon heads during self-play. The point
of distillation is to put the net somewhere far better than random init so
TD can find good moves immediately.

Pipeline:
  1. Heuristic-vs-heuristic self-play; sample every visited position.
  2. Label each as a soft 4-vector via `heuristic_to_4vector`.
  3. Mini-batch SGD with per-head BCE.

Usage:
  python3 distill.py --positions 200000 --epochs 30 --out runs/distill
"""
from __future__ import annotations
import argparse
import math
import os
import random
import time

import torch

from engine import (
    starting_position, mirror, generate_plays, check_win,
)
from encoding import encode, INPUT_SIZE
from net_torch import Net, heuristic_to_4vector
from bench import heuristic_value, pick_with


def collect_positions(target: int, rng: random.Random) -> list:
    """Heuristic-vs-heuristic self-play; return a list of Positions."""
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
            out.append(after)
            if len(out) >= target:
                return out
            win = check_win(after)
            if win is not None:
                break
            p = mirror(after)
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
    return out


def train_distill(positions: list, hidden, epochs: int,
                  lr: float, batch: int, out_dir: str, seed: int = 1,
                  init: str | None = None):
    rng = torch.Generator().manual_seed(seed)
    py_rng = random.Random(seed)

    N = len(positions)
    X = torch.zeros((N, INPUT_SIZE), dtype=torch.float32)
    Y = torch.zeros((N, 4), dtype=torch.float32)
    print(f"encoding {N} positions...", flush=True)
    for i, p in enumerate(positions):
        X[i] = torch.from_numpy(encode(p))
        Y[i] = torch.tensor(heuristic_to_4vector(heuristic_value(p)))
    # Clamp away from 0/1 for log-stability.
    Y.clamp_(1e-4, 1.0 - 1e-4)
    print(f"  Y means per head: "
          f"p_w={Y[:, 0].mean():.3f}  p_gw={Y[:, 1].mean():.3f}  "
          f"p_l={Y[:, 2].mean():.3f}  p_gl={Y[:, 3].mean():.3f}",
          flush=True)

    if init:
        net = Net.load_json(init)
        if net.hidden_layers != ([hidden] if isinstance(hidden, int) else list(hidden)):
            print(f"  WARN: --init hidden_layers={net.hidden_layers} != "
                  f"--hidden {hidden}; using init shape", flush=True)
        print(f"  initialized from {init}", flush=True)
    else:
        net = Net(hidden=hidden, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  net hidden_layers={net.hidden_layers}", flush=True)

    t0 = time.time()
    for epoch in range(epochs):
        progress = epoch / max(1, epochs - 1)
        cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        perm = torch.randperm(N, generator=rng)
        loss_sum = 0.0
        for s in range(0, N, batch):
            bi = perm[s:s + batch]
            xb = X[bi]                                # (B, INPUT)
            yb = Y[bi]                                # (B, 4)
            B = xb.shape[0]
            # Forward: cache pre-tanh inputs (xb plus each post-tanh h) per layer.
            inputs = [xb]
            for li in range(net._L):
                z = inputs[-1] @ net.W[li].T + net.b[li]
                inputs.append(torch.tanh(z))
            z_out = inputs[-1] @ net.W[net._L].T + net.b[net._L]
            yhat = torch.sigmoid(z_out)               # (B, 4)
            loss = (-yb * torch.log(yhat + 1e-9)
                    - (1 - yb) * torch.log(1 - yhat + 1e-9)).mean()
            loss_sum += float(loss.item()) * B
            # Backward (manual; gradient of sigmoid+BCE is (yhat - y) / 4 since
            # loss is the mean over 4 heads).
            dZ_out = (yhat - yb) / 4.0                # (B, 4)
            # Output layer grads.
            dW_out = dZ_out.T @ inputs[-1] / B        # (4, H_last)
            db_out = dZ_out.mean(dim=0)               # (4,)
            # Backprop through hidden layers.
            dH = dZ_out @ net.W[net._L]               # (B, H_last)
            hidden_grads_W = [None] * net._L
            hidden_grads_b = [None] * net._L
            for li in range(net._L - 1, -1, -1):
                h_i = inputs[li + 1]                  # post-tanh of layer li
                x_i = inputs[li]                      # input to layer li
                dZ = dH * (1.0 - h_i * h_i)
                hidden_grads_W[li] = dZ.T @ x_i / B    # (H_i, H_{i-1})
                hidden_grads_b[li] = dZ.mean(dim=0)
                if li > 0:
                    dH = dZ @ net.W[li]                # backprop to previous layer
            # Apply updates.
            net.W[net._L] -= cur_lr * dW_out
            net.b[net._L] -= cur_lr * db_out
            for li in range(net._L):
                net.W[li] -= cur_lr * hidden_grads_W[li]
                net.b[li] -= cur_lr * hidden_grads_b[li]
        avg_loss = loss_sum / N
        # Report MSE per head and equity-MSE.
        with torch.no_grad():
            yfull = net.value_batched(X)               # (N, 4)
        mse = ((yfull - Y) ** 2).mean(dim=0)
        equity_pred = yfull[:, 0] + yfull[:, 1] - yfull[:, 2] - yfull[:, 3]
        equity_true = Y[:, 0] + Y[:, 1] - Y[:, 2] - Y[:, 3]
        eq_mse = float(((equity_pred - equity_true) ** 2).mean().item())
        print(f"epoch {epoch + 1}/{epochs}  ce={avg_loss:.5f}  "
              f"head_mse=[{mse[0]:.4f},{mse[1]:.4f},{mse[2]:.4f},{mse[3]:.4f}]  "
              f"eq_mse={eq_mse:.5f}  lr={cur_lr:.4f}  "
              f"elapsed={time.time() - t0:.1f}s", flush=True)

    net.save_json(os.path.join(out_dir, "weights-distill.json"))
    net.save_json(os.path.join(out_dir, "weights-latest.json"))
    print(f"saved {out_dir}/weights-distill.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/distill")
    ap.add_argument("--positions", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", default="120",
                    help="int (one hidden layer) or comma list like '200,200'")
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

    hidden_arg = args.hidden
    if "," in hidden_arg:
        hidden_layers = [int(s) for s in hidden_arg.split(",") if s]
    else:
        hidden_layers = int(hidden_arg)
    train_distill(positions, hidden_layers, args.epochs, args.lr,
                  args.batch, args.out, seed=args.seed, init=args.init)


if __name__ == "__main__":
    main()
