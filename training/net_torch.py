"""PyTorch CPU net with arbitrary-depth tanh trunk and 4 sigmoid output heads,
with per-head TD(λ) eligibility traces.

Architecture: x (198) -> [Linear(H_i) -> tanh]^L -> Linear(4) -> sigmoid
   Outputs: (p_win, p_gammon_win, p_loss, p_gammon_loss) for the player on roll.
   Cubeless equity: p_win + p_gammon_win - p_loss - p_gammon_loss.

Phase A used L=1 (one hidden layer of 120). Phase B uses L=2 (two hidden layers
of 200 each). The Net constructor takes `hidden: int | list[int]`; an int is
interpreted as a single-layer width for backward compatibility.

Eligibility traces:
- Hidden-layer params (shared across heads) have per-head traces; the update
  sums per-head td_error * trace.
- Output-layer params (each row k owned by head k) have per-head traces with
  per-head updates (no sum).

JSON format (current):
   {"input": 198, "hidden": [200, 200], "output": 4,
    "W": [matrix_in_to_h1, matrix_h1_to_h2, matrix_h2_to_out],
    "b": [vector_h1, vector_h2, vector_out]}
The loader also accepts the Phase A "W1/b1/W2/b2" format with "hidden" as an int.

We avoid torch.nn.Module and autograd — the analytic gradient below is
trivial and far cheaper than autograd's bookkeeping for this size of net.
"""
from __future__ import annotations
import json
import math
from typing import List, Sequence, Union

import torch

INPUT = 198
HIDDEN_DEFAULT: List[int] = [120]
OUTPUT = 4

# Indices into the 4-output vector.
P_W, P_GW, P_L, P_GL = 0, 1, 2, 3


def equity_from_probs(probs: torch.Tensor) -> torch.Tensor:
    # probs shape: (..., 4)
    return probs[..., P_W] + probs[..., P_GW] - probs[..., P_L] - probs[..., P_GL]


def _normalize_hidden(hidden: Union[int, Sequence[int]]) -> List[int]:
    if isinstance(hidden, int):
        return [hidden]
    return list(hidden)


class Net:
    def __init__(self, hidden: Union[int, Sequence[int]] = HIDDEN_DEFAULT,
                 seed: int = 0):
        self.hidden_layers: List[int] = _normalize_hidden(hidden)
        gen = torch.Generator().manual_seed(seed)
        # Layer sizes: [INPUT, H_1, H_2, ..., H_L, OUTPUT]
        sizes = [INPUT] + self.hidden_layers + [OUTPUT]
        self.W: List[torch.Tensor] = []
        self.b: List[torch.Tensor] = []
        for i in range(len(sizes) - 1):
            W = torch.randn(sizes[i + 1], sizes[i], generator=gen) * 0.10
            b = torch.zeros(sizes[i + 1])
            self.W.append(W)
            self.b.append(b)
        # Output layer index.
        self._L = len(self.W) - 1  # number of hidden layers
        self.reset_traces()

    # Back-compat property for code that reads "hidden" as a scalar.
    @property
    def hidden(self) -> int:
        return self.hidden_layers[0] if len(self.hidden_layers) == 1 else -1

    def reset_traces(self):
        sizes = [INPUT] + self.hidden_layers + [OUTPUT]
        self.eW: List[torch.Tensor] = []
        self.eb: List[torch.Tensor] = []
        for i in range(len(sizes) - 1):
            out_size = sizes[i + 1]
            in_size = sizes[i]
            if i < self._L:
                # Hidden layer: per-head 3D trace (output, out_size, in_size).
                self.eW.append(torch.zeros(OUTPUT, out_size, in_size))
                self.eb.append(torch.zeros(OUTPUT, out_size))
            else:
                # Output layer: per-head trace for that head's row of W
                # (output, in_size); per-head scalar for bias.
                self.eW.append(torch.zeros(OUTPUT, in_size))
                self.eb.append(torch.zeros(OUTPUT))

    @staticmethod
    def _as_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x, dtype=torch.float32)

    # --- forward -----------------------------------------------------------

    def forward(self, x):
        """Single-sample forward. Returns (y, h_cache, x_tensor).

        y: (4,) sigmoid outputs.
        h_cache: list of post-tanh tensors [h_1, ..., h_L], used by td_step.
        x_tensor: input echoed back as tensor for td_step.
        """
        x_t = self._as_tensor(x)
        h_cache: List[torch.Tensor] = []
        cur = x_t
        for i in range(self._L):
            cur = torch.tanh(self.W[i] @ cur + self.b[i])
            h_cache.append(cur)
        z_out = self.W[self._L] @ cur + self.b[self._L]
        y = torch.sigmoid(z_out)
        return y, h_cache, x_t

    def value(self, x) -> torch.Tensor:
        y, _, _ = self.forward(x)
        return y

    def equity(self, x) -> float:
        y, _, _ = self.forward(x)
        return float(equity_from_probs(y).item())

    def value_batched(self, X) -> torch.Tensor:
        """Batched forward. X shape (B, INPUT) -> (B, 4) sigmoid probs."""
        cur = self._as_tensor(X)
        for i in range(self._L):
            cur = torch.tanh(cur @ self.W[i].T + self.b[i])
        return torch.sigmoid(cur @ self.W[self._L].T + self.b[self._L])

    def equity_batched(self, X) -> torch.Tensor:
        return equity_from_probs(self.value_batched(X))

    # --- TD step -----------------------------------------------------------

    def td_step(self, x, y, h_cache, td_error, alpha: float, lam: float):
        """Apply one TD(λ) update step.

        x: (INPUT,) input from this step.
        y: (4,) sigmoid outputs.
        h_cache: list of (H_i,) post-tanh hiddens from forward().
        td_error: (4,) per-head TD error.
        """
        x_t = self._as_tensor(x)
        td = self._as_tensor(td_error).reshape(OUTPUT)

        # Layer inputs: layer 0 sees x, layer i sees h_cache[i-1].
        # The output layer (index self._L) sees h_cache[-1] (or x if no hidden).
        inputs = [x_t] + h_cache  # length self._L + 1

        # --- Update output layer trace + propagate dz back into top hidden ---
        s = y * (1.0 - y)                                  # (output,)
        # Output layer: W shape (output, in_out), bias (output,).
        h_top = inputs[self._L]                            # (in_out,)
        self.eW[self._L].mul_(lam).add_(s.unsqueeze(1) * h_top.unsqueeze(0))
        self.eb[self._L].mul_(lam).add_(s)
        W_out = self.W[self._L]                            # (output, in_out)
        one_minus_h2_top = 1.0 - h_top * h_top if self._L > 0 else None
        # ds_at_h_top[k, j] = s[k] * W_out[k, j]
        ds_at_h = s.unsqueeze(1) * W_out                   # (output, in_out)

        # --- Propagate through hidden layers, top-down (L-1 ... 0) ---
        for i in range(self._L - 1, -1, -1):
            h_i = inputs[i + 1]                            # (H_i,)
            x_i = inputs[i]                                # (H_{i-1},) or x
            one_minus_h2 = 1.0 - h_i * h_i
            # ds_at_z_i = ds_at_h * (1 - h_i^2), shape (output, H_i)
            ds_at_z = ds_at_h * one_minus_h2.unsqueeze(0)
            # Trace W_i: per-head outer(ds_at_z[k], x_i)
            # shape (output, H_i, H_{i-1})
            self.eW[i].mul_(lam).add_(
                ds_at_z.unsqueeze(2) * x_i.unsqueeze(0).unsqueeze(0)
            )
            self.eb[i].mul_(lam).add_(ds_at_z)
            # Propagate ds back to inputs of this layer for the next iteration.
            if i > 0:
                # ds_at_h_{i-1}[k, q] = sum_j(ds_at_z[k, j] * W_i[j, q])
                # = ds_at_z @ W_i, shape (output, H_{i-1})
                ds_at_h = ds_at_z @ self.W[i]
            # else: we don't need to propagate to x.

        # --- Apply updates ----------------------------------------------------
        # Output layer (per-head, no sum over heads):
        self.W[self._L].add_(alpha * td.unsqueeze(1) * self.eW[self._L])
        self.b[self._L].add_(alpha * td * self.eb[self._L])
        # Hidden layers (sum over heads, scaled by per-head td_error):
        for i in range(self._L):
            self.W[i].add_(alpha * (td.unsqueeze(1).unsqueeze(2) * self.eW[i]).sum(dim=0))
            self.b[i].add_(alpha * (td.unsqueeze(1) * self.eb[i]).sum(dim=0))

    # --- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "input": INPUT,
            "hidden": list(self.hidden_layers),
            "output": OUTPUT,
            "W": [W.tolist() for W in self.W],
            "b": [b.tolist() for b in self.b],
        }

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, d: dict) -> "Net":
        out_dim = d.get("output")
        if out_dim != OUTPUT:
            raise ValueError(f"net_torch.Net expects output={OUTPUT}, got {out_dim}")
        if d.get("input") != INPUT:
            raise ValueError(f"net_torch.Net expects input={INPUT}, got {d.get('input')}")
        hidden = d["hidden"]
        if isinstance(hidden, int):
            # Legacy Phase A format with W1/b1/W2/b2.
            n = cls(hidden=hidden)
            n.W[0] = torch.tensor(d["W1"], dtype=torch.float32)
            n.b[0] = torch.tensor(d["b1"], dtype=torch.float32)
            n.W[1] = torch.tensor(d["W2"], dtype=torch.float32)
            n.b[1] = torch.tensor(d["b2"], dtype=torch.float32)
            return n
        # Multi-layer format.
        n = cls(hidden=hidden)
        Ws = d["W"]; bs = d["b"]
        if len(Ws) != len(n.W) or len(bs) != len(n.b):
            raise ValueError(
                f"layer count mismatch: file has {len(Ws)} W tensors, "
                f"net config has {len(n.W)}"
            )
        for i in range(len(n.W)):
            n.W[i] = torch.tensor(Ws[i], dtype=torch.float32)
            n.b[i] = torch.tensor(bs[i], dtype=torch.float32)
        return n

    @classmethod
    def load_json(cls, path: str) -> "Net":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # --- snapshot helpers for parallel training ---------------------------

    def snapshot(self) -> List[torch.Tensor]:
        """Return a flat list of parameter tensor clones (W_0, b_0, W_1, b_1, ...).

        Used by parallel_train to send weights between processes.
        """
        out: List[torch.Tensor] = []
        for W, b in zip(self.W, self.b):
            out.append(W.clone())
            out.append(b.clone())
        return out

    def load_snapshot(self, params: Sequence[torch.Tensor]):
        """Load a snapshot (flat [W_0, b_0, W_1, b_1, ...]) into this Net.
        If the snapshot's layer shapes differ from this Net's current shapes,
        we resize: hidden_layers, W, b, and eligibility traces are all rebuilt.
        """
        if len(params) % 2 != 0:
            raise ValueError(f"snapshot length must be even, got {len(params)}")
        new_n_layers = len(params) // 2
        # Derive hidden_layers from snapshot shapes: each hidden layer's W has
        # shape (h_i, h_{i-1}); the last layer is (OUTPUT, h_last).
        hidden_from_snap: List[int] = []
        for i in range(new_n_layers - 1):
            hidden_from_snap.append(int(params[2 * i].shape[0]))
        # If shape changed, rebuild internal state.
        if hidden_from_snap != self.hidden_layers:
            self.hidden_layers = hidden_from_snap
            self.W = []
            self.b = []
            for i in range(new_n_layers):
                self.W.append(torch.zeros_like(params[2 * i]))
                self.b.append(torch.zeros_like(params[2 * i + 1]))
            self._L = new_n_layers - 1
            self.reset_traces()
        for i in range(new_n_layers):
            self.W[i] = params[2 * i].clone()
            self.b[i] = params[2 * i + 1].clone()


# --- distillation helpers ---------------------------------------------------

def heuristic_to_4vector(heuristic_equity: float, gammon_rate: float = 0.18) -> List[float]:
    """Soft 4-vector target derived from a scalar heuristic equity in [-1, 1].

    Maps the scalar onto (p_w, p_gw, p_l, p_gl) using a logistic on the equity
    to recover P(win), and a fixed empirical gammon rate for the conditional
    P(gammon | win). TD reshapes the heads in self-play.
    """
    p_w = 1.0 / (1.0 + math.exp(-2.0 * heuristic_equity))
    p_l = 1.0 - p_w
    p_gw = gammon_rate * p_w
    p_gl = gammon_rate * p_l
    return [p_w, p_gw, p_l, p_gl]
