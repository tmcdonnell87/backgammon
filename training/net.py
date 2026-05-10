"""1-hidden-layer net with TD(λ) eligibility traces, in numpy.

Architecture: x (198) -> tanh hidden (H) -> sigmoid output (1)
   Output = P(player-on-roll wins from this position). Since input is from
   "us" perspective, output is "P(we win)".

Forward:
   z1 = W1 @ x + b1            (H,)
   h  = tanh(z1)               (H,)
   z2 = W2 @ h + b2            scalar
   y  = sigmoid(z2)            scalar in (0, 1)

Loss: TD error e_t = r_t + gamma * y_{t+1} - y_t (with gamma=1, r=0 for
non-terminal, r=outcome at terminal). Eligibility traces accumulate
gradient of y_t over time.

Sutton's TD(λ) update for general parameter w:
   trace_w := gamma * lambda * trace_w + dY/dw
   w       := w + alpha * e_t * trace_w
We treat each game as one episode and reset traces between games.
"""
from __future__ import annotations
import json
import numpy as np

INPUT = 198
HIDDEN = 80


class Net:
    def __init__(self, hidden: int = HIDDEN, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Small init keeps tanh in linear regime initially.
        self.W1 = rng.standard_normal((hidden, INPUT)).astype(np.float32) * 0.10
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.standard_normal(hidden).astype(np.float32) * 0.10
        self.b2 = np.float32(0.0)
        self.hidden = hidden
        self.reset_traces()

    def reset_traces(self):
        self.eW1 = np.zeros_like(self.W1)
        self.eb1 = np.zeros_like(self.b1)
        self.eW2 = np.zeros_like(self.W2)
        self.eb2 = np.float32(0.0)

    def forward(self, x: np.ndarray):
        z1 = self.W1 @ x + self.b1
        h = np.tanh(z1)
        z2 = float(self.W2 @ h + self.b2)
        # Stable sigmoid:
        if z2 >= 0:
            ez = np.exp(-z2)
            y = 1.0 / (1.0 + ez)
        else:
            ez = np.exp(z2)
            y = ez / (1.0 + ez)
        return y, h, x

    def value(self, x: np.ndarray) -> float:
        y, _, _ = self.forward(x)
        return y

    def td_step(self, x: np.ndarray, y: float, h: np.ndarray,
                td_error: float, alpha: float, lam: float):
        """Update traces with grad of y_t at (x, h, y) and apply alpha*td_error*trace.

        Gradient of y = sigmoid(W2.h + b2) wrt parameters:
           dy/dz2 = y * (1 - y)
           dy/dW2 = y(1-y) * h
           dy/db2 = y(1-y)
           dy/dh  = y(1-y) * W2
           dh/dz1 = 1 - h^2     (elementwise)
           dy/dz1 = y(1-y) * W2 * (1 - h^2)
           dy/dW1 = outer(dy/dz1, x)
           dy/db1 = dy/dz1
        """
        s = y * (1.0 - y)
        dW2 = s * h
        db2 = s
        dz1 = s * self.W2 * (1.0 - h * h)
        dW1 = np.outer(dz1, x)
        db1 = dz1

        # Decay traces, accumulate gradient.
        self.eW1 = lam * self.eW1 + dW1
        self.eb1 = lam * self.eb1 + db1
        self.eW2 = lam * self.eW2 + dW2
        self.eb2 = lam * self.eb2 + db2

        # Apply update.
        delta = alpha * td_error
        self.W1 += delta * self.eW1
        self.b1 += delta * self.eb1
        self.W2 += delta * self.eW2
        self.b2 = np.float32(self.b2 + delta * self.eb2)

    def to_dict(self) -> dict:
        return {
            "input": INPUT,
            "hidden": self.hidden,
            "output": 1,
            "W1": self.W1.astype(np.float32).tolist(),
            "b1": self.b1.astype(np.float32).tolist(),
            "W2": self.W2.astype(np.float32).tolist(),
            "b2": float(self.b2),
        }

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, d: dict) -> "Net":
        n = cls(hidden=d["hidden"])
        n.W1 = np.asarray(d["W1"], dtype=np.float32)
        n.b1 = np.asarray(d["b1"], dtype=np.float32)
        n.W2 = np.asarray(d["W2"], dtype=np.float32)
        n.b2 = np.float32(d["b2"])
        return n

    @classmethod
    def load_json(cls, path: str) -> "Net":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def save_npz(net: Net, path: str):
    np.savez(path, W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2,
             hidden=net.hidden)


def load_npz(path: str) -> Net:
    z = np.load(path)
    n = Net(hidden=int(z["hidden"]))
    n.W1 = z["W1"].astype(np.float32)
    n.b1 = z["b1"].astype(np.float32)
    n.W2 = z["W2"].astype(np.float32)
    n.b2 = np.float32(z["b2"])
    return n
