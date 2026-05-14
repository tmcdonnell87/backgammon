"""Build a Match Equity Table (MET) for cubeful match play.

This builds the *dead-cube* MET by direct DP over single-game outcomes:

  MET[a][b] = expected MWC for the player who is `a` away (needs `a` more
              points), against an opponent who is `b` away, averaged over
              one game's worth of single/gammon/backgammon outcomes, then
              recursing into the resulting score.

Boundary:
  MET[0][b] = 1.0   for b >= 1 (we already won)
  MET[a][0] = 0.0   for a >= 1 (they already won)

Single-game outcome distribution (us perspective):
  single win:    p_w * (1 - g)         -> us scores 1 point
  gammon win:    p_w * (g - bg)        -> us scores 2
  backgammon w:  p_w * bg              -> us scores 3
  single loss:   (1-p_w) * (1 - g)     -> opp scores 1
  gammon loss:   (1-p_w) * (g - bg)    -> opp scores 2
  backgammon l:  (1-p_w) * bg          -> opp scores 3

`cube_efficiency` is recorded in the output so the cube-decision module
(which lives in TS at `src/ai/cubeDecision.ts` and Python at
`training/cube_decision.py`) can apply the Janowski live-cube formula on
top of this table. It is unused at MET build time in the dead-cube model.

The default gammon rate 0.26 is the empirical per-win gammon fraction in
strong backgammon play; backgammon rate 0.01 follows the same. After Phase
F we re-bake with measured rates from the trained net (see plan).

Run:
  cd training
  ../.venv/bin/python met_build.py            # writes ../public/weights/met.json
  ../.venv/bin/python met_build.py --matches 25 --gammon-rate 0.28
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List


def build_met(matches: int = 15,
              p_w: float = 0.5,
              gammon_rate: float = 0.26,
              backgammon_rate: float = 0.01,
              cube_efficiency: float = 0.7) -> Dict[str, object]:
    """Construct the MET as a nested list [a][b] for a, b in [0, matches].

    Single-game outcome probabilities multiply through the recursion. Direct
    DP works because each MET[a][b] depends only on entries with strictly
    smaller a or strictly smaller b.
    """
    N = matches
    if N < 1:
        raise ValueError(f"matches must be >= 1, got {N}")
    if not 0.0 < p_w < 1.0:
        raise ValueError(f"p_w must be in (0, 1), got {p_w}")
    if not 0.0 <= backgammon_rate <= gammon_rate <= 1.0:
        raise ValueError(
            f"need 0 <= backgammon_rate ({backgammon_rate}) "
            f"<= gammon_rate ({gammon_rate}) <= 1")

    p_sw = p_w * (1.0 - gammon_rate)
    p_gw = p_w * (gammon_rate - backgammon_rate)
    p_bw = p_w * backgammon_rate
    p_sl = (1.0 - p_w) * (1.0 - gammon_rate)
    p_gl = (1.0 - p_w) * (gammon_rate - backgammon_rate)
    p_bl = (1.0 - p_w) * backgammon_rate

    # MET[a][b] for a, b in [0, N]
    met: List[List[float]] = [[0.0] * (N + 1) for _ in range(N + 1)]
    for b in range(N + 1):
        met[0][b] = 1.0
    for a in range(1, N + 1):
        met[a][0] = 0.0

    # Fill in order of increasing a + b; each entry only depends on smaller
    # a or smaller b, so a single sweep is exact (no fixed-point iteration).
    for s in range(2, 2 * N + 1):
        a_lo = max(1, s - N)
        a_hi = min(N, s - 1)
        for a in range(a_lo, a_hi + 1):
            b = s - a
            if b < 1 or b > N:
                continue
            met[a][b] = (
                p_sw * met[max(0, a - 1)][b]
                + p_gw * met[max(0, a - 2)][b]
                + p_bw * met[max(0, a - 3)][b]
                + p_sl * met[a][max(0, b - 1)]
                + p_gl * met[a][max(0, b - 2)]
                + p_bl * met[a][max(0, b - 3)]
            )

    return {
        "version": 1,
        "model": "dead-cube",
        "matches": N,
        "p_w": p_w,
        "gammon_rate": gammon_rate,
        "backgammon_rate": backgammon_rate,
        "cube_efficiency": cube_efficiency,
        # Index as met[a][b]: a is us-away, b is them-away
        "met": met,
    }


def met_entry(met: Dict[str, object], away_us: int, away_them: int) -> float:
    """Look up MWC for us at score (away_us, away_them), with cube-state-free
    averaging (next-game starts with cube centered at 1)."""
    N = int(met["matches"])  # type: ignore[index]
    table: List[List[float]] = met["met"]  # type: ignore[assignment]
    a = max(0, min(N, away_us))
    b = max(0, min(N, away_them))
    if a == 0:
        return 1.0
    if b == 0:
        return 0.0
    return float(table[a][b])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=15,
                    help="largest match length supported (default: 15)")
    ap.add_argument("--p-w", type=float, default=0.5,
                    help="cubeless win probability assumed symmetric (default: 0.5)")
    ap.add_argument("--gammon-rate", type=float, default=0.26,
                    help="fraction of wins that are gammons-or-better (default: 0.26)")
    ap.add_argument("--backgammon-rate", type=float, default=0.01,
                    help="fraction of wins that are backgammons (default: 0.01)")
    ap.add_argument("--cube-efficiency", type=float, default=0.7,
                    help="live-cube efficiency τ for downstream cube decisions (default: 0.7)")
    ap.add_argument("--out", default="../public/weights/met.json",
                    help="output path (default: ../public/weights/met.json)")
    args = ap.parse_args()

    met = build_met(
        matches=args.matches,
        p_w=args.p_w,
        gammon_rate=args.gammon_rate,
        backgammon_rate=args.backgammon_rate,
        cube_efficiency=args.cube_efficiency,
    )
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(met, f, separators=(",", ":"))
    print(f"wrote MET (matches={args.matches}, g={args.gammon_rate}, "
          f"bg={args.backgammon_rate}, p_w={args.p_w}) to {out}")
    print("MET[a][b] preview (5x5 corner; a = us-away, b = them-away):")
    table: List[List[float]] = met["met"]  # type: ignore[assignment]
    header = "      " + "  ".join(f"b={b:<5}" for b in range(1, 6))
    print(header)
    for a in range(1, 6):
        row = "  ".join(f"{table[a][b]:.4f}" for b in range(1, 6))
        print(f"  a={a}: {row}")


if __name__ == "__main__":
    main()
