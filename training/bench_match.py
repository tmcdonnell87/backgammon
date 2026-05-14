"""AI-vs-AI cubeful match-play benchmark.

Plays N matches to a fixed length between two nets, with Janowski
dead-cube doubling decisions driven by the MET in public/weights/met.json.
Reports per-A match win rate with Wilson CI, plus per-game stats
(gammon rate, average ending cube, drop rate).

Usage:
   cd training
   ../.venv/bin/python bench_match.py \\
       --A ../public/weights/expert.json \\
       --B ../public/weights/expert.phase0.json \\
       --matches 400 --match-length 7 --seed 1

For 1-output legacy nets (Phase 0), gammon probabilities are zeroed at
cube-decision time (the net has no gammon information). This honestly
penalizes the gammon-blind net in cubeful play — exactly the point of
the comparison.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from engine import (
    starting_position, mirror, generate_plays, check_win, Position,
)
from encoding import encode
from race_filter import still_in_contact
from cube_decision import (
    decide_cube_action, decide_take_drop, load_met, Met,
)


OutcomesFn = Callable[[Position], Tuple[float, float, float, float]]
EquityFn = Callable[[Position], float]


def load_net_with_outcomes(path: str) -> Tuple[object, EquityFn, OutcomesFn]:
    """Like match_nets.load_net_auto, but also returns an outcomes_fn that
    yields (p_w, p_gw, p_l, p_gl) from the position's perspective."""
    with open(path) as f:
        d = json.load(f)

    if d.get("version") == 2 and "contact" in d and "race" in d:
        base = os.path.dirname(path)
        contact_path = d["contact"] if os.path.isabs(d["contact"]) \
            else os.path.join(base, d["contact"])
        race_path = d["race"] if os.path.isabs(d["race"]) \
            else os.path.join(base, d["race"])
        _c, eq_c, out_c = load_net_with_outcomes(contact_path)
        _r, eq_r, out_r = load_net_with_outcomes(race_path)

        def eq(pos):
            return eq_c(pos) if still_in_contact(pos) else eq_r(pos)

        def out(pos):
            return out_c(pos) if still_in_contact(pos) else out_r(pos)
        return (_c, _r), eq, out

    out_dim = d.get("output", 1)
    if out_dim == 4:
        from net_torch import Net as TorchNet
        net = TorchNet.from_dict(d)

        def eq(pos, _net=net):
            return _net.equity(encode(pos))

        def out(pos, _net=net):
            y = _net.value(encode(pos)).tolist()
            return (float(y[0]), float(y[1]), float(y[2]), float(y[3]))
        return net, eq, out

    if out_dim == 1:
        from net import Net as NumpyNet
        net = NumpyNet.from_dict(d)

        def eq(pos, _net=net):
            return 2.0 * _net.value(encode(pos)) - 1.0

        def out(pos, _net=net):
            # No gammon information; cube decisions stay conservative.
            p_w = float(_net.value(encode(pos)))
            return (p_w, 0.0, 1.0 - p_w, 0.0)
        return net, eq, out

    raise ValueError(f"Unknown net output dim: {out_dim}")


def pick_with_equity(p: Position, plays, equity_in_opp_frame: EquityFn) -> int:
    """Pick play index maximizing our (us-frame) equity."""
    if len(plays) == 1:
        return 0
    best_i, best = 0, -math.inf
    for i, (_pl, after) in enumerate(plays):
        win = check_win(after)
        if win is not None:
            _winner_abs, base_pts = win
            us_eq = float(base_pts)
        else:
            opp = mirror(after)
            us_eq = -equity_in_opp_frame(opp)
        if us_eq > best:
            best = us_eq
            best_i = i
    return best_i


# ---------------- match state + stats ----------------

@dataclass
class MatchState:
    score_a: int = 0           # match score for net A
    score_b: int = 0           # match score for net B
    match_length: int = 7
    cube_value: int = 1
    cube_owner_is_a: Optional[bool] = None   # None=centered, True=A, False=B
    crawford: bool = False
    _was_crawford: bool = False  # internal: was the *previous* game a Crawford game?

    def start_new_game(self):
        # Crawford: first game where one side reaches match_length-1.
        # Once a Crawford game has been played, post-Crawford resumes regular
        # cube; we never re-enter Crawford in the same match.
        if self.crawford:
            self._was_crawford = True
            self.crawford = False
        elif (not self._was_crawford
              and self.match_length > 1
              and max(self.score_a, self.score_b) == self.match_length - 1):
            self.crawford = True
        self.cube_value = 1
        self.cube_owner_is_a = None


@dataclass
class GameStats:
    games: int = 0
    matches: int = 0
    a_match_wins: int = 0
    a_wins: int = 0
    b_wins: int = 0
    gammons: int = 0
    backgammons: int = 0
    drops: int = 0
    doubles_offered: int = 0
    doubles_taken: int = 0
    cube_value_sum: int = 0       # accumulated final cube_value over games
    cube_value_max_in_match: int = 0


# ---------------- single-game play ----------------

@dataclass
class GameResult:
    a_won: bool         # did A win the game?
    base_points: int    # 1=single, 2=gammon, 3=backgammon; 0 if dropped
    cube_value: int     # cube value when the game ended (used for scoring)
    dropped: bool       # game ended on a drop (no roll-out)


def _play_one_game(a_is_side0: bool,
                   eq_A: EquityFn, out_A: OutcomesFn,
                   eq_B: EquityFn, out_B: OutcomesFn,
                   ms: MatchState,
                   met: Met,
                   rng: random.Random,
                   stats: GameStats) -> GameResult:
    """Play one game to completion (or to a drop).

    `a_is_side0` controls which engine side (0 or 1) A plays in this game.
    Side 0 always gets the first roll under our convention.
    """
    p = starting_position()
    # Opening roll: doubles disallowed.
    d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
    while d1 == d2:
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)

    first_turn = True

    while True:
        mover_is_a = (p.turn == 0) if a_is_side0 else (p.turn == 1)
        # Cube decision before the (non-opening) roll, gated on cube legality.
        if (not first_turn
                and ms.match_length > 1
                and not ms.crawford
                and (ms.cube_owner_is_a is None
                     or ms.cube_owner_is_a == mover_is_a)):
            out_self = out_A if mover_is_a else out_B
            out_opp = out_B if mover_is_a else out_A
            o4 = out_self(p)
            score_self = ms.score_a if mover_is_a else ms.score_b
            score_opp = ms.score_b if mover_is_a else ms.score_a
            # Normalize cube owner to decider-frame: 0 if decider owns,
            # 1 if opp owns, None if centered.
            if ms.cube_owner_is_a is None:
                cube_owner_norm = None
            elif ms.cube_owner_is_a == mover_is_a:
                cube_owner_norm = 0
            else:
                cube_owner_norm = 1
            action = decide_cube_action(
                side=0,
                score=(score_self, score_opp),
                match_length=ms.match_length,
                crawford=ms.crawford,
                cube_value=ms.cube_value,
                cube_owner=cube_owner_norm,
                outcomes=o4,
                met=met,
            )
            if action != "no_double":
                stats.doubles_offered += 1
                # Opp responds. Outcomes from opp's perspective: mirror p.
                p_recv = mirror(p)
                o4_recv = out_opp(p_recv)
                t = decide_take_drop(
                    side=0,
                    score=(score_opp, score_self),
                    match_length=ms.match_length,
                    crawford=ms.crawford,
                    cube_value=ms.cube_value,
                    outcomes=o4_recv,
                    met=met,
                )
                if t == "drop":
                    return GameResult(
                        a_won=mover_is_a,
                        base_points=0,
                        cube_value=ms.cube_value,
                        dropped=True,
                    )
                stats.doubles_taken += 1
                ms.cube_value *= 2
                ms.cube_owner_is_a = not mover_is_a
        first_turn = False

        plays = generate_plays(p, d1, d2)
        if len(plays) == 1 and len(plays[0][0]) == 0:
            after = p
        else:
            eq_fn = eq_A if mover_is_a else eq_B
            idx = pick_with_equity(p, plays, eq_fn)
            _, after = plays[idx]
        win = check_win(after)
        if win is not None:
            winner_abs, base_pts = win
            # Did A win?
            a_won = (winner_abs == 0 and a_is_side0) or \
                    (winner_abs == 1 and not a_is_side0)
            return GameResult(
                a_won=a_won,
                base_points=int(base_pts),
                cube_value=ms.cube_value,
                dropped=False,
            )
        p = mirror(after)
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)


def play_one_match(a_starts_side0: bool,
                   eq_A: EquityFn, out_A: OutcomesFn,
                   eq_B: EquityFn, out_B: OutcomesFn,
                   match_length: int,
                   met: Met,
                   rng: random.Random,
                   stats: GameStats) -> bool:
    """Play one match. Returns True if A wins it."""
    ms = MatchState(match_length=match_length)
    game_idx = 0
    max_cube_in_match = 1
    while ms.score_a < match_length and ms.score_b < match_length:
        ms.start_new_game()
        # A alternates sides each game.
        a_is_side0 = a_starts_side0 if (game_idx % 2 == 0) \
            else (not a_starts_side0)
        gr = _play_one_game(
            a_is_side0=a_is_side0,
            eq_A=eq_A, out_A=out_A, eq_B=eq_B, out_B=out_B,
            ms=ms, met=met, rng=rng, stats=stats,
        )
        # Score award:
        #  drop: doubler wins cube_value points (pre-doubling cube value;
        #        our flow returns ms.cube_value unchanged on drop).
        #  normal: winner wins base_points * cube_value.
        if gr.dropped:
            pts = gr.cube_value
        else:
            pts = gr.base_points * gr.cube_value
        if gr.a_won:
            ms.score_a += pts
            stats.a_wins += 1
        else:
            ms.score_b += pts
            stats.b_wins += 1
        stats.games += 1
        stats.cube_value_sum += gr.cube_value
        if gr.cube_value > max_cube_in_match:
            max_cube_in_match = gr.cube_value
        if gr.dropped:
            stats.drops += 1
        else:
            if gr.base_points == 2:
                stats.gammons += 1
            elif gr.base_points >= 3:
                stats.backgammons += 1
        game_idx += 1

    if max_cube_in_match > stats.cube_value_max_in_match:
        stats.cube_value_max_in_match = max_cube_in_match
    return ms.score_a >= match_length


# ---------------- main ----------------

def _wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.5, 0.5
    p = wins / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True, help="weights for net A")
    ap.add_argument("--B", required=True, help="weights for net B")
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--match-length", type=int, default=7)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--met", default="../public/weights/met.json")
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    met_path = args.met if os.path.isabs(args.met) \
        else os.path.normpath(os.path.join(here, args.met))
    met = load_met(met_path)

    print(f"loading A: {args.A}")
    _A, eq_A, out_A = load_net_with_outcomes(args.A)
    print(f"loading B: {args.B}")
    _B, eq_B, out_B = load_net_with_outcomes(args.B)
    print(f"MET: matches={met.matches} model={'dead-cube'} "
          f"g={met.gammon_rate} bg={met.backgammon_rate}")
    print(f"Playing {args.matches} match-to-{args.match_length} games, seed={args.seed}")

    rng = random.Random(args.seed)
    stats = GameStats()
    t0 = time.time()
    for m in range(args.matches):
        # A starts as side 0 on even matches; side 1 on odd matches.
        a_starts_side0 = (m % 2 == 0)
        a_won = play_one_match(
            a_starts_side0=a_starts_side0,
            eq_A=eq_A, out_A=out_A, eq_B=eq_B, out_B=out_B,
            match_length=args.match_length, met=met, rng=rng, stats=stats,
        )
        stats.matches += 1
        if a_won:
            stats.a_match_wins += 1
        if (m + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            wr = stats.a_match_wins / stats.matches
            print(f"  match {stats.matches}/{args.matches}  "
                  f"A win rate {wr:.3f}  "
                  f"games/match {stats.games / stats.matches:.1f}  "
                  f"elapsed {elapsed:.1f}s "
                  f"({stats.games / elapsed:.1f} g/s)",
                  flush=True)

    wr = stats.a_match_wins / stats.matches
    lo, hi = _wilson_ci(stats.a_match_wins, stats.matches)
    print()
    print(f"A wins {stats.a_match_wins}/{stats.matches} matches = {wr:.3f}  "
          f"(Wilson 95% CI [{lo:.3f}, {hi:.3f}])")
    print(f"Games played: {stats.games}  ({stats.games / stats.matches:.1f} per match)")
    if stats.games:
        print(f"  A game wins: {stats.a_wins} / {stats.games} = "
              f"{stats.a_wins / stats.games:.3f}")
        print(f"  Gammons:     {stats.gammons} ({stats.gammons / stats.games:.3f})")
        print(f"  Backgammons: {stats.backgammons} ({stats.backgammons / stats.games:.3f})")
        print(f"  Drops:       {stats.drops} ({stats.drops / stats.games:.3f})")
        print(f"  Doubles offered: {stats.doubles_offered} "
              f"({stats.doubles_offered / stats.games:.2f} per game)")
        if stats.doubles_offered:
            print(f"  Doubles taken:   {stats.doubles_taken} "
                  f"({stats.doubles_taken / stats.doubles_offered:.3f} of offers)")
        print(f"  Average final cube value: "
              f"{stats.cube_value_sum / stats.games:.3f}")


if __name__ == "__main__":
    main()
