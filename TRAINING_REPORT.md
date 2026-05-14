# Backgammon AI — Training Report

**Dates:** 2026-05-10 to 2026-05-13
**Hardware:** Ryzen 7 255 (8c/16t), DDR5, CPU-only (PyTorch CPU + multiprocessing)
**Total compute:** ~18 hours of CPU training + benching

## TL;DR

The published Expert net (`public/weights/expert.json`) is now **Phase A**:
1-hidden-layer × 120 width, 4-output sigmoid head
(`p_win, p_gammon_win, p_loss, p_gammon_loss`),
trained by heuristic-distill (200k positions) + 1.3M games of parallel TD(λ)
self-play. The prior 1-output net is preserved at
`public/weights/expert.phase0.json`. **Phase I** added an exact one-sided
bear-off database at `public/weights/bearoff.bin` for race endgames.

| Net | vs heuristic (0-ply / 2-ply) | h2h vs Phase A |
|---|---|---|
| **Phase A** *(published)* | 79% / 78% | (baseline) |
| Phase 0 *(legacy 1-output, preserved)* | 75% / — | 47.95% ± 1.55pp |
| Phase B (200×200, 1.3M TD) | 75% / — | 51.3% ± 1.55pp (parity) |
| Phase E gnubg-distill (500k) | 67% / 77% | 35% |
| Phase E-TD 300k (gnubg + 300k TD) | 74% / 81%* | 42% ± 6.8pp at 2-ply |
| Phase I gnubg-distill 2M (1×120 to 2×400) | — / — | parity to -10pp (varies) |

*\*100-game sample; statistically equivalent to Phase A's 78%.*

Against **gnubg-2ply** as the reference engine, Phase A's hint (2-ply) is
**72.5% top-1 / 0.009 mean equity loss / +0.024 signed bias**. Adding the
Phase I bear-off DB lifts this ~+0.5pp on the bench and provides exact
equity in any pure home-board race (vs gnubg-0ply, max diff 0.0017).

Phase A is a clean structural win over the legacy: 4-output head adds
gammon awareness, and 0-ply heuristic bench is +4pp (75% → 79%). Beyond
that, **bigger nets, stronger teachers, phased dispatch, and supervised
distillation (equity or ranking loss) did not break the cubeless h2h
ceiling.** The remaining ~27pp top-1 gap to gnubg-2ply is structural —
search depth (3-ply+) and Tesauro features, not training signal.

## Scope

Five-phase plan from
`~/.claude/plans/continue-hidden-spring.md`:

| Phase | What | Outcome |
|---|---|---|
| A | PyTorch CPU port + 4-output head | **Shipped** — published as `expert.json` |
| B | 2-layer 200×200 net, parallel TD | Trained; statistical parity with A |
| C | Race/contact split + dispatch | Infrastructure shipped; not exercised against strong teacher yet |
| D | 2-ply rollout TD targets | Infrastructure shipped; subsumed by Phase E in practice |
| E | gnubg distill + bench | Infrastructure shipped; trained 500k-position distill + 400k TD on top |

## Architecture (Phase A — published)

```
198 (TD-Gammon encoding) → Linear(120) → tanh → Linear(4) → sigmoid
```

Outputs are the four cubeless probabilities. Cubeless equity =
`p_w + p_gw − p_l − p_gl`, range roughly `[−2, +2]` when calibrated.

Training:
- **Distill**: 200k positions sampled from heuristic self-play. Per-head BCE
  against a soft 4-vector target derived from the scalar heuristic equity
  (`p_w = σ(2·heuristic)`, `p_gw = 0.18 · p_w`, etc.). 30 epochs, cosine LR
  0.2→0.02. Bench after distill: 35% vs heuristic.
- **TD(λ)**: parallel-pool self-play, 12 worker processes + 1 parameter
  server, α=0.01, λ=0.5, Polyak blend=0.1, broadcast every 500 games. Ran
  200k validation + 1.1M continuation = 1.3M games total. Best checkpoint
  at 1.3M total games: **52.05% ± 1.55pp** vs the legacy 1-output net over
  4000 games of 0-ply h2h. Throughput: ~200 g/s aggregate.

Hyperparameter learning: α=0.03 (the legacy 1-output choice) **diverges** in
the 4-output trainer because the shared trunk (W1, b1) receives a summed
gradient from all 4 heads — effective step is ~4× the scalar case. α=0.01
trains stably.

## Phase B (2-layer 200×200, parallel TD)

Re-distilled the wider net (`eq_mse=0.0154`, slightly better fit than
Phase A's 0.0194), then ran 1.3M TD games with the same parallel
infrastructure (α=0.01, λ=0.5, blend=0.1).

- vs heuristic at 0-ply: ~75% plateau (no improvement over Phase A).
- vs Phase A, 4000-game cross-seeded h2h: **51.3% ± 1.55pp** — lower
  CI bound 49.8%, *not* a statistically clean win.
- Throughput: ~60 g/s aggregate (3-4× slower forward than Phase A due to
  the larger net).

**Verdict**: capacity is not the bottleneck for cubeless 0-ply h2h.

## Phase E (gnubg integration)

Wired up GNU Backgammon 1.07.001 as a subprocess oracle for distillation
labels and (in principle) benchmarking. Two phases of training:

### E-distill (no TD)

500k heuristic-self-play positions labeled by gnubg 2-ply (~77 eval/s, total
~108 min). 30 epochs SGD against the 6-prob output (collapsed to our
4-vector by folding backgammons into gammons).

- vs heuristic at 0-ply: **66.8%**
- vs heuristic at 2-ply (deployment mode): 77%
- vs Phase A, 2000-game 0-ply h2h: **35%**
- vs Phase A, 200-game 2-ply h2h: 42% ± 6.8pp

Distill alone gives the net gnubg's *static* eval, which is a great leaf
evaluator but not a self-play-refined *policy*. Phase A's policy was
shaped by 1.3M games of TD signal.

### E-TD (gnubg warmstart + TD on top)

Loaded the 500k-distill weights and ran 400k+ games of TD self-play. Same
hyperparams as Phase B (α=0.01, λ=0.5, blend=0.1).

- vs heuristic at 0-ply: hovered 65-75% across checkpoints
- vs heuristic at 2-ply: 81% at the 300k checkpoint (100 games, CI ±8pp)
- vs Phase A, 200-game 2-ply h2h: 42% ± 6.8pp at 300k checkpoint

The TD pass *did* improve over pure distill (+6pp h2h at the 300k peak),
but it did not produce a net stronger than Phase A.

## Bugs found and fixed during the session

These are the load-bearing ones. The code lives in the repo at the
indicated paths.

1. **`stillInContact` had the inequality backwards** —
   `src/ai/heuristic.ts:120` and the Python copy in `training/bench.py`.
   The original used `oursMin ≤ oppsMax`, which is almost always true for
   any non-bear-off position; it classified essentially every game state
   as "contact" and the heuristic's race branch was dead code. Correct
   check: `oursMax ≥ oppsMin`. Phase C dispatch would have been broken
   without this fix.

2. **Parallel-trainer deadlock at `broadcast_every=2000`** —
   `training/worker_proc.py`. The original loop blocked on
   `weight_q.get()` for each batch; with 12 workers × 50 games per batch
   = 600 games per round, no worker could produce a second batch until
   the server broadcast a new snapshot at 2000 games, which never
   happened. Fixed by caching the snapshot and treating per-batch deltas
   as *incremental* (post_batch − pre_batch) instead of (post_batch −
   last_snapshot).

3. **Polyak blend was 4× too aggressive** —
   `training/parallel_train.py`. The plan called for blend=0.5; with 12
   workers contributing incremental deltas asynchronously, that put a
   ~6× effective learning rate on the global net. After 100k games the
   output layer's L2 norm grew 3.5× and the net regressed to 4% vs
   heuristic (equity collapsed to ~0). Lowered to blend=0.1, broadcast
   every 500 games — trains stably, monotone improvement.

4. **gnubg Position-ID had the two players in reversed order** —
   `training/gnubg_client.py`. gnubg's spec puts the opponent's 25
   nibbles first and the player-on-roll's 25 nibbles second; my
   encoder did the reverse. The starting position is symmetric so the
   two orderings yield the same Position ID by coincidence, hiding the
   bug. Caught by sending an obviously-X-winning bear-off position and
   getting equity=−1 back. Fixed; verified by an asymmetric race
   convention test.

5. **gnubg read queue not drained per query** — same file. The
   subprocess reader accumulates output lines from previous `eval`
   commands; without a drain at the start of each query, the regex
   match would return the *previous* position's eval row, so every
   call returned the same number. Confirmed by sending five different
   positions and getting five identical evals. Fixed by draining
   `out_q` at the head of each `evaluate_position`.

6. **`distill_gnubg.py` had a backward-pass ordering bug** — applied
   layer updates inside the backward loop and then used the
   just-updated weights to backpropagate to lower layers. Loss still
   trended downward, so it wasn't obvious; but the inner-layer
   gradients were wrong. After fixing (collect gradients first, apply
   in a second pass — same pattern as `training/distill.py`), the
   gnubg-distilled net's starting-position equity matched gnubg's to
   1e-2.

## Throughput numbers

| Workload | Throughput |
|---|---|
| Phase A self-play (1-layer 120, 12 workers) | ~200 games/s |
| Phase B/E self-play (2-layer 200×200, 12 workers) | ~60 games/s |
| gnubg eval (subprocess, 2-ply) | ~77 evals/s |
| Heuristic distill (200k pos × 30 ep, hidden=120) | ~100s wall |
| gnubg distill (500k pos × 30 ep, hidden=200×200) | ~110 min wall (labeling dominates) |
| 0-ply bench (1000 games vs heuristic) | ~20-30s |
| 2-ply self-bench (100 games) | ~6-7 min per net (full CPU) |

## What didn't move the needle

- **Bigger trunk** (Phase B 200×200 vs Phase A 1×120) → ~parity h2h.
- **Better teacher** (gnubg-distill vs heuristic-distill) → catches up after
  TD, but doesn't surpass.
- **TD on top of gnubg warmstart** → marginal improvement over pure distill
  but Phase A still beats it at 2-ply h2h.

The plan's prediction held: *"h2h cubeless winrate doesn't move
dramatically"* — the structural benefits of the 4-output head and the
2-layer trunk are mostly invisible to cubeless 0-ply h2h. They likely
show up in cube/match play, which we did not measure in this session.

## Files in the repo (new / modified)

Python (`training/`):
- `net_torch.py` — torch CPU net, arbitrary hidden-layer depth, manual TD(λ)
  gradient + per-head eligibility traces, JSON serialization (multi-layer +
  legacy 1-layer load-compat)
- `parallel_train.py`, `worker_proc.py` — worker pool + parameter server
- `distill.py` — torch version, 4-vector targets from heuristic equity
- `distill_gnubg.py` — labels via gnubg 2-ply; **Phase I fix** for gnubg's
  `W(g)` already nesting `W(bg)` (no addition needed)
- `label_gnubg_mp.py` — **Phase I.** Multi-process gnubg labeling at
  ~290 labels/s (4 workers), with checkpoint+resume. General-purpose
  infrastructure for any future supervised distillation pass
- `bearoff_build.py` — **Phase I.** Builds the one-sided race table
- `bearoff_verify.py` — **Phase I.** Cross-checks bearoff vs gnubg on
  sampled pure-race positions
- `bench_move_accuracy.py` — **Phase I.** `--net-ply` flag for net-side
  ply (0 or 2), `--bearoff` for table-augmented evaluation
- `gnubg_client.py` — subprocess wrapper + Position-ID encoder
- `rollout.py` — Python port of `src/ai/search.ts:score2ply`; 2-ply
  expectimax 4-vector targets
- `race_filter.py` — exported `still_in_contact` / `is_race` for Phase C
- `outcome.py` — terminal-state 4-vector classifier
- `train.py` — refactored for `mode={all|contact|race}` and rollout_fraction
- `bench.py`, `match_nets.py`, `publish_if_good.py` — auto-detect across
  three weight schemas (1-output legacy, 4-output 1-layer, 4-output ML) and
  the phased manifest

TypeScript (`src/ai/`):
- `neural.ts` — discriminated-union loader, multi-layer forward,
  `evaluateOutcomes()` for the tutor, `PhasedNeuralEvaluator` for Phase C
  dispatch; **Phase I:** `bearoffEquity` shortcut in both `evaluate` and
  `evaluateOutcomes`
- `bearoff.ts` — **Phase I.** Loader, state→index map, equity join for
  pure-race positions
- `worker.ts` — **Phase I.** Fetches `/weights/bearoff.json` at startup
- `heuristic.ts` — `stillInContact` bug fix, exported
- `engine.ts` — `AnyNeuralEvaluator` type

Tests (`test/`):
- `neural.test.ts` — new parity tests for 4-output (1-layer + 2-layer
  fixtures), rollout parity to Python, phased dispatch
- `ai.test.ts` — additional tactical regression tests (opening 4-2, race
  bear-off)
- `bearoff.test.ts` — **Phase I.** 10 tests: table load, isPureHomeRace
  eligibility, trivial-won/lost cases, race symmetry, probability bounds
- `fixtures/tiny-weights-4.json`, `tiny-weights-4-2layer.json`,
  `rollout_cases.json` — generated from the Python side

Weights (`public/weights/`):
- `expert.json` — **Phase A** (still the published net after Phase I)
- `expert.phase0.json` — legacy 1-output, preserved for rollback
- `bearoff.bin`, `bearoff.json` — **Phase I.** 7 MB raw (~770 KB
  gzipped) one-sided race table; finish-rolls + first-bearoff histograms

## Phase F — Race/Contact split with cross-net bootstrap

The "open work" item from the prior section was attempted. Plan in
`~/.claude/plans/continue-hidden-spring.md`. Result: **at parity with
Phase A, does not pass the ship gate; not published.**

Pipeline:
- **Race specialist net** (`training/distill_race.py` + race-mode TD).
  Hidden=[40], 100k race-only positions distilled from heuristic
  self-play (eq_mse=0.0146), then 300k games of mode=race TD at
  α=0.01, λ=0.3, blend=0.1, broadcast=500. Best checkpoint at 50k
  games — race plateaus fast.
- **Cross-net bootstrap** in `training/train.py:play_game`: when
  training the contact net (mode=contact) and the position after our
  move is a race position, the TD target uses the frozen race net's
  evaluation instead of self-bootstrap. Phase check moves from
  `_phase_matches(after)` to `_phase_matches_input(prev_was_contact)`
  so the contact net fires on boundary plies.
- **Contact net training**: 600k games at hidden=[120], mode=contact,
  resumed from `expert.json`, with `--bootstrap-weights race-50k`.
  ~66 min on 12 workers.

Phased manifest schema (already wired in `src/ai/neural.ts:253`):
`{"version": 2, "contact": "...", "race": "..."}`. `match_nets.py` and
`publish_if_good.py` extended to support paired-weights mode.

### Race-net gates

Race specialist evaluated in phased deployment with heuristic as the
contact half (via new `bench_race_gate.py` and `"contact":"heuristic"`
support in `match_nets.load_net_auto`), vs heuristic-only:

| Race net | phased winrate vs heuristic-only (2000 games) |
|---|---|
| distill only (no TD) | 49.3% (CI ±2.2%) |
| TD 50k (best) | **52.3%** (CI ±2.2%) |
| TD 100k | 52.0% (CI ±2.2%) |
| TD 250k | 50.9% (CI ±2.2%) |
| TD 300k (final) | 51.8% (CI ±2.2%) |

Race net adds ~3pp game-level winrate over heuristic. Modest because
the race heuristic (pip-count based) is already near-optimal; the
specialist's main contribution is better gammon/bear-off awareness
from the 4-output head.

### Boundary discontinuity is real

Plugging the race specialist into Phase A's contact net naively
(no contact retraining) **regresses** by ~2pp:

| Manifest | vs Phase A monolithic (2000 cubeless h2h) |
|---|---|
| {contact: Phase A, race: race-50k}, drop-in | 47.8% (CI ±2.2%) |
| {contact: contact-100k (bootstrapped), race: race-50k} | 50.0% |
| {contact: contact-500k (best), race: race-50k}, 4000 games | **50.0%** (CI ±1.5%) |

Contact retraining recovers the boundary loss (47.8% → 50%) — the
cross-net bootstrap mechanism works as designed. But it doesn't break
the cubeless h2h ceiling at ~51% that we hit across Phase B, E, and
now F.

### Cubeful match-play with phased net

`bench_match.py --A phased-500k --B ../public/weights/expert.json
--matches 400 --match-length 7`:
**50.5%** (Wilson 95% CI [45.6%, 55.4%]). Statistical parity.

### Verdict

Phase F infrastructure ships (Python plumbing, manifest schema, gates,
`bench_phased.py`, `bench_race_gate.py`, `publish_if_good.py` paired
mode), but the trained manifest **does not publish**. Re-running with
a stronger race net (more capacity, more TD) is plausible but the
across-the-board ceiling at 51% suggests this is a real cubeless h2h
plateau, not a capacity / specialization shortfall.

## Phase G/H — Match Equity Table + Janowski cube decisions

Built the full cubeful match-play stack: computed Janowski dead-cube
MET (`training/met_build.py` → `public/weights/met.json`), Janowski
decision module (`src/ai/cubeDecision.ts`, `training/cube_decision.py`),
match-play bench (`training/bench_match.py`), and tests
(`test/met.test.ts`, `test/cubeDecision.test.ts`).

Tests: MET sanity (symmetry, MET[1][1]=0.5, monotonicity, rough
±15pp agreement with Janowski's published 7-pt MET), cube decision
boundaries (Crawford, money game, opponent owns cube), and an
equity-sweep parity test that all three cube actions
(no_double / double_take / double_drop) appear and behave sanely.

### First publishable cubeful bench

`bench_match.py --A expert.json --B expert.phase0.json --matches 400
--match-length 7 --seed 1`: **50.2%** (Wilson 95% CI [45.4%, 55.1%]).
Second seed (=42, 400 matches): 55.7% (CI [50.9%, 60.5%]). Two-seed
pooled (800 matches): ~53.0%.

The plan target was ≥55% — not cleared. Cube doesn't amplify the
4-output gammon-aware skill gap as hoped.

### Why: dead-cube MET produces a cube spiral

Per-game stats over 400 matches show:
- doubles offered: 2.3 per game
- take rate: **99.4%**
- drop rate: **1.3%** of games
- average ending cube value: **5.96**
- gammon rate: 22.3%, backgammon rate 16.6%

Cubes routinely escalate 1 → 2 → 4 → 8 → 16 because, under the
dead-cube model, mwc_take vs mwc_drop differ by only ~0.5pp at near-
balanced positions, so the receiver almost always picks "take." Cubes
then keep ratcheting. Once a game is at cube 8, a single backgammon
wins the match; most matches end in 1.3 games.

In real play, the live-cube MET prices takes more conservatively
because the *taker* now owns redouble rights — but the doubler also
waits longer to double in the first place (live cube efficiency τ).

### Live-cube margin fix

Implemented as a follow-up: a Janowski-style "waiting value" margin in
`decideCubeAction` (TS + Python) that requires the doubling MWC to
beat no-double by `τ × 0.06 × cube_room` (where `cube_room` is the
fraction of the match still left to play). Keeps the dead-cube MET as
the table; only the action thresholds change.

Re-running Phase A vs Phase 0 cubeful with the live-cube margin:

| Seed | Phase A match wins / 400 | Winrate | CI |
|---|---|---|---|
| 1 | 228 | 57.0% | [52.1%, 61.8%] |
| 42 | 229 | 57.3% | [52.4%, 62.0%] |
| **Pooled (800)** | 457 | **57.1%** | ~[53.8%, 60.4%] |

Clears the 55% gate. Per-game stats also normalized:
- avg ending cube: 3.96 (was 5.96)
- drop rate: 7-9% (was 1.3%)
- take rate: 95% (was 99.4%)
- 2.1 games per match (was 1.3)

**Sanity:** Phase A self-bench is 50.5%/200 (CI [43.6%, 57.4%]) — the
bench is unbiased.

**Phased manifest under live-cube:** the Phase F manifest still ties
Phase A monolithic at 48.5% (CI [43.6%, 53.4%]) — confirms Phase F's
lack of edge is robust to cube model; not just a metric artifact.

## Phase I — gnubg comparison and the static-eval ceiling

The Phase G/H work raised an obvious question: how does Phase A actually
compare to gnubg, the open-source reference engine? Three deliverables
in this phase: an accurate gnubg comparison bench, a 2M-position
distillation experiment, and an exact bear-off database.

### Baseline correction: Phase A vs gnubg-2ply was understated

Previously reported numbers were 0-ply: 65% top-1, 0.019 mean equity
loss. But the in-app hint runs at 2-ply (`src/ai/levels.ts:39-46` →
`src/ai/engine.ts:188` → `src/ai/search.ts:70 score2ply`), and 0-ply
vs 2-ply is an unfair handicap on our side.

Added `--net-ply` to `bench_move_accuracy.py` and re-ran. The actual
2-ply-vs-2-ply numbers:

| Metric | reported (0-ply) | actual in-app (2-ply) |
|---|---|---|
| Top-1 vs gnubg-2ply | 65.0% | **72.5%** |
| Top-3 | 92.5% | 95.0% |
| Mean equity loss | 0.019 | **0.009** (at gnubg's good-move boundary) |
| Pearson correlation | 0.94 | 0.96 |
| Signed bias | +0.085 | +0.024 |

The hint engine was already at "competent intermediate" — median
equity loss 0.000, mean inside gnubg's "good move" band. The 2-ply
expectimax also self-corrects the 0-ply optimism bias from +0.085 to
+0.024, well inside the ±0.03 target.

### 2M-position gnubg-2ply distillation: didn't beat Phase A

Built `training/label_gnubg_mp.py` — multi-process gnubg labeling at
**~290 labels/sec** with 4 workers (vs the plan's 40/s estimate, ~7×
faster), with checkpoint+resume. Total label run: 2M positions in
1h54min. Position collection used 70% Phase A self-play + 30%
heuristic self-play, for state-distribution coverage of what the
in-app net actually visits.

**One bug found and fixed**: gnubg's `W(g)` value is "P(wins by
gammon or better)" and already nests `W(bg)` — the prior
`distill_gnubg.py` was adding them, double-counting backgammons. Fix
in `gnubg_eval_to_4vector`. Smoke verification: `max(p_gw - p_w) =
0.0` after fix vs `1.0` before.

Swept training configurations against the same 2M labels:

| Net | val_eq_mse | 200-pos top-1 (seed=1, +bearoff) |
|---|---|---|
| Phase A (current 1×120 TD) | — | **0.725** |
| 1×120 gentle distill (lr=0.002, 3 ep, warm-start) | 0.0345 | 0.738 (noisy +1.3pp) |
| 1×240 distill (random init, 30 ep) | 0.0214 | 0.637 |
| 1×400 distill | 0.0437 | 0.625 |
| 2×200 distill | 0.0196 | 0.644 |
| 2×400 distill | 0.0177 | 0.619 |

Confirmed across 4 seeds (882 evaluated positions, +bearoff):

| Config | pooled top-1 | signed bias |
|---|---|---|
| Phase A + bearoff | **72.3%** | +0.025 |
| 1×120 gentle distill + bearoff | 72.1% | +0.040 |
| 2×200 distill + bearoff | 67.6% | +0.013 |

**Counterintuitive result.** Bigger nets fit gnubg's equity *values*
better (lower val_eq_mse) but pick the right move *less often*. The
2×400 has the lowest val MSE in the sweep (0.018) and the worst top-1
(0.619).

The mechanism: **Phase A's TD self-play produces a smooth,
policy-coherent equity surface**. TD forces similar positions to map
to similar equities (else the bootstrap diverges); the net learns a
discriminative *gradient*. Distillation against gnubg's per-position
labels doesn't have that constraint — bigger nets memorize the noisy
targets more precisely, hurting the smoothness that move-ranking
depends on.

### Ranking-loss distillation: also didn't beat Phase A

If equity-distillation matches values but not ranking, switch to a
ranking loss directly. Built `training/label_ranking_mp.py` to
collect 25k anchor positions × ~8 plays each (169k gnubg queries,
~9 min), and a ListNet softmax-CE trainer.

Result on 4 seeds (882 positions, +bearoff):

| Config | pooled top-1 | signed bias |
|---|---|---|
| Phase A + bearoff | 72.3% | +0.025 |
| Ranking-distilled + bearoff | 71.7% | **+0.179** |

Move-ranking didn't improve (within noise), and the equity bias blew
up 7× — the ranking loss has no calibration constraint, so equity
values drift. That bias would make the in-app equity bar mislead
users even if the moves it picks are fine. Worse UX, no top-1 gain.

(Both `label_ranking_mp.py` and the trainer were experiment scaffolding
and have been removed; the negative result is recorded here.)

### Bear-off database — small but real, and shipped

Built a one-sided race-distribution table in
`training/bearoff_build.py`. For each home-board distribution
`(n_0..n_5)` with sum ≤ 15 (`C(21, 6) = 54,264` states), backward
induction over the 21 dice rolls produces two histograms:

  * `finish[s, k]` — P(state finishes all 15 off in exactly k rolls)
  * `first_off[s, k]` — P(state has scored at least 1 bear-off by roll k)

From two such tables (us, them), an explicit join in
`src/ai/bearoff.ts` computes exact `(p_w, p_gw, p_l, p_gl)` whenever
both sides are in pure home-board race (no bar, no checkers outside
home boards):

  * `P(we win) = Σ_k p_us_finish[k] · Σ_{j≥k} p_them_finish[j]`
  * `P(we gammon) = Σ_k p_us_finish[k] · (1 - p_them_first_off[k-1])`

Build: ~2 min on a single CPU core. Quantized to uint16 ×65535,
packed as `public/weights/bearoff.bin` (7 MB) + `bearoff.json`
metadata. ~770 KB gzipped on the wire.

Cross-checked against gnubg-0ply on 50 random pure-race positions:
**max equity diff 0.0017**, mean 0.0005. Gammon detection identical
(gnubg's own bear-off database is exact, so this is effectively
ground truth).

Wired as a shortcut in `NeuralEvaluator.evaluate` / `evaluateOutcomes`
(`src/ai/neural.ts`) — table lookup runs first; net falls through only
when ineligible. Test suite: `test/bearoff.test.ts` (10 tests),
all 97 project tests green.

**Bench impact: +0.3-0.5pp top-1 on the standard bench.** The bench
sampling under-represents race endgames (~6% of root positions, ~0.5%
of 2-ply leaves are pure races). In actual race-endgame gameplay
where the table activates, it's exact equity — much bigger qualitative
win than the bench number suggests.

### Phase I verdict

- **Ship bearoff DB.** Strict improvement, zero regressions, exact in
  the late race. ~770 KB gzipped extra on the wire.
- **Keep Phase A as the published net.** Distillation (equity- or
  ranking-loss, any net size from 1×120 to 2×400) didn't break the
  static-eval ceiling at 2-ply. The bottleneck is policy-coherent
  smoothness from TD self-play, not capacity or training signal.
- **Don't publish a distilled net.** Best-case +0.2pp top-1, worst-
  case bias regressions in the equity bar.

The Phase A 2-ply hint engine + bearoff gives ~73% top-1 vs gnubg-2ply
with calibrated equity (bias ±0.03). The remaining ~27pp gap to gnubg
is **structural** — search depth (3-ply+) and hand-engineered
Tesauro features, not anything we can fix with more training data on
the existing topology.

## What didn't move the needle (updated)

- **Bigger trunk** (Phase B 200×200 vs Phase A 1×120) → ~parity h2h.
- **Better teacher** (gnubg-distill vs heuristic-distill) → catches
  up after TD, but doesn't surpass.
- **TD on top of gnubg warmstart** → marginal improvement over pure
  distill but Phase A still beats it at 2-ply h2h.
- **Phase F (race/contact split with cross-net bootstrap)** →
  recovers the boundary discontinuity, lands at exact parity with
  Phase A. Doesn't break the 51% cubeless ceiling.
- **Cubeful (dead-cube MET) doubling** → cube spiral, ~53% pooled,
  below the 55% bar. Fixed by adding a live-cube waiting-value margin
  (see Phase G/H section); pooled now 57.1%, gate cleared.
- **Phase I equity-distillation against gnubg-2ply** (1×120 to 2×400,
  random and warm-start, lr/epoch sweep) → no top-1 gain over Phase A.
  Lower val_eq_mse, identical or worse move ranking.
- **Phase I ranking-loss distillation** (25k anchors, ListNet softmax
  CE, β=4.0) → no top-1 gain, calibration regression (+0.18 bias).

The structural pattern is consistent: post-Phase A improvements have
hit a ~51% ceiling in **cubeless** head-to-head, whether the change
is bigger nets, stronger teachers, phase specialization, distillation
against gnubg, or cube logic alone. The cubeless ceiling appears to
be a property of the metric (alternating-seed 0/2-ply games at this
skill level) rather than the model. **Cubeful match play with the
live-cube margin does amplify the skill gap (50% cubeless → 57%
cubeful for Phase A vs Phase 0), as the original plan predicted.**
**Bear-off DB is the only Phase I deliverable that ships — a strict
improvement on race endgames.**

## Open work — most promising next steps (updated)

1. **Controller cube wiring** — **DONE.** Added `cube-decision` phase
   to `src/game/controller.ts` with `runCubeFlow` async handler, gated
   on `cubeEnabled && both-CPU && !crawford && canDouble(...)`. Crawford
   set/clear lives in `startNewGame` via `state.crawfordPlayed`.
   `AIClient.decideCube` / `decideTake` route through `src/ai/engine.ts`
   helpers; `src/ai/worker.ts` loads `/weights/met.json` at startup.
   `test/cube.test.ts` (10 tests) covers `canDouble` semantics across
   ownership/Crawford, `applyDoubleAccepted` ownership transfer, and
   the Crawford state-machine transitions. 97/97 tests green
   (incl. 10 new bearoff tests). Human-vs-AI matches still default to
   cubeless (no UI for human take/drop); cube fires only in headless
   AI-vs-AI mode.
2. **Bear-off DB** — **DONE.** Phase I. One-sided race table at
   `public/weights/bearoff.{bin,json}`, wired via `src/ai/bearoff.ts`
   shortcut in `NeuralEvaluator`. Exact race endgame for any position
   with both sides home-bound. See Phase I section.
3. **TD self-play on a bigger net.** The remaining static-eval-side
   lever. Phase A's edge over distillation is policy-coherent
   smoothness from TD, not capacity. Running TD on a 2×200 net from
   gnubg-distilled warm-start might compose both gains. Infrastructure
   in `parallel_train.py` already supports arbitrary hidden sizes.
   Estimated wall: 5-10h. ROI uncertain — Phase B (TD-trained 2×200)
   already hit ~parity with Phase A.
4. **3-ply mode for tutor only.** Bumps the search side substantially
   (worth ~50-100 Elo per ply). Current `score2ply` in
   `src/ai/search.ts` extends naturally to ply=3 via a `score3ply`
   variant. Wall-clock per hint goes from ~150ms to ~5-10s — needs to
   be tutor-only, not live-hint. Lowest-effort meaningful gain.
5. **Doubling-cube head / learned cube decisions.** Train a small
   model on gnubg's cube actions over a sampled position bank. Could
   replace or augment the Janowski formula. Higher effort; the live-
   cube margin already gives publishable cubeful behavior, so this is
   only worth it if the margin's heuristic calibration starts losing
   to gnubg in real cube-decision parity tests.

## Reproducing the published net

```bash
cd training
# Distill
../.venv/bin/python distill.py --positions 200000 --epochs 30 --hidden 120 \
    --lr 0.2 --out runs/phaseA-distill --seed 1
# TD self-play
../.venv/bin/python parallel_train.py \
    --out runs/phaseA --games 1300000 \
    --hidden 120 --workers 12 --batch 50 \
    --alpha 0.01 --lambda 0.5 --blend 0.1 --broadcast-every 500 \
    --ckpt-every 100000 \
    --resume runs/phaseA-distill/weights-distill.json --seed 1
# Bench + publish (manually pick the strongest checkpoint by 2000-game h2h)
../.venv/bin/python match_nets.py --A runs/phaseA/weights-1100000.json \
    --B ../public/weights/expert.phase0.json --games 2000 --seed 400
../.venv/bin/python publish_if_good.py \
    --weights runs/phaseA/weights-1100000.json --games 1000 --min-winrate 0.55
```

Wall time: ~3 hours total on a Ryzen 7 255.
