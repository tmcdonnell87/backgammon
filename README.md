# Backgammon

Offline-capable web app for backgammon. Play vs. computer at multiple
difficulty levels, pass-and-play with a friend, and turn on tutor mode to rate
your moves against the engine's best play.

## Run

```bash
pnpm install
pnpm dev      # dev server on http://localhost:5173
pnpm test     # vitest
pnpm build    # production build into dist/
pnpm preview  # serve the built app
```

Self-play sanity check (runs the AI against itself):

```bash
pnpm selfplay expert beginner 20
```

## Status

Implemented:

- Rules engine (move generation, hit/bear-off, gammon/backgammon detection),
  with tests
- SVG board with click-to-move, dice, pass-and-play overlay, undo
- Computer opponent at four difficulty levels (Beginner / Casual / Strong /
  Expert) backed by a hand-tuned positional evaluator and 2-ply expectimax
  search at the top tier
- Tutor mode: per-move equity loss vs. the engine's best play, plus a
  post-game performance-rating summary
- PWA shell: `manifest.webmanifest` + cache-first service worker, installable

Deferred (planned for follow-up):

- Doubling cube UI and cube-decision AI (Phase 7)

## Training the neural-net evaluator

The shipped `public/weights/expert.json` was trained by TD(λ) self-play with
heuristic distillation as a warm start. Reproducing the run:

```bash
python3 -m pip install numpy
cd training

# 1. Distill the heuristic into the net (~3 min): collects 200k positions
#    via heuristic self-play and regresses against heuristic equity.
python3 distill.py --out runs/distill --positions 200000 --epochs 30 \
                   --lr 0.2 --hidden 120

# 2. Resume TD-lambda from the distilled checkpoint (~1.5 hours per 100k
#    games on CPU). SIGTERM cleanly writes a final checkpoint.
mkdir -p runs/expert
cp runs/distill/ckpt-distill.npz runs/expert/ckpt-init.npz
python3 train.py --out runs/expert --games 1000000 --ckpt-every 5000 \
                 --hidden 120 --alpha 0.02 --lambda 0.5 \
                 --resume runs/expert/ckpt-init.npz

# 3. Bench candidates and head-to-head to decide what to ship:
python3 bench.py --weights runs/expert/weights-N.json --games 800
python3 match_nets.py --A runs/expert/weights-N.json \
                      --B runs/expert/weights-PUBLISHED.json --games 1000

# 4. Publish to the web app:
cp runs/expert/weights-N.json ../public/weights/expert.json
```

Hyperparameters that mattered: starting from scratch with `alpha=0.1` (the
TD-Gammon default-ish) diverges around 5-10k games — weight norms blow up and
the policy regresses to near-random. `alpha=0.03 lambda=0.5` (effective
per-trace step `alpha/(1-lambda) = 0.06`) trains stably. The heuristic
warm-start cuts the time-to-parity from "many hours" to about 15k self-play
games. The shipped checkpoint reaches ~75% vs heuristic over 800 games at 0
ply; gains plateau there with the current architecture.

Encoding (198 inputs: 4 features per point per side + bar/off/turn) and the
forward pass are mirrored exactly between Python (`training/encoding.py`,
`training/net.py`) and TypeScript (`src/ai/neural.ts`); a parity test in
`test/neural.test.ts` pins them.

## Layout

```
src/
  engine/   board model, move generation, rules, cube state
  ai/       evaluator, search (0-ply / 2-ply), worker, levels
  game/     turn-loop controller, persistence
  ui/       SVG board, layout math, menu, tutor panel
  main.ts   bootstrap
test/       vitest specs for engine + AI
scripts/
  selfplay.ts   AI-vs-AI sanity runs
public/
  sw.js                  service worker
  manifest.webmanifest   PWA manifest
  icon.svg
```

The AI runs in a Web Worker (`src/ai/worker.ts`); the main thread stays
responsive while the engine searches. The worker is the single AI entry
point for both the computer opponent and the tutor.
