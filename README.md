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

The Expert tier loads weights from `public/weights/expert.json` if present and
falls back to the heuristic otherwise. The weights are produced by a TD(λ)
self-play trainer (TD-Gammon style) under `training/`:

```bash
python3 -m pip install numpy
cd training
python3 train.py --out runs/expert --games 1000000 --ckpt-every 5000 --hidden 80
# checkpoints land in runs/expert/{ckpt-N.npz, weights-N.json, weights-latest.json}

# publish the latest checkpoint to the web app:
./publish.sh
```

`train.py` writes a CSV log (`log.csv`) and prints progress every 50 games.
Average plies per game drops sharply as the net learns to bear off and avoid
dancing, which is a useful coarse signal. Sustained throughput on a CPU is
~20 games/s at hidden=80; an overnight run yields ~1M games. Training is
resumable (`--resume runs/expert/ckpt-final.npz`), and `SIGTERM` triggers a
clean final checkpoint.

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
