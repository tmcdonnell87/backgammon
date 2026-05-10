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
- Trained neural-net evaluator for "Expert" tier (Phase 5; ships behind the
  current heuristic until weights are trained)

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
