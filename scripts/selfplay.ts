// Quick self-play sanity check: play one full game between two AI levels.
// Run with: pnpm selfplay [level1] [level2] [count]
//
// Verifies the engine runs cleanly to completion (no infinite loops, no
// invalid positions, checker invariant holds).

import { mirror, startingPosition, CHECKERS_PER_SIDE } from "../src/engine/position";
import { generatePlays, applyPlay } from "../src/engine/moves";
import { checkWin, rollDice, rollOpening } from "../src/engine/rules";
import { pickMove } from "../src/ai/engine";
import { Difficulty } from "../src/ai/levels";

function checkerInvariant(p: ReturnType<typeof startingPosition>): boolean {
  let us = p.barUs + p.offUs;
  let them = p.barThem + p.offThem;
  for (let i = 0; i < p.points.length; i++) {
    if (p.points[i] > 0) us += p.points[i];
    else if (p.points[i] < 0) them += -p.points[i];
  }
  return us === CHECKERS_PER_SIDE && them === CHECKERS_PER_SIDE;
}

function playOne(white: Difficulty, black: Difficulty): { winner: 0 | 1; turns: number } {
  let pos = startingPosition();
  // Opening
  let [d1, d2] = rollOpening();
  let firstSide: 0 | 1 = d1 > d2 ? 0 : 1;
  if (firstSide === 1) pos = mirror(pos);
  pos.dice = [d1, d2];
  pos.turn = firstSide;

  let turns = 0;
  while (turns < 1000) {
    if (!checkerInvariant(pos)) {
      throw new Error(`Checker invariant violated after turn ${turns}`);
    }
    const dice = pos.dice;
    if (!dice) throw new Error("No dice on roll");
    const [a, b] = dice;
    const plays = generatePlays(pos, a, b);
    const onRoll: 0 | 1 = pos.turn as 0 | 1;
    const lvl = onRoll === 0 ? white : black;
    const play = plays.length === 0 ? [] : pickMove(pos, plays, lvl);
    pos = applyPlay(pos, play);
    pos.dice = null;
    const r = checkWin(pos);
    if (r) {
      return { winner: r.winner as 0 | 1, turns: turns + 1 };
    }
    pos = mirror(pos);
    [d1, d2] = rollDice();
    pos.dice = [d1, d2];
    turns++;
  }
  throw new Error(`Game did not finish in 1000 turns`);
}

const argLevels = (process.argv.slice(2).filter((a) => !/^\d+$/.test(a)) as Difficulty[]);
const argCount = parseInt(process.argv.slice(2).find((a) => /^\d+$/.test(a)) ?? "10", 10);

const a: Difficulty = argLevels[0] ?? "casual";
const b: Difficulty = argLevels[1] ?? "casual";

let aWins = 0;
let totalTurns = 0;
const t0 = performance.now();
for (let i = 0; i < argCount; i++) {
  const r = playOne(a, b);
  if (r.winner === 0) aWins++;
  totalTurns += r.turns;
}
const dt = (performance.now() - t0) / 1000;
console.log(`${a} vs ${b}: ${aWins}/${argCount} (${((aWins / argCount) * 100).toFixed(0)}%)`);
console.log(`Avg turns/game: ${(totalTurns / argCount).toFixed(1)}, total time: ${dt.toFixed(1)}s`);
