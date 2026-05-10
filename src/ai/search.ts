import { Position, mirror } from "../engine/position";
import { Play, applyPlay, generatePlays } from "../engine/moves";
import { checkWin } from "../engine/rules";
import { Evaluator } from "./evaluator";

export interface ScoredPlay {
  play: Play;
  equity: number;
}

// Evaluate a terminal position (after a play). Returns equity in "us" perspective.
// If game is over, +/-1 with magnitude scaled by gammon/backgammon.
function terminalEquityAfterMyPlay(p: Position): number | null {
  // After our play and we haven't yet mirrored. Check win.
  const r = checkWin(p);
  if (!r) return null;
  // r.winner is in absolute terms. p.turn is still us at this point (we haven't mirrored).
  const sign = r.winner === p.turn ? 1 : -1;
  const mag = r.basePoints / 1; // 1 / 2 / 3
  return sign * mag;
}

// 0-ply rank: evaluate each candidate play by applying it, then evaluating the
// resulting position from the opponent's perspective (negated to give us-equity).
export function score0ply(p: Position, plays: Play[], ev: Evaluator): ScoredPlay[] {
  const out: ScoredPlay[] = [];
  for (const play of plays) {
    const after = applyPlay(p, play);
    const term = terminalEquityAfterMyPlay(after);
    let eq: number;
    if (term !== null) {
      eq = term;
    } else {
      const oppView = mirror(after);
      eq = -ev.evaluate(oppView);
    }
    out.push({ play, equity: eq });
  }
  return out;
}

const ALL_ROLLS: ReadonlyArray<readonly [number, number, number]> = (() => {
  const rolls: [number, number, number][] = [];
  for (let i = 1; i <= 6; i++) {
    for (let j = i; j <= 6; j++) {
      // Probability: doubles = 1/36, non-doubles = 2/36 (since (i,j) and (j,i) both valid)
      const prob = i === j ? 1 / 36 : 2 / 36;
      rolls.push([i, j, prob]);
    }
  }
  return rolls;
})();

// For each candidate play, value = E_over_opp_rolls[ -max_over_opp_plays(eval after opp play) ].
// The expectation is taken over all 21 distinct dice rolls weighted by frequency.
export function score2ply(p: Position, plays: Play[], ev: Evaluator): ScoredPlay[] {
  const out: ScoredPlay[] = [];
  for (const play of plays) {
    const after = applyPlay(p, play);
    const term = terminalEquityAfterMyPlay(after);
    if (term !== null) {
      out.push({ play, equity: term });
      continue;
    }
    const oppView = mirror(after);
    let total = 0;
    for (const [d1, d2, prob] of ALL_ROLLS) {
      const oppPlays = generatePlays(oppView, d1, d2);
      // Opponent picks play that maximizes their equity (= bad for us)
      let bestOpp = -Infinity;
      for (const oplay of oppPlays) {
        const oppAfter = applyPlay(oppView, oplay);
        const oTerm = terminalEquityAfterMyPlay(oppAfter);
        let oeq: number;
        if (oTerm !== null) {
          oeq = oTerm;
        } else {
          const usAgain = mirror(oppAfter);
          oeq = -ev.evaluate(usAgain);
        }
        if (oeq > bestOpp) bestOpp = oeq;
      }
      total += prob * -bestOpp;
    }
    out.push({ play, equity: total });
  }
  return out;
}

// Rank plays at the requested ply. With noise, perturbs the equities and
// optionally returns top-K candidates as if the AI hesitated.
export function rankPlays(
  p: Position,
  plays: Play[],
  ev: Evaluator,
  plies: 0 | 2,
): ScoredPlay[] {
  if (plays.length === 0 || (plays.length === 1 && plays[0].length === 0)) {
    return [{ play: plays[0] ?? [], equity: 0 }];
  }
  const scored = plies === 2 ? score2ply(p, plays, ev) : score0ply(p, plays, ev);
  scored.sort((a, b) => b.equity - a.equity);
  return scored;
}

// Pick a play with optional noise: perturb each equity by Gaussian noise and pick max.
// Higher `noise` simulates a weaker opponent who occasionally chooses sub-optimally.
export function pickWithNoise(scored: ScoredPlay[], noise: number, rng = Math.random): Play {
  if (scored.length === 1) return scored[0].play;
  if (noise <= 0) return scored[0].play;
  let bestI = 0;
  let bestV = -Infinity;
  for (let i = 0; i < scored.length; i++) {
    const n = noise * gauss(rng);
    const v = scored[i].equity + n;
    if (v > bestV) {
      bestV = v;
      bestI = i;
    }
  }
  return scored[bestI].play;
}

function gauss(rng: () => number): number {
  // Box–Muller
  const u1 = Math.max(rng(), 1e-9);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
