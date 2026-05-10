import { POINTS, Position, mirror, pipCount, pipCountThem } from "../engine/position";
import { Evaluator, Equity } from "./evaluator";

// Hand-tuned positional evaluator. Returns equity in roughly [-1, 1].
//
// Implementation strategy: every feature is computed for "us" only, and the
// final equity is `sided(p) - sided(mirror(p))`. That guarantees the function
// is antisymmetric (eq(p) = -eq(mirror(p))) and fairer to compare across
// candidate plays.

const W_PIP = 0.0035;
const W_OFF = 0.07;
const W_BAR = 0.18;
const W_POINT_MID = 0.018;
const W_POINT_GOLDEN = 0.045; // our 5-pt
const W_POINT_BAR = 0.040; // our 7-pt (bar pt)
const W_POINT_HOME = 0.030;
const W_HOME_BUILDERS = 0.008;
const W_BLOT_BASE = 0.012;
const W_OUR_HOME_BLOT = 0.018;
const W_BLOT_SAFE = 0.004;
const W_OPP_HOME_ANCHOR = 0.06;
const W_PRIME_PER = 0.08;
const W_FULL_PRIME = 0.18;
const W_CLOSEOUT = 0.30;
const W_BACK_LONE = 0.025;
const W_HIT_OPP_BLOT = 0.05;

export function evaluateHeuristic(p: Position): Equity {
  if (!stillInContact(p)) {
    // Pure race: only pip difference and borne-off matter
    const pipDiff = pipCountThem(p) - pipCount(p);
    return Math.tanh(0.045 * pipDiff + 0.6 * (p.offUs - p.offThem) / 15);
  }
  return Math.tanh(sidedScore(p) - sidedScore(mirror(p)));
}

// One-sided score: features for "us" only. The full evaluation differences this
// against the mirrored position to ensure antisymmetry.
function sidedScore(p: Position): number {
  let s = 0;
  s += -W_PIP * pipCount(p);
  s += W_OFF * p.offUs;
  s += -W_BAR * p.barUs;

  for (let i = 0; i < POINTS; i++) {
    const v = p.points[i];
    if (v >= 2) {
      let w = W_POINT_MID;
      if (i <= 5) w = W_POINT_HOME;
      if (i === 4) w = W_POINT_GOLDEN;
      if (i === 6) w = W_POINT_BAR;
      s += w;
      if (v > 2) s += W_HOME_BUILDERS * (v - 2);
    } else if (v === 1) {
      const shotProb = blotShotProbability(p, i);
      let w = W_BLOT_BASE;
      if (i <= 5) w = W_OUR_HOME_BLOT;
      if (shotProb < 0.05) w = W_BLOT_SAFE;
      s -= w * (1 + shotProb * 6);
    }
  }

  // Anchors in opponent's home (idx 18..23): valuable for back game / containment
  for (let i = 18; i < POINTS; i++) {
    if (p.points[i] >= 2) s += W_OPP_HOME_ANCHOR;
  }

  // Prime: longest run of consecutive made points in idx 0..7
  let primeRun = 0;
  let bestPrime = 0;
  for (let i = 0; i <= 7; i++) {
    if (p.points[i] >= 2) {
      primeRun++;
      if (primeRun > bestPrime) bestPrime = primeRun;
    } else {
      primeRun = 0;
    }
  }
  if (bestPrime >= 2) s += W_PRIME_PER * (bestPrime - 1);
  if (bestPrime >= 6) s += W_FULL_PRIME;

  // Closeout: full home + opp on bar
  let closed = true;
  for (let i = 0; i <= 5; i++) {
    if (p.points[i] < 2) {
      closed = false;
      break;
    }
  }
  if (closed && p.barThem > 0) s += W_CLOSEOUT;

  // Single back checker (no anchor) is exposed
  for (let i = 18; i < POINTS; i++) {
    if (p.points[i] === 1) s -= W_BACK_LONE;
  }

  // Hits we have available against opp blots
  for (let i = 0; i < POINTS; i++) {
    if (p.points[i] !== -1) continue;
    let canHit = false;
    if (p.barUs > 0) {
      const enterDie = 24 - i;
      if (enterDie >= 1 && enterDie <= 6) canHit = true;
    } else {
      for (let d = 1; d <= 6; d++) {
        const src = i + d;
        if (src >= 0 && src < POINTS && p.points[src] >= 1) {
          canHit = true;
          break;
        }
      }
    }
    if (canHit) s += W_HIT_OPP_BLOT;
  }

  return s;
}

function stillInContact(p: Position): boolean {
  if (p.barUs > 0 || p.barThem > 0) return true;
  let oursMin = POINTS;
  let oppsMax = -1;
  for (let i = 0; i < POINTS; i++) {
    if (p.points[i] > 0 && i < oursMin) oursMin = i;
    if (p.points[i] < 0 && i > oppsMax) oppsMax = i;
  }
  if (oursMin === POINTS || oppsMax === -1) return false;
  return oursMin <= oppsMax;
}

function blotShotProbability(p: Position, i: number): number {
  const distances = new Set<number>();
  if (p.barThem > 0) {
    if (i >= 0 && i <= 5) distances.add(i + 1);
  } else {
    for (let j = 0; j < POINTS; j++) {
      if (p.points[j] >= 0) continue;
      const d = i - j;
      if (d > 0 && d <= 24) distances.add(d);
    }
  }
  return shotsToProb(distances);
}

function shotsToProb(needed: Set<number>): number {
  if (needed.size === 0) return 0;
  let count = 0;
  for (let i = 1; i <= 6; i++) {
    for (let j = 1; j <= 6; j++) {
      const sums = new Set<number>([i, j]);
      if (i === j) {
        sums.add(2 * i);
        sums.add(3 * i);
        sums.add(4 * i);
      } else {
        sums.add(i + j);
      }
      let hit = false;
      for (const n of needed) {
        if (sums.has(n)) {
          hit = true;
          break;
        }
      }
      if (hit) count++;
    }
  }
  return count / 36;
}

export const heuristicEvaluator: Evaluator = {
  name: "heuristic",
  evaluate(p) {
    return evaluateHeuristic(p);
  },
};
