import { Position, mirror, hashPosition } from "../engine/position";
import { Play, applyPlay, generatePlays } from "../engine/moves";
import { checkWin } from "../engine/rules";
import { Evaluator } from "./evaluator";

export interface ScoredPlay {
  play: Play;
  equity: number;
}

// Deduplicate plays by final position. Equivalent orderings (same final state,
// different sub-move sequence) collapse to one representative, since the AI
// scores positions, not orderings. The UI keeps the un-deduped list so the
// player can still play either order.
function dedupePlaysByFinal(p: Position, plays: Play[]): Play[] {
  const seen = new Map<string, Play>();
  for (const play of plays) {
    const after = applyPlay(p, play);
    const h = hashPosition(after);
    if (!seen.has(h)) seen.set(h, play);
  }
  return [...seen.values()];
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

export const ALL_ROLLS: ReadonlyArray<readonly [number, number, number]> = (() => {
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
      // Opp's legalPlays is also un-deduped (UI semantics). Dedupe by final
      // here — the inner search only cares about reachable positions.
      const oppPlays = dedupePlaysByFinal(oppView, generatePlays(oppView, d1, d2));
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
//
// We dedupe by final position before scoring — equivalent orderings cost the
// AI nothing extra. Returns one ScoredPlay per unique final.
export function rankPlays(
  p: Position,
  plays: Play[],
  ev: Evaluator,
  plies: 0 | 2,
): ScoredPlay[] {
  if (plays.length === 0 || (plays.length === 1 && plays[0].length === 0)) {
    return [{ play: plays[0] ?? [], equity: 0 }];
  }
  const unique = dedupePlaysByFinal(p, plays);
  const scored = plies === 2 ? score2ply(p, unique, ev) : score0ply(p, unique, ev);
  scored.sort((a, b) => b.equity - a.equity);
  return scored;
}

// ---------------------------------------------------------------------------
// Filtered deep search (tutor / hint tier).
//
// score2ply scores EVERY legal play with a full 21-roll opponent expansion and
// no filtering — fine at 2-ply, but a 3rd ply over every play is infeasible.
// rankPlaysDeep instead (1) prefilters candidates with a cheap 0-ply pass,
// (2) deep-searches only the survivors, and (3) bounds the branching at each
// interior node to the top `innerKeep` replies. A per-call memo cache keyed by
// board hash collapses the heavily-overlapping leaf positions.
//
// Depth mapping (matches the existing convention): a candidate play is scored
// as `-positionValue(mirror(after), plies-1)`. So plies=1 reproduces score0ply
// and plies=2 reproduces score2ply exactly (with no inner filtering).
// ---------------------------------------------------------------------------

export interface SearchOpts {
  plies: 1 | 2 | 3;
  keepTop?: number; // root survivors that get the deep search (default 10)
  keepWindow?: number; // also keep root plays within this equity of the best (default 0.08)
  innerKeep?: number; // top-N replies expanded at each interior node (default 8)
  cache?: Map<string, number>; // memo: hashPosition -> static equity
}

function cachedEval(q: Position, ev: Evaluator, cache: Map<string, number>): number {
  // ev.evaluate is a pure function of the board (dice/cube/score ignored), so
  // memoizing by hashPosition is exact. It also applies the bearoff shortcut.
  const h = hashPosition(q);
  const c = cache.get(h);
  if (c !== undefined) return c;
  const v = ev.evaluate(q);
  cache.set(h, v);
  return v;
}

// 0-ply value of each play from `q`'s perspective (q to move): apply, then
// negate the opponent-POV static eval. Used as the interior-node prefilter.
function zeroPlyScored(
  q: Position,
  plays: Play[],
  ev: Evaluator,
  cache: Map<string, number>,
): { play: Play; eq: number }[] {
  const out: { play: Play; eq: number }[] = [];
  for (const play of plays) {
    const after = applyPlay(q, play);
    const term = terminalEquityAfterMyPlay(after);
    const eq = term !== null ? term : -cachedEval(mirror(after), ev, cache);
    out.push({ play, eq });
  }
  return out;
}

// Value of position `q` (q is on roll, about to roll dice) from q's own
// perspective, looking `depth` rolls ahead. depth=0 is the static leaf eval.
function positionValue(
  q: Position,
  ev: Evaluator,
  depth: number,
  innerKeep: number,
  cache: Map<string, number>,
): number {
  if (depth <= 0) return cachedEval(q, ev, cache);
  let total = 0;
  for (const [d1, d2, prob] of ALL_ROLLS) {
    const plays = dedupePlaysByFinal(q, generatePlays(q, d1, d2));
    // Filter to the top `innerKeep` replies by a cheap 0-ply pass when there
    // are more than that; otherwise keep all (so parity with score2ply holds
    // when innerKeep is large).
    let cand: Play[];
    if (plays.length <= innerKeep) {
      cand = plays;
    } else {
      const z = zeroPlyScored(q, plays, ev, cache);
      z.sort((a, b) => b.eq - a.eq);
      cand = z.slice(0, innerKeep).map((s) => s.play);
    }
    let best = -Infinity;
    for (const pl of cand) {
      const after = applyPlay(q, pl);
      const term = terminalEquityAfterMyPlay(after);
      const v = term !== null ? term : -positionValue(mirror(after), ev, depth - 1, innerKeep, cache);
      if (v > best) best = v;
    }
    total += prob * best;
  }
  return total;
}

// Rank plays with a filtered deep search. Returns one ScoredPlay per unique
// final, covering every input play's final (non-survivors keep their 0-ply
// prefilter equity — a play the prefilter drops is already a clear error, so a
// cheap grade for it is acceptable).
export function rankPlaysDeep(
  p: Position,
  plays: Play[],
  ev: Evaluator,
  opts: SearchOpts,
): ScoredPlay[] {
  if (plays.length === 0 || (plays.length === 1 && plays[0].length === 0)) {
    return [{ play: plays[0] ?? [], equity: 0 }];
  }
  const unique = dedupePlaysByFinal(p, plays);
  const depthAfter = opts.plies - 1;
  const cache = opts.cache ?? new Map<string, number>();

  // 0-ply: just the static pass.
  if (depthAfter <= 0) {
    const scored = score0ply(p, unique, ev);
    scored.sort((a, b) => b.equity - a.equity);
    return scored;
  }

  // Prefilter by 0-ply, then split into survivors (deep-searched) and the rest.
  const pre = score0ply(p, unique, ev);
  pre.sort((a, b) => b.equity - a.equity);
  const keepTop = opts.keepTop ?? 10;
  const keepWindow = opts.keepWindow ?? 0.08;
  const innerKeep = opts.innerKeep ?? 8;
  const best0 = pre.length ? pre[0].equity : -Infinity;

  const out: ScoredPlay[] = [];
  for (let i = 0; i < pre.length; i++) {
    const sp = pre[i];
    if (i < keepTop || sp.equity >= best0 - keepWindow) {
      const after = applyPlay(p, sp.play);
      const term = terminalEquityAfterMyPlay(after);
      const eq = term !== null ? term : -positionValue(mirror(after), ev, depthAfter, innerKeep, cache);
      out.push({ play: sp.play, equity: eq });
    } else {
      out.push(sp); // keep the cheap prefilter equity
    }
  }
  out.sort((a, b) => b.equity - a.equity);
  return out;
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
