// Exact bear-off lookup for pure-race endgame positions.
//
// Backed by a one-sided race table built in training/bearoff_build.py:
//   finish[state][k]    = P(state finishes all 15 off in exactly k rolls)
//   first_off[state][k] = P(state has scored at least one bear-off by roll k)
//
// `bearoffEquity(pos)` returns the exact 4-vec (p_w, p_gw, p_l, p_gl) when
// both sides are in pure home-board race (no bar, no checkers outside the
// home boards). Otherwise returns null and the caller falls through to the
// net.
//
// State convention matches engine: point indices 0..5 are our home, 18..23
// are opponent's home (which is their 1..6 reading the other way).

import { POINTS, Position } from "../engine/position";
import { OutcomeProbs } from "./neural";

interface BearoffMeta {
  version: number;
  max_rolls: number;
  n_states: number;
  home_points: number;
  checkers_per_side: number;
  bin_file: string;
  bin_layout: Array<{
    name: string;
    shape: number[];
    dtype: string;
    scale: number;
  }>;
}

interface BearoffData {
  maxRolls: number;        // K = MAX_ROLLS
  histLen: number;         // K + 1
  finish: Float32Array;    // (n_states * histLen)
  firstOff: Float32Array;  // (n_states * histLen)
  stateIndex: Map<string, number>;
}

let cached: BearoffData | null = null;
let cachedFetch: Promise<BearoffData | null> | null = null;

function stateKey(n: number[]): string {
  return `${n[0]},${n[1]},${n[2]},${n[3]},${n[4]},${n[5]}`;
}

async function loadBearoffCore(jsonUrl: string): Promise<BearoffData | null> {
  const res = await fetch(jsonUrl);
  if (!res.ok) return null;
  const meta = (await res.json()) as BearoffMeta;
  if (meta.version !== 1 || meta.home_points !== 6) return null;
  // Resolve bin file relative to the json URL.
  const lastSlash = jsonUrl.lastIndexOf("/");
  const binUrl = (lastSlash >= 0 ? jsonUrl.slice(0, lastSlash + 1) : "")
    + meta.bin_file;
  const binRes = await fetch(binUrl);
  if (!binRes.ok) return null;
  const buf = await binRes.arrayBuffer();
  const histLen = meta.max_rolls + 1;
  const nStates = meta.n_states;
  // Layout: states (n_states*6 uint8) | finish (n_states*histLen uint16) | first_off (same)
  const statesByteLen = nStates * 6;
  const finishByteLen = nStates * histLen * 2;
  const expectedLen = statesByteLen + 2 * finishByteLen;
  if (buf.byteLength !== expectedLen) {
    console.warn(`bearoff.bin size mismatch: got ${buf.byteLength}, expected ${expectedLen}`);
    return null;
  }
  const statesView = new Uint8Array(buf, 0, statesByteLen);
  const finishView = new Uint16Array(buf, statesByteLen, nStates * histLen);
  const firstOffView = new Uint16Array(buf, statesByteLen + finishByteLen,
    nStates * histLen);
  // Dequantize. /65535 with uint16.
  const scale = 1 / 65535;
  const finish = new Float32Array(nStates * histLen);
  const firstOff = new Float32Array(nStates * histLen);
  for (let i = 0; i < finish.length; i++) {
    finish[i] = finishView[i] * scale;
    firstOff[i] = firstOffView[i] * scale;
  }
  // Build state -> index map from the shipped state table.
  const stateIndex = new Map<string, number>();
  for (let i = 0; i < nStates; i++) {
    const base = i * 6;
    const key = `${statesView[base]},${statesView[base + 1]},${statesView[base + 2]},`
      + `${statesView[base + 3]},${statesView[base + 4]},${statesView[base + 5]}`;
    stateIndex.set(key, i);
  }
  return { maxRolls: meta.max_rolls, histLen, finish, firstOff, stateIndex };
}

export async function loadBearoff(jsonUrl = "/weights/bearoff.json"):
    Promise<BearoffData | null> {
  if (cached) return cached;
  if (cachedFetch) return cachedFetch;
  cachedFetch = (async () => {
    try {
      const r = await loadBearoffCore(jsonUrl);
      if (r) cached = r;
      return r;
    } catch (e) {
      console.warn("bearoff load failed:", e);
      return null;
    }
  })();
  return cachedFetch;
}

export function setBearoff(data: BearoffData | null) {
  cached = data;
}

export function getBearoff(): BearoffData | null {
  return cached;
}

// Returns true if every checker is in a home board and there's nothing on
// the bar. (Includes positions where one or both sides have borne off some
// checkers — that's still "pure race no contact".)
function isPureHomeRace(p: Position): boolean {
  if (p.barUs !== 0 || p.barThem !== 0) return false;
  // Our checkers must be in 0..5; opponent's in 18..23.
  for (let i = 6; i < 18; i++) {
    if (p.points[i] !== 0) return false;
  }
  for (let i = 0; i < 6; i++) {
    if (p.points[i] < 0) return false;
  }
  for (let i = 18; i < POINTS; i++) {
    if (p.points[i] > 0) return false;
  }
  return true;
}

function ourState(p: Position): number[] {
  // (n_0..n_5) for the player on roll.
  return [
    p.points[0], p.points[1], p.points[2],
    p.points[3], p.points[4], p.points[5],
  ];
}

function theirState(p: Position): number[] {
  // opp's home is points[18..23]; their 1-pt is our 24-pt = points[23],
  // their 6-pt = points[18]. So n_i = -points[23-i].
  return [
    -p.points[23], -p.points[22], -p.points[21],
    -p.points[20], -p.points[19], -p.points[18],
  ];
}

// Returns exact 4-vec for the player on roll, or null when ineligible.
export function bearoffEquity(p: Position): OutcomeProbs | null {
  if (!cached) return null;
  if (!isPureHomeRace(p)) return null;
  const us = ourState(p);
  const them = theirState(p);
  // If either side has already won, return terminal. (Shouldn't normally
  // happen — the engine ends the game before evaluation — but be defensive.)
  let usSum = 0, themSum = 0;
  for (let i = 0; i < 6; i++) { usSum += us[i]; themSum += them[i]; }
  // Sanity: sums must be in [0..15].
  if (usSum > 15 || themSum > 15 || usSum < 0 || themSum < 0) return null;
  if (usSum === 0) {
    // We've already borne off all 15. Already won.
    return { pWin: 1, pGammonWin: themSum === 15 ? 1 : 0, pLoss: 0, pGammonLoss: 0 };
  }
  if (themSum === 0) {
    return { pWin: 0, pGammonWin: 0, pLoss: 1, pGammonLoss: usSum === 15 ? 1 : 0 };
  }
  const usIdx = cached.stateIndex.get(stateKey(us));
  const themIdx = cached.stateIndex.get(stateKey(them));
  if (usIdx === undefined || themIdx === undefined) {
    // Out-of-table state (>15 checkers? bug). Defensively fail.
    return null;
  }
  const K = cached.histLen;
  const usFinish = cached.finish.subarray(usIdx * K, usIdx * K + K);
  const themFinish = cached.finish.subarray(themIdx * K, themIdx * K + K);
  const usFirstOff = cached.firstOff.subarray(usIdx * K, usIdx * K + K);
  const themFirstOff = cached.firstOff.subarray(themIdx * K, themIdx * K + K);

  // Side on roll: our k-th finishing roll happens at game-roll 2k-1; their
  // j-th at 2j. We win iff k <= j. After our finish at our k-th roll, they
  // have had k-1 their-rolls; after their finish at their j-th roll, we
  // have had j our-rolls (since their roll 2j > our roll 2j-1).
  //
  //   P(we win)     = sum_k p_us[k] * sum_{j>=k} p_them[j]
  //   P(we gammon)  = sum_k p_us[k] * (1 - first_off_them[k-1])
  //   P(they win)   = 1 - P(we win)
  //   P(they gammon) = sum_j p_them[j] * (1 - first_off_us[j])

  // Build P(them finish in j >= k) cumulative suffix-sum.
  // (Iterate from end so suffix sum is easy.)
  const themSuffix = new Float64Array(K + 1);
  themSuffix[K] = 0;
  for (let j = K - 1; j >= 0; j--) {
    themSuffix[j] = themSuffix[j + 1] + themFinish[j];
  }

  let pWin = 0;
  let pGammonWin = 0;
  for (let k = 0; k < K; k++) {
    const pk = usFinish[k];
    if (pk === 0) continue;
    pWin += pk * themSuffix[k];
    // first_off_them[k-1]: 0 when k=0 (no rolls => no offs), else lookup.
    const themFirstByK1 = k === 0 ? 0 : themFirstOff[k - 1];
    pGammonWin += pk * (1 - themFirstByK1);
  }

  let pGammonLoss = 0;
  for (let j = 0; j < K; j++) {
    const pj = themFinish[j];
    if (pj === 0) continue;
    // After their j-th finishing roll, we've had j our-rolls.
    const usFirstByJ = j === 0 ? 0 : usFirstOff[j];  // index by j our-rolls
    pGammonLoss += pj * (1 - usFirstByJ);
  }
  const pLoss = 1 - pWin;

  // Clamp tiny FP overshoots.
  return {
    pWin: clamp01(pWin),
    pGammonWin: clamp01(pGammonWin),
    pLoss: clamp01(pLoss),
    pGammonLoss: clamp01(pGammonLoss),
  };
}

function clamp01(x: number): number {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}
