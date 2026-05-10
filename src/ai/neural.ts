// Neural evaluator: loads JSON weights produced by training/train.py and runs
// the same 198-input -> tanh hidden -> sigmoid forward pass.
//
// Output of the net is P(we win) in [0, 1]. We map it to equity in [-1, 1] as
// 2 * y - 1 so the rest of the search code can treat it uniformly.

import { POINTS, Position } from "../engine/position";
import { Evaluator } from "./evaluator";

export const NEURAL_INPUT_SIZE = 198;

export interface NeuralWeights {
  input: number;
  hidden: number;
  output: number; // 1
  W1: number[][]; // [hidden][input]
  b1: number[];   // [hidden]
  W2: number[];   // [hidden]
  b2: number;
}

// Encoding mirrors training/encoding.py exactly. Order matters.
export function encodePosition(p: Position, out?: Float32Array): Float32Array {
  const x = out ?? new Float32Array(NEURAL_INPUT_SIZE);
  x.fill(0);
  // Us: 4 features per point at [i*4 .. i*4+3]
  for (let i = 0; i < POINTS; i++) {
    const n = Math.max(p.points[i], 0);
    const base = i * 4;
    if (n >= 1) x[base] = 1;
    if (n >= 2) x[base + 1] = 1;
    if (n >= 3) x[base + 2] = 1;
    if (n >= 4) x[base + 3] = (n - 3) / 2;
  }
  // Them: at index POINTS-1-i (so they encode in their forward direction)
  for (let i = 0; i < POINTS; i++) {
    const n = Math.max(-p.points[POINTS - 1 - i], 0);
    const base = 96 + i * 4;
    if (n >= 1) x[base] = 1;
    if (n >= 2) x[base + 1] = 1;
    if (n >= 3) x[base + 2] = 1;
    if (n >= 4) x[base + 3] = (n - 3) / 2;
  }
  x[192] = p.barUs / 2;
  x[193] = p.barThem / 2;
  x[194] = p.offUs / 15;
  x[195] = p.offThem / 15;
  x[196] = 1;
  x[197] = 0;
  return x;
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

export class NeuralEvaluator implements Evaluator {
  readonly name = "neural";
  private W1: Float32Array[]; // hidden rows of input
  private b1: Float32Array;
  private W2: Float32Array;
  private b2: number;
  private hidden: number;
  private xBuf = new Float32Array(NEURAL_INPUT_SIZE);
  private hBuf: Float32Array;

  constructor(w: NeuralWeights) {
    if (w.input !== NEURAL_INPUT_SIZE) {
      throw new Error(`expected input ${NEURAL_INPUT_SIZE}, got ${w.input}`);
    }
    if (w.output !== 1) {
      throw new Error(`expected output 1, got ${w.output}`);
    }
    this.hidden = w.hidden;
    this.W1 = w.W1.map((row) => Float32Array.from(row));
    this.b1 = Float32Array.from(w.b1);
    this.W2 = Float32Array.from(w.W2);
    this.b2 = w.b2;
    this.hBuf = new Float32Array(this.hidden);
  }

  // Returns equity in [-1, 1].
  evaluate(p: Position): number {
    const x = encodePosition(p, this.xBuf);
    const h = this.hBuf;
    for (let j = 0; j < this.hidden; j++) {
      const row = this.W1[j];
      let z = this.b1[j];
      for (let i = 0; i < NEURAL_INPUT_SIZE; i++) z += row[i] * x[i];
      h[j] = Math.tanh(z);
    }
    let z2 = this.b2;
    for (let j = 0; j < this.hidden; j++) z2 += this.W2[j] * h[j];
    const y = sigmoid(z2);
    return 2 * y - 1;
  }
}

// Module-level cache so we only fetch+parse once.
let cachedEvaluator: NeuralEvaluator | null = null;
let cachedFetch: Promise<NeuralEvaluator | null> | null = null;

export async function loadNeuralEvaluator(url = "/weights/expert.json"):
    Promise<NeuralEvaluator | null> {
  if (cachedEvaluator) return cachedEvaluator;
  if (cachedFetch) return cachedFetch;
  cachedFetch = (async () => {
    try {
      const res = await fetch(url);
      if (!res.ok) return null;
      const w = (await res.json()) as NeuralWeights;
      cachedEvaluator = new NeuralEvaluator(w);
      return cachedEvaluator;
    } catch {
      return null;
    }
  })();
  return cachedFetch;
}

// For tests / Node consumers: build directly from a parsed weights object.
export function neuralEvaluatorFromWeights(w: NeuralWeights): NeuralEvaluator {
  return new NeuralEvaluator(w);
}
