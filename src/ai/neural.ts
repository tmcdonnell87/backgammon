// Neural evaluator: loads JSON weights produced by training/{train,parallel_train}.py
// and runs the same forward pass.
//
// Three weight schemas are supported (loader auto-detects):
//   * Legacy 1-output (training/net.py, numpy): single sigmoid P(we win) in [0,1],
//     mapped to equity = 2y - 1. Format: { input, hidden:int, output:1, W1, b1, W2, b2 }
//     where W2 is a flat hidden-length array and b2 is a scalar.
//   * Phase A 4-output, 1 hidden layer (training/net_torch.py, old): four sigmoid
//     heads. Format: { input, hidden:int, output:4, W1, b1, W2:[4][hidden], b2:[4] }.
//   * Phase B+ 4-output, arbitrary depth (training/net_torch.py): {
//       input, hidden:[h1,h2,...], output:4, W:[matrices], b:[vectors] }
//     where the last entry in W,b is the output layer with shape (4, hLast)/(4,).
// Internally all three are normalized into a uniform per-layer representation.

import { POINTS, Position } from "../engine/position";
import { Evaluator } from "./evaluator";
import { stillInContact } from "./heuristic";
import { bearoffEquity } from "./bearoff";

export const NEURAL_INPUT_SIZE = 198;

// Legacy 1-output (numpy training/net.py).
export interface NeuralWeights1 {
  input: number;
  hidden: number;
  output: 1;
  W1: number[][]; // [hidden][input]
  b1: number[];   // [hidden]
  W2: number[];   // [hidden]
  b2: number;
}

// Phase A: 4-output, 1 hidden layer with W1/b1/W2/b2 naming.
export interface NeuralWeights4Legacy {
  input: number;
  hidden: number;
  output: 4;
  W1: number[][]; // [hidden][input]
  b1: number[];   // [hidden]
  W2: number[][]; // [4][hidden]
  b2: number[];   // [4]
}

// Phase B+: 4-output, arbitrary depth, list-of-layers shape.
export interface NeuralWeightsML {
  input: number;
  hidden: number[];     // e.g. [200, 200]
  output: 4;
  W: number[][][];      // each entry: [outDim][inDim]
  b: number[][];        // each entry: [outDim]
}

export type NeuralWeights = NeuralWeights1 | NeuralWeights4Legacy | NeuralWeightsML;

// Per-outcome probabilities for the 4-output net.
export interface OutcomeProbs {
  pWin: number;
  pGammonWin: number;
  pLoss: number;
  pGammonLoss: number;
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

// Normalize any supported weight schema into a uniform list of layers.
// Returns hiddenLayers (sizes of hidden layers), Wlayers (per-layer weight rows),
// blayers (per-layer biases), and outputDim.
function normalizeWeights(w: NeuralWeights): {
  hiddenLayers: number[];
  Wlayers: Float32Array[][];
  blayers: Float32Array[];
  outputDim: 1 | 4;
} {
  if (w.input !== NEURAL_INPUT_SIZE) {
    throw new Error(`expected input ${NEURAL_INPUT_SIZE}, got ${w.input}`);
  }
  const out = (w as { output: number }).output;
  if (out !== 1 && out !== 4) {
    throw new Error(`expected output 1 or 4, got ${out}`);
  }

  // Multi-layer ML format (Phase B+): hidden is an array.
  if (Array.isArray((w as NeuralWeightsML).hidden)) {
    const ml = w as NeuralWeightsML;
    const hidden = ml.hidden;
    if (ml.W.length !== hidden.length + 1 || ml.b.length !== hidden.length + 1) {
      throw new Error(
        `multi-layer: W/b length ${ml.W.length}/${ml.b.length} mismatches `
        + `expected ${hidden.length + 1}`,
      );
    }
    const Wlayers = ml.W.map((mat) => mat.map((row) => Float32Array.from(row)));
    const blayers = ml.b.map((v) => Float32Array.from(v));
    return { hiddenLayers: hidden, Wlayers, blayers, outputDim: out };
  }

  // Legacy single-hidden-layer formats (either output=1 or output=4 with W1/b1/W2/b2).
  const hidden = (w as { hidden: number }).hidden;
  const W1 = (w as NeuralWeights1 | NeuralWeights4Legacy).W1
    .map((row) => Float32Array.from(row));
  const b1 = Float32Array.from((w as NeuralWeights1 | NeuralWeights4Legacy).b1);
  if (out === 1) {
    const w1 = w as NeuralWeights1;
    const W2Rows = [Float32Array.from(w1.W2)];          // [[w0,...,w_{H-1}]]
    const b2 = Float32Array.from([w1.b2]);              // [scalar]
    return {
      hiddenLayers: [hidden],
      Wlayers: [W1, W2Rows],
      blayers: [b1, b2],
      outputDim: 1,
    };
  }
  // output === 4 legacy
  const w4 = w as NeuralWeights4Legacy;
  const W2Rows = w4.W2.map((row) => Float32Array.from(row));
  const b2 = Float32Array.from(w4.b2);
  return {
    hiddenLayers: [hidden],
    Wlayers: [W1, W2Rows],
    blayers: [b1, b2],
    outputDim: 4,
  };
}

export class NeuralEvaluator implements Evaluator {
  readonly name = "neural";
  readonly outputDim: 1 | 4;
  readonly hiddenLayers: number[];
  private Wlayers: Float32Array[][]; // layer i: rows of length size_{i-1}
  private blayers: Float32Array[];
  private xBuf = new Float32Array(NEURAL_INPUT_SIZE);
  // Reusable hidden buffers, one per hidden layer.
  private hBufs: Float32Array[];

  constructor(w: NeuralWeights) {
    const { hiddenLayers, Wlayers, blayers, outputDim } = normalizeWeights(w);
    this.outputDim = outputDim;
    this.hiddenLayers = hiddenLayers;
    this.Wlayers = Wlayers;
    this.blayers = blayers;
    this.hBufs = hiddenLayers.map((h) => new Float32Array(h));
  }

  // Returns the post-tanh activation of the last hidden layer.
  private forwardThroughHidden(p: Position): Float32Array {
    const x = encodePosition(p, this.xBuf);
    let cur: Float32Array = x;
    for (let layer = 0; layer < this.hiddenLayers.length; layer++) {
      const W = this.Wlayers[layer];
      const b = this.blayers[layer];
      const h = this.hBufs[layer];
      const inSize = cur.length;
      for (let j = 0; j < h.length; j++) {
        const row = W[j];
        let z = b[j];
        for (let i = 0; i < inSize; i++) z += row[i] * cur[i];
        h[j] = Math.tanh(z);
      }
      cur = h;
    }
    return cur;
  }

  // Returns the raw sigmoid head outputs. Length === outputDim.
  private forwardOutput(p: Position): number[] {
    const h = this.forwardThroughHidden(p);
    const Wout = this.Wlayers[this.Wlayers.length - 1];
    const bout = this.blayers[this.blayers.length - 1];
    const probs: number[] = new Array<number>(this.outputDim);
    for (let k = 0; k < this.outputDim; k++) {
      const row = Wout[k];
      let z = bout[k];
      for (let j = 0; j < h.length; j++) z += row[j] * h[j];
      probs[k] = sigmoid(z);
    }
    return probs;
  }

  // Returns cubeless equity. Range:
  //  * 1-output: [-1, +1]   (= 2y - 1)
  //  * 4-output: [-2, +2]   (= p_w + p_gw - p_l - p_gl)
  evaluate(p: Position): number {
    const bo = bearoffEquity(p);
    if (bo) return bo.pWin + bo.pGammonWin - bo.pLoss - bo.pGammonLoss;
    const probs = this.forwardOutput(p);
    if (this.outputDim === 1) {
      return 2 * probs[0] - 1;
    }
    return probs[0] + probs[1] - probs[2] - probs[3];
  }

  // For UI / tutor display: returns all four outcome probabilities. For a
  // 1-output net, p_win is reconstructed from the scalar and the gammon heads
  // are reported as 0.
  evaluateOutcomes(p: Position): OutcomeProbs {
    const bo = bearoffEquity(p);
    if (bo) return bo;
    const probs = this.forwardOutput(p);
    if (this.outputDim === 1) {
      return { pWin: probs[0], pGammonWin: 0, pLoss: 1 - probs[0], pGammonLoss: 0 };
    }
    return {
      pWin: probs[0], pGammonWin: probs[1], pLoss: probs[2], pGammonLoss: probs[3],
    };
  }
}

// --- Phased (race/contact) evaluator (Phase C) ---------------------------

// Manifest format produced by Phase C: instead of weight tensors, the JSON
// is a tiny dispatch table pointing at the contact and race sub-weights.
// Detected by `version: 2` (or absence of `input`).
export interface PhasedManifest {
  version: 2;
  // Sub-weight URLs, resolved relative to the manifest URL (or absolute).
  contact: string;
  race: string;
}

export class PhasedNeuralEvaluator implements Evaluator {
  readonly name = "neural-phased";
  constructor(
    private contactNet: NeuralEvaluator,
    private raceNet: NeuralEvaluator,
  ) {}
  private pick(p: Position): NeuralEvaluator {
    return stillInContact(p) ? this.contactNet : this.raceNet;
  }
  evaluate(p: Position): number {
    return this.pick(p).evaluate(p);
  }
  evaluateOutcomes(p: Position): OutcomeProbs {
    return this.pick(p).evaluateOutcomes(p);
  }
}

// Net evaluator that may be a single-net (NeuralEvaluator) or a phased pair.
// The interface stays Evaluator-compatible.
export type AnyNeuralEvaluator = NeuralEvaluator | PhasedNeuralEvaluator;

// Module-level cache so we only fetch+parse once.
let cachedEvaluator: AnyNeuralEvaluator | null = null;
let cachedFetch: Promise<AnyNeuralEvaluator | null> | null = null;

function looksLikeManifest(obj: unknown): obj is PhasedManifest {
  if (!obj || typeof obj !== "object") return false;
  const o = obj as Record<string, unknown>;
  return o.version === 2 && typeof o.contact === "string"
      && typeof o.race === "string";
}

function resolveRelativeUrl(base: string, target: string): string {
  // If target is absolute (http(s) or starts with /), use as-is.
  if (/^(https?:)?\//.test(target)) return target;
  const lastSlash = base.lastIndexOf("/");
  return (lastSlash >= 0 ? base.slice(0, lastSlash + 1) : "") + target;
}

export async function loadNeuralEvaluator(url = "/weights/expert.json"):
    Promise<AnyNeuralEvaluator | null> {
  if (cachedEvaluator) return cachedEvaluator;
  if (cachedFetch) return cachedFetch;
  cachedFetch = (async () => {
    try {
      const res = await fetch(url);
      if (!res.ok) return null;
      const obj = await res.json();
      if (looksLikeManifest(obj)) {
        const contactUrl = resolveRelativeUrl(url, obj.contact);
        const raceUrl = resolveRelativeUrl(url, obj.race);
        const [contactRes, raceRes] = await Promise.all([
          fetch(contactUrl),
          fetch(raceUrl),
        ]);
        if (!contactRes.ok || !raceRes.ok) return null;
        const [contactW, raceW] = await Promise.all([
          contactRes.json() as Promise<NeuralWeights>,
          raceRes.json() as Promise<NeuralWeights>,
        ]);
        cachedEvaluator = new PhasedNeuralEvaluator(
          new NeuralEvaluator(contactW),
          new NeuralEvaluator(raceW),
        );
      } else {
        const w = obj as NeuralWeights;
        cachedEvaluator = new NeuralEvaluator(w);
      }
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

// For tests / Node: build a phased evaluator directly from two parsed
// weights objects (skipping the manifest fetch).
export function phasedEvaluatorFromWeights(
  contact: NeuralWeights,
  race: NeuralWeights,
): PhasedNeuralEvaluator {
  return new PhasedNeuralEvaluator(
    new NeuralEvaluator(contact),
    new NeuralEvaluator(race),
  );
}
