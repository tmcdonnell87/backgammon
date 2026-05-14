import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { POINTS, Position } from "../src/engine/position";
import {
  encodePosition,
  NEURAL_INPUT_SIZE,
  neuralEvaluatorFromWeights,
  phasedEvaluatorFromWeights,
  NeuralWeights,
} from "../src/ai/neural";
import { stillInContact } from "../src/ai/heuristic";

interface Case {
  name: string;
  points: number[];
  bar_us: number;
  bar_them: number;
  off_us: number;
  off_them: number;
  turn: number;
  x: number[];
}

function buildPosition(c: Case): Position {
  const pts = new Int8Array(POINTS);
  for (let i = 0; i < POINTS; i++) pts[i] = c.points[i];
  return {
    points: pts,
    barUs: c.bar_us,
    barThem: c.bar_them,
    offUs: c.off_us,
    offThem: c.off_them,
    turn: c.turn as 0 | 1,
    dice: null,
    cube: { value: 1, owner: null },
    score: [0, 0],
    matchLength: 1,
    crawford: false,
  };
}

describe("neural encoding parity with Python", () => {
  const cases = JSON.parse(
    readFileSync(join(__dirname, "fixtures", "encode_cases.json"), "utf8"),
  ) as Case[];

  for (const c of cases) {
    it(`matches python encoding: ${c.name}`, () => {
      const p = buildPosition(c);
      const x = encodePosition(p);
      expect(x.length).toBe(NEURAL_INPUT_SIZE);
      expect(x.length).toBe(c.x.length);
      for (let i = 0; i < x.length; i++) {
        expect(x[i]).toBeCloseTo(c.x[i], 6);
      }
    });
  }
});

describe("neural forward pass parity with Python", () => {
  it("matches python net.value() on the start position (1-output)", () => {
    const wPath = join(__dirname, "fixtures", "tiny-weights.json");
    const w = JSON.parse(readFileSync(wPath, "utf8")) as NeuralWeights & {
      _expected: { name: string; y: number }[];
    };
    const ev = neuralEvaluatorFromWeights(w);
    const cases = JSON.parse(
      readFileSync(join(__dirname, "fixtures", "encode_cases.json"), "utf8"),
    ) as Case[];
    for (const c of cases) {
      const p = buildPosition(c);
      const eq = ev.evaluate(p);
      const y = (eq + 1) / 2;
      const expected = w._expected.find((e) => e.name === c.name);
      expect(expected).toBeTruthy();
      expect(y).toBeCloseTo(expected!.y, 5);
    }
  });

  function runParity(fixtureName: string, expectedHidden: number[]) {
    const wPath = join(__dirname, "fixtures", fixtureName);
    const w = JSON.parse(readFileSync(wPath, "utf8")) as NeuralWeights & {
      _expected: {
        name: string;
        p_w: number;
        p_gw: number;
        p_l: number;
        p_gl: number;
        equity: number;
      }[];
    };
    expect(w.output).toBe(4);
    const ev = neuralEvaluatorFromWeights(w);
    expect(ev.outputDim).toBe(4);
    expect(ev.hiddenLayers).toEqual(expectedHidden);
    const cases = JSON.parse(
      readFileSync(join(__dirname, "fixtures", "encode_cases.json"), "utf8"),
    ) as Case[];
    for (const c of cases) {
      const p = buildPosition(c);
      const expected = w._expected.find((e) => e.name === c.name);
      expect(expected).toBeTruthy();

      const eq = ev.evaluate(p);
      expect(eq).toBeCloseTo(expected!.equity, 5);

      const probs = ev.evaluateOutcomes(p);
      expect(probs.pWin).toBeCloseTo(expected!.p_w, 5);
      expect(probs.pGammonWin).toBeCloseTo(expected!.p_gw, 5);
      expect(probs.pLoss).toBeCloseTo(expected!.p_l, 5);
      expect(probs.pGammonLoss).toBeCloseTo(expected!.p_gl, 5);
    }
  }

  it("matches python net_torch equity on the test positions (4-output, 1 hidden layer)",
    () => runParity("tiny-weights-4.json", [8]));

  it("matches python net_torch equity on the test positions (4-output, 2 hidden layers)",
    () => runParity("tiny-weights-4-2layer.json", [10, 6]));
});

describe("python rollout parity (score2ply)", () => {
  // Hex hash of (points + bar/off counts), matches training/engine.board_hash.
  function hexBoardHash(p: Position): string {
    const bytes = new Uint8Array(28);
    for (let i = 0; i < POINTS; i++) bytes[i] = (p.points[i] + 16) & 0xff;
    bytes[24] = p.barUs;
    bytes[25] = p.barThem;
    bytes[26] = p.offUs;
    bytes[27] = p.offThem;
    let s = "";
    for (let i = 0; i < 28; i++) s += bytes[i].toString(16).padStart(2, "0");
    return s;
  }

  interface RolloutCase {
    name: string;
    dice: [number, number];
    pos: {
      points: number[];
      bar_us: number;
      bar_them: number;
      off_us: number;
      off_them: number;
      turn: number;
    };
    results: { after_hash: string; us_equity: number }[];
  }

  it("matches python rollout_target_us_frame on a fixed test set", async () => {
    // Need to import lazily so we can use the shared search code.
    const { score2ply } = await import("../src/ai/search");
    const { generatePlays, applyPlay } = await import("../src/engine/moves");

    const w = JSON.parse(
      readFileSync(join(__dirname, "fixtures", "tiny-weights-4.json"), "utf8"),
    ) as NeuralWeights;
    const ev = neuralEvaluatorFromWeights(w);
    const cases = JSON.parse(
      readFileSync(join(__dirname, "fixtures", "rollout_cases.json"), "utf8"),
    ) as RolloutCase[];

    for (const c of cases) {
      const p: Position = buildPosition({
        name: c.name,
        points: c.pos.points,
        bar_us: c.pos.bar_us,
        bar_them: c.pos.bar_them,
        off_us: c.pos.off_us,
        off_them: c.pos.off_them,
        turn: c.pos.turn,
        x: [],
      });
      const plays = generatePlays(p, c.dice[0], c.dice[1]);
      const scored = score2ply(p, plays, ev);
      // For each scored play, look up the python expected equity by hash.
      for (let i = 0; i < plays.length; i++) {
        const after = applyPlay(p, plays[i]);
        const hash = hexBoardHash(after);
        const expected = c.results.find((r) => r.after_hash === hash);
        expect(expected, `no match for ${c.name} play ${i}`).toBeTruthy();
        expect(scored[i].equity).toBeCloseTo(expected!.us_equity, 5);
      }
    }
  });
});

describe("phased evaluator dispatches by stillInContact", () => {
  it("uses contact net on contact positions and race net on race positions", () => {
    const w1 = JSON.parse(
      readFileSync(join(__dirname, "fixtures", "tiny-weights-4.json"), "utf8"),
    ) as NeuralWeights;
    const w2 = JSON.parse(
      readFileSync(join(__dirname, "fixtures", "tiny-weights-4-2layer.json"),
        "utf8"),
    ) as NeuralWeights;
    // Mark "contact" with one tiny net and "race" with the other so the two
    // branches produce visibly different outputs on the same position.
    const phased = phasedEvaluatorFromWeights(w1, w2);
    const single1 = neuralEvaluatorFromWeights(w1);
    const single2 = neuralEvaluatorFromWeights(w2);

    // Build a contact position (the starting position is contact).
    const startCase: Case = {
      name: "start", points: [
        2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5,
        -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2,
      ],
      bar_us: 0, bar_them: 0, off_us: 0, off_them: 0, turn: 0,
      x: [],
    };
    const contactPos = buildPosition(startCase);
    expect(stillInContact(contactPos)).toBe(true);
    expect(phased.evaluate(contactPos)).toBeCloseTo(single1.evaluate(contactPos), 8);

    // Build a race position (no overlap).
    const racePos: Position = {
      ...contactPos,
      points: new Int8Array([
        2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, -2, -2, -2, -2, -2, -2,
      ]),
    };
    expect(stillInContact(racePos)).toBe(false);
    expect(phased.evaluate(racePos)).toBeCloseTo(single2.evaluate(racePos), 8);
  });
});
