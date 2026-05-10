import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { POINTS, Position } from "../src/engine/position";
import {
  encodePosition,
  NEURAL_INPUT_SIZE,
  neuralEvaluatorFromWeights,
  NeuralWeights,
} from "../src/ai/neural";

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
    postCrawford: false,
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
  it("matches python net.value() on the start position", () => {
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
});
