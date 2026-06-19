import { describe, it, expect } from "vitest";
import { startingPosition } from "../src/engine/position";
import { generatePlays } from "../src/engine/moves";
import { analyzeMove } from "../src/ai/engine";

// The tutor / equity bar must grade with the SAME strength no matter which
// opponent difficulty you chose to play against. analyzeMove now uses
// TUTOR_CONFIG, not LEVELS[difficulty], so its output must be identical across
// difficulties. (No book is loaded here, so this exercises the search path.)
describe("tutor analysis is decoupled from opponent difficulty", () => {
  it("analyzeMove returns identical results for beginner and expert", () => {
    const p = startingPosition();
    p.dice = [6, 4];
    const plays = generatePlays(p, 6, 4);

    const beginner = analyzeMove(p, plays, "beginner");
    const expert = analyzeMove(p, plays, "expert");
    const casual = analyzeMove(p, plays, "casual");

    expect(beginner.bestPlay).toEqual(expert.bestPlay);
    expect(beginner.equities).toEqual(expert.equities);
    expect(beginner.bestEquity).toEqual(expert.bestEquity);
    // And casual (the old weak-tutor case) matches too.
    expect(casual.equities).toEqual(expert.equities);
  });
});
