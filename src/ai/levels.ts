export type Difficulty = "beginner" | "casual" | "strong" | "expert";

export interface LevelConfig {
  difficulty: Difficulty;
  evaluator: "random" | "heuristic" | "pubeval" | "neural";
  plies: 0 | 2;
  // Random noise added to evaluations to introduce realistic mistakes
  noise: number;
  // For tutor analysis: prefer higher precision than for live play
  analysisEvaluator: "heuristic" | "pubeval" | "neural";
  analysisPlies: 0 | 2;
}

// Tutor / hint analysis configuration. Deliberately DECOUPLED from the
// opponent's difficulty: the tutor always grades and the equity bar always
// reads using the strongest evaluator (neural) at a deep, filtered search —
// regardless of whether you're playing Beginner or Expert. Playing a weak
// opponent must not give you weak coaching.
//
// `plies` is the root decision depth. plies=2 (my move + opp roll/move + static
// leaf) is the default: it's full, unfiltered expectimax — what the old "Expert"
// analysis used — and runs in tens of ms with the net, so it's safe to run
// eagerly on every move. plies=3 adds my reply ply; it's a genuine accuracy
// gain but pure-JS 3-ply expectimax is ~1–5s per move (too slow to run on every
// turn / in the integration tests), so it's gated behind the filter params
// below and OFF by default. Flip `plies` to 3 to trade latency for accuracy.
// When plies===3, `keepTop`/`keepWindow` bound the root candidates that get the
// deep search and `innerKeep` bounds the replies expanded at each interior node.
export interface TutorConfig {
  evaluator: "heuristic" | "neural";
  plies: 1 | 2 | 3;
  keepTop: number;
  keepWindow: number;
  innerKeep: number;
}

export const TUTOR_CONFIG: TutorConfig = {
  evaluator: "neural",
  plies: 2,
  keepTop: 8,
  keepWindow: 0.06,
  innerKeep: 6,
};

export const LEVELS: Record<Difficulty, LevelConfig> = {
  beginner: {
    difficulty: "beginner",
    evaluator: "heuristic",
    plies: 0,
    noise: 0.08,
    analysisEvaluator: "pubeval",
    analysisPlies: 0,
  },
  casual: {
    difficulty: "casual",
    evaluator: "pubeval",
    plies: 0,
    noise: 0.02,
    analysisEvaluator: "pubeval",
    analysisPlies: 0,
  },
  strong: {
    difficulty: "strong",
    evaluator: "pubeval",
    plies: 0,
    noise: 0,
    analysisEvaluator: "pubeval",
    analysisPlies: 2,
  },
  expert: {
    difficulty: "expert",
    evaluator: "neural",
    plies: 2,
    noise: 0,
    analysisEvaluator: "neural",
    analysisPlies: 2,
  },
};
