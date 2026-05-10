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
    evaluator: "pubeval",
    plies: 2,
    noise: 0,
    analysisEvaluator: "pubeval",
    analysisPlies: 2,
  },
};
