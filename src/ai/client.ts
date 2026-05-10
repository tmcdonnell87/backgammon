// Main-thread AI client. Spawns a Worker and provides promise-based RPC.
// Falls back to in-process execution when Worker isn't available (e.g., tests).

import { AIClient, MoveAnalysis } from "./api";
import { Play } from "../engine/moves";
import { Position } from "../engine/position";
import { Difficulty } from "./levels";
import { analyzeMove, pickMove } from "./engine";

type Pending = {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
};

export function createWorkerClient(): AIClient {
  let worker: Worker | null = null;
  try {
    worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  } catch {
    worker = null;
  }

  if (!worker) return createMainThreadClient();

  let nextId = 1;
  const pending = new Map<number, Pending>();
  worker.onmessage = (ev: MessageEvent<{ id: number; type: string; [k: string]: unknown }>) => {
    const m = ev.data;
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    if (m.type === "error") p.reject(new Error(String(m.message)));
    else p.resolve(m);
  };

  const send = <T>(req: Record<string, unknown>): Promise<T> => {
    return new Promise<T>((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      worker!.postMessage({ id, ...req });
    });
  };

  return {
    async pickMove(p: Position, legalPlays: Play[], difficulty: Difficulty): Promise<Play> {
      const r = await send<{ play: Play }>({
        type: "pick",
        pos: serialize(p),
        legalPlays,
        difficulty,
      });
      return r.play;
    },
    async analyze(p: Position, legalPlays: Play[], difficulty: Difficulty): Promise<MoveAnalysis> {
      const r = await send<{ bestPlay: Play; bestEquity: number; equities: number[] }>({
        type: "analyze",
        pos: serialize(p),
        legalPlays,
        difficulty,
      });
      return { bestPlay: r.bestPlay, bestEquity: r.bestEquity, equities: r.equities };
    },
  };
}

export function createMainThreadClient(): AIClient {
  return {
    async pickMove(p, legalPlays, difficulty) {
      // Yield to event loop so UI updates before computation
      await new Promise<void>((res) => setTimeout(res, 0));
      return pickMove(p, legalPlays, difficulty);
    },
    async analyze(p, legalPlays, difficulty) {
      await new Promise<void>((res) => setTimeout(res, 0));
      const a = analyzeMove(p, legalPlays, difficulty);
      return { bestPlay: a.bestPlay, bestEquity: a.bestEquity, equities: a.equities };
    },
  };
}

function serialize(p: Position): Position {
  // Position is structured-cloneable already; this is for clarity.
  return p;
}
