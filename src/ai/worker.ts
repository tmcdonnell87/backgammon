// Web Worker entry. Hosts the AI engine off the main thread.

import { analyzeMove, pickMove, setNeuralEvaluator } from "./engine";
import { Difficulty } from "./levels";
import { Play } from "../engine/moves";
import { Position } from "../engine/position";
import { loadNeuralEvaluator } from "./neural";

// Best-effort load of trained Expert weights. If absent, neural tiers
// transparently fall back to the heuristic.
loadNeuralEvaluator("/weights/expert.json")
  .then((ev) => {
    if (ev) setNeuralEvaluator(ev);
  })
  .catch(() => {});

type Req =
  | { id: number; type: "pick"; pos: Position; legalPlays: Play[]; difficulty: Difficulty }
  | { id: number; type: "analyze"; pos: Position; legalPlays: Play[]; difficulty: Difficulty };

type Resp =
  | { id: number; type: "pick"; play: Play }
  | { id: number; type: "analyze"; bestPlay: Play; bestEquity: number; equities: number[] }
  | { id: number; type: "error"; message: string };

self.addEventListener("message", (ev: MessageEvent<Req>) => {
  const msg = ev.data;
  try {
    // Position arrived via structured clone: points is an Int8Array which survives, but
    // we re-wrap to be safe.
    const pos: Position = {
      ...msg.pos,
      points: new Int8Array(msg.pos.points),
    };
    if (msg.type === "pick") {
      const play = pickMove(pos, msg.legalPlays, msg.difficulty);
      const r: Resp = { id: msg.id, type: "pick", play };
      (self as DedicatedWorkerGlobalScope).postMessage(r);
    } else {
      const a = analyzeMove(pos, msg.legalPlays, msg.difficulty);
      const r: Resp = {
        id: msg.id,
        type: "analyze",
        bestPlay: a.bestPlay,
        bestEquity: a.bestEquity,
        equities: a.equities,
      };
      (self as DedicatedWorkerGlobalScope).postMessage(r);
    }
  } catch (e) {
    const r: Resp = { id: msg.id, type: "error", message: e instanceof Error ? e.message : String(e) };
    (self as DedicatedWorkerGlobalScope).postMessage(r);
  }
});
