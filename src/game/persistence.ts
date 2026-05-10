import { get, set, del } from "idb-keyval";
import { GameSettings } from "./controller";

const KEY_SETTINGS = "bg.settings";
const KEY_GAME = "bg.game";

const DEFAULT_SETTINGS: GameSettings = {
  matchLength: 1,
  cubeEnabled: false,
  whitePlayer: "human",
  blackPlayer: "cpu",
  whiteName: "You",
  blackName: "Computer",
  cpuDifficulty: "casual",
  tutorEnabled: false,
  hidePassAndPlay: false,
};

export async function loadSettings(): Promise<GameSettings> {
  try {
    const v = (await get(KEY_SETTINGS)) as Partial<GameSettings> | undefined;
    return { ...DEFAULT_SETTINGS, ...(v ?? {}) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export async function saveSettings(s: GameSettings): Promise<void> {
  try {
    await set(KEY_SETTINGS, s);
  } catch {
    // ignore
  }
}

// Saved game blob — opaque to persistence. Controller serializes/deserializes.
export async function saveGame(blob: unknown): Promise<void> {
  try {
    await set(KEY_GAME, blob);
  } catch {
    // ignore
  }
}

export async function loadGame(): Promise<unknown | null> {
  try {
    return (await get(KEY_GAME)) ?? null;
  } catch {
    return null;
  }
}

export async function clearGame(): Promise<void> {
  try {
    await del(KEY_GAME);
  } catch {
    // ignore
  }
}
