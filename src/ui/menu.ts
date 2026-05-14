import { GameSettings, PlayerKind } from "../game/controller";
import { Difficulty } from "../ai/levels";

export interface MenuCallbacks {
  onStart: (s: GameSettings) => void;
  onClose?: () => void;
}

export function showMenu(initial: GameSettings, cb: MenuCallbacks): HTMLElement {
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";

  const modal = document.createElement("div");
  modal.className = "modal";
  modal.innerHTML = `
    <h2>New Game</h2>
    <div class="row">
      <label>
        <span>Opponent</span>
        <select name="opponent">
          <option value="cpu">Computer</option>
          <option value="human">Pass &amp; play (2 humans)</option>
        </select>
      </label>
    </div>
    <div class="row" data-only-cpu>
      <label>
        <span>Difficulty</span>
        <select name="difficulty">
          <option value="beginner">Beginner</option>
          <option value="casual">Casual</option>
          <option value="strong">Strong</option>
          <option value="expert">Expert</option>
        </select>
      </label>
      <label>
        <span>Your color</span>
        <select name="yourcolor">
          <option value="white">White</option>
          <option value="black">Black</option>
        </select>
      </label>
    </div>
    <div class="row" data-only-human>
      <label>
        <span>White's name</span>
        <input type="text" name="whiteName" maxlength="20" />
      </label>
      <label>
        <span>Black's name</span>
        <input type="text" name="blackName" maxlength="20" />
      </label>
    </div>
    <div class="row">
      <label>
        <span>Match length</span>
        <select name="matchLength">
          <option value="1">1 point (single game)</option>
          <option value="3">3 points</option>
          <option value="5">5 points</option>
          <option value="7">7 points</option>
          <option value="9">9 points</option>
          <option value="11">11 points</option>
        </select>
      </label>
    </div>
    <div class="actions">
      <button type="button" data-action="cancel">Cancel</button>
      <button type="button" class="primary" data-action="start">Start</button>
    </div>
  `;

  // Initial values
  const opp = modal.querySelector<HTMLSelectElement>('select[name="opponent"]')!;
  const diff = modal.querySelector<HTMLSelectElement>('select[name="difficulty"]')!;
  const yourColor = modal.querySelector<HTMLSelectElement>('select[name="yourcolor"]')!;
  const whiteName = modal.querySelector<HTMLInputElement>('input[name="whiteName"]')!;
  const blackName = modal.querySelector<HTMLInputElement>('input[name="blackName"]')!;
  const matchLen = modal.querySelector<HTMLSelectElement>('select[name="matchLength"]')!;

  const bothHumanInitially = initial.whitePlayer === "human" && initial.blackPlayer === "human";
  opp.value = bothHumanInitially ? "human" : "cpu";
  diff.value = initial.cpuDifficulty;
  yourColor.value = initial.whitePlayer === "human" ? "white" : "black";
  whiteName.value = initial.whiteName;
  blackName.value = initial.blackName;
  matchLen.value = String(initial.matchLength);

  const updateVis = (): void => {
    const isCpu = opp.value === "cpu";
    modal.querySelectorAll<HTMLElement>("[data-only-cpu]").forEach((el) => {
      el.style.display = isCpu ? "" : "none";
    });
    modal.querySelectorAll<HTMLElement>("[data-only-human]").forEach((el) => {
      el.style.display = !isCpu ? "" : "none";
    });
  };
  opp.addEventListener("change", updateVis);
  updateVis();

  modal.querySelector<HTMLButtonElement>('[data-action="cancel"]')!.addEventListener("click", () => {
    backdrop.remove();
    cb.onClose?.();
  });
  modal.querySelector<HTMLButtonElement>('[data-action="start"]')!.addEventListener("click", () => {
    const isCpu = opp.value === "cpu";
    const yourIsWhite = yourColor.value === "white";
    const settings: GameSettings = {
      matchLength: parseInt(matchLen.value, 10),
      cubeEnabled: false, // doubling cube UI/logic deferred until Phase 7 ships
      whitePlayer: isCpu ? (yourIsWhite ? "human" : "cpu") : ("human" as PlayerKind),
      blackPlayer: isCpu ? (yourIsWhite ? "cpu" : "human") : ("human" as PlayerKind),
      whiteName: isCpu ? (yourIsWhite ? "You" : "Computer") : whiteName.value || "White",
      blackName: isCpu ? (yourIsWhite ? "Computer" : "You") : blackName.value || "Black",
      cpuDifficulty: diff.value as Difficulty,
      // Tutor mode is now exclusively toggled in-game from the settings modal,
      // so we preserve whatever was last saved here.
      tutorEnabled: initial.tutorEnabled,
      // Sticky in-game-only settings — preserved across game starts.
      showPipCount: initial.showPipCount,
      showEquity: initial.showEquity,
    };
    backdrop.remove();
    cb.onStart(settings);
  });

  backdrop.appendChild(modal);
  return backdrop;
}
