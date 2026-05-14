import { GameController } from "../game/controller";

export function openSettingsModal(controller: GameController, overlayContainer: HTMLElement): void {
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  const modal = document.createElement("div");
  modal.className = "modal settings-modal";
  const s = controller.state.settings;
  modal.innerHTML = `
    <h2>Settings</h2>
    <label class="checkbox">
      <input type="checkbox" name="tutor" ${s.tutorEnabled ? "checked" : ""}/>
      <span>Tutor mode <em>— flags errors and blunders after each move</em></span>
    </label>
    <label class="checkbox">
      <input type="checkbox" name="pip" ${s.showPipCount ? "checked" : ""}/>
      <span>Show pip count <em>— remaining pips for both players</em></span>
    </label>
    <label class="checkbox">
      <input type="checkbox" name="equity" ${s.showEquity ? "checked" : ""}/>
      <span>Show game equity <em>— live position evaluation</em></span>
    </label>
    <div class="actions">
      <button data-action="done" class="primary">Done</button>
    </div>
  `;
  backdrop.appendChild(modal);
  overlayContainer.appendChild(backdrop);

  const tutorBox = modal.querySelector<HTMLInputElement>('input[name="tutor"]')!;
  const pipBox = modal.querySelector<HTMLInputElement>('input[name="pip"]')!;
  const equityBox = modal.querySelector<HTMLInputElement>('input[name="equity"]')!;

  // Apply each toggle immediately so the user sees the effect in real time.
  const applyOne = (patch: Partial<typeof s>): void => {
    controller.setSettings({ ...controller.state.settings, ...patch });
  };
  tutorBox.addEventListener("change", () => applyOne({ tutorEnabled: tutorBox.checked }));
  pipBox.addEventListener("change", () => applyOne({ showPipCount: pipBox.checked }));
  equityBox.addEventListener("change", () => applyOne({ showEquity: equityBox.checked }));

  const close = (): void => backdrop.remove();
  modal.querySelector<HTMLButtonElement>('[data-action="done"]')!.addEventListener("click", close);
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) close();
  });
}
