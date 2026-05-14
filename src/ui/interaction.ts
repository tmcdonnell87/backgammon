import {
  GameController,
  Phase,
  legalNextTargets,
  remainingPlays,
  remainingDice,
} from "../game/controller";

// Translate a tap on a board zone (point 0-23, BAR=24, OFF=-1) into the right
// controller action.
//
// When something is already picked up the priority is:
//   1. Tapped cell is a legal destination of the selected source → move (pickTo).
//   2. Tapped cell is the selected source itself → deselect.
//   3. Tapped cell is some other legal source → switch selection there (and
//      auto-execute if that source has a unique destination).
//   4. Otherwise → try pickTo (handles unique-source resolution).
//
// When nothing is picked up:
//   1. Tapped cell is a legal source → pickFrom (and auto-execute if unique).
//   2. Otherwise → pickTo (resolves a destination when only one legal source).
//
// Note rule 1 (priority of dest over source) fixes the historical "click-hijack"
// bug where tapping point X to land a piece silently re-picked-up from X when X
// was itself a legal source.
export function handleTap(controller: GameController, idx: number): void {
  if (controller.state.phase.kind !== "play") return;
  const phase = controller.state.phase as Extract<Phase, { kind: "play" }>;
  const targets = legalNextTargets(phase.legalPlays, controller.state.pendingPlay);
  const selected = controller.state.selectedFrom;

  if (selected !== null) {
    const dests = targets.get(selected);
    if (dests && dests.has(idx)) {
      controller.pickTo(idx);
      return;
    }
    if (idx === selected) {
      controller.clearSelection();
      return;
    }
    if (targets.has(idx)) {
      // Silent pickFrom — autoExec will emit via pickTo. Avoids a brief
      // "source selected, no move yet" frame that flashes between renders.
      controller.pickFrom(idx, true);
      autoExecIfUnique(controller, idx);
      return;
    }
    controller.pickTo(idx);
    return;
  }

  if (targets.has(idx)) {
    controller.pickFrom(idx);
    autoExecIfUnique(controller, idx);
    return;
  }
  controller.pickTo(idx);
}

// Auto-execute the picked-up source's move, preferring the move that uses the
// leftmost remaining die. Always fires when at least one legal sub-move exists
// from `from` (i.e., we never leave a source dangling with nothing to do).
// If the leftmost die has no move from this source, fall back to the other die.
function autoExecIfUnique(controller: GameController, from: number): void {
  if (controller.state.phase.kind !== "play") return;
  if (controller.state.selectedFrom !== from) return;
  const phase = controller.state.phase as Extract<Phase, { kind: "play" }>;
  const remaining = remainingPlays(phase.legalPlays, controller.state.pendingPlay);
  const dice = controller.state.position.dice;
  if (!dice) return;
  const rolledAll = dice[0] === dice[1] ? [dice[0], dice[0], dice[0], dice[0]] : [dice[0], dice[1]];
  const remainingDiceValues = remainingDice(rolledAll, controller.state.pendingPlay);
  if (remainingDiceValues.length === 0) return;

  // Walk the remaining die values left-to-right; pick the first whose value
  // can move this source. (For doubles all values are equal, so this loop just
  // picks the first matching sub-move.)
  const seenDice = new Set<number>();
  for (const die of remainingDiceValues) {
    if (seenDice.has(die)) continue;
    seenDice.add(die);
    for (const play of remaining) {
      if (play.length <= controller.state.pendingPlay.length) continue;
      const sub = play[controller.state.pendingPlay.length];
      if (sub.from === from && sub.die === die) {
        controller.pickTo(sub.to);
        return;
      }
    }
  }
  // Final fallback: any legal next dest from `from`.
  const targets = legalNextTargets(phase.legalPlays, controller.state.pendingPlay);
  const dests = targets.get(from);
  if (!dests || dests.size === 0) return;
  const first = dests.values().next().value as number;
  controller.pickTo(first);
}
