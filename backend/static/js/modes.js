/**
 * Mode registry: the single extension point for new recognition modes (for example "Record").
 *
 * A mode is a panel (`.insight[data-panel]` in index.html), a button in `#modeSwitch`, and one
 * `registerMode()` call. The shell (main.js) handles the switch, the panel visibility, the title,
 * the number-key shortcut, the empty-state text and the API query; the mode only renders its panel.
 *
 * @typedef {object} ModeDef
 * @property {string}  id          'objects' | 'faces' | 'sign' | your own; also `data-mode` and the `?mode=` value
 * @property {number}  order       position in the switch; the number key is `order` (1 to 9)
 * @property {string}  title       panel heading
 * @property {string|function(object): string} sub  panel subheading (a function receives the shared `state`)
 * @property {string}  empty       one line shown in the empty state ("what does this mode do")
 * @property {string}  panel       id of the panel element to show, e.g. 'insightObjects'
 * @property {function(object): string} [query]   query string for POST /api/analyze; default `mode=<id>`
 * @property {boolean} [draws]     true (default) to draw boxes/skeletons from the detections
 * @property {function(object): void} [onEnter]   the mode became active (ctx = {state})
 * @property {function(object): void} [onLeave]   the mode is being left
 * @property {function(Array, object, object): void} [onResult]  (dets, data, ctx) one analysed frame arrived
 * @property {function(object): void} [onReset]   the feed was reset: clear the panel
 */
import { state, emit } from './store.js';
import { $ } from './util.js';

const modes = new Map();

/** Register (or replace) a mode. Call it at module load, before main.js initialises. */
export function registerMode(def) {
  if (!def || !def.id || !def.panel) throw new Error('registerMode needs {id, panel}');
  modes.set(def.id, { draws: true, query: () => `mode=${def.id}`, ...def });
}

export function getMode(id) { return modes.get(id); }
export function activeMode() { return modes.get(state.mode); }
/** Registered modes sorted by `order`. */
export function listModes() { return [...modes.values()].sort((a, b) => (a.order || 99) - (b.order || 99)); }
export function modeIds() { return listModes().map((m) => m.id); }

/** Show the panel of the active mode and hide the others. */
export function showPanel(id) {
  for (const m of modes.values()) $(m.panel).classList.toggle('hidden', m.id !== id);
}

/**
 * Switch the active mode. Runs onLeave/onEnter, then emits 'mode'. Does nothing if nothing changed
 * (except when `force` is set). main.js listens to 'mode' to refresh the shell.
 */
export function switchMode(id, { sign, force = false } = {}) {
  const next = modes.get(id);
  if (!next) return false;
  const changed = id !== state.mode || (sign && sign !== state.sign);
  if (!changed && !force) return false;
  const prev = modes.get(state.mode);
  if (prev && prev.id !== id && prev.onLeave) prev.onLeave({ state });
  state.mode = id;
  if (sign) state.sign = sign;
  if (next.onEnter && changed) next.onEnter({ state });
  emit('mode', { mode: state.mode, sign: state.sign, changed });
  return changed;
}
