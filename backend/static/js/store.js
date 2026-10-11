/**
 * Shared UI state plus a tiny event bus. Modules read `state` and react to events instead of
 * importing each other, which keeps the dependency graph flat.
 *
 * Events (payload):
 *   'mode'    {mode, sign}          the recognition mode or sign sub-mode changed
 *   'camera'  {status, facing}      camera status: off | starting | live | paused | blocked | nocamera | busy | insecure | error
 *   'media'   {shown, w, h}         video frames appeared, went away, or the video size changed
 *   'filter'  {minConf, hidden}     confidence threshold or hidden-class set changed
 *   'overlay' {overlay}             detections shown or hidden
 *   'result'  {dets, data, size}    one analysed frame arrived (dets are already normalised)
 *   'theme'   {theme}               light or dark
 */
export const state = {
  mode: 'objects',
  sign: 'spell',            // 'spell' | 'gestures'
  camera: 'off',            // see the 'camera' event
  facing: 'user',           // 'user' (front, shown mirrored) | 'environment' (back)
  mirror: true,             // display mirroring: true for the front camera. Detections are drawn mirrored too; frames are sent unmirrored.
  overlay: true,
  minConf: 0.35,            // objects below this are hidden (slider in the objects panel)
  hiddenClasses: new Set(), // object classes switched off with the filter chips
  embed: document.documentElement.dataset.embed === 'true',
};

const listeners = new Map();

/** Subscribe to an event. Returns an unsubscribe function. */
export function on(evt, fn) {
  if (!listeners.has(evt)) listeners.set(evt, new Set());
  listeners.get(evt).add(fn);
  return () => listeners.get(evt).delete(fn);
}

/** Publish an event to every subscriber (errors in one subscriber do not stop the others). */
export function emit(evt, payload) {
  for (const fn of listeners.get(evt) || []) {
    try { fn(payload); } catch (err) { console.error(`listener for "${evt}" failed`, err); }
  }
}
