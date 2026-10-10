# Web UI modules

Plain ES modules, no build step. `../app.js` is the entry (`<script type="module">`).

| Module | Role |
| --- | --- |
| `util.js` | DOM helpers, `countUp`, `reducedMotion` |
| `store.js` | shared `state` and a tiny event bus (`on`, `emit`) |
| `config.js` | colours, limits, the ASL alphabet |
| `modes.js` | **mode registry**, the extension point for new modes |
| `overlay.js` | canvas tracks, smart label placement, confidence and class filter |
| `snapshot.js` | download the annotated frame as a PNG |
| `insights.js` | Objects and Expressions panels (registers both modes) |
| `sign.js` | Sign panel: word builder, ASL reference chart, "Try" challenge (registers `sign`) |
| `stats.js` | FPS, latency, model, device |
| `toast.js` | transient messages |
| `media.js` | camera (start, stop, pause, flip), frame loop |
| `main.js` | shell: switches, theme, shortcuts, embed mode, URL sync |

## Adding a mode (for example "Record")

1. `index.html`: add a radio button to `#modeSwitch` (`data-mode="record"`), and a panel
   `<div class="insight hidden" id="insightRecord" data-panel="record">` inside `#panelBody`
   (see the `RECORD HOOK` comment).
2. Create `js/record.js` and call `registerMode({ id: 'record', order: 4, title, sub, empty, panel: 'insightRecord',
   query: () => 'mode=record', draws: true, onResult(dets, data) { /* render the panel */ }, onEnter, onLeave, onReset })`.
   The shell then handles the switch, panel visibility, title, the `4` shortcut and the API query.
3. Add `import './js/record.js';` to `../app.js`.
4. Optional: subscribe with `on('result', ...)`, `on('media', ...)` from `store.js` for cross-cutting needs,
   and use `toast()` for feedback. The server-side `mode` value and any extra query parameters are yours
   (`query(state)`); the frame loop in `media.js` sends whatever you return.

Modules never import each other circularly: `util`, `store`, `config`, `modes` are leaves; everything
else may import them and, where needed, `toast` and `overlay`. `media.js` and `main.js` sit on top.
