/** Objects and facial-expression panels plus their mode registrations. */
import { state, emit, on } from './store.js';
import { registerMode } from './modes.js';
import { $, h, countUp, hash, pct } from './util.js';
import { PALETTE, EMOTION_COLORS, EXPRESSIONS, MAX_FACES } from './config.js';
import { visibleDets, hiddenCount, requestDraw } from './overlay.js';

let lastDets = [], lastSummary = '', pendingSummary = null, announceTimer = 0;
const seenClasses = new Map();
const colorFor = (name) => PALETTE[hash(name) % PALETTE.length];
function render() { renderObjects(lastDets); requestDraw(); }

function renderObjects(dets) {
  const visible = visibleDets(dets), groups = new Map();
  dets.forEach((d) => seenClasses.set(String(d.class), Date.now()));
  visible.forEach((d) => { const name = String(d.class), g = groups.get(name) || { name, n: 0, sum: 0 }; g.n++; g.sum += Number(d.confidence) || 0; groups.set(name, g); });
  const rows = [...groups.values()].sort((a, b) => b.n - a.n || b.sum / b.n - a.sum / a.n), max = rows[0]?.n || 1;
  const list = $('objBars'), existing = new Map([...list.children].map((li) => [li.dataset.key, li]));
  rows.forEach((g, i) => {
    let li = existing.get(g.name);
    if (!li) { li = h('li', 'bar-row'); li.dataset.key = g.name; li.append(h('span', 'bar-name', g.name), h('span', 'bar-track'), h('span', 'bar-count'), h('span', 'bar-conf')); li.children[1].append(h('span', 'bar-fill')); li.style.setProperty('--c', colorFor(g.name)); }
    existing.delete(g.name); li.children[1].firstChild.style.setProperty('--w', `${Math.round(g.n / max * 100)}%`);
    li.children[2].textContent = `×${g.n}`; li.children[3].textContent = pct(g.sum / g.n); li.children[3].title = 'Average confidence';
    if (list.children[i] !== li) list.insertBefore(li, list.children[i] || null);
  });
  existing.forEach((li) => li.remove());
  countUp($('objCount'), visible.length); countUp($('objClasses'), rows.length);
  const all = $('classChips'), old = new Map([...all.querySelectorAll('button[data-class]')].map((b) => [b.dataset.class, b]));
  const allButton = all.querySelector('.chip-all') || makeAllChip();
  all.prepend(allButton); allButton.setAttribute('aria-pressed', String(state.hiddenClasses.size === 0));
  const names = new Set([...seenClasses.keys(), ...state.hiddenClasses]);
  for (const name of names) {
    const recentlySeen = Date.now() - (seenClasses.get(name) || 0) < 3000;
    if (!recentlySeen && !state.hiddenClasses.has(name)) continue;
    let chip = old.get(name);
    if (!chip) chip = makeChip(name);
    old.delete(name); chip.style.setProperty('--c', colorFor(name));
    const n = dets.filter((d) => String(d.class) === name).length;
    chip.querySelector('.chip-n').textContent = String(n);
    chip.setAttribute('aria-pressed', String(!state.hiddenClasses.has(name)));
    all.append(chip);
  }
  old.forEach((b) => b.remove());
  const conf = Math.round(state.minConf * 100), hidden = hiddenCount(dets);
  const average = visible.length ? visible.reduce((sum, d) => sum + (Number(d.confidence) || 0), 0) / visible.length : null;
  if (average == null) $('objConf').textContent = '–'; else countUp($('objConf'), Math.round(average * 100), { suffix: '%' });
  $('minConf').value = String(conf); $('minConfOut').textContent = `${conf}%`;
  $('minConf').style.setProperty('--range-progress', `${(conf / 95) * 100}%`);   // filled part of the slider track
  $('hiddenNote').textContent = hidden ? `${hidden} ${hidden === 1 ? "detection" : "detections"} hidden below ${conf}% or by filters.` : `Showing everything above ${conf}%.`;
  $('objNone').textContent = dets.length && !visible.length ? `Nothing above ${conf}%, lower the threshold to see more.` : 'Nothing detected yet.';
  $('objNone').classList.toggle('hidden', visible.length > 0);
}
function makeAllChip() {
  const b = h('button', 'chip chip-all'); b.type = 'button'; b.textContent = 'All';
  b.addEventListener('click', () => { state.hiddenClasses.clear(); emit('filter', { minConf: state.minConf, hidden: state.hiddenClasses }); render(); }); return b;
}
function makeChip(name) {
  const b = h('button', 'chip'); b.type = 'button'; b.dataset.class = name;
  b.append(h('span', 'chip-dot'), h('span', 'chip-name', name), h('span', 'chip-n', '0'));
  b.addEventListener('click', () => {
    if (state.hiddenClasses.has(name)) state.hiddenClasses.delete(name); else state.hiddenClasses.add(name);
    emit('filter', { minConf: state.minConf, hidden: state.hiddenClasses }); render();
  }); return b;
}
$('minConf').addEventListener('input', (event) => {
  state.minConf = Math.max(0, Math.min(0.95, Number(event.target.value) / 100));
  emit('filter', { minConf: state.minConf, hidden: state.hiddenClasses }); render();
});
function emotionScores(d) { return d.scores || d.top3 || { [d.class]: d.confidence }; }
function renderFaces(dets) {
  const list = $('faceList'), faces = dets.slice().sort((a, b) => a.bbox.x1 - b.bbox.x1).slice(0, MAX_FACES);
  while (list.children.length > faces.length) list.lastChild.remove();
  faces.forEach((d, i) => {
    let card = list.children[i];
    if (!card) {
      card = h('article', 'face'); const head = h('header', 'face-head'); head.append(h('span', 'face-name'), h('span', 'face-top'));
      const ul = h('ul', 'emo');
      for (const emotion of EXPRESSIONS) { const li = h('li', 'emo-row'); li.dataset.emo = emotion; li.style.setProperty('--c', EMOTION_COLORS[emotion]); const track = h('span', 'emo-track'); track.append(h('span', 'emo-fill')); li.append(h('span', 'emo-name', emotion), track, h('span', 'emo-val')); ul.append(li); }
      card.append(head, ul); list.append(card);
    }
    const scores = emotionScores(d); card.children[0].children[0].textContent = faces.length > 1 ? `Face ${i + 1}` : 'Face';
    card.children[0].children[1].textContent = `${d.class} ${pct(d.confidence)}`; card.children[0].children[1].style.setProperty('--c', EMOTION_COLORS[d.class] || '#2563eb');
    for (const li of card.children[1].children) { const value = Number(scores[li.dataset.emo]) || 0; li.children[1].firstChild.style.setProperty('--w', `${Math.round(value * 100)}%`); li.children[2].textContent = pct(value); li.classList.toggle('is-top', li.dataset.emo === d.class); }
  });
  const more = dets.length - faces.length; $('faceNone').textContent = more > 0 ? `+${more} more` : 'No face in view.'; $('faceNone').classList.toggle('hidden', faces.length > 0 && !more);
}

/** Debounced screen-reader summary, shared with sign mode. */
export function announce(dets) {
  let summary;
  if (!dets.length) summary = state.mode === 'sign' ? 'No hand in view' : state.mode === 'faces' ? 'No face in view' : 'Nothing detected';
  else if (state.mode === 'objects') { const counts = new Map(); dets.forEach((d) => counts.set(d.class, (counts.get(d.class) || 0) + 1)); summary = [...counts].map(([name, n]) => `${n} ${name}`).join(', '); }
  else summary = dets.reduce((a, d) => !a || d.confidence > a.confidence ? d : a, null).class;
  if (summary === lastSummary) { clearTimeout(announceTimer); pendingSummary = null; return; }
  if (summary === pendingSummary) return;
  pendingSummary = summary; clearTimeout(announceTimer);
  announceTimer = setTimeout(() => { lastSummary = summary; pendingSummary = null; $('liveSummary').textContent = summary; }, 900);
}
export function resetAnnounce() { clearTimeout(announceTimer); pendingSummary = null; lastSummary = ''; $('liveSummary').textContent = ''; }
function resetObjects() { lastDets = []; $('objBars').replaceChildren(); $('classChips').replaceChildren(); $('objNone').classList.remove('hidden'); $('objNone').textContent = 'Nothing detected yet.'; countUp($('objCount'), 0); countUp($('objClasses'), 0); $('objConf').textContent = '–'; renderObjects([]); resetAnnounce(); }
function onObjects(dets) { lastDets = dets; renderObjects(dets); announce(dets); }
function onFaces(dets) { lastDets = dets; renderFaces(dets); announce(dets); }
on('filter', render);
on('result', ({ dets }) => { if (state.mode === 'objects') onObjects(dets); else if (state.mode === 'faces') onFaces(dets); });
registerMode({ id: 'objects', order: 1, panel: 'insightObjects', title: 'Objects', sub: '80 everyday object classes', empty: 'Finds and labels everyday objects with a box and a confidence score.', onResult: onObjects, onReset: resetObjects });
registerMode({ id: 'faces', order: 2, panel: 'insightFaces', title: 'Expressions', sub: 'Seven expressions, scored per face', empty: 'Reads one of seven expressions per face, with a score for each.', onResult: onFaces, onReset() { lastDets = []; $('faceList').replaceChildren(); $('faceNone').textContent = 'No face in view.'; $('faceNone').classList.remove('hidden'); resetAnnounce(); } });
