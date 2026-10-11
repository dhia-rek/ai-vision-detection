/** Sign-language mode: word builder, ASL chart, hold indicator and letter challenge. */
import { $, clamp, countUp } from './util.js';
import { state, on } from './store.js';
import { ASL_LETTERS, ASL_IMG, MOTION_LETTERS, EXPRESSIONS, SIGN_COLOR } from './config.js';
import { registerMode } from './modes.js';
import { announce } from './insights.js';

const ringCircumference = 2 * Math.PI * 52;
let tryTarget = 'B', streak = 0, solved = new Set(), holdSince = null, successTimer = 0, trySucceeded = false;
const tryPool = ASL_LETTERS.filter((letter) => !MOTION_LETTERS.includes(letter));
const builder = {
  holdS: 0.8, spaceS: 1.5, minConf: 0.6, text: '', current: null, since: 0,
  committed: false, lastSeen: null,
  reset() { this.current = null; this.since = 0; this.committed = false; this.lastSeen = null; },
  feed(letter, conf, t) {
    if (letter == null || conf < this.minConf) {
      this.current = null; this.committed = false;
      if (this.lastSeen !== null && t - this.lastSeen >= this.spaceS && this.text && !this.text.endsWith(' ')) this.text += ' ';
    } else {
      this.lastSeen = t;
      if (letter !== this.current) { this.current = letter; this.since = t; this.committed = false; }
      else if (!this.committed && t - this.since >= this.holdS) { this.committed = true; this.text += letter; pulseRing(); }
    }
    this.render(t);
  },
  render(t = performance.now() / 1000) {
    $('wordText').textContent = this.text;
    $('wordBox').classList.toggle('is-empty', !this.text);
    if (spellActive()) setRing(this.current ? (this.committed ? 1 : (t - this.since) / this.holdS) : 0);
  },
  edit(fn) { this.text = fn(this.text); this.render(); },
  clear() { this.text = ''; this.reset(); this.render(); },
};
function pulseRing() {
  const ring = $('holdRing'); ring.classList.remove('pulse'); void ring.offsetWidth; ring.classList.add('pulse');
}
function setRing(progress) {
  const value = $('ringValue');
  value.style.strokeDasharray = ringCircumference.toFixed(2);
  value.style.strokeDashoffset = (ringCircumference * (1 - clamp(progress, 0, 1))).toFixed(2);
}
export function spellActive() { return state.mode === 'sign' && state.sign === 'spell'; }
export function resetSpell() { builder.reset(); setRing(0); resetTryProgress(); }
export function feedSpell(best, tSeconds) {
  const letter = best ? String(best.class).toUpperCase() : null;
  const confidence = best ? Number(best.confidence) || 0 : 0;
  builder.feed(letter, confidence, tSeconds);
  if (state.mode === 'sign') announce(best ? [best] : []);
  feedTry(best, tSeconds);
  updateChartLive(letter);
}
function resetTryProgress() {
  holdSince = null; trySucceeded = false; clearTimeout(successTimer);
  $('tryCard').dataset.state = 'waiting';
  $('tryBar').style.setProperty('--p', '0');
  $('trySuccess').hidden = true;
  $('tryHint').textContent = idleHint();
}
function idleHint() {
  return state.camera === 'live' || state.camera === 'paused' ? 'Copy the hand shape and hold it.' : 'Start the camera to play.';
}
function setTarget(letter) {
  if (!tryPool.includes(letter)) return;
  tryTarget = letter;
  resetTryProgress(); renderTry(); renderChart();
}
function renderTry() {
  $('tryImg').src = ASL_IMG(tryTarget);
  $('tryImg').alt = `ASL hand shape for the letter ${tryTarget}`;
  $('tryLetter').textContent = tryTarget;
  $('tryTitle').setAttribute('aria-label', `Try: ${tryTarget}`);
}
function chooseNext() {
  const choices = tryPool.filter((letter) => letter !== tryTarget);
  setTarget(choices[Math.floor(Math.random() * choices.length)]);
}
function succeed() {
  trySucceeded = true;
  $('tryCard').dataset.state = 'success';
  $('tryBar').style.setProperty('--p', '1');
  $('trySuccessText').textContent = `Correct: ${tryTarget}`;
  $('trySuccess').hidden = false;
  solved.add(tryTarget); streak++;
  countUp($('tryStreakN'), streak, { duration: 350 });
  renderChart();
  successTimer = setTimeout(chooseNext, 2200);
}
function feedTry(best, t) {
  if (!spellActive() || trySucceeded) return;
  if (!best || Number(best.confidence) < 0.6) {
    holdSince = null;
    $('tryCard').dataset.state = 'waiting';
    $('tryBar').style.setProperty('--p', '0');
    $('tryHint').textContent = idleHint();
    return;
  }
  const seen = String(best.class).toUpperCase();
  if (seen !== tryTarget) {
    holdSince = null; $('tryCard').dataset.state = 'wrong';
    $('tryHint').textContent = `I see ${seen}. Looking for ${tryTarget}.`;
    $('tryBar').style.setProperty('--p', '0'); return;
  }
  if (holdSince === null) holdSince = t;
  const progress = clamp((t - holdSince) / 0.8, 0, 1);
  $('tryCard').dataset.state = 'holding';
  $('tryHint').textContent = 'Keep holding the hand shape.';
  $('tryBar').style.setProperty('--p', String(progress));
  if (progress >= 1) succeed();
}
function renderChart() {
  const grid = $('chartGrid');
  if (!grid.childElementCount) {
    for (const letter of ASL_LETTERS) {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'chart-tile';
      button.dataset.letter = letter; button.setAttribute('aria-label', `Practise ${letter}`); button.setAttribute('aria-pressed', 'false');
      const holder = document.createElement('span'); holder.className = 'chart-img';
      const img = document.createElement('img'); img.src = ASL_IMG(letter); img.alt = ''; img.loading = 'lazy';
      holder.append(img); button.append(holder);
      const label = document.createElement('span'); label.className = 'chart-letter'; label.textContent = letter; button.append(label);
      if (MOTION_LETTERS.includes(letter)) {
        button.classList.add('is-motion'); button.disabled = true;
        button.setAttribute('aria-label', `${letter} requires motion and cannot be practised here`);
        const tag = document.createElement('span'); tag.className = 'chart-tag'; tag.textContent = 'motion'; button.append(tag);
      } else button.addEventListener('click', () => setTarget(letter));
      grid.append(button);
    }
  }
  for (const tile of grid.querySelectorAll('[data-letter]')) {
    const letter = tile.dataset.letter;
    tile.classList.toggle('is-target', letter === tryTarget);
    tile.classList.toggle('is-done', solved.has(letter));
    tile.setAttribute('aria-pressed', String(letter === tryTarget));
  }
}
function updateChartLive(letter) {
  for (const tile of $('chartGrid').querySelectorAll('[data-letter]')) tile.classList.toggle('is-live', tile.dataset.letter === letter);
}
function wireBuilder() {
  $('spaceBtn').addEventListener('click', () => { if (builder.text && !builder.text.endsWith(' ')) builder.edit((s) => `${s} `); });
  $('backBtn').addEventListener('click', () => builder.edit((s) => s.slice(0, -1)));
  $('clearBtn').addEventListener('click', () => builder.clear());
  $('tryNext').addEventListener('click', chooseNext);
}

registerMode({
  id: 'sign', order: 3, title: 'Sign language', sub: (s) => s.sign === 'spell' ? 'ASL fingerspelling, letters A to Z' : 'Seven static hand gestures',
  empty: 'Spells words from ASL hand signs.', panel: 'insightSign',
  onEnter() { resetSpell(); }, onLeave() { resetTryProgress(); },
  onReset() { resetSpell(); builder.clear(); },
  onResult(dets) {
    const best = dets.reduce((a, d) => !a || d.confidence > a.confidence ? d : a, null);
    const spell = state.sign === 'spell';
    $('signLabel').textContent = spell ? 'Current letter' : 'Current gesture';
    $('tryCard').classList.toggle('hidden', !spell);
    $('speller').classList.toggle('hidden', !spell);
    $('signMotionNote').classList.toggle('hidden', !spell);
    const main = $('signMain');
    main.textContent = best ? best.class : '–';
    $('signConf').textContent = best ? `${Math.round((Number(best.confidence) || 0) * 100)}% confidence${best.low_confidence ? ' (low)' : ''}` : 'No hand in view';
    main.dataset.long = String(main.textContent.length > 2); main.classList.toggle('is-empty', !best);
    if (spell) feedSpell(best, performance.now() / 1000);
  },
});
on('camera', ({ status }) => { if (!trySucceeded && !['live', 'paused'].includes(status)) resetTryProgress(); });
wireBuilder(); renderChart(); renderTry(); builder.render(); setRing(0); resetTryProgress();
