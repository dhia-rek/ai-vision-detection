/** App shell: mode radios, theme, help, keyboard shortcuts, URL and camera-state presentation. */
import { $ } from './util.js';
import { state, on, emit } from './store.js';
import { listModes, getMode, switchMode, showPanel } from './modes.js';
import { initMedia, startCamera, stopCamera, togglePause, flipCamera } from './media.js';
import { initStats, resetStats } from './stats.js';
import { initOverlay } from './overlay.js';
import { initSnapshot, takeSnapshot } from './snapshot.js';
import './sign.js';

const body = document.body, root = document.documentElement;
const modes = listModes();
const cameraCopy = {
  blocked: ['The camera is blocked', 'Your browser is not letting this page use the camera. Allow it for this site, then try again.'],
  nocamera: ['No camera found', 'Connect a webcam, check that another app is not hiding it, then try again.'],
  busy: ['The camera is busy', 'Another app or tab may be using it. Close it and try again.'],
  insecure: ['The camera needs a secure page', 'Browsers only allow camera access on HTTPS or localhost.'],
  error: ['Could not start the camera', ''],
};

function setRadios(group, attr, value) {
  for (const button of group.querySelectorAll('[role="radio"]')) {
    const active = button.dataset[attr] === value;
    button.setAttribute('aria-checked', String(active)); button.tabIndex = active ? 0 : -1;
  }
}
function wireRadios(group, attr, onPick) {
  const buttons = [...group.querySelectorAll('[role="radio"]')];
  buttons.forEach((button) => button.addEventListener('click', () => onPick(button.dataset[attr])));
  group.addEventListener('keydown', (event) => {
    const direction = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (!direction) return;
    event.preventDefault();
    const i = buttons.indexOf(document.activeElement);
    const next = buttons[(Math.max(0, i) + direction + buttons.length) % buttons.length];
    next.focus(); onPick(next.dataset[attr]);
  });
}
function syncUrl() {
  const params = new URLSearchParams(location.search);
  params.set('mode', state.mode);
  if (state.mode === 'sign') params.set('sign_mode', state.sign); else params.delete('sign_mode');
  try { history.replaceState(null, '', `${location.pathname}?${params}${location.hash}`); } catch { /* Sandboxed frames may deny history access. */ }
}
function syncMode() {
  const mode = getMode(state.mode);
  if (!mode) return;
  body.dataset.mode = state.mode;
  $('panel').dataset.view = state.mode;
  $('panel').dataset.sign = state.sign;
  $('panelTitle').textContent = mode.title;
  $('panelSub').textContent = typeof mode.sub === 'function' ? mode.sub(state) : mode.sub;
  $('emptyTitle').textContent = mode.title;
  $('emptyText').textContent = mode.empty;
  for (const item of modes) $(item.panel).classList.toggle('hidden', item.id !== mode.id);
  showPanel(mode.id);
  $('signSwitch').classList.toggle('hidden', state.mode !== 'sign');
  setRadios($('modeSwitch'), 'mode', state.mode);
  setRadios($('signSwitch'), 'sign', state.sign);
  $('tryCard').classList.toggle('hidden', state.mode !== 'sign' || state.sign !== 'spell');
  $('speller').classList.toggle('hidden', state.mode !== 'sign' || state.sign !== 'spell');
  $('signMotionNote').classList.toggle('hidden', state.mode !== 'sign' || state.sign !== 'spell');
  syncUrl();
}
function syncCamera(status) {
  const pill = $('statusPill'), labels = {
    off: ['Camera off', 'idle'], starting: ['Starting camera', 'busy'], live: ['Live', 'running'],
    paused: ['Paused', 'paused'], blocked: ['Camera blocked', 'error'], nocamera: ['Camera off', 'error'],
    busy: ['Camera off', 'error'], insecure: ['Camera off', 'error'], error: ['Camera off', 'error'],
  };
  const [text, kind] = labels[status] || labels.off;
  $('statusText').textContent = text; pill.dataset.kind = kind;
  const starting = status === 'starting';
  // While flipping cameras the last frame stays up, so the empty card must not cover it.
  $('empty').classList.toggle('hidden', !(status === 'off' || (starting && document.body.dataset.media !== 'live')));
  $('startBtn').disabled = starting;
  $('startBtn').querySelector('span').textContent = starting ? 'Starting camera' : 'Start camera';
  const help = cameraCopy[status];
  $('camHelp').classList.toggle('hidden', !help);
  if (help) {
    $('camHelp').dataset.kind = status;
    $('cameraHelpTitle').textContent = help[0]; $('helpText').textContent = status === 'error' ? (document.body.dataset.cameraMessage || help[1]) : help[1];
    const showSteps = status === 'blocked';
    $('helpSteps').classList.toggle('hidden', !showSteps);
    for (const item of $('helpSteps').querySelectorAll('.howto')) item.open = false;
    if (showSteps) {
      const ua = navigator.userAgent.toLowerCase();
      const browser = /iphone|ipad|ipod/.test(ua) ? 'ios' : /android/.test(ua) ? 'android' : /firefox/.test(ua) ? 'firefox' : /safari/.test(ua) && !/chrome|chromium|android/.test(ua) ? 'safari' : 'chrome';
      const howto = $('helpSteps').querySelector(`[data-browser="${browser}"]`);
      if (howto) howto.open = true;
      if (window.top !== window.self) { $('helpText').textContent += ' If this page is embedded, the host page must allow the camera (allow="camera").'; $('openFull').classList.remove('hidden'); }
    }
    requestAnimationFrame(() => $('helpTitle').focus({ preventScroll: true }));
  }
  const live = status === 'live' || status === 'paused';
  document.body.dataset.media = live || (starting && document.body.dataset.media === 'live') ? 'live' : 'none';
  $('playBtn').dataset.state = status === 'paused' ? 'paused' : 'live';
  $('playLabel').textContent = status === 'paused' ? 'Resume' : 'Pause';
  $('stopBtn').disabled = !live && status !== 'starting';
  $('flipBtn').disabled = !live;
}
function setTheme(theme, persist = false) {
  if (!['light', 'dark'].includes(theme)) theme = 'light';
  root.dataset.theme = theme;
  $('themeToggle').setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  if (persist) { try { localStorage.setItem('avd-theme', theme); } catch { /* Storage can be disabled. */ } }
  emit('theme', { theme });
}
function toggleTheme() { setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark', true); }
function toggleOverlay() {
  state.overlay = !state.overlay;
  emit('overlay', { overlay: state.overlay });
}
function typingTarget(target) { return !!target.closest('input, textarea, select, [contenteditable="true"]'); }
function handleKeys(event) {
  if (event.ctrlKey || event.metaKey || event.altKey || $('helpDialog').open || typingTarget(event.target)) return;
  if (/^[1-9]$/.test(event.key)) {
    const mode = modes[Number(event.key) - 1]; if (mode) switchMode(mode.id); return;
  }
  const onControl = !!event.target.closest('button, a, [role="radio"]');
  switch (event.key) {
    case ' ': if (!onControl) { event.preventDefault(); void togglePause(); } break;
    case 'f': case 'F': void flipCamera(); break;
    case 'o': case 'O': toggleOverlay(); break;
    case 's': case 'S': takeSnapshot(); break;
    case 't': case 'T': toggleTheme(); break;
    case '?': $('helpDialog').showModal(); break;
    case 'Backspace': if (state.mode === 'sign' && !onControl) { event.preventDefault(); $('backBtn').click(); } break;
    default: break;
  }
}
function initHelp() {
  const dialog = $('helpDialog');
  $('helpBtn').addEventListener('click', () => dialog.showModal());
  dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
}
function init() {
  const params = new URLSearchParams(location.search);
  const wanted = params.get('mode'); if (getMode(wanted)) state.mode = wanted;
  if (['spell', 'gestures'].includes(params.get('sign_mode'))) state.sign = params.get('sign_mode');
  state.embed = params.get('embed') === '1';
  root.dataset.embed = String(state.embed);
  if (state.embed) { const full = new URL(location.href); full.searchParams.delete('embed'); $('openFull').href = full.href; $('openFull').classList.remove('hidden'); }
  let theme = params.get('theme');
  if (!['light', 'dark'].includes(theme)) { try { theme = localStorage.getItem('avd-theme'); } catch { theme = null; } }
  setTheme(theme || 'light');
  initOverlay({ viewport: $('viewport'), canvas: $('overlay') });
  initSnapshot({ button: $('snapBtn'), video: $('video') });
  initStats(); resetStats();
  initMedia();
  wireRadios($('modeSwitch'), 'mode', (id) => { if (getMode(id)) switchMode(id); });
  wireRadios($('signSwitch'), 'sign', (sign) => switchMode('sign', { sign }));
  $('startBtn').addEventListener('click', () => void startCamera());
  $('retryBtn').addEventListener('click', () => void startCamera());
  $('stopBtn').addEventListener('click', stopCamera);
  $('playBtn').addEventListener('click', () => void togglePause());
  $('flipBtn').addEventListener('click', () => void flipCamera());
  $('overlayToggle').addEventListener('click', toggleOverlay);
  $('themeToggle').addEventListener('click', toggleTheme);
  $('errorClose').addEventListener('click', () => $('errorBox').classList.add('hidden'));
  initHelp(); document.addEventListener('keydown', handleKeys);
  on('mode', syncMode); on('camera', ({ status }) => syncCamera(status));
  on('media', ({ shown, w, h }) => { document.body.dataset.media = shown ? 'live' : 'none'; if (shown && w && h) $('viewport').style.setProperty('--ar', `${w} / ${h}`); });
  on('overlay', ({ overlay }) => { state.overlay = overlay; $('overlayToggle').setAttribute('aria-pressed', String(overlay)); $('viewport').classList.toggle('overlay-off', !overlay); });
  $('overlayToggle').setAttribute('aria-pressed', String(state.overlay));
  $('viewport').classList.toggle('overlay-off', !state.overlay);
  syncMode(); syncCamera(state.camera);
}
init();
