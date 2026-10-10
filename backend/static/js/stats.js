/** Live inference performance counters and sparkline. */
import { $ } from './util.js';
import { on } from './store.js';
import { SPARK_POINTS } from './config.js';
let times = [], series = [];
export function initStats() { on('camera', ({ status }) => { if (status === 'starting' || status === 'off') resetStats(); }); resetStats(); }
export function resetStats() {
  times = []; series = []; $('fps').textContent = '–'; $('sparkLine').setAttribute('points', '');
  $('latency').textContent = '–'; $('latencyStat').title = 'Round trip per frame'; $('modelName').textContent = '–'; $('deviceText').textContent = 'CPU';
}
export function recordFrame(t1, roundTrip, data) {
  times.push(t1); if (times.length > 9) times.shift();
  if (times.length > 1 && t1 > times[0]) {
    const fps = (times.length - 1) * 1000 / (t1 - times[0]); $('fps').textContent = fps.toFixed(1);
    series.push(fps); if (series.length > SPARK_POINTS) series.shift();
    const top = Math.max(5, ...series) * 1.15;
    $('sparkLine').setAttribute('points', series.map((v, i) => `${(i / (SPARK_POINTS - 1) * 80).toFixed(1)},${(21 - v / top * 20).toFixed(1)}`).join(' '));
  }
  recordLatency(roundTrip, data);
}
export function recordLatency(roundTrip, data = {}) {
  $('latency').textContent = `${Math.round(roundTrip)} ms`;
  $('latencyStat').title = `Round trip ${Math.round(roundTrip)} ms, model ${data.inference_ms != null ? Math.round(data.inference_ms) : '?'} ms`;
  if (data.model_used) $('modelName').textContent = data.model_used;
  $('deviceText').textContent = String(data.device || 'cpu').toUpperCase();
}
