/** Accessible transient notifications. */
import { $, h, svgUse } from './util.js';
const visible = [];
export function toast(message, { kind = 'info', duration = 3200 } = {}) {
  const host = $('toasts'); if (!host) return;
  while (visible.length >= 3) remove(visible[0]);
  const icon = kind === 'success' ? 'check' : kind === 'error' ? 'alert' : 'activity';
  const node = h('div', 'toast'); node.dataset.kind = ['info', 'success', 'error'].includes(kind) ? kind : 'info';
  node.append(svgUse(icon), h('span', 'toast-text', String(message))); host.appendChild(node); visible.push(node);
  let timer = setTimeout(() => dismiss(node), Math.max(0, duration));
  node.addEventListener('pointerenter', () => clearTimeout(timer));
  node.addEventListener('pointerleave', () => { timer = setTimeout(() => dismiss(node), 500); });
  node.addEventListener('animationend', (event) => { if (node.classList.contains('is-leaving') && event.target === node) remove(node); });
}
function dismiss(node) {
  if (!node?.isConnected) return remove(node);
  node.classList.add('is-leaving'); setTimeout(() => remove(node), 450);
}
function remove(node) { if (!node) return; node.remove(); const i = visible.indexOf(node); if (i >= 0) visible.splice(i, 1); }
