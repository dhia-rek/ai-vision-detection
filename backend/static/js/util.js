/** Small DOM and maths helpers shared by every module. */

export const $ = (id) => document.getElementById(id);
export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const pct = (v) => `${Math.round((Number(v) || 0) * 100)}%`;
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Whether the user asked for less motion (JS-driven animations must honour this like the CSS does). */
export function reducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Create an element with an optional class name and text. */
export function h(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

/** An inline icon that points at a symbol of the sprite in index.html (`#i-<name>`). */
export function svgUse(name) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('class', 'icon');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(ns, 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.appendChild(use);
  return svg;
}

/** FNV-1a hash of a string, so a class name always maps to the same palette slot. */
export function hash(s) {
  let x = 2166136261;
  for (let i = 0; i < s.length; i++) { x ^= s.charCodeAt(i); x = Math.imul(x, 16777619); }
  return x >>> 0;
}

/**
 * Animate a number inside `el` from what it shows now to `to` (ease-out, ~450 ms).
 * Sets the final text immediately under reduced motion. A newer call cancels the running one.
 * @param {HTMLElement} el target element (its textContent is replaced)
 * @param {number} to final value
 * @param {{decimals?: number, suffix?: string, duration?: number}} [opts]
 */
export function countUp(el, to, opts = {}) {
  const { decimals = 0, suffix = '', duration = 450 } = opts;
  const text = (v) => `${v.toFixed(decimals)}${suffix}`;
  cancelAnimationFrame(el._countRaf || 0);
  const from = Number.isFinite(el._countValue) ? el._countValue : 0;
  el._countValue = to;
  if (reducedMotion() || from === to || duration <= 0) { el.textContent = text(to); return; }
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / duration);
    const e = k === 1 ? 1 : 1 - 2 ** (-10 * k);
    el.textContent = text(from + (to - from) * e);
    if (k < 1) el._countRaf = requestAnimationFrame(step);
  };
  el._countRaf = requestAnimationFrame(step);
}
