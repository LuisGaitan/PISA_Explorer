/* PISA Explorer chart library — shared by the chat UI and the admin dashboard.
 * Plain SVG built with DOM APIs (textContent only — labels are untrusted data).
 * Forms: bars with 95%-CI whiskers (league-table mode past 12 rows), dumbbells
 * for cross-cycle change (2018 → 2022 → 2025, any two or three cycles),
 * diverging bars around zero, heatmaps for crosstabs.
 * Tokens (--series1, --ink2, …) come from the host page's CSS.
 */
"use strict";

const METRICS = new Set(["estimate", "se", "n_pv", "change", "se_change", "cycle", "rank"]);
const CYCLE_COL = /^(estimate|se)_(\d{4})$/;          // estimate_2018, se_2025, …
const isMetric = c => METRICS.has(c) || CYCLE_COL.test(c);
/* Oldest → newest cycle: soft, base, strong blue (sequential = time order). */
const CYCLE_COLORS = {
  2: ["var(--series1-soft)", "var(--series1)"],
  3: ["var(--series1-soft)", "var(--series1)", "var(--series1-strong)"],
};

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}
const svgNS = "http://www.w3.org/2000/svg";
function sv(tag, attrs) {
  const node = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  return node;
}
const fmt = (x, d = 1) => (x === null || x === undefined || Number.isNaN(+x))
  ? "–" : (+x).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: 0 });

/* ---------- tooltip (one shared element, created on demand) ---------- */
let $tip = null;
function tipEl() {
  if (!$tip) {
    $tip = el("div"); $tip.id = "tooltip";
    document.body.appendChild($tip);
  }
  return $tip;
}
function showTip(evt, valueText, labelText, extraLines) {
  const t = tipEl();
  t.replaceChildren();
  t.appendChild(el("div", "val", valueText));
  t.appendChild(el("div", "lab", labelText));
  for (const line of extraLines || []) t.appendChild(el("div", "lab", line));
  t.style.display = "block";
  const pad = 14, w = t.offsetWidth;
  let x = evt.clientX + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  t.style.left = x + "px";
  t.style.top = Math.max(8, evt.clientY - 10) + "px";
}
function hideTip() { if ($tip) $tip.style.display = "none"; }

/* ---------- scales & ticks ---------- */
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let t = start; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}

function roundedBar(x0, x1, y, h) {   // square at baseline x0, 4px-round data end
  const r = Math.min(4, Math.abs(x1 - x0), h / 2), right = x1 > x0;
  if (Math.abs(x1 - x0) < 0.5) return "";
  return right
    ? `M${x0},${y} H${x1 - r} A${r},${r} 0 0 1 ${x1},${y + r} V${y + h - r} A${r},${r} 0 0 1 ${x1 - r},${y + h} H${x0} Z`
    : `M${x0},${y} H${x1 + r} A${r},${r} 0 0 0 ${x1},${y + r} V${y + h - r} A${r},${r} 0 0 0 ${x1 + r},${y + h} H${x0} Z`;
}

/* ---------- chart: horizontal bars with 95% CI whiskers ---------- */
function barChart(rows, catCols, valueKey, seKey, title, opts = {}) {
  const league = rows.length > 12;
  if (league) rows = [...rows].sort((a, b) => (+b[valueKey]) - (+a[valueKey]));
  const cats = rows.map(r => catCols.map(c => r[c]).join(" · "));
  const vals = rows.map(r => +r[valueKey]);
  const ses  = rows.map(r => seKey ? +r[seKey] || 0 : 0);
  let lo = Math.min(0, ...vals.map((v, i) => v - 1.96 * ses[i]));
  let hi = Math.max(0, ...vals.map((v, i) => v + 1.96 * ses[i]));
  if (lo < 0) lo -= (hi - lo) * 0.09;
  const diverging = vals.some(v => v < 0) && vals.some(v => v > 0) || vals.every(v => v < 0);

  const W = 860, labelW = opts.labelW || 150, padR = 60, axisH = 26, padT = 6;
  const rowH = league ? 22 : 30, barH = league ? 15 : 20;
  const H = padT + rows.length * rowH + axisH;
  const x = v => labelW + (v - lo) / (hi - lo || 1) * (W - labelW - padR);
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}` });
  svg.setAttribute("role", "img");

  let ticks = niceTicks(lo, hi);
  if (opts.integer) {   // counts: whole-number ticks only
    ticks = [...new Set(ticks.map(Math.round))].filter(t => t >= lo && t <= hi);
    if (ticks.length < 2) ticks = [0, Math.max(1, Math.round(hi))];
  }
  for (const t of ticks) {
    const g = sv("line", { x1: x(t), x2: x(t), y1: padT, y2: H - axisH, "stroke-width": 1 });
    g.style.stroke = "var(--grid)";
    svg.appendChild(g);
    const tt = sv("text", { x: x(t), y: H - axisH + 16, "text-anchor": "middle", class: "ticktext" });
    tt.style.fill = "var(--muted)"; tt.textContent = fmt(t, 2);
    svg.appendChild(tt);
  }
  const zero = sv("line", { x1: x(0), x2: x(0), y1: padT, y2: H - axisH, "stroke-width": 1 });
  zero.style.stroke = "var(--baseline)";
  svg.appendChild(zero);

  rows.forEach((r, i) => {
    const y = padT + i * rowH + (rowH - barH) / 2;
    const v = vals[i], se = ses[i];
    const isRef = cats[i] === "OECD avg";
    const color = isRef ? "var(--muted)"
      : diverging ? (v >= 0 ? "var(--series1)" : "var(--neg)") : "var(--series1)";

    const label = sv("text", { x: labelW - 8, y: y + barH / 2 + 4, "text-anchor": "end" });
    label.style.fill = "var(--ink2)";
    if (isRef) label.style.fontWeight = "650";
    label.textContent = cats[i].length > 26 ? cats[i].slice(0, 25) + "…" : cats[i];
    svg.appendChild(label);

    const bar = sv("path", { d: roundedBar(x(0), x(v), y, barH) });
    bar.style.fill = color;
    svg.appendChild(bar);

    if (se > 0) {
      const a = x(v - 1.96 * se), b = x(v + 1.96 * se), cy = y + barH / 2;
      for (const [x1, x2, y1, y2] of [[a, b, cy, cy], [a, a, cy - 4, cy + 4], [b, b, cy - 4, cy + 4]]) {
        const wline = sv("line", { x1, x2, y1, y2, "stroke-width": 1.5 });
        wline.style.stroke = "var(--ink2)";
        svg.appendChild(wline);
      }
    }
    if (rows.length <= 12) {
      const outward = v >= 0;
      const tipX = outward ? Math.max(x(v), x(v + 1.96 * se)) + 6
                           : Math.min(x(v), x(v - 1.96 * se)) - 6;
      const fits = outward ? tipX <= W - 4 : tipX >= labelW + 36;
      if (fits) {
        const vt = sv("text", { x: tipX, y: y + barH / 2 + 4, class: "ticktext",
                                "text-anchor": outward ? "start" : "end" });
        vt.style.fill = "var(--ink2)"; vt.textContent = fmt(v, opts.decimals ?? 1);
        svg.appendChild(vt);
      }
    }
    const hit = sv("rect", { x: 0, y: padT + i * rowH, width: W, height: rowH, fill: "transparent" });
    hit.addEventListener("pointermove", e => {
      bar.style.filter = "brightness(1.12)";
      showTip(e, fmt(v, 2), cats[i],
        se > 0 ? [`SE ${fmt(se, 2)} · 95% CI [${fmt(v - 1.96 * se)}, ${fmt(v + 1.96 * se)}]`] : []);
    });
    hit.addEventListener("pointerleave", () => { bar.style.filter = ""; hideTip(); });
    svg.appendChild(hit);
  });

  return wrapChart(svg, title, "bar");
}

/* ---------- chart: dumbbell / connected dots across cycles ---------- */
function dumbbellChart(rows, catCols, cycles, title) {
  const cats = rows.map(r => catCols.map(c => r[c]).join(" · "));
  const val = (r, c) => { const v = r[`estimate_${c}`]; return v === null || v === undefined ? NaN : +v; };
  const se  = (r, c) => +r[`se_${c}`] || 0;
  const all = [];
  for (const r of rows) for (const c of cycles) {
    const v = val(r, c);
    if (Number.isFinite(v)) all.push(v - 1.96 * se(r, c), v + 1.96 * se(r, c));
  }
  const lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.06 || 1;
  const colors = CYCLE_COLORS[cycles.length] || CYCLE_COLORS[3];
  const first = cycles[0], last = cycles[cycles.length - 1];

  const W = 860, labelW = 150, padR = 56, rowH = 32, axisH = 26, padT = 6;
  const H = padT + rows.length * rowH + axisH;
  const x = v => labelW + (v - (lo - pad)) / ((hi + pad) - (lo - pad)) * (W - labelW - padR);
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}` });

  for (const t of niceTicks(lo - pad, hi + pad)) {
    const g = sv("line", { x1: x(t), x2: x(t), y1: padT, y2: H - axisH, "stroke-width": 1 });
    g.style.stroke = "var(--grid)"; svg.appendChild(g);
    const tt = sv("text", { x: x(t), y: H - axisH + 16, "text-anchor": "middle", class: "ticktext" });
    tt.style.fill = "var(--muted)"; tt.textContent = fmt(t, 0);
    svg.appendChild(tt);
  }

  rows.forEach((r, i) => {
    const cy = padT + i * rowH + rowH / 2;
    const label = sv("text", { x: labelW - 8, y: cy + 4, "text-anchor": "end" });
    label.style.fill = "var(--ink2)"; label.textContent = cats[i];
    svg.appendChild(label);

    const present = cycles.filter(c => Number.isFinite(val(r, c)));
    const vs = present.map(c => val(r, c));
    if (vs.length >= 2) {
      const conn = sv("line", { x1: x(Math.min(...vs)), x2: x(Math.max(...vs)), y1: cy, y2: cy, "stroke-width": 1.5 });
      conn.style.stroke = "var(--baseline)";
      svg.appendChild(conn);
    }
    // draw newest last so it sits on top when dots overlap
    present.forEach(c => {
      const dot = sv("circle", { cx: x(val(r, c)), cy, r: 5.5, "stroke-width": 2 });
      dot.style.fill = colors[cycles.indexOf(c)]; dot.style.stroke = "var(--surface)";
      svg.appendChild(dot);
    });
    const vLast = val(r, last), vFirst = val(r, first);
    if (rows.length <= 8 && Number.isFinite(vLast)) {
      const outer = !Number.isFinite(vFirst) || vLast >= vFirst;
      const vt = sv("text", { x: x(vLast) + (outer ? 10 : -10), y: cy + 4,
                              "text-anchor": outer ? "start" : "end", class: "ticktext" });
      vt.style.fill = "var(--ink2)"; vt.textContent = fmt(vLast);
      svg.appendChild(vt);
    }
    const change = r.change, seCh = r.se_change;
    const hit = sv("rect", { x: 0, y: padT + i * rowH, width: W, height: rowH, fill: "transparent" });
    hit.addEventListener("pointermove", e => {
      const lines = cycles.map(c => Number.isFinite(val(r, c))
        ? `${c}: ${fmt(val(r, c), 2)} (SE ${fmt(se(r, c), 2)})` : `${c}: no estimate`);
      if (change !== undefined && change !== null && Number.isFinite(+change)) {
        const sig = Math.abs(+change) > 1.96 * (+seCh || 0) ? "significant" : "not significant";
        lines.push(`change ${first} → ${last}: ${(+change >= 0 ? "+" : "") + fmt(change, 2)} (SE ${fmt(seCh, 2)}) — ${sig}`);
      }
      showTip(e, cats[i], cycles.join(" → "), lines);
    });
    hit.addEventListener("pointerleave", hideTip);
    svg.appendChild(hit);
  });

  const wrap = wrapChart(svg, title, "dumbbell");
  const legend = el("div", "legend");
  cycles.forEach((c, k) => {
    const item = el("span");
    const dot = el("span", "dot"); dot.style.background = colors[k];
    item.appendChild(dot); item.appendChild(document.createTextNode(`PISA ${c}`));
    legend.appendChild(item);
  });
  wrap.insertBefore(legend, wrap.querySelector(".chartwrap"));
  return wrap;
}

/* ---------- chart: heatmap for weighted crosstabs ---------- */
const RAMP_LO = [0xcd, 0xe2, 0xfb], RAMP_HI = [0x10, 0x42, 0x81];  // sequential blue
function rampColor(t) {
  const c = RAMP_LO.map((lo, i) => Math.round(lo + (RAMP_HI[i] - lo) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function heatmapChart(rows, title) {
  const rlab = r => String(r.row_label ?? r.row);
  const clab = r => String(r.col_label ?? r.col);
  const rowCats = [...new Set(rows.map(rlab))];
  const colCats = [...new Set(rows.map(clab))];
  const vals = rows.map(r => +r.estimate);
  const vmin = Math.min(...vals), vmax = Math.max(...vals);

  const W = 860, labelW = 190, headH = 26, cellH = 34, axisPad = 6;
  const cellW = (W - labelW - 10) / colCats.length;
  const H = headH + rowCats.length * cellH + axisPad;
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}` });

  colCats.forEach((c, j) => {
    const t = sv("text", { x: labelW + j * cellW + cellW / 2, y: headH - 8, "text-anchor": "middle" });
    t.style.fill = "var(--ink2)";
    t.textContent = c.length > 22 ? c.slice(0, 21) + "…" : c;
    svg.appendChild(t);
  });
  rowCats.forEach((r, i) => {
    const t = sv("text", { x: labelW - 8, y: headH + i * cellH + cellH / 2 + 4, "text-anchor": "end" });
    t.style.fill = "var(--ink2)";
    t.textContent = r.length > 28 ? r.slice(0, 27) + "…" : r;
    svg.appendChild(t);
  });

  for (const cell of rows) {
    const i = rowCats.indexOf(rlab(cell)), j = colCats.indexOf(clab(cell));
    if (i < 0 || j < 0) continue;
    const v = +cell.estimate;
    const t = vmax > vmin ? (v - vmin) / (vmax - vmin) : 0.5;
    const x = labelW + j * cellW, y = headH + i * cellH;
    const rect = sv("rect", { x: x + 1, y: y + 1, width: cellW - 2, height: cellH - 2, rx: 3, fill: rampColor(t) });
    svg.appendChild(rect);
    const label = sv("text", { x: x + cellW / 2, y: y + cellH / 2 + 4, "text-anchor": "middle", class: "ticktext" });
    label.setAttribute("fill", t > 0.55 ? "#ffffff" : "#0b0b0b");
    label.textContent = fmt(v);
    svg.appendChild(label);
    const hit = sv("rect", { x, y, width: cellW, height: cellH, fill: "transparent" });
    hit.addEventListener("pointermove", e => {
      rect.style.filter = "brightness(1.12)";
      showTip(e, fmt(v, 2) + "%", `${rlab(cell)} × ${clab(cell)}`,
              cell.se != null ? [`SE ${fmt(+cell.se, 2)}`] : []);
    });
    hit.addEventListener("pointerleave", () => { rect.style.filter = ""; hideTip(); });
    svg.appendChild(hit);
  }
  return wrapChart(svg, title, "heatmap");
}

/* Charts keep a minimum drawn width and scroll sideways on narrow screens,
 * so text stays legible on phones instead of shrinking to nothing. */
function wrapChart(svg, title, kind) {
  const card = el("div", "card chart");
  card.dataset.chart = kind;
  if (title) card.appendChild(el("p", "chart-title", title));
  const scroller = el("div", "chartwrap");
  scroller.appendChild(svg);
  card.appendChild(scroller);
  return card;
}

/* Pick the form from the result's shape. Returns a card element or null.
 * The returned card carries data-chart = bar | league | dumbbell | heatmap. */
function buildChart(table, plan) {
  if (!table || !table.rows.length) return null;
  const cols = table.columns;
  const catCols = cols.filter(c => !isMetric(c));
  if (!catCols.length && table.rows.length > 1) return null;
  const explain = (plan && plan.explanation) || "";
  // note: +null coerces to 0, so a null check must come first
  const finite = keys => table.rows.filter(
    r => keys.every(k => r[k] !== null && r[k] !== undefined && Number.isFinite(+r[k])));
  if (cols.includes("row") && cols.includes("col") && cols.includes("estimate")) {
    const rows = finite(["estimate"]);
    return rows.length && rows.length <= 40
      ? heatmapChart(rows, explain + "  (row percentages; hover for SE)") : null;
  }
  const cycles = cols.filter(c => /^estimate_\d{4}$/.test(c)).map(c => c.slice(9)).sort();
  if (cycles.length >= 2) {
    // a row needs at least two cycles to draw a change; single-cycle rows stay in the table
    const rows = table.rows.filter(r => cycles.filter(
      c => r[`estimate_${c}`] !== null && r[`estimate_${c}`] !== undefined
           && Number.isFinite(+r[`estimate_${c}`])).length >= 2);
    return rows.length && rows.length <= 40
      ? dumbbellChart(rows, catCols.length ? catCols : cols.slice(0, 1), cycles, explain) : null;
  }
  if (cols.includes("estimate")) {
    const rows = finite(["estimate"]);
    if (!catCols.length || !rows.length || rows.length > 90) return null;
    const card = barChart(rows, catCols, "estimate", "se",
                          explain + "  (whiskers: 95% confidence interval)");
    if (rows.length > 12) card.dataset.chart = "league";
    return card;
  }
  return null;
}

/* Simple counts → bars, for the admin dashboard (no SEs). */
function countBars(pairs, title, decimals = 0) {
  const rows = pairs.map(([label, n]) => ({ label: String(label), estimate: +n }));
  if (!rows.length) return el("div", "card muted", "no data yet");
  const card = barChart(rows, ["label"], "estimate", null, title,
                        { labelW: 220, decimals, integer: decimals === 0 });
  card.dataset.chart = "count";
  return card;
}
