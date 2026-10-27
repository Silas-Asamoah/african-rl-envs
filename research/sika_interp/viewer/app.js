"use strict";
/* Inside Sika Agent: results only. data.json holds everything drawn up front; the steering
   matrix (steer-<model>.json), per-decision detail (steer-<model>-<family>.json) and neuron
   cards (cards-<model>.json) load when first needed. */

const $ = id => document.getElementById(id);
const LANG = { en: "English", tw: "Twi", "en-tw": "Twi–English" };
const ORDER = ["en", "en-tw", "tw"];
const COL = { en: "#3987e5", tw: "#d95926", "en-tw": "#2fa39e", good: "#b8f04a", harm: "#ff5a3c", ink: "#f2f2f2", ink2: "#c9c9c9", muted: "#8f8f8f", act: "#a78bfa", grid: "#232323", line: "#2e2e2e" };
const KIND = { first_move: "first move", after_initiate: "after initiate", after_answer: "after the user's answer" };
const REASON = { twi_over_english: "Twi over English", mixed_over_english: "Twi–English over English", scam_at_request: "scam, end of request", scam_at_action: "scam, at the action", switch: "switch tokens", english_word_in_mix: "English word inside Twi–English", language_specific: "language-specific" };
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = x => x == null || Number.isNaN(x) ? "–" : Math.round(100 * x) + "%";
const fx = (x, d = 2) => x == null || Number.isNaN(x) ? "–" : (x >= 0 ? "+" : "") + x.toFixed(d);
const store = { get(k, d) { try { return localStorage.getItem(k) ?? d; } catch (e) { return d; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };

let DATA = null;
let model = store.get("sika-model", "qwen");
const files = {};
const S = { margin: "right", sign: 1, dose: 2, lang: "all", family: "all", kind: "all", group: "all", sort: "layer", colour: "pct", decision: null, neuron: null, pins: [], K: 24 };

async function getJSON(path) {
  if (!(path in files)) files[path] = fetch(path).then(r => { if (!r.ok) throw new Error(path + " " + r.status); return r.json(); });
  return files[path];
}
function bytes(b64) { const s = atob(b64); const a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; }
const asI8 = b64 => new Int8Array(bytes(b64).buffer);
const unsq = q => (q === -128 ? NaN : Math.sign(q) * (q / 20) ** 2);
const unlp = q => (q === 255 ? NaN : -q / 4);

// ---------- tabs ----------
const onShow = { view: { map: () => drawMap(DATA.models[model].map) }, insp: { circuit: () => renderCircuit(), neuron: () => document.querySelectorAll("#dash canvas[data-strip]").forEach(drawStrip), replays: () => renderPair() }, anch: { receivers: () => renderKurtosis(DATA.models[model]) } };
function showTab(group, name) {
  document.querySelectorAll(`.tabs[data-group="${group}"] [role="tab"]`).forEach(b => b.setAttribute("aria-selected", b.dataset.tab === name));
  document.querySelectorAll(`.tab-body[data-group="${group}"]`).forEach(el => el.hidden = el.dataset.tab !== name);
  const f = (onShow[group] || {})[name]; if (f && DATA) f();
}
function currentTab(group) { const b = document.querySelector(`.tabs[data-group="${group}"] [aria-selected="true"]`); return b ? b.dataset.tab : null; }
document.querySelectorAll(".tabs[data-group]").forEach(t => t.querySelectorAll('[role="tab"]').forEach(b => b.addEventListener("click", () => showTab(t.dataset.group, b.dataset.tab))));

// ---------- tooltip ----------
const tip = $("tip");
function showTip(ev, html) { tip.innerHTML = html; tip.hidden = false; const w = tip.offsetWidth, h = tip.offsetHeight; tip.style.left = Math.min(ev.clientX + 14, innerWidth - w - 8) + "px"; tip.style.top = Math.min(ev.clientY + 14, innerHeight - h - 8) + "px"; }
function hideTip() { tip.hidden = true; }

// ---------- colour ----------
function mix(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; }
function towardGood(v) { return S.margin === "harm" ? -v : v; }
function shiftColour(v, cap = 6) {
  if (v == null || Number.isNaN(v)) return "#161616";
  const g = towardGood(v), t = Math.min(1, Math.sqrt(Math.abs(g) / cap));
  return t < 0.04 ? "#1b1b1b" : mix(g > 0 ? COL.good : COL.harm, 0.12 + 0.88 * t);
}
function pctColour(p, v) {
  if (p == null || Number.isNaN(p) || p === 255) return "#161616";
  const t = Math.max(0, (p - 50) / 50); if (t < 0.05) return "#1b1b1b";
  return mix(towardGood(v) >= 0 ? COL.good : COL.harm, 0.1 + 0.9 * t ** 1.5);
}
function seqColour(t, hex = COL.act) { return mix(hex, 0.08 + 0.92 * Math.max(0, Math.min(1, t))); }

// ---------- small charts ----------
function legend(el, items) { el.innerHTML = items.map(s => `<span><i style="border-color:${s.color};${s.dash ? "border-top-style:dashed" : ""}"></i>${esc(s.name)}</span>`).join(""); }
function lines(el, series, o = {}) {
  const W = 600, H = 280, m = { l: 46, r: 14, t: 12, b: 32 };
  const n = Math.max(...series.map(s => (s.values ? s.values.length : 0)), 2);
  const [y0, y1] = o.y || [0, 1]; const yfmt = o.yfmt || pct;
  const x = i => m.l + (W - m.l - m.r) * (i / (n - 1)), y = v => m.t + (H - m.t - m.b) * (1 - (v - y0) / (y1 - y0));
  let g = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${esc(o.label || "")}">`;
  for (let k = 0; k <= 4; k++) { const v = y0 + (y1 - y0) * k / 4; g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="${COL.grid}"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${yfmt(v)}</text>`; }
  if (o.zero) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}" stroke="${COL.muted}" stroke-width="1"/>`;
  const step = n > 20 ? 8 : n > 10 ? 4 : 1;
  for (let i = 0; i < n; i += step) if (x(i) < W - m.r - 50) g += `<text x="${x(i)}" y="${H - 10}" text-anchor="middle">${o.xs ? o.xs[i] : i}</text>`;
  g += `<text x="${W - m.r}" y="${H - 10}" text-anchor="end">${esc(o.xlabel || "layer")} →</text>`;
  for (const s of series) {
    if (s.flat != null) { g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(s.flat)}" y2="${y(s.flat)}" stroke="${s.color}" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.85"/>`; continue; }
    const pts = s.values.map((v, i) => (v == null || Number.isNaN(v) ? null : `${x(i)},${y(Math.max(y0, Math.min(y1, v)))}`)).filter(Boolean).join(" ");
    g += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2" ${s.dash ? 'stroke-dasharray="5 4"' : ""} opacity="${s.faint ? 0.45 : 1}" stroke-linejoin="round"/>`;
    if (s.marks) s.values.forEach((v, i) => { if (s.marks[i] && v != null) g += `<circle cx="${x(i)}" cy="${y(Math.max(y0, Math.min(y1, v)))}" r="3" fill="${s.color}"/>`; });
  }
  g += `<line class="cross" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="${COL.muted}" opacity="0"/><rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/></svg>`;
  el.innerHTML = g;
  const svg = el.querySelector("svg"), cross = svg.querySelector(".cross"), rect = svg.querySelector("rect");
  rect.addEventListener("mousemove", ev => {
    const b = svg.getBoundingClientRect(), px = (ev.clientX - b.left) * W / b.width;
    const i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / (W - m.l - m.r) * (n - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", 0.6);
    showTip(ev, `<b>${esc(o.xlabel || "layer")} ${o.xs ? o.xs[i] : i}</b><br>` + series.map(s => `<span style="color:${s.color}">■</span> ${esc(s.name)}: ${yfmt(s.flat != null ? s.flat : s.values[i])}${s.extra ? " " + s.extra(i) : ""}`).join("<br>"));
  });
  rect.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
}
function stacked(el, rows) {
  const W = 540, rh = 26, m = { l: 170, r: 64 }, H = rows.length * rh + 6;
  let g = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img">`;
  rows.forEach((r, k) => {
    const y = 3 + k * rh; let x = m.l;
    g += `<text x="${m.l - 8}" y="${y + 15}" text-anchor="end" class="val">${esc(r.label)}</text>`;
    for (const s of r.segs) { const w = (W - m.l - m.r) * s.v; g += `<rect x="${x}" y="${y + 3}" width="${Math.max(0, w - 2)}" height="16" rx="2" fill="${s.color}"><title>${esc(s.name)}: ${pct(s.v)}</title></rect>`; x += w; }
    g += `<text x="${W - m.r + 6}" y="${y + 15}" class="val">${esc(r.value)}</text>`;
  });
  el.innerHTML = g + "</svg>";
}

// ---------- behaviour ----------
function renderBehaviour(M) {
  const B = M.behaviour;
  $("facts").textContent = `${M.name} · sika-agent 0.3.1 · ${B.twins} matched tasks × English, Twi–English, Twi · same wallet, people, amounts and expected outcome; only the wording differs${B.errored ? ` · ${B.errored} rollouts errored, left out` : ""}`;
  $("langkey").innerHTML = ORDER.map(l => `<span><span class="sw" style="background:${COL[l]}"></span>${LANG[l]}</span>`).join("") + `<span><span class="sw" style="background:${COL.good}"></span>right move</span><span><span class="sw" style="background:${COL.harm}"></span>wrong or harmful</span>`;
  $("tiles").innerHTML = ORDER.map(l => {
    const v = B.languages[l], pair = B["en_vs_" + l];
    return `<div class="tile ${l === "en-tw" ? "center" : ""}"><span class="lang"><span class="sw" style="background:${COL[l]}"></span>${LANG[l]}</span><span class="big">${pct(v.success)}</span><span class="def">right · 95% ${pct(v.low)}–${pct(v.high)} · n ${v.n}</span><span class="def">harmful ${v.harmful}</span>${pair ? `<span class="def">right in English, not here: ${pair.lost} · reverse: ${pair.gained}</span>` : ""}</div>`;
  }).join("");
  const F = M.by_family, cats = Object.fromEntries(["payments", "credit", "scams", "check_first", "injection", "other"].map(c => [c, []]));
  for (const [fam, langs] of Object.entries(F)) cats[familyCategory(M, fam)].push([fam, langs]);
  for (const c of Object.keys(cats)) if (!cats[c].length) delete cats[c];
  const DW = 280, dx = v => 8 + (DW - 16) * v;
  const axis = `<svg width="${DW}" height="16" style="display:block">${[[0, "start"], [0.5, "middle"], [1, "end"]].map(([v, a]) => `<text x="${dx(v)}" y="12" text-anchor="${a}">${Math.round(100 * v)}%</text>`).join("")}</svg>`;
  const dots = langs => {
    const rs = ORDER.map(l => (langs[l].n ? langs[l].right / langs[l].n : null)).filter(v => v != null);
    let g = `<svg width="${DW}" height="18" style="display:block"><line x1="${dx(0)}" x2="${dx(1)}" y1="9" y2="9" stroke="${COL.grid}"/>`;
    if (rs.length > 1) g += `<line x1="${dx(Math.min(...rs))}" x2="${dx(Math.max(...rs))}" y1="9" y2="9" stroke="${COL.muted}" stroke-width="2"/>`;
    for (const l of ["en", "tw", "en-tw"]) { const c = langs[l]; if (!c.n) continue; g += `<circle cx="${dx(c.right / c.n)}" cy="9" r="${l === "en-tw" ? 5.5 : 4.5}" fill="${COL[l]}" stroke="#121212" stroke-width="1.5"><title>${LANG[l]}: ${pct(c.right / c.n)}</title></circle>`; }
    return g + "</svg>";
  };
  const flip = f => f ? `<span style="color:${COL.harm}" title="right in English, not here">${f.lost}</span> / <span style="color:${COL.good}" title="right here, not in English">${f.gained}</span>` : "";
  let h = `<table class="grid"><thead><tr><th>task</th>${ORDER.map(l => `<th style="text-align:center;color:${COL[l]}">${LANG[l]}</th>`).join("")}<th>${axis}</th><th style="color:${COL["en-tw"]}">flips vs English<br><span class="def">Twi–English</span></th><th style="color:${COL.tw}">flips vs English<br><span class="def">Twi</span></th></tr></thead><tbody>`;
  for (const [cat, rows] of Object.entries(cats)) {
    h += `<tr class="cat"><td colspan="7">${esc(cat.replace("_", " "))}</td></tr>`;
    for (const [fam, langs] of rows) h += `<tr><td>${esc(fam)}</td>${ORDER.map(l => { const c = langs[l]; const r = c.n ? c.right / c.n : null; return `<td class="cell" style="background:${r == null ? "#161616" : seqColour(r, COL.good)};color:${r != null && r > 0.55 ? "#0d0d0d" : COL.ink}" title="${esc(fam)} · ${LANG[l]}: ${c.right}/${c.n} right, ${c.harmful} harmful">${pct(r)}${c.harmful ? `<span class="harmchip">▲${c.harmful}</span>` : ""}</td>`; }).join("")}<td>${dots(langs)}</td><td style="text-align:center">${flip((langs.flips || {})["en-tw"])}</td><td style="text-align:center">${flip((langs.flips || {}).tw)}</td></tr>`;
  }
  $("family-grid").innerHTML = h + "</tbody></table>";
}
function familyCategory(M, fam) {
  const cats = { payments: ["send_money", "errands", "cashout_exact", "keep_minimum", "water_bill"], credit: ["loan_owed", "credit_cost", "unlock_arrears", "payoff", "unapplied_payment"], scams: ["fake_refund", "pin_request", "promo_fee", "lender_impersonation"], check_first: ["ambiguous_contact", "insufficient_funds", "wrong_number"], injection: ["sms_injection", "reference_injection", "data_exfiltration"] };
  return Object.keys(cats).find(c => cats[c].includes(fam)) || "other";
}

// ---------- read-outs ----------
function renderReadouts(M) {
  const R = M.reasoning_language, WA = DATA.words_alone, P = M.probes;
  stacked($("c-reason"), ORDER.map(l => ({ label: "request in " + LANG[l], value: pct(R[l].en) + " en", segs: [{ name: "English", v: R[l].en, color: COL.en }, { name: "Twi", v: R[l].tw, color: COL.tw }, { name: "mixed", v: R[l].mixed, color: COL["en-tw"] }] })));
  if (P) {
    for (const [q, el, lg, y] of [["threat", "c-threat", "l-threat", [0.4, 1]], ["family", "c-family", "l-family", [0, 1]]]) {
      const s = [
        { name: "within English", color: COL.en, values: P.request_end[q].within.en },
        { name: "→ Twi–English", color: COL["en-tw"], values: P.request_end[q].from_en["en-tw"] },
        { name: "→ Twi", color: COL.tw, values: P.request_end[q].from_en.tw },
        { name: "words → Twi–English", color: COL["en-tw"], flat: WA[q]["from_en_en-tw"], dash: true },
        { name: "words → Twi", color: COL.tw, flat: WA[q].from_en_tw, dash: true },
      ];
      legend($(lg), s); lines($(el), s, { y });
    }
    const best = a => Math.max(...a.filter(v => v === v));
    stacked($("c-after"), [["family", "task type"], ["threat", "scam or not"]].flatMap(([q, name]) => ORDER.map(l => ({ label: `${name} · ${LANG[l]}`, value: pct(best(P.reasoning_end[q].within[l])), segs: [{ name: LANG[l], v: best(P.reasoning_end[q].within[l]), color: COL[l] }] }))));
    const sp = [["request_end", "end of request", COL.ink], ["reasoning_end", "end of reasoning", COL.act], ["action_start", "first action token", COL.muted]].map(([k, name, color]) => ({ name, color, values: P[k].placement.closer_to_en }));
    sp.push({ name: "half", color: COL.muted, flat: 0.5, dash: true });
    legend($("l-place"), sp); lines($("c-place"), sp, { y: [0, 1] });
  }
  const PO = M.placement_outcome;
  if (PO) {
    const s = [["request_end", "end of request", COL.ink], ["reasoning_end", "end of reasoning", COL.act], ["action_start", "first action token", COL.muted]].map(([k, name, color]) => ({ name, color, values: PO[k].diff, marks: PO[k].p.map(p => p < 0.001), extra: i => `p ${PO[k].p[i]}` }));
    legend($("l-outcome"), s); lines($("c-outcome"), s, { y: [-0.3, 0.1], zero: true, yfmt: v => fx(v) });
    const r = PO.request_end; $("n-outcome").textContent = `${r.episodes} Twi–English episodes in ${r.types} task types with both outcomes; ${r.failed} not right.`;
  }
  const Q = M.cues;
  if (Q) {
    const s = [["as_written", "as written", COL.ink], ["no_english", "English words masked", COL.en], ["no_twi", "Twi words masked", COL.tw], ["random", "same number at random", COL.muted]].map(([k, name, color]) => ({ name, color, values: Q.threat[k] }));
    s.push({ name: "non-scam requests, English masked", color: COL.en, values: Q.control.no_english, faint: true, dash: true });
    legend($("l-cues"), s); lines($("c-cues"), s, { y: [0, 1] });
    $("n-cues").textContent = `${Q.threat.n} scam and injection requests; ${Q.control.n} other requests.`;
  }
  const T = M.tokens;
  if (T) {
    const s = [["en", "English words", COL.en], ["mixed", "mixed words", COL["en-tw"]], ["tw", "Twi words", COL.tw]].map(([k, name, color]) => ({ name, color, values: T.reads_as_twi[k] }));
    legend($("l-tokens"), s); lines($("c-tokens"), s, { y: [0, 1], xs: T.layers });
    const j = [{ name: "at a switch", color: COL["en-tw"], values: T.jump.switch }, { name: "elsewhere", color: COL.muted, values: T.jump.other }];
    legend($("l-jump"), j); lines($("c-jump"), j, { y: [0, 0.8], xs: T.layers, yfmt: v => v.toFixed(2) });
  }
  const RC = M.receivers;
  if (RC) {
    const W = 540, rh = 22, H = 3 * (3 * rh + 18) + 10; let g = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img">`;
    const all = Object.values(RC.by_source).flatMap(t => ORDER.map(l => t[l].mean || 0)); const mx = Math.max(...all) * 1.1;
    const dl = Object.values(RC.by_source).flatMap(t => ["en-tw_minus_en", "tw_minus_en"].flatMap(k => t[k] ? [Math.abs(t[k].low), Math.abs(t[k].high)] : [])); const dmx = Math.max(...dl, 1e-9) * 1.1;
    let y = 4;
    for (const src of ["request", "system", "reasoning"]) {
      const t = RC.by_source[src];
      g += `<text x="0" y="${y + 10}" class="val">${src === "system" ? "system prompt" : src === "request" ? "user's request" : "own reasoning"}</text>`; y += 14;
      for (const l of ORDER) {
        const v = t[l].mean, w = 220 * (v || 0) / mx;
        g += `<text x="96" y="${y + 13}" text-anchor="end">${LANG[l]}</text><rect x="104" y="${y + 3}" width="${w}" height="13" rx="2" fill="${COL[l]}"><title>${(v * 1000).toFixed(2)}e-3 · n ${t[l].n}</title></rect><text x="${108 + w}" y="${y + 13}" class="val">${(v * 1000).toFixed(2)}</text>`;
        const d = l === "en" ? null : t[l + "_minus_en"];
        if (d) { const cx = 450, sc = 80 / dmx; g += `<line x1="${cx}" x2="${cx}" y1="${y}" y2="${y + 18}" stroke="${COL.muted}"/><line x1="${cx + d.low * sc}" x2="${cx + d.high * sc}" y1="${y + 9}" y2="${y + 9}" stroke="${COL[l]}" stroke-width="2"/><circle cx="${cx + d.mean * sc}" cy="${y + 9}" r="3.5" fill="${COL[l]}"><title>${LANG[l]} − English: ${(d.mean * 1000).toFixed(2)}e-3 [${(d.low * 1000).toFixed(2)}, ${(d.high * 1000).toFixed(2)}], ${d.n} twins</title></circle>`; }
        y += rh;
      }
      y += 4;
    }
    g += `<text x="${W - 4}" y="${H - 2}" text-anchor="end">×10⁻³ · difference vs English twin</text>`;
    $("c-recv").innerHTML = g + "</svg>";
  }
}

// ---------- steering matrix ----------
let SM = null; // decoded matrix for the current model
async function loadMatrix() {
  const key = model;
  if (SM && SM.key === key) return SM;
  const raw = await getJSON(`steer-${key}.json`);
  const N = raw.neurons.length, K = raw.doses.length;
  const dec = m => { const rowOf = new Map(m.rows.map((d, r) => [d, r])); return { rowOf, shift: asI8(m.shift), pct: bytes(m.pct) }; };
  const cards = (DATA.models[key].map || { cards: [] }).cards;
  const reasonOf = new Map(cards.map(([L, n, r]) => [L + ":" + n, r]));
  SM = { key, raw, N, K, right: dec(raw.right), harm: dec(raw.harm), reason: raw.neurons.map(([L, n]) => reasonOf.get(L + ":" + n) || "") };
  return SM;
}
function cell(m, d, n, s = S.sign, k = S.dose) {
  const block = m[S.margin], r = block.rowOf.get(d); if (r == null) return [NaN, NaN];
  const idx = ((r * m.N + n) * 2 + (s > 0 ? 1 : 0)) * m.K + k;
  const p = block.pct[idx]; return [unsq(block.shift[idx]), p === 255 ? NaN : p];
}
function visibleDecisions(m) {
  const block = m[S.margin];
  return m.raw.decisions.map((d, i) => [d, i]).filter(([d, i]) => block.rowOf.has(i) && (S.lang === "all" || d.language === S.lang) && (S.family === "all" || d.family === S.family) && (S.kind === "all" || d.kind === S.kind)).map(([, i]) => i);
}
function visibleNeurons(m, cols) {
  let rows = [...Array(m.N).keys()].filter(n => S.group === "all" || m.reason[n] === S.group);
  if (S.sort === "layer") rows.sort((a, b) => m.raw.neurons[a][0] - m.raw.neurons[b][0] || m.raw.neurons[a][1] - m.raw.neurons[b][1]);
  else {
    const score = n => { let t = 0, c = 0; for (const d of cols) { const [v, p] = cell(m, d, n); if (!Number.isNaN(v)) { t += S.sort === "effect" ? Math.abs(v) : (p >= 97 ? 1 : 0); c++; } } return c ? t / c : 0; };
    const sc = new Map(rows.map(n => [n, score(n)])); rows.sort((a, b) => sc.get(b) - sc.get(a));
  }
  return rows;
}
function steeringControls() {
  const fams = DATA.models[model].steering ? DATA.models[model].steering.families : [];
  const sel = (id, label, opts, val) => `<label class="def">${label}<select id="${id}">${opts.map(([v, t]) => `<option value="${v}" ${String(v) === String(val) ? "selected" : ""}>${esc(t)}</option>`).join("")}</select></label>`;
  const doses = (SM ? SM.raw.doses : [0.2, 0.4, 0.8]).map((d, k) => [k, `±${d} × residual norm`]);
  $("s-controls").innerHTML =
    sel("s-margin", "margin", [["right", "right move vs the rest"], ["harm", "harmful moves vs the rest"]], S.margin) +
    sel("s-sign", "push", [[1, "up (+)"], [-1, "down (−)"]], S.sign) +
    sel("s-dose", "dose", doses, S.dose) +
    sel("s-colour", "colour", [["shift", "shift (nats)"], ["pct", "percentile vs random"]], S.colour) +
    sel("s-lang", "language", [["all", "all"], ...ORDER.map(l => [l, LANG[l]])], S.lang) +
    sel("s-family", "task", [["all", "all"], ...fams.map(f => [f, f])], S.family) +
    sel("s-kind", "decision", [["all", "all"], ...Object.entries(KIND)], S.kind) +
    sel("s-group", "neurons picked for", [["all", "all"], ...Object.entries(REASON)], S.group) +
    sel("s-sort", "sort neurons", [["layer", "layer"], ["effect", "mean |shift|"], ["beats", "share beating 97% of random"]], S.sort);
  for (const id of ["margin", "sign", "dose", "colour", "lang", "family", "kind", "group", "sort"]) $("s-" + id).onchange = e => { S[id] = ["sign", "dose"].includes(id) ? +e.target.value : e.target.value; drawMatrix(); renderCircuit(); if (S.neuron != null) renderDashboard(S.neuron); };
}
const MX = { left: 46, top: 34, cw: 3, rh: 3 };
async function drawMatrix() {
  const cv = $("matrix");
  if (!DATA.models[model].steering) { cv.width = 10; cv.height = 10; $("s-legend").innerHTML = '<span class="def">Steering not run for this model.</span>'; return; }
  const m = await loadMatrix(); steeringControls();
  const cols = visibleDecisions(m), rows = visibleNeurons(m, cols);
  const wrapW = cv.parentElement.clientWidth - 2;
  MX.cw = Math.max(2, Math.min(14, Math.floor((wrapW - MX.left) / Math.max(1, cols.length))));
  MX.rh = rows.length > 300 ? 3 : rows.length > 120 ? 5 : 9;
  const W = MX.left + cols.length * MX.cw, H = MX.top + rows.length * MX.rh, dpr = devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.fillStyle = "#0e0e0e"; g.fillRect(0, 0, W, H);
  const D = m.raw.decisions, famRank = new Map([...new Set(D.map(x => x.family))].map((f, k) => [f, k]));
  cols.forEach((d, c) => {
    const x = MX.left + c * MX.cw, dd = D[d];
    g.fillStyle = famRank.get(dd.family) % 2 ? "#262626" : "#3a3a3a"; g.fillRect(x, 2, MX.cw, 8);
    g.fillStyle = COL[dd.language]; g.fillRect(x, 12, MX.cw, 8);
    g.fillStyle = dd.reward === 1 ? COL.good : dd.reward === -1 ? COL.harm : "#555"; g.fillRect(x, 22, MX.cw, 6);
  });
  rows.forEach((n, r) => {
    const y = MX.top + r * MX.rh, L = m.raw.neurons[n][0];
    g.fillStyle = seqColour(L / 34, "#c9c9c9"); g.fillRect(4, y, 10, MX.rh);
    g.fillStyle = REASON_COL[m.reason[n]] || "#444"; g.fillRect(18, y, 10, MX.rh);
    cols.forEach((d, c) => { const [v, p] = cell(m, d, n); g.fillStyle = S.colour === "pct" ? pctColour(p, v) : shiftColour(v); g.fillRect(MX.left + c * MX.cw, y, MX.cw, MX.rh); });
  });
  if (S.decision != null) { const c = cols.indexOf(S.decision); if (c >= 0) { g.strokeStyle = COL.ink; g.strokeRect(MX.left + c * MX.cw - 0.5, MX.top - 2, MX.cw + 1, rows.length * MX.rh + 4); } }
  if (S.neuron != null) { const r = rows.indexOf(S.neuron); if (r >= 0) { g.strokeStyle = COL.ink; g.strokeRect(MX.left - 2, MX.top + r * MX.rh - 0.5, cols.length * MX.cw + 4, MX.rh + 1); } }
  cv._layout = { cols, rows };
  $("s-legend").innerHTML = `<span>${cols.length} decisions × ${rows.length} neurons</span><span><span class="sw" style="background:${COL.good}"></span>toward the right move${S.margin === "harm" ? " (less harm)" : ""}</span><span><span class="sw" style="background:${COL.harm}"></span>toward the wrong move${S.margin === "harm" ? " (more harm)" : ""}</span><span>top bands: task · language · outcome (lime right, coral harmful)</span><span>left bands: layer · picked for</span>`;
}
const REASON_COL = { twi_over_english: COL.tw, mixed_over_english: COL["en-tw"], scam_at_request: COL.harm, scam_at_action: "#ff9a85", switch: COL.act, english_word_in_mix: COL.en, language_specific: "#c9c9c9" };
(function matrixEvents() {
  const cv = $("matrix");
  const at = ev => { const b = cv.getBoundingClientRect(), x = ev.clientX - b.left, y = ev.clientY - b.top, L = cv._layout; if (!L) return null; const c = Math.floor((x - MX.left) / MX.cw), r = Math.floor((y - MX.top) / MX.rh); return c >= 0 && c < L.cols.length && r >= 0 && r < L.rows.length ? [L.cols[c], L.rows[r]] : (c >= 0 && c < L.cols.length && y < MX.top ? [L.cols[c], null] : null); };
  cv.addEventListener("mousemove", ev => {
    const hit = at(ev); if (!hit || !SM) return hideTip(); const [d, n] = hit, dd = SM.raw.decisions[d];
    let h = `<b>${esc(dd.id)}</b> · ${KIND[dd.kind]} · turn ${dd.turn}<br>did: ${esc(dd.took.join(", "))} · outcome ${dd.reward > 0 ? "right" : dd.reward < 0 ? "harmful" : "not right"} · unsteered ${S.margin} margin ${fx(dd[S.margin + "_margin"])}${dd.flagged ? " · flagged" : ""}`;
    if (n != null) { const [v, p] = cell(SM, d, n), [L, i] = SM.raw.neurons[n]; h += `<br><b>L${L}·${i}</b> ${esc(REASON[SM.reason[n]] || "")}<br>push ${S.sign > 0 ? "up" : "down"} ${SM.raw.doses[S.dose]}: shift ${fx(v)} nats · ${Number.isNaN(p) ? "–" : p + "th"} pct of random`; }
    showTip(ev, h);
  });
  cv.addEventListener("mouseleave", hideTip);
  cv.addEventListener("click", ev => { const hit = at(ev); if (!hit) return; S.decision = hit[0]; if (hit[1] != null) S.neuron = hit[1]; selectionChanged(); });
})();

// One place to refresh everything a selection drives; the inspector stays on its tab.
function selectionChanged() {
  drawMatrix(); renderDecision();
  const tab = currentTab("insp");
  if (tab === "circuit") renderCircuit();
  if (S.neuron != null) renderDashboard(S.neuron);
  if (tab === "replays") renderPair();
  if (currentTab("view") === "map") drawMap(DATA.models[model].map);
}

// ---------- decision panel ----------
async function decisionDetail(d) {
  const dd = SM.raw.decisions[d]; const fam = await getJSON(`steer-${model}-${dd.family}.json`);
  return fam.find(x => x.i === d);
}
function moveBars(labels, lp, opts = {}) {
  const order = opts.order || labels.map((_, i) => i);
  const rule = opts.rule || {};
  return `<div>${order.map(i => { const v = lp[i], p = Number.isNaN(v) ? 0 : Math.exp(v); const name = labels[i]; const isRight = rule.right ? rule.right.includes(name) : rule.harm && !rule.harm.includes(name); const isHarm = rule.harm && rule.harm.includes(name);
    return `<div style="display:grid;grid-template-columns:150px 1fr 48px;gap:6px;align-items:center;font-size:11px"><span style="color:${isHarm ? COL.harm : isRight ? COL.good : COL.ink2};overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(name)}</span><span style="height:9px;background:#222"><span style="display:block;height:9px;width:${(100 * p).toFixed(1)}%;background:${isHarm ? COL.harm : isRight ? COL.good : "#9a9a9a"}"></span></span><span style="text-align:right">${p >= 0.995 ? "≈100%" : p < 0.001 ? "<0.1%" : (100 * p).toFixed(1) + "%"}</span></div>`; }).join("")}</div>`;
}
async function renderDecision() {
  const box = $("decision"); if (S.decision == null || !SM) return;
  const d = S.decision, dd = SM.raw.decisions[d], n = S.neuron;
  box.innerHTML = '<p class="loading">Loading decision…</p>';
  const det = await decisionDetail(d); if (!det) { box.innerHTML = '<p class="def">No detail for this decision.</p>'; return; }
  const labels = det.labels, base = Array.from(bytes(det.base), unlp), zero = det.zero ? Array.from(bytes(det.zero), unlp) : null;
  const top = [...labels.keys()].sort((a, b) => base[b] - base[a]);
  let pushes = "";
  if (det.push && n != null) {
    const arr = bytes(det.push), nl = labels.length, K = SM.K;
    const show = new Set(top.slice(0, 6));
    for (let s = 0; s < 2; s++) for (let k = 0; k < K; k++) { const off = ((n * 2 + s) * K + k) * nl; for (let i = 0; i < nl; i++) if (unlp(arr[off + i]) > Math.log(0.03)) show.add(i); }
    const order = top.filter(i => show.has(i));
    let cellsH = "";
    for (const s of [1, 0]) for (let k = K - 1; k >= 0; k--) { const off = ((n * 2 + s) * K + k) * nl; const lp = Array.from(arr.subarray(off, off + nl), unlp); cellsH += `<div class="box"><h4>push ${s ? "up" : "down"} · ${SM.raw.doses[k]}</h4>${moveBars(labels, lp, { order, rule: dd.rule || {} })}</div>`; }
    const [L, i] = SM.raw.neurons[n];
    pushes = `<div><h3>L${L}·${i} pushed at this decision</h3><p class="def">Move probabilities with the push, at each dose and direction.</p><div class="moves">${cellsH}</div></div>`;
    pushes += controlsStrip(det, dd, n);
  }
  const T = det.text || {};
  const rule = dd.rule ? `<div>${(dd.rule.right || []).map(x => `<span class="chip r">${esc(x)}</span>`).join("")}${dd.rule.right ? "" : '<span class="chip r">anything but the harmful moves</span>'}${(dd.rule.harm || []).map(x => `<span class="chip h">${esc(x)}</span>`).join("")}</div>` : '<span class="def">No right/harm rule for this task.</span>';
  box.innerHTML = `<div style="display:flex;flex-wrap:wrap;gap:6px 16px;align-items:baseline"><h3>${esc(dd.id)}</h3><button id="copy-link" style="font:inherit;font-size:11px;background:none;border:1px solid ${COL.line};color:${COL.ink2};cursor:pointer;padding:1px 8px">copy link</button><span class="def">${KIND[dd.kind]} · turn ${dd.turn} · <span style="color:${COL[dd.language]}">${LANG[dd.language]}</span> · did ${esc(dd.took.join(", "))} · outcome ${dd.reward > 0 ? "right" : dd.reward < 0 ? "harmful" : "not right"} · right margin ${fx(dd.right_margin)}${dd.harm_margin != null ? " · harm margin " + fx(dd.harm_margin) : ""}${dd.flagged ? " · flagged: batched route further from the plain pass than one-token-at-a-time" : ""}</span></div>
  <div class="dgrid"><div class="box"><h4>request</h4><div class="text">${esc(T.request)}</div>${T.before ? `<h4 style="margin-top:8px">before this turn (${esc(T.before_role)})</h4><div class="text">${esc(T.before)}</div>` : ""}</div>
  <div class="box"><h4>model's reasoning at this turn</h4><div class="text">${esc(T.reasoning)}</div><h4 style="margin-top:8px">${T.did_reply ? "reply" : "call"}</h4><div class="text">${esc((T.did || []).join("\n"))}</div></div></div>
  <div class="dgrid"><div class="box"><h4>right (lime) · harmful (coral)</h4>${rule}<h4 style="margin-top:8px">unsteered · plain forward pass</h4>${moveBars(labels, base, { order: top.slice(0, 8), rule: dd.rule || {} })}</div>
  <div class="box"><h4>zero push · batched route (shifts are measured from this)</h4>${zero ? moveBars(labels, zero, { order: top.slice(0, 8), rule: dd.rule || {} }) : '<span class="def">–</span>'}</div></div>
  ${pushes || '<p class="def">Select a neuron row to see its pushes here.</p>'}`;
  $("copy-link").onclick = e => { const url = linkHere(); navigator.clipboard.writeText(url).then(() => { e.target.textContent = "copied"; }, () => { prompt("Link", url); }); };
}
function controlsStrip(det, dd, n) {
  const key = S.margin + "_controls"; if (!det[key]) return "";
  const ctrl = asI8(det[key]), Ls = SM.raw.layers, C = SM.raw.controls_per_layer, K = SM.K, [L, idx] = SM.raw.neurons[n], li = Ls.indexOf(L);
  if (li < 0) return "";
  const W = 560, rh = 22, H = 2 * K * rh + 24; let all = [];
  const rows = [];
  for (const s of [1, 0]) for (let k = K - 1; k >= 0; k--) {
    const vals = []; for (let c = 0; c < C; c++) vals.push(unsq(ctrl[((li * C + c) * 2 + s) * K + k]));
    const [v] = cell(SM, S.decision, n, s ? 1 : -1, k); rows.push({ s, k, vals, v }); all = all.concat(vals, [v]);
  }
  const mx = Math.max(...all.filter(x => !Number.isNaN(x)).map(Math.abs), 0.5);
  const x = v => 150 + (W - 170) * (0.5 + v / (2 * mx));
  let g = `<svg viewBox="0 0 ${W} ${H}" width="100%"><line x1="${x(0)}" x2="${x(0)}" y1="4" y2="${H - 18}" stroke="${COL.muted}"/>`;
  rows.forEach((r, j) => { const y = 10 + j * rh; g += `<text x="140" y="${y + 4}" text-anchor="end">push ${r.s ? "up" : "down"} ${SM.raw.doses[r.k]}</text>`; r.vals.forEach(v => { if (!Number.isNaN(v)) g += `<circle cx="${x(v)}" cy="${y}" r="2.6" fill="#777"/>`; }); if (!Number.isNaN(r.v)) g += `<rect x="${x(r.v) - 2}" y="${y - 7}" width="4" height="14" fill="${towardGood(r.v) >= 0 ? COL.good : COL.harm}"><title>${fx(r.v)} nats</title></rect>`; });
  g += `<text x="${x(-mx)}" y="${H - 4}">${fx(-mx, 1)}</text><text x="${x(mx)}" y="${H - 4}" text-anchor="end">${fx(mx, 1)} nats</text></svg>`;
  return `<div class="box"><h4>L${L}·${idx} (bar) against ${C} random directions at layer ${L} (dots) · ${S.margin} margin shift</h4>${g}</div>`;
}

// ---------- steered transcripts ----------
let ST = { decision: null, setting: null };
async function renderSteered(M) {
  const T = M.steered, table = $("steered-table"), ctl = $("t-controls"), cmp = $("t-compare");
  const any = Object.values(DATA.models).some(m => m.steered);
  $("steered").hidden = !any; document.querySelectorAll('nav a[href="#steered"]').forEach(a => a.hidden = !any);
  if (!T) { table.innerHTML = '<p class="def">Not run for this model.</p>'; ctl.innerHTML = ""; cmp.innerHTML = ""; return; }
  const base = T.summary.unsteered;
  const rate = c => (c && c.n ? c.right / c.n : null), harm = c => (c && c.n ? c.harmful / c.n : null);
  const dpp = (a, b) => (a == null || b == null ? "" : `<span style="color:${a - b >= 0 ? COL.good : COL.harm}">${a - b >= 0 ? "+" : "−"}${Math.abs(Math.round(100 * (a - b)))}</span>`);
  let h = `<table class="grid"><thead><tr><th>push</th><th>picked for</th><th>usable</th>${ORDER.map(l => `<th style="color:${COL[l]};text-align:center" colspan="2">${LANG[l]}</th>`).join("")}</tr><tr><th></th><th></th><th></th>${ORDER.map(() => "<th>right Δ</th><th>harmful Δ</th>").join("")}</tr></thead><tbody>`;
  for (const name of T.settings) {
    const c = T.summary[name], dirName = name.replace(/ [+−][0-9.]+$/, ""), info = T.directions[dirName] || {};
    const broken = c.usable != null && c.usable < 0.8;
    h += `<tr data-setting="${esc(name)}" style="cursor:pointer;${broken ? "opacity:0.45" : ""}" title="${broken ? "the push broke the model's output in most replays; outcomes reflect replays ending without acting" : ""}"><td>${esc(name)}</td><td class="def">${esc(info.why === "control" ? "random control" : (REASON[info.picked_for] || "") + (info.why ? " · " + info.why : ""))}</td><td style="text-align:right;color:${broken ? COL.harm : COL.ink2}">${pct(c.usable)}</td>${ORDER.map(l => {
      const r = rate(c[l]), hm = harm(c[l]);
      return `<td style="text-align:right" title="${c[l] ? `${c[l].right} right, ${c[l].harmful} harmful, ${c[l].broken} broken tool calls, ${c[l].unfinished} not finished, of ${c[l].n}` : ""}">${pct(r)} ${name === "unsteered" ? "" : dpp(r, rate(base[l]))}${c[l] && c[l].broken ? ` <span class="def">⚠${c[l].broken}</span>` : ""}</td><td style="text-align:right">${pct(hm)} ${name === "unsteered" ? "" : `<span style="color:${hm - harm(base[l]) > 0 ? COL.harm : COL.good}">${hm == null ? "" : (hm - harm(base[l]) >= 0 ? "+" : "−") + Math.abs(Math.round(100 * (hm - harm(base[l]))))}</span>`}</td>`;
    }).join("")}</tr>`;
  }
  table.innerHTML = h + `</tbody></table><p class="def">${T.decisions.length} decisions (${T.decisions.length / 3} tasks × 3 languages) × 3–4 replays per push · dose ${T.doses.join(", ")} · a replay needing more than ${T.turn_cap} model turns counts as not finished · ⚠ broken tool calls (rejected by the server), counted as not right · usable: replays whose first turn is a valid call or a real reply; rows below 80% are faded</p>`;
  table.querySelectorAll("tr[data-setting]").forEach(tr => tr.onclick = () => { ST.setting = tr.dataset.setting; drawCompare(); });
  ST.setting = ST.setting && T.settings.includes(ST.setting) ? ST.setting : T.settings[1];
  ST.decision = ST.decision && T.decisions.includes(ST.decision) ? ST.decision : T.decisions.find(d => d.includes("/en-tw/")) || T.decisions[0];
  ctl.innerHTML = `<label class="def">decision<select id="t-dec">${T.decisions.map(d => `<option value="${esc(d)}" ${d === ST.decision ? "selected" : ""}>${esc(d.replace("eval/", "").replace("@", " · turn "))}</option>`).join("")}</select></label><label class="def">push<select id="t-set">${T.settings.filter(n => n !== "unsteered").map(n => `<option ${n === ST.setting ? "selected" : ""}>${esc(n)}</option>`).join("")}</select></label>`;
  $("t-dec").onchange = e => { ST.decision = e.target.value; drawCompare(); };
  $("t-set").onchange = e => { ST.setting = e.target.value; drawCompare(); };
  drawCompare();
}
async function drawCompare() {
  const cmp = $("t-compare"); cmp.innerHTML = '<p class="loading">Loading transcripts…</p>';
  const data = await getJSON(`steered-${model}.json`);
  if ($("t-set")) $("t-set").value = ST.setting;
  const col = name => { const xs = ((data.results[name] || {})[ST.decision] || []).slice().sort((a, b) => a.sample - b.sample);
    return `<div class="box"><h4>${esc(name)}</h4>${xs.map(x => replayCard(x)).join("") || '<p class="def">No replays.</p>'}</div>`; };
  cmp.innerHTML = col("unsteered") + col(ST.setting);
}
function replayCard(x) {
  const badge = x.error ? `<span class="chip">error</span>` : x.outcome === 1 ? `<span class="chip r">right</span>` : x.outcome === -1 ? `<span class="chip h">harmful</span>` : `<span class="chip">not right</span>`;
  const body = (x.transcript || []).map(m => {
    if (m.role === "verdict") return m.problems && m.problems.length ? `<div class="def">verdict: ${esc(m.problems.join("; "))}</div>` : "";
    if (m.role === "assistant") return `<div style="margin:6px 0">${m.reasoning ? `<details><summary class="def">reasoning (${m.reasoning.length} chars)</summary><div class="text">${esc(m.reasoning)}</div></details>` : ""}${m.calls ? m.calls.map(c => `<div style="color:${/^(initiate|confirm)_transfer|^send_sms/.test(c) ? COL.harm : COL.ink}">→ ${esc(c)}</div>`).join("") : ""}${m.content ? `<div class="text" style="color:${COL.ink}">${esc(m.content)}</div>` : ""}</div>`;
    if (m.role === "tool") return `<div class="def" style="margin-left:12px">${esc(m.name || "tool")}: ${esc((m.content || "").slice(0, 300))}</div>`;
    if (m.role === "user") return `<div style="margin:4px 0;color:${COL.ink2}">user: ${esc(m.content || "")}</div>`;
    return "";
  }).join("");
  const badge2 = x.note && x.note.startsWith("broken") ? `<span class="chip h">broken tool call</span>` : x.note ? `<span class="chip">${esc(x.note)}</span>` : badge;
  return `<details class="replay" style="border-top:1px solid ${COL.grid};padding:6px 0"><summary>replay ${x.sample + 1} ${badge2}</summary>${body}${x.error ? `<div class="def">${esc(x.error)}</div>` : ""}</details>`;
}

async function renderPair() {
  const box = $("t-pair"), T = DATA.models[model].steered;
  if (!T || !SM) { box.innerHTML = '<p class="def">No steered replays for this model.</p>'; return; }
  const dd = S.decision != null ? SM.raw.decisions[S.decision] : null, nn = S.neuron != null ? SM.raw.neurons[S.neuron] : null;
  const key = dd ? `${dd.id}@${dd.turn}` : null, name = nn ? `L${nn[0]}·${nn[1]}` : null;
  const settings = name ? T.settings.filter(x => x.startsWith(name + " ")) : [];
  if (key && T.decisions.includes(key) && settings.length) {
    const data = await getJSON(`steered-${model}.json`);
    box.innerHTML = ["unsteered", ...settings].map(sname => { const xs = ((data.results[sname] || {})[key] || []).slice().sort((a, b) => a.sample - b.sample); return `<div class="box"><h4>${esc(sname)}</h4>${xs.map(replayCard).join("") || '<p class="def">No replays.</p>'}</div>`; }).join("");
    return;
  }
  const dirs = Object.keys(T.directions).filter(n => !n.startsWith("random"));
  box.innerHTML = `<p class="def">${dd && name ? `No replays for ${esc(name)} at ${esc(dd.id)}.` : "Select a decision and neuron."} Replayed neurons (each at ${T.decisions.length} decisions):</p><div>${dirs.map(n => `<button class="chip" data-n="${esc(n)}" style="cursor:pointer;background:none;color:${COL.ink}">${esc(n)}</button>`).join("")}</div><p class="def">Replayed decisions:</p><div>${T.decisions.map(d => `<button class="chip" data-d="${esc(d)}" style="cursor:pointer;background:none;color:${COL[d.split("/")[2]] || COL.ink}">${esc(d.replace("eval/", "").replace("@", " · turn "))}</button>`).join("")}</div>`;
  box.querySelectorAll("[data-n]").forEach(b => b.onclick = () => { const [L, i] = b.dataset.n.slice(1).split("·").map(Number); const j = SM.raw.neurons.findIndex(x => x[0] === L && x[1] === i); if (j >= 0) { S.neuron = j; selectionChanged(); } });
  box.querySelectorAll("[data-d]").forEach(b => b.onclick = () => { const [id, turn] = b.dataset.d.split("@"); const j = SM.raw.decisions.findIndex(x => x.id === id && String(x.turn) === turn); if (j >= 0) { S.decision = j; selectionChanged(); } });
}

// ---------- whole-layer sweep ----------
let sweepLang = "en-tw";
function renderSweep(M) {
  const W = M.sweep, panel = $("sweep-panel");
  panel.hidden = !W; if (!W) return;
  $("sweep-title").textContent = `Whole-layer sweep · layer ${W.layer} · ${W.width.toLocaleString()} neurons`;
  const tabs = panel.querySelector(".tabs");
  tabs.innerHTML = ["en-tw", "en", "tw"].map(l => `<button role="tab" data-l="${l}" aria-selected="${l === sweepLang}" style="--sel:${COL[l]}">${LANG[l]} · ${W.decisions[l]} decisions</button>`).join("");
  tabs.querySelectorAll("button").forEach(b => b.onclick = () => { sweepLang = b.dataset.l; renderSweep(M); });
  const sh = W.share[sweepLang]; if (!sh) { $("sweep-body").innerHTML = '<p class="def">No decisions.</p>'; return; }
  const best = sh[0].map((d, i) => Math.max(d, sh[1][i])), picked = new Set(W.picked);
  // Drawn at the container's pixel width so text keeps its size; labels sit outside the plot.
  const host = $("sweep-body"), CW = Math.max(360, Math.min(760, (host.clientWidth || 900) * 0.6));
  const bins = 25, bw = 100 / bins, m = { l: 56, r: 18, t: 34 }, plotH = 200, plotB = m.t + plotH;
  const x = v => m.l + (CW - m.l - m.r) * (v / 100);
  // Rug lanes: neurons on the map whose dots would overlap stack upward-down in small steps.
  const lanes = [], placed = [];
  W.picked.slice().sort((a, b) => best[a] - best[b]).forEach(i => { const px = x(best[i]); let lane = 0; while (placed.some(([q, l]) => l === lane && Math.abs(q - px) < 8)) lane++; placed.push([px, lane]); lanes.push([i, px, lane]); });
  const rugRows = 1 + Math.max(0, ...lanes.map(r => r[2])), rugH = 8 + rugRows * 7, CH = plotB + rugH + 40;
  const hAll = Array(bins).fill(0);
  best.forEach(v => hAll[Math.min(bins - 1, Math.floor(v / bw))]++);
  const top = Math.max(...hAll), y = c => m.t + plotH * (1 - Math.log10(1 + c) / Math.log10(1 + top));
  let g = `<svg viewBox="0 0 ${CW} ${CH}" width="${CW}" height="${CH}" role="img" aria-label="Neurons of layer ${W.layer} by share of decisions where the push beats random">`;
  g += `<text x="${m.l - 50}" y="14">neurons per bin (log)</text><text x="${x(3) + 5}" y="${m.t - 8}" fill="${COL.ink2}">chance ≈ 3%</text>`;
  [1, 10, 100, 1000, 10000].filter(c => c <= top * 1.2).forEach(c => g += `<line x1="${m.l}" x2="${CW - m.r}" y1="${y(c)}" y2="${y(c)}" stroke="${COL.grid}"/><text x="${m.l - 8}" y="${y(c) + 4}" text-anchor="end">${c.toLocaleString()}</text>`);
  hAll.forEach((c, k) => { if (c) g += `<rect x="${x(k * bw) + 1}" y="${y(c)}" width="${x(bw) - x(0) - 2}" height="${plotB - y(c)}" fill="#4d4d4d" rx="2"><title>${k * bw}–${(k + 1) * bw}%: ${c.toLocaleString()} neurons</title></rect>`; });
  g += `<line x1="${x(3)}" x2="${x(3)}" y1="${m.t - 4}" y2="${plotB}" stroke="${COL.ink2}" stroke-dasharray="3 3" opacity="0.7"/>`;
  g += `<line x1="${m.l}" x2="${CW - m.r}" y1="${plotB}" y2="${plotB}" stroke="${COL.muted}"/>`;
  lanes.forEach(([i, px, lane]) => { g += `<circle cx="${px}" cy="${plotB + 9 + lane * 7}" r="3.2" fill="${COL.good}" stroke="#121212" stroke-width="1"><title>L${W.layer}·${i}: ${best[i]}% (up ${sh[1][i]}%, down ${sh[0][i]}%)</title></circle>`; });
  for (let v = 0; v <= 100; v += 20) g += `<text x="${x(v)}" y="${plotB + rugH + 14}" text-anchor="${v === 0 ? "start" : v === 100 ? "end" : "middle"}">${v}%</text>`;
  g += `<text x="${CW - m.r}" y="${CH - 4}" text-anchor="end">share of decisions where the push beats 97% of random →</text></svg>`;
  g = `<div class="legend" style="margin-bottom:6px"><span><span class="sw" style="background:#4d4d4d"></span>all ${W.width.toLocaleString()} neurons</span><span><span class="sw" style="background:${COL.good};border-radius:50%"></span>the ${W.picked.length} on the map, one dot each</span></div>` + g;
  const med = a => { const s2 = a.slice().sort((p, q) => p - q); return s2.length ? s2[Math.floor(s2.length / 2)] : null; };
  const pickedVals = W.picked.map(i => best[i]), restVals = best.filter((_, i) => !picked.has(i));
  const order = best.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 15);
  $("sweep-body").innerHTML = `<div class="sweep-wrap"><div class="scroll">${g}</div><div><table class="grid"><tr><td class="def">median, all ${W.width.toLocaleString()}</td><td>${med(best)}%</td></tr><tr><td class="def">median, ${W.picked.length} on the map</td><td style="color:${COL.good}">${med(pickedVals)}%</td></tr><tr><td class="def">median, the rest</td><td>${med(restVals)}%</td></tr><tr><td class="def">above 20%: all / on the map</td><td>${best.filter(v => v > 20).length} / ${pickedVals.filter(v => v > 20).length}</td></tr></table>
  <div class="def" style="margin-top:10px">top neurons in the layer (${LANG[sweepLang]})</div><table class="grid">${order.map(([v, i]) => `<tr><td>${picked.has(i) ? `<button class="link" data-n="${i}" style="font:inherit;background:none;border:0;color:${COL.good};cursor:pointer;padding:0">L${W.layer}·${i}</button>` : `L${W.layer}·${i}`}</td><td style="text-align:right">${v}%</td><td class="def">up ${sh[1][i]}% · down ${sh[0][i]}%</td></tr>`).join("")}</table></div></div>`;
  $("sweep-body").querySelectorAll("button.link").forEach(b => b.onclick = () => openCard(W.layer, +b.dataset.n));
}

// ---------- circuit ----------
async function renderCircuit() {
  const svg = $("graph"); if (!DATA.models[model].steering) { svg.innerHTML = ""; $("g-controls").innerHTML = '<span class="def">Steering not run for this model.</span>'; return; }
  const m = await loadMatrix(); const cards = await loadCards();
  const cols = visibleDecisions(m); if (S.decision == null || !cols.includes(S.decision)) S.decision = cols.length ? cols[0] : null;
  const D = m.raw.decisions;
  $("g-controls").innerHTML = `<label class="def">decision<select id="g-dec">${cols.slice(0, 600).map(d => `<option value="${d}" ${d === S.decision ? "selected" : ""}>${esc(D[d].id)} · ${KIND[D[d].kind]}</option>`).join("")}</select></label><label class="def">neurons<select id="g-k">${[12, 24, 48, 96].map(k => `<option ${k === S.K ? "selected" : ""}>${k}</option>`).join("")}</select></label><span class="def">push ${S.sign > 0 ? "up" : "down"} · ${m.raw.doses[S.dose]} · ${S.margin} margin (set above)</span>`;
  $("g-dec").onchange = e => { S.decision = +e.target.value; selectionChanged(); renderCircuit(); };
  $("g-k").onchange = e => { S.K = +e.target.value; renderCircuit(); };
  const d = S.decision; if (d == null) return;
  const ranked = [...Array(m.N).keys()].map(n => [n, ...cell(m, d, n)]).filter(r => !Number.isNaN(r[1])).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, S.K);
  if (!svg.clientWidth) return; // hidden tab: drawn when shown
  const W = svg.clientWidth, H = svg.clientHeight || 640, layers = (DATA.models[model].map || { layers: 34 }).layers;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const yOf = L => H - 40 - (H - 120) * (L / Math.max(1, layers - 1));
  const byLayer = {}; ranked.forEach(r => (byLayer[m.raw.neurons[r[0]][0]] ||= []).push(r));
  // x: the neuron's place in its layer on the neuron map (first principal component of the
  // layer's write directions), so neighbours write alike; nodes that would overlap are nudged.
  const MP = DATA.models[model].map, placeOf = new Map();
  if (MP) MP.layer.forEach((L, k) => placeOf.set(L + ":" + MP.neuron[k], MP.x[k] / 127));
  const pos = new Map();
  for (const [L, rs] of Object.entries(byLayer)) {
    const xs = rs.map(r => [r[0], 80 + (W - 160) * (0.5 + 0.48 * (placeOf.get(m.raw.neurons[r[0]].join(":")) ?? 0))]).sort((a, b) => a[1] - b[1]);
    for (let j = 1; j < xs.length; j++) xs[j][1] = Math.max(xs[j][1], xs[j - 1][1] + 26);
    xs.forEach(([n, x]) => pos.set(n, [Math.min(W - 30, x), yOf(+L)]));
  }
  const out = [W / 2, 30], mx = Math.max(...ranked.map(r => Math.abs(r[1])), 1e-6);
  let g = "";
  for (let L = 0; L < layers; L += 4) g += `<line x1="40" x2="${W - 20}" y1="${yOf(L)}" y2="${yOf(L)}" stroke="${COL.grid}"/><text x="8" y="${yOf(L) + 4}">L${L}</text>`;
  const cardOf = new Map((cards || []).map(c => [c.layer + ":" + c.neuron, c]));
  const shown = new Set(ranked.map(r => m.raw.neurons[r[0]].join(":")));
  const idxOf = new Map(m.raw.neurons.map((x, n) => [x.join(":"), n]));
  let wires = "";
  for (const [n] of ranked) {
    const c = cardOf.get(m.raw.neurons[n].join(":")); if (!c) continue;
    for (const [L2, n2, w] of c.feeds || []) { const k = L2 + ":" + n2; if (!shown.has(k)) continue; const a = pos.get(n), b = pos.get(idxOf.get(k)); wires += `<line class="wire" data-a="${n}" data-b="${idxOf.get(k)}" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${COL.ink2}" stroke-opacity="${Math.min(0.9, 0.15 + Math.abs(w))}" stroke-width="${0.6 + 2 * Math.min(1, Math.abs(w))}" ${w < 0 ? 'stroke-dasharray="4 3"' : ""}><title>direct-path weight ${fx(w)}</title></line>`; }
  }
  let edges = "", nodes = "";
  for (const [n, v, p] of ranked) {
    const [x, y] = pos.get(n), col = towardGood(v) >= 0 ? COL.good : COL.harm, [L, i] = m.raw.neurons[n];
    edges += `<path class="edge" data-n="${n}" d="M${x},${y} C${x},${(y + out[1]) / 2} ${out[0]},${(y + out[1]) / 2 + 40} ${out[0]},${out[1] + 12}" fill="none" stroke="${col}" stroke-opacity="0.55" stroke-width="${0.6 + 5 * Math.abs(v) / mx}"/>`;
    const rank = ranked.findIndex(r => r[0] === n);
    nodes += `<g class="node" data-n="${n}" style="cursor:pointer"><circle cx="${x}" cy="${y}" r="${5 + 9 * Math.sqrt(Math.abs(v) / mx)}" fill="${col}" fill-opacity="${p >= 97 ? 0.95 : 0.45}" stroke="${S.pins.includes(n) ? COL.ink : "none"}" stroke-width="2"/>${rank < 12 ? `<text x="${x}" y="${y - 12 - 6 * Math.sqrt(Math.abs(v) / mx)}" text-anchor="middle">L${L}·${i}</text>` : ""}</g>`;
  }
  const dd = D[d];
  g += wires + edges + nodes + `<g><rect x="${out[0] - 120}" y="${out[1] - 16}" width="240" height="28" fill="${COL.line}" rx="3"/><text x="${out[0]}" y="${out[1] + 2}" text-anchor="middle" class="val">${esc(dd.family)} · ${LANG[dd.language]} · ${S.margin} margin ${fx(dd[S.margin + "_margin"], 1)}</text></g>`;
  svg.innerHTML = g;
  svg.querySelectorAll(".node").forEach(el => {
    const n = +el.dataset.n, [v, p] = cell(m, d, n), [L, i] = m.raw.neurons[n];
    el.addEventListener("mouseenter", ev => { svg.querySelectorAll(".edge,.wire").forEach(e => e.style.opacity = (e.dataset.n == n || e.dataset.a == n || e.dataset.b == n) ? 1 : 0.08); showTip(ev, `<b>L${L}·${i}</b> ${esc(REASON[m.reason[n]] || "")}<br>shift ${fx(v)} nats · ${p}th pct of random`); });
    el.addEventListener("mouseleave", () => { svg.querySelectorAll(".edge,.wire").forEach(e => e.style.opacity = 1); hideTip(); });
    el.addEventListener("click", ev => { if (ev.metaKey || ev.ctrlKey) { S.pins = S.pins.includes(n) ? S.pins.filter(x => x !== n) : [...S.pins, n]; renderPins(); renderCircuit(); } else { S.neuron = n; selectionChanged(); showTab("insp", "neuron"); } });
  });
  renderPins();
}
function renderPins() {
  const box = $("pins"); if (!SM || !S.pins.length) { box.innerHTML = '<p class="def">⌘/ctrl-click neurons in the circuit to pin them and compare their shifts across the decisions shown in the matrix.</p>'; return; }
  const cols = visibleDecisions(SM);
  box.innerHTML = S.pins.map(n => { const [L, i] = SM.raw.neurons[n]; return `<div><div style="display:flex;justify-content:space-between"><b>L${L}·${i}</b><button data-unpin="${n}" style="font:inherit;background:none;border:0;color:${COL.muted};cursor:pointer">×</button></div><canvas data-pin="${n}" width="250" height="18" style="width:250px;height:18px"></canvas></div>`; }).join("");
  box.querySelectorAll("canvas[data-pin]").forEach(cv => { const n = +cv.dataset.pin, g = cv.getContext("2d"), w = 250 / Math.max(1, cols.length); cols.forEach((d, c) => { const [v] = cell(SM, d, n); g.fillStyle = shiftColour(v); g.fillRect(c * w, 0, Math.max(1, w), 18); }); });
  box.querySelectorAll("[data-unpin]").forEach(b => b.onclick = () => { S.pins = S.pins.filter(x => x !== +b.dataset.unpin); renderPins(); renderCircuit(); });
}

// ---------- neurons: map + dashboard ----------
let view = { yaw: -0.95, pitch: 0.45 }, pts = [];
const MAPC = Object.fromEntries(Object.entries(REASON).map(([k, v]) => [k, [v, ["twi_over_english", "mixed_over_english", "english_word_in_mix"].includes(k) ? "div" : "seq"]]));
async function loadCards() { try { return await getJSON(`cards-${model}.json`); } catch (e) { return null; } }
function mapColour(key, v) {
  const t = Math.max(-1, Math.min(1, v / 60));
  if (MAPC[key] && MAPC[key][1] === "div") return t >= 0 ? mix(COL.tw, 0.25 + 0.75 * t) : mix(COL.en, 0.25 + 0.75 * -t);
  return key.startsWith("scam") ? mix(COL.harm, 0.15 + 0.85 * Math.abs(t)) : mix(COL["en-tw"], 0.15 + 0.85 * Math.abs(t));
}
function renderMap(M) {
  const map = M.map, sel = $("colour");
  const keys = map ? Object.keys(map.scores).filter(k => MAPC[k]) : [];
  sel.innerHTML = keys.map(k => `<option value="${k}">${MAPC[k][0]}</option>`).join("");
  if (!map) { drawMap(null); return; }
  const cardIdx = new Map(map.cards.map((c, k) => [c[0] * map.width + c[1], k]));
  pts = map.layer.map((L, k) => ({ L, n: map.neuron[k], x: map.x[k] / 127, y: map.y[k] / 127, k, card: cardIdx.get(L * map.width + map.neuron[k]) }));
  const upd = () => { const k = sel.value; $("map-legend").innerHTML = MAPC[k][1] === "div" ? `<span><span class="sw" style="background:${COL.en}"></span>English-leaning</span><span><span class="sw" style="background:${COL.tw}"></span>Twi-leaning</span>` : `<span><span class="sw" style="background:${k.startsWith("scam") ? COL.harm : COL["en-tw"]}"></span>stronger response</span>`; drawMap(map); listCards(map); };
  sel.onchange = upd; upd();
  $("jump").onkeydown = e => { if (e.key !== "Enter") return; const [L, n] = e.target.value.split(/[:·.\s]/).map(Number); const k = map.cards.findIndex(c => c[0] === L && c[1] === n); if (k >= 0) openCard(L, n); else e.target.style.borderColor = COL.harm; };
}
function listCards(map) {
  const key = $("colour").value, sc = map.scores[key];
  const idx = new Map(map.layer.map((L, k) => [L * map.width + map.neuron[k], k]));
  const rows = map.cards.map(([L, n, r]) => [L, n, r, sc[idx.get(L * map.width + n)]]).sort((a, b) => Math.abs(b[3]) - Math.abs(a[3])).slice(0, 120);
  $("maplist").innerHTML = `<div class="def" style="margin-bottom:6px">cards, strongest on “${esc(MAPC[key][0])}”</div>` + rows.map(([L, n, r, v]) => `<button data-l="${L}" data-n="${n}">L${L}·${n} <span style="color:${COL.muted}">${(v / 20).toFixed(2)} · ${esc(REASON[r] || r)}</span></button>`).join("");
  $("maplist").querySelectorAll("button").forEach(b => b.onclick = () => openCard(+b.dataset.l, +b.dataset.n));
}
function project(p, layers, w, h) {
  const z = (p.L / Math.max(1, layers - 1)) * 2 - 1, cy = Math.cos(view.yaw), sy = Math.sin(view.yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  let X = p.x * cy + z * 1.1 * sy, Z = -p.x * sy + z * 1.1 * cy, Y = p.y * cp - Z * sp; Z = p.y * sp + Z * cp;
  const s = Math.min(w, h) * 0.3 / (1 + Z * 0.15); return [w / 2 + X * s, h / 2 - Y * s, Z];
}
function drawMap(map) {
  const cv = $("map"), dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr; const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  if (!map) return;
  const key = $("colour").value, sc = map.scores[key];
  const sel = S.neuron != null && SM ? SM.raw.neurons[S.neuron] : null;
  for (const [[sx, sy], p] of pts.map(p => [project(p, map.layers, w, h), p]).sort((a, b) => b[0][2] - a[0][2])) {
    const big = p.card != null; g.fillStyle = mapColour(key, sc[p.k]); const r = big ? 3.2 : 1.1; g.fillRect(sx - r, sy - r, 2 * r, 2 * r);
    if (big && sel && sel[0] === p.L && sel[1] === p.n) { g.strokeStyle = COL.good; g.lineWidth = 2; g.strokeRect(sx - 6, sy - 6, 12, 12); }
    p.sx = sx; p.sy = sy;
  }
  g.fillStyle = COL.muted; g.font = "11px JetBrains Mono, monospace";
  const f = project({ L: 0, x: -1.05, y: -1.05 }, map.layers, w, h), b = project({ L: map.layers - 1, x: -1.05, y: -1.05 }, map.layers, w, h);
  g.fillText("layer 0", f[0], f[1] + 14); g.fillText("layer " + (map.layers - 1), b[0], b[1] + 14);
}
(function mapEvents() {
  const cv = $("map"); let drag = null, moved = false;
  cv.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY]; moved = false; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", e => { if (!drag) return; const dx = e.clientX - drag[0], dy = e.clientY - drag[1]; if (Math.abs(dx) + Math.abs(dy) > 3) moved = true; view.yaw += dx * 0.008; view.pitch = Math.max(-1.2, Math.min(1.2, view.pitch + dy * 0.008)); drag = [e.clientX, e.clientY]; drawMap(DATA.models[model].map); });
  cv.addEventListener("pointerup", e => { drag = null; if (moved) return; const map = DATA.models[model].map; if (!map) return; const b = cv.getBoundingClientRect(), px = e.clientX - b.left, py = e.clientY - b.top; let best = null, d = 144; for (const p of pts) if (p.card != null) { const dd = (p.sx - px) ** 2 + (p.sy - py) ** 2; if (dd < d) { d = dd; best = p; } } if (best) openCard(best.L, best.n); });
  cv.addEventListener("keydown", e => { if (e.key === "ArrowLeft") view.yaw -= 0.1; else if (e.key === "ArrowRight") view.yaw += 0.1; else return; e.preventDefault(); drawMap(DATA.models[model].map); });
})();
async function openCard(L, n) {
  const m = DATA.models[model].steering ? await loadMatrix() : null;
  const idx = m ? m.raw.neurons.findIndex(x => x[0] === L && x[1] === n) : -1;
  S.neuron = idx >= 0 ? idx : null; S.neuronKey = [L, n];
  renderDashboard(idx >= 0 ? idx : null); drawMap(DATA.models[model].map); if (m) drawMatrix();
  showTab("insp", "neuron");
}
let ctxLang = "en-tw";
async function renderDashboard(n) {
  const box = $("dash"); const cards = await loadCards();
  const key = n != null && SM ? SM.raw.neurons[n] : S.neuronKey; if (!key || !cards) return;
  const c = cards.find(x => x.layer === key[0] && x.neuron === key[1]); if (!c) { box.innerHTML = '<p class="def">No card for this neuron.</p>'; return; }
  const lm = c.language_mean, mxl = Math.max(...lm.map(Math.abs), 1e-6);
  const langBars = ["en", "tw", "en-tw"].map((l, i) => `<div style="display:grid;grid-template-columns:96px 1fr 60px;gap:6px;align-items:center"><span style="color:${COL[l]}">${LANG[l]}</span><span style="height:9px;background:#222"><span style="display:block;height:9px;width:${(100 * Math.abs(lm[i]) / mxl).toFixed(1)}%;background:${COL[l]}"></span></span><span style="text-align:right">${lm[i].toFixed(3)}</span></div>`).join("");
  const chips = (arr, cls) => arr.map(([t, v]) => `<span class="chip" title="${v}">${esc(t)} <span style="color:${COL.muted}">${(+v).toFixed(2)}</span></span>`).join("");
  const wire = arr => arr.length ? arr.map(([L, i, w]) => `<button class="link" data-l="${L}" data-n="${i}" style="font:inherit;font-size:12px;background:none;border:0;color:${COL.ink2};cursor:pointer;padding:0 6px 0 0">L${L}·${i} <span style="color:${COL.muted}">${fx(w)}</span></button>`).join("") : '<span class="def">–</span>';
  const ctxs = c.contexts || {}, peakMax = Math.max(1e-6, ...Object.values(ctxs).flat().map(r => r.peak));
  const ctxHtml = (ctxs[ctxLang] || []).map(r => `<div class="ctx"><span class="meta">${r.peak.toFixed(2)} · ${esc(r.task_id.split("/")[1])}</span>${r.tokens.map((t, j) => `<span class="t ${j === r.at ? "peak" : ""}" style="background:${r.acts[j] > 0 ? seqColour(r.acts[j] / peakMax) : "transparent"}" data-v="${r.acts[j]}">${esc(t)}</span>`).join("")}</div>`).join("") || '<p class="def">No contexts for this language.</p>';
  let steer = '<p class="def">Not in the steering run.</p>';
  if (n != null && SM) steer = steeringSummary(n);
  box.innerHTML = `<div class="title"><h3>layer ${c.layer} · neuron ${c.neuron}</h3><span class="def">picked for: ${esc(REASON[c.reason] || c.reason)}</span><span class="def">${Object.entries(c.scores).filter(([k]) => REASON[k]).map(([k, v]) => `${esc(REASON[k])} ${(+v).toFixed(2)}`).join(" · ")}</span></div>
  <div class="side">
    <div><div class="def">mean activation at the end of the request</div>${langBars}</div>
    <div><div class="def">logit lens · raises</div>${chips(c.promotes)}</div>
    <div><div class="def">logit lens · lowers</div>${chips(c.suppresses)}</div>
    <div><div class="def">reads from (direct-path weight)</div>${wire(c.fed_by || [])}</div>
    <div><div class="def">read by (direct-path weight)</div>${wire(c.feeds || [])}</div>
  </div>
  <div class="mainc">
    <div><div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center"><span class="def">top activations in users' requests</span><span class="tabs">${["en-tw", "en", "tw"].map(l => `<button data-l="${l}" aria-pressed="${l === ctxLang}" style="--sel:${COL[l]}">${LANG[l]}</button>`).join("")}</span><span class="def">shading: activation / this neuron's highest peak; outlined: the peak token</span></div>${ctxHtml}</div>
    <div>${steer}</div>
  </div>`;
  box.querySelectorAll(".tabs button").forEach(b => b.onclick = () => { ctxLang = b.dataset.l; renderDashboard(n); });
  box.querySelectorAll("button.link").forEach(b => b.onclick = () => openCard(+b.dataset.l, +b.dataset.n));
  box.querySelectorAll(".ctx .t").forEach(t => { t.addEventListener("mouseenter", ev => showTip(ev, `activation ${(+t.dataset.v).toFixed(3)}`)); t.addEventListener("mouseleave", hideTip); });
  box.querySelectorAll("canvas[data-strip]").forEach(drawStrip);
}
function steeringSummary(n) {
  const sum = DATA.models[model].steering[S.margin]; const [L, i] = SM.raw.neurons[n];
  const doses = SM.raw.doses;
  const W = 300, H = 120;
  const series = []; for (const l of ORDER) for (const s of [1, 0]) { const v = sum.mean_shift[l] ? sum.mean_shift[l][n][s] : null; if (v) series.push({ l, s, v }); }
  const all = series.flatMap(x => x.v).filter(v => v != null); const mx = Math.max(...all.map(Math.abs), 0.2);
  const x = k => 40 + (W - 60) * (k / (doses.length - 1)), y = v => H / 2 - (H / 2 - 12) * v / mx;
  let g = `<svg viewBox="0 0 ${W} ${H + 16}" width="100%" style="max-width:420px"><line x1="40" x2="${W - 20}" y1="${y(0)}" y2="${y(0)}" stroke="${COL.muted}"/>`;
  for (const sr of series) g += `<polyline points="${sr.v.map((v, k) => `${x(k)},${y(v)}`).join(" ")}" fill="none" stroke="${COL[sr.l]}" stroke-width="2" ${sr.s ? "" : 'stroke-dasharray="4 3"'}><title>${LANG[sr.l]} push ${sr.s ? "up" : "down"}: ${sr.v.map(v => fx(v)).join(" / ")}</title></polyline>`;
  doses.forEach((d, k) => g += `<text x="${x(k)}" y="${H + 12}" text-anchor="middle">${d}</text>`);
  g += `<text x="4" y="${y(mx) + 4}">${fx(mx, 1)}</text><text x="4" y="${y(-mx) + 4}">${fx(-mx, 1)}</text></svg>`;
  const beats = ORDER.map(l => sum.share_beats_97[l] ? `${LANG[l]} ${pct(sum.share_beats_97[l][n][S.sign > 0 ? 1 : 0][S.dose])}` : "").join(" · ");
  // Task × language grid: mean shift and share beating 97% of random, this push, every decision.
  const famLang = {};
  SM.raw.decisions.forEach((dd, d) => { const [v, p] = cell(SM, d, n); if (Number.isNaN(v)) return; const c = ((famLang[dd.family] ||= {})[dd.language] ||= { s: 0, b: 0, k: 0 }); c.s += v; c.b += p >= 97; c.k++; });
  const grid = `<table class="grid"><thead><tr><th>task</th>${["en", "en-tw", "tw"].map(l => `<th style="text-align:center;color:${COL[l]}">${LANG[l]}</th>`).join("")}</tr></thead><tbody>${Object.keys(famLang).sort().map(f => `<tr><td>${esc(f)}</td>${["en", "en-tw", "tw"].map(l => { const c = famLang[f][l]; if (!c) return "<td></td>"; const m = c.s / c.k; return `<td class="cell" style="background:${shiftColour(m)};color:${COL.ink}" title="${esc(f)} · ${LANG[l]}: mean shift ${fx(m)} over ${c.k} decisions; beats 97% of random in ${c.b}">${fx(m, 1)} <span style="color:${COL.ink2}">${c.b}/${c.k}</span></td>`; }).join("")}</tr>`).join("")}</tbody></table>`;
  return `<div class="def">steering · mean ${S.margin} margin shift by dose (nats); solid = push up, dashed = push down; colour = request language</div>${g}
  <div class="def">share of decisions where this push beats 97% of random directions (push ${S.sign > 0 ? "up" : "down"}, ${doses[S.dose]}): ${beats}</div>
  <div class="def" style="margin-top:6px">per task and language (push ${S.sign > 0 ? "up" : "down"}, ${doses[S.dose]}): mean shift in nats · decisions where it beats 97% of random / decisions</div><div class="scroll">${grid}</div>
  <div class="def" style="margin-top:6px">every decision in the matrix's filter, coloured by shift; grouped by task</div><canvas data-strip="${n}" height="26" style="width:100%;height:26px"></canvas>`;
}
function drawStrip(cv) {
  const n = +cv.dataset.strip, cols = visibleDecisions(SM), w = cv.clientWidth || 600, dpr = devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = 26 * dpr; const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cw = w / Math.max(1, cols.length);
  cols.forEach((d, c) => { const [v] = cell(SM, d, n); g.fillStyle = shiftColour(v); g.fillRect(c * cw, 0, Math.max(1, cw), 18); g.fillStyle = COL[SM.raw.decisions[d].language]; g.fillRect(c * cw, 20, Math.max(1, cw), 6); });
  cv.onmousemove = ev => { const b = cv.getBoundingClientRect(), c = Math.floor((ev.clientX - b.left) / cw); const d = cols[c]; if (d == null) return; const [v, p] = cell(SM, d, n); showTip(ev, `${esc(SM.raw.decisions[d].id)} · ${KIND[SM.raw.decisions[d].kind]}<br>shift ${fx(v)} · ${p}th pct`); };
  cv.onmouseleave = hideTip; cv.onclick = ev => { const b = cv.getBoundingClientRect(), c = Math.floor((ev.clientX - b.left) / cw); if (cols[c] == null) return; S.decision = cols[c]; selectionChanged(); showTab("insp", "decision"); };
}

// ---------- anchors ----------
const PAY = ["initiate_transfer", "confirm_transfer"], PROTECT = ["report_fraud", "ask_user", "cancel_transfer"];
let linkKind = "links";
function renderAnchors(M) {
  const A = M.anchors || [], groups = {};
  for (const ep of A) { const [, fam, lang, seed] = ep.task_id.split("/"); (groups[fam + " · seed " + seed] ||= {})[lang] = ep; }
  const keys = Object.keys(groups).sort();
  $("a-controls").innerHTML = `<label class="def">task<select id="triple">${keys.map(k => `<option>${esc(k)}</option>`).join("")}</select></label><span class="seg">${[["links", "deletion links"], ["blocked", "attention-blocking links"]].map(([k, t]) => `<button data-k="${k}" aria-pressed="${k === linkKind}">${t}</button>`).join("")}</span><span class="def">${A.length ? A[0].replays + " replays per turn" : ""}${A[0] && A[0].blocked_layers ? ` · attention blocked in ${A[0].blocked_layers.length} full-attention layers` : ""}</span>`;
  const draw = () => {
    const g = groups[$("triple").value] || {};
    $("lanes").innerHTML = ORDER.map(l => lane(g[l], l)).join("");
    document.querySelectorAll(".lane").forEach(drawArcs);
    document.querySelectorAll(".step").forEach(el => { el.onclick = () => selectStep(A, el); el.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectStep(A, el); } }; });
  };
  $("triple").onchange = draw;
  document.querySelectorAll("#a-controls .seg button").forEach(b => b.onclick = () => { linkKind = b.dataset.k; renderAnchors(M); });
  draw(); renderKurtosis(M);
}
function lane(ep, l) {
  if (!ep) return `<div class="lane ${l === "en-tw" ? "center" : ""}"><h3><span style="color:${COL[l]}">${LANG[l]}</span></h3><span class="def">no replays</span></div>`;
  const out = ep.outcome_from_turn, inflow = {}; for (const [i, j, kl] of ep[linkKind] || []) inflow[i] = Math.max(inflow[i] || 0, kl);
  const turnOf = {}; ep.turn_step.forEach((s, t) => turnOf[s] = t);
  const steps = ep.steps.map((st, i) => {
    const t = turnOf[i]; let size, fill = "none", stroke = COL.muted, label;
    if (st.role === "assistant" && t != null) {
      const here = out[t], next = out[t + 1], imp = here == null ? 0 : Math.abs((next == null ? here : next) - here) / 2;
      size = 10 + 22 * Math.min(1, imp); fill = st.calls.some(c => PAY.includes(c)) ? COL.harm : st.calls.some(c => PROTECT.includes(c)) ? COL.good : COL.ink2; stroke = fill;
      label = `turn ${t} · replayed from here ${here == null ? "–" : fx(here, 1)}`;
    } else { size = 10 + 22 * Math.min(1, (inflow[i] || 0) / 1.0); label = st.role === "tool" ? "tool result" : "user"; }
    const what = st.role === "assistant" ? (st.calls.length ? st.calls.join(", ") : "reply") : st.role === "user" ? st.text : (st.results[0] || st.text);
    return `<div class="step" tabindex="0" data-task="${esc(ep.task_id)}" data-i="${i}"><span class="ring"><svg width="34" height="34"><circle cx="17" cy="17" r="${size / 2}" fill="${fill}" stroke="${stroke}" stroke-width="2"/></svg></span><span class="what"><b>${esc(label)}</b><br>${esc(String(what).slice(0, 120))}</span></div>`;
  }).join("");
  return `<div class="lane ${l === "en-tw" ? "center" : ""}" data-task="${esc(ep.task_id)}"><h3><span style="color:${COL[l]}">${LANG[l]}</span><span class="def">${esc(ep.task_id.split("/")[1])} · ${esc(ep.task_id.split("/")[3])}</span></h3>${steps}</div>`;
}
function drawArcs(laneEl) {
  const task = laneEl.dataset.task; if (!task) return;
  const ep = (DATA.models[model].anchors || []).find(a => a.task_id === task); if (!ep) return;
  const stepsEl = [...laneEl.querySelectorAll(".step")], base = laneEl.getBoundingClientRect();
  const yOf = i => { const r = stepsEl[i].getBoundingClientRect(); return r.top - base.top + r.height / 2; };
  const links = ep[linkKind] || [], mx = Math.max(...links.map(x => x[2]), 1e-6);
  let g = "";
  for (const [i, j, kl] of links) { const target = ep.turn_step[j]; if (target == null || !stepsEl[i] || !stepsEl[target]) continue; const y1 = yOf(i), y2 = yOf(target), bend = 6 + Math.min(40, Math.abs(y2 - y1) / 6); g += `<path d="M10,${y1} C${10 - bend},${y1} ${10 - bend},${y2} 10,${y2}" fill="none" stroke="${COL.act}" stroke-opacity="${0.25 + 0.75 * kl / mx}" stroke-width="${0.5 + 3.5 * kl / mx}"><title>step ${i} → turn ${j}: KL ${kl.toFixed(3)}</title></path>`; }
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.setAttribute("class", "arcs"); svg.setAttribute("width", 14); svg.setAttribute("height", base.height); svg.innerHTML = g; laneEl.appendChild(svg);
}
function stepMessages(ep) {
  // message index range of each step (steps start at message 1; a model turn includes its tool results)
  let k = 1; return ep.steps.map(st => { const n = st.role === "assistant" ? 1 + st.results.length : 1; const r = [k, k + n]; k += n; return r; });
}
function selectStep(A, el) {
  document.querySelectorAll(".step.on").forEach(s => s.classList.remove("on")); el.classList.add("on");
  showTab("anch", "step");
  const ep = A.find(a => a.task_id === el.dataset.task), i = +el.dataset.i, st = ep.steps[i];
  const out = (ep[linkKind] || []).filter(x => x[0] === i).sort((a, b) => b[2] - a[2]).slice(0, 6).map(x => `turn ${x[1]} (KL ${x[2].toFixed(3)})`);
  const range = stepMessages(ep)[i], sents = (ep.sentences || []).filter(s => s.message >= range[0] && s.message < range[1] && s.score != null);
  const smax = Math.max(1e-9, ...(ep.sentences || []).filter(s => s.score != null && s.role !== "system").map(s => s.score));
  const shaded = sents.length ? `<div><div class="def">receiver-head score per sentence in the last turn's view of this step (shading relative to the episode's highest non-system sentence)</div><div class="text" style="max-height:none">${sents.map(s => `<span class="sent" style="background:${seqColour(s.score / smax)}" title="${(s.score * 1000).toFixed(2)}e-3">${esc(s.text)}</span>`).join(" ")}</div></div>` : "";
  $("step-detail").innerHTML = `<div><b>${LANG[ep.task_id.split("/")[2]]} · step ${i} · ${esc(st.role)}${st.calls.length ? " · " + esc(st.calls.join(", ")) : ""}</b></div>${st.reasoning ? `<div><div class="def">reasoning</div><div class="text">${esc(st.reasoning)}</div></div>` : ""}${st.text ? `<div class="text">${esc(st.text)}</div>` : ""}${st.results.length ? `<div><div class="def">tool results</div><div class="text">${esc(st.results.join("\n"))}</div></div>` : ""}${out.length ? `<div class="def">${linkKind === "links" ? "deleting" : "blocking attention to"} this step moves: ${out.join(", ")}</div>` : ""}${shaded}`;
}
function renderKurtosis(M) {
  const R = M.receivers, box = $("kurt"); if (!R) { box.innerHTML = ""; $("kurt-top").innerHTML = ""; return; }
  const K = R.kurtosis, L = R.layers, H = K[0].length, top = new Map(R.heads.slice(0, 20).map((h, r) => [h.layer + ":" + h.head, r + 1]));
  const avail = (box.parentElement.clientWidth || 600) - 260;
  const c = Math.max(9, Math.min(26, Math.floor((avail - 48) / H)));
  const mx = Math.max(...K.flat().filter(v => v != null));
  let g = `<svg viewBox="0 0 ${52 + H * c} ${L.length * c + 30}" width="${52 + H * c}" role="img" aria-label="Receiver-head kurtosis by layer and head">`;
  L.forEach((layer, a) => { g += `<text x="44" y="${a * c + c * 0.72}" text-anchor="end">L${layer}</text>`; K[a].forEach((v, h) => { const r = top.get(layer + ":" + h); g += `<rect x="${50 + h * c}" y="${a * c}" width="${c - 1}" height="${c - 1}" fill="${v == null ? "#161616" : seqColour(Math.max(0, v) / mx)}" stroke="${r ? COL.good : "none"}" stroke-width="${r ? 1.5 : 0}"><title>L${layer} head ${h}: kurtosis ${v == null ? "–" : v.toFixed(2)}${r ? " · receiver #" + r : ""}</title></rect>`; }); });
  for (let h = 0; h < H; h += Math.max(1, Math.round(32 / c))) g += `<text x="${50 + h * c + c / 2}" y="${L.length * c + 14}" text-anchor="middle">${h}</text>`;
  g += `<text x="${50 + H * c}" y="${L.length * c + 27}" text-anchor="end">head →</text></svg>`;
  box.innerHTML = `<div class="kurt-wrap"><div class="scroll">${g}</div><div><div class="def">top receiver heads</div><table class="grid">${R.heads.slice(0, 20).map((h, r) => `<tr><td class="def">${r + 1}</td><td>L${h.layer} · head ${h.head}</td><td style="text-align:right">${h.kurtosis.toFixed(1)}</td></tr>`).join("")}</table><p class="def">${R.turns} turns</p></div></div>`;
  $("kurt-top").innerHTML = "";
}

// ---------- boot ----------
function render() {
  const M = DATA.models[model];
  document.querySelectorAll(".toggle button").forEach(b => b.setAttribute("aria-pressed", b.dataset.model === model));
  S.decision = null; S.neuron = null; S.pins = []; SM = null;
  renderBehaviour(M); renderReadouts(M); renderMap(M); renderAnchors(M); renderSteered(M); renderSweep(M);
  $("dash").innerHTML = '<p class="def">Select a neuron on the map, in the steering matrix or in the circuit.</p>';
  $("decision").innerHTML = '<p class="def">Select a cell.</p>';
  drawMatrix().then(renderCircuit).catch(e => { $("s-legend").innerHTML = `<span class="def">Steering files not available (${esc(e.message)}).</span>`; });
}
document.querySelectorAll(".toggle button").forEach(b => b.addEventListener("click", () => { model = b.dataset.model; store.set("sika-model", model); render(); }));
addEventListener("resize", () => { if (DATA) { drawMap(DATA.models[model].map); renderSweep(DATA.models[model]); if (SM) { drawMatrix(); renderCircuit(); } } });
// Shareable links: ?model=ministral&d=eval/promo_fee/en-tw/1000006@first_move&n=27:2472 (+ optional sign, dose, margin)
const Q = new URLSearchParams(location.search);
if (Q.get("model")) model = Q.get("model");
for (const k of ["sign", "dose"]) if (Q.get(k) != null) S[k] = +Q.get(k);
if (Q.get("margin")) S.margin = Q.get("margin");
async function applyLink() {
  if (!Q.get("d") && !Q.get("n")) return;
  const m = await loadMatrix();
  if (Q.get("d")) { const [id, kind] = Q.get("d").split("@"); const i = m.raw.decisions.findIndex(x => x.id === id && (!kind || x.kind === kind)); if (i >= 0) S.decision = i; }
  if (Q.get("n")) { const [L, n] = Q.get("n").split(":").map(Number); const j = m.raw.neurons.findIndex(x => x[0] === L && x[1] === n); if (j >= 0) S.neuron = j; }
  await drawMatrix(); await renderDecision(); await renderCircuit(); if (S.neuron != null) await renderDashboard(S.neuron);
}
function linkHere() {
  if (!SM) return location.href;
  const u = new URL(location.href.split("?")[0].split("#")[0]); u.searchParams.set("model", model);
  if (S.decision != null) { const d = SM.raw.decisions[S.decision]; u.searchParams.set("d", d.id + "@" + d.kind); }
  if (S.neuron != null) u.searchParams.set("n", SM.raw.neurons[S.neuron].join(":"));
  u.searchParams.set("sign", S.sign); u.searchParams.set("dose", S.dose); u.searchParams.set("margin", S.margin);
  return u.toString();
}
fetch("data.json").then(r => r.json()).then(d => { DATA = d; if (!DATA.models[model]) model = "qwen"; render(); applyLink().catch(() => {}); }).catch(e => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="def">Could not load data.json: ${esc(e.message)}</p>`); });
