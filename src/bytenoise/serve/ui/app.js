"use strict";
// All server-provided strings are inserted with textContent (never innerHTML).
const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k.includes("-")) e.setAttribute(k, v);
    else if (v !== false && v != null) e[k] = v;
  }
  for (const c of kids.flat()) e.append(c instanceof Node ? c : document.createTextNode(String(c ?? "")));
  return e;
};
const fmt = (v) => (v == null ? "-" : typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toPrecision(4)) : String(v));
const age = (t) => { const s = Date.now() / 1000 - t; return s < 90 ? `${s | 0}s` : s < 5400 ? `${(s / 60) | 0}m` : s < 172800 ? `${(s / 3600) | 0}h` : `${(s / 86400) | 0}d`; };

const token = $("token");
token.value = localStorage.getItem("bn_token") || "";
token.addEventListener("change", () => localStorage.setItem("bn_token", token.value));

window.addEventListener("unhandledrejection", (e) => e.preventDefault()); // api() already toasted the error
let errTimer;
function showErr(msg) {
  const e = $("err"); e.textContent = msg; e.hidden = false;
  clearTimeout(errTimer); errTimer = setTimeout(() => (e.hidden = true), 6000);
}
async function api(path, body) {
  const headers = {};
  if (token.value) headers.Authorization = `Bearer ${token.value}`;
  const opt = { headers };
  if (body !== undefined) { opt.method = "POST"; headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(`/api/${path}`, opt);
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) { showErr(j.error || `HTTP ${r.status}`); throw new Error(j.error); }
  return j;
}

// ---------- chart ----------
const PALETTE = ["#6aa7ff", "#4cc38a", "#e5b64a", "#ef6b6b", "#b48cff", "#5ed1d1", "#ff9f5a", "#c0c6d4"];
function drawChart(canvas, steps, series) {
  const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const g = canvas.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const pad = { l: 56, r: 12, t: 10, b: 22 };
  const pts = series.flatMap((s) => s.y.filter((v) => v != null && isFinite(v)));
  if (!pts.length) { g.fillStyle = "#8b93a7"; g.fillText("no data", 12, 20); return; }
  let lo = Math.min(...pts), hi = Math.max(...pts); if (lo === hi) { lo -= 1; hi += 1; }
  const allSteps = series.flatMap((s) => s.steps || steps);
  const x0 = Math.min(...allSteps), x1 = Math.max(...allSteps) || 1;
  const X = (x) => pad.l + ((x - x0) / (x1 - x0 || 1)) * (W - pad.l - pad.r);
  const Y = (y) => H - pad.b - ((y - lo) / (hi - lo)) * (H - pad.t - pad.b);
  g.strokeStyle = "#2a2f3a"; g.fillStyle = "#8b93a7"; g.font = "11px system-ui";
  for (let i = 0; i <= 4; i++) { const v = lo + ((hi - lo) * i) / 4, y = Y(v); g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke(); g.fillText(v.toPrecision(3), 4, y + 3); }
  g.fillText(String(x0), pad.l, H - 6); g.fillText(String(x1), W - pad.r - 30, H - 6);
  series.forEach((s, k) => {
    g.strokeStyle = PALETTE[k % PALETTE.length]; g.lineWidth = 1.5; g.beginPath(); let pen = false;
    s.y.forEach((v, i) => { if (v == null || !isFinite(v)) return; const px = X((s.steps || steps)[i]), py = Y(v); pen ? g.lineTo(px, py) : g.moveTo(px, py); pen = true; });
    g.stroke();
  });
}
function legend(series) {
  return el("div", { class: "keys" }, series.map((s, k) => el("span", {}, colorDot(PALETTE[k % PALETTE.length]), s.name)));
}
function colorDot(c) { const e = el("span", {}, "■ "); e.style.color = c; return e; }

// ---------- tabs ----------
const TABS = ["runs", "run", "train", "test", "data"];
const loaders = { runs: loadRuns, run: () => {}, train: loadTrain, test: loadTest, data: loadData };
let current = "runs", selectedRun = null, runsCache = [];
function setTab(name) {
  current = name;
  for (const t of TABS) $(`tab-${t}`).hidden = t !== name;
  for (const b of $("tabs").children) b.className = b.dataset.tab === name ? "on" : "";
  loaders[name]();
}
for (const t of TABS) $("tabs").append(el("button", { "data-tab": t, onclick: () => setTab(t) }, t[0].toUpperCase() + t.slice(1)));

// ---------- runs ----------
const compare = new Set();
async function loadRuns() {
  const root = $("tab-runs");
  const d = await api("runs"); runsCache = d.runs;
  const filter = el("input", { placeholder: "filter id / data / arch", oninput: () => draw() });
  const status = el("select", { onchange: () => draw() }, ["all", "running", "done", "timed_out", "incomplete", "failed", "stopped"].map((s) => el("option", { value: s }, s)));
  const body = el("tbody"), cmp = el("div");
  function draw() {
    const f = filter.value.toLowerCase();
    body.replaceChildren(...runsCache.filter((r) => (status.value === "all" || r.status === status.value) && `${r.id} ${r.data} ${r.arch} ${r.denoiser}`.toLowerCase().includes(f)).slice(0, 400).map((r) =>
      el("tr", { class: "click", onclick: (e) => { if (e.target.type !== "checkbox") openRun(r.id); } },
        el("td", {}, el("input", { type: "checkbox", checked: compare.has(r.id), onchange: (e) => { e.target.checked ? compare.add(r.id) : compare.delete(r.id); } })),
        el("td", {}, r.id), el("td", { class: `s-${r.status}` }, r.status), el("td", {}, [r.arch, r.denoiser].filter(Boolean).join("/") || "-"),
        el("td", {}, r.data || "-"), el("td", {}, r.step == null ? "-" : r.target_steps ? `${r.step}/${r.target_steps}` : r.step),
        el("td", {}, fmt(r.last_loss)), el("td", { class: "dim" }, age(r.mtime)))));
  }
  root.replaceChildren(
    el("div", { class: "row" }, filter, status, el("button", { class: "act sec", onclick: loadRuns }, "Refresh"),
      el("button", { class: "act", onclick: () => compareRuns(cmp) }, "Compare loss of checked"), el("span", { class: "dim" }, `${d.stats.runs} runs, ${d.stats.skipped} non-run entries skipped`)),
    cmp,
    el("table", {}, el("thead", {}, el("tr", {}, ["", "run", "status", "model", "data", "step", "loss", "age"].map((h) => el("th", {}, h)))), body));
  draw();
}
async function compareRuns(box) {
  if (!compare.size) return showErr("check at least one run");
  const ids = [...compare].slice(0, 8), series = [];
  let steps = [];
  for (const id of ids) {
    const m = await api(`runs/${id}/metrics?max=600`);
    if (!m.series.loss) continue;
    steps = m.steps.length > steps.length ? m.steps : steps;
    series.push({ name: id, steps: m.steps, y: m.series.loss });
  }
  const canvas = el("canvas");
  box.replaceChildren(el("div", { class: "card" }, el("b", {}, "loss"), canvas, legend(series)));
  drawChart(canvas, steps, series);
}

// ---------- run ----------
async function openRun(id) { selectedRun = id; setTab("run"); const root = $("tab-run"); root.replaceChildren(el("p", { class: "dim" }, "loading"));
  const [d, m] = await Promise.all([api(`runs/${id}`), api(`runs/${id}/metrics`)]);
  const keys = Object.keys(m.series).sort(), on = new Set(keys.includes("loss") ? ["loss"] : keys.slice(0, 1));
  const canvas = el("canvas"), keysBox = el("div", { class: "keys" }), leg = el("div");
  function paint() {
    const series = [...on].map((k) => ({ name: k, y: m.series[k] }));
    drawChart(canvas, m.steps, series); leg.replaceChildren(legend(series));
  }
  keysBox.append(...keys.map((k) => el("label", {}, el("input", { type: "checkbox", checked: on.has(k), onchange: (e) => { e.target.checked ? on.add(k) : on.delete(k); paint(); } }), " " + k)));
  const info = d.info;
  const evals = el("div", {}, d.evals.length ? d.evals.map((e) => el("button", { class: "act sec", onclick: () => showEval(e, evOut) }, e.file)) : el("span", { class: "dim" }, "no evals yet"));
  const evOut = el("pre", { hidden: true });
  root.replaceChildren(
    el("h2", {}, id), el("div", { class: "card" }, el("table", {}, Object.entries(info).filter(([k]) => k !== "extra").map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", {}, k === "mtime" ? new Date(v * 1000).toLocaleString() : fmt(v)))))),
    el("div", { class: "card" }, canvas, leg, keysBox), el("div", { class: "card" }, el("b", {}, "Eval results"), el("div", { class: "row" }, evals), evOut),
    d.config ? el("div", { class: "card" }, el("b", {}, "Resolved config"), el("pre", {}, JSON.stringify(d.config, null, 1))) : "");
  paint();
}
async function showEval(e, out) { const r = await api(`evals/${e.run}/${e.file}`); out.hidden = false; out.textContent = JSON.stringify(r.results, null, 1); }

// ---------- train ----------
let jobTimer;
async function loadTrain() {
  const root = $("tab-train");
  const exps = await api("experiments");
  const exp = el("select", {}, exps.map((e) => el("option", { value: e }, e)));
  const sets = el("textarea", { rows: 3, placeholder: "overrides, one per line: train.max_updates=50" });
  const dev = el("select", {}, ["cpu", "cuda"].map((d) => el("option", { value: d }, d)));
  const jobsBox = el("div"), log = el("pre", { hidden: true });
  async function launch() {
    const j = await api("jobs", { experiment: exp.value, sets: sets.value.split("\n").map((s) => s.trim()).filter(Boolean), device: dev.value });
    if (j.blocked) showErr(`queued: ${j.blocked}`);
    refresh();
  }
  async function refresh() {
    if (current !== "train") return;
    const js = await api("jobs");
    jobsBox.replaceChildren(el("table", {}, el("thead", {}, el("tr", {}, ["job", "experiment", "device", "status", "runs", ""].map((h) => el("th", {}, h)))),
      el("tbody", {}, js.map((j) => el("tr", { class: "click", onclick: () => showLog(j.id) },
        el("td", {}, j.id), el("td", {}, j.experiment), el("td", {}, j.device), el("td", { class: `s-${j.status}` }, j.status), el("td", {}, j.runs.join(", ")),
        el("td", {}, ["queued", "running"].includes(j.status) ? el("button", { class: "act sec", onclick: async (e) => { e.stopPropagation(); await api(`jobs/${j.id}/stop`, {}); refresh(); } }, "stop") : ""))))));
    clearTimeout(jobTimer); jobTimer = setTimeout(refresh, 4000);
  }
  async function showLog(id) { const l = await api(`jobs/${id}/log`); log.hidden = false; log.textContent = l.log || "(empty)"; }
  root.replaceChildren(
    el("div", { class: "card" }, el("div", { class: "row" }, el("label", {}, "experiment", exp), el("label", {}, "device", dev), el("button", { class: "act", onclick: launch }, "Launch")), sets),
    jobsBox, log);
  refresh();
}

// ---------- test ----------
async function loadTest() {
  const root = $("tab-test");
  const [d, protos, models] = await Promise.all([api("runs"), api("protocols"), api("models")]);
  const runs = d.runs.filter((r) => r.has_checkpoint);
  const optsFor = () => runs.map((r) => el("option", { value: r.id }, `${r.id} (${r.status})`));
  const runA = el("select", {}, optsFor()), runB = el("select", {}, el("option", { value: "" }, "(none)"), optsFor());
  const which = el("select", {}, ["ema", "raw"].map((w) => el("option", { value: w }, w)));
  const prompt = el("textarea", { rows: 3, placeholder: "prompt (empty for unconditional)" });
  const nBytes = el("input", { type: "number", value: 128, min: 1, max: 4096 }), seed = el("input", { type: "number", value: 0 });
  const temp = el("input", { type: "number", value: 0.8, step: 0.1, min: 0 }), topk = el("input", { type: "number", value: 40, min: 0 });
  const out = el("div", { class: "cols" });
  const proto = el("select", {}, protos.map((p) => el("option", { value: p }, p))), data = el("input", { placeholder: "dataset (optional)" });
  const evOut = el("pre", { hidden: true });
  const familyOf = (id) => { const r = runs.find((x) => x.id === id); return r && (r.arch === "diffusion" ? "diffusion" : "ar_lm"); };
  async function gen() {
    out.replaceChildren();
    for (const id of [runA.value, runB.value].filter(Boolean)) {
      const body = { run: id, which: which.value, prompt: prompt.value, n_bytes: +nBytes.value, seed: +seed.value };
      if (familyOf(id) === "ar_lm") body.sampler = { temperature: +temp.value, top_k: +topk.value };
      const card = el("div", { class: "card" }, el("b", {}, id), el("pre", {}, "running..."));
      out.append(card);
      try { const r = await api("generate", body); card.lastChild.textContent = r.text; } catch (e) { card.lastChild.textContent = String(e.message || e); }
    }
  }
  async function runEval() {
    evOut.hidden = false; evOut.textContent = "running...";
    try { const r = await api("eval", { run: runA.value, which: which.value, protocol: proto.value, data: data.value || undefined }); evOut.textContent = JSON.stringify(r.result.results, null, 1); }
    catch (e) { evOut.textContent = String(e.message || e); }
  }
  root.replaceChildren(
    el("div", { class: "card" }, el("b", {}, "Model"), el("div", { class: "row" }, el("label", {}, "run A", runA), el("label", {}, "run B (A/B)", runB), el("label", {}, "weights", which)),
      el("p", { class: "dim" }, `families: ${models.map((m) => `${m.family} [${m.capabilities.join(", ")}]`).join("; ")}. Runs on CPU.`)),
    el("div", { class: "card" }, el("b", {}, "Playground"), prompt,
      el("div", { class: "row" }, el("label", {}, "bytes", nBytes), el("label", {}, "seed", seed), el("label", {}, "temperature (AR)", temp), el("label", {}, "top_k (AR)", topk), el("button", { class: "act", onclick: gen }, "Generate")), out),
    el("div", { class: "card" }, el("b", {}, "Protocol eval (run A)"), el("div", { class: "row" }, el("label", {}, "protocol", proto), el("label", {}, "dataset", data), el("button", { class: "act", onclick: runEval }, "Run")), evOut));
}

// ---------- data ----------
async function loadData() {
  const ds = await api("datasets");
  $("tab-data").replaceChildren(el("table", {}, el("thead", {}, el("tr", {}, ["dataset", "train", "val", "test", "mix", "manifest"].map((h) => el("th", {}, h)))),
    el("tbody", {}, ds.map((d) => el("tr", {}, el("td", {}, d.name), ...["train", "val", "test"].map((s) => el("td", {}, d.bytes[s] == null ? "-" : `${(d.bytes[s] / 1e6).toFixed(1)} MB`)), el("td", {}, d.mix ? "yes" : ""), el("td", {}, d.manifest ? "yes" : ""))))));
}

async function pollGpu() {
  try {
    const g = await api("gpu");
    $("gpu").textContent = g.available ? `${g.name}: ${g.mem_used_mb}/${g.mem_total_mb} MB, ${g.util_pct}%${g.foreign_python && g.foreign_python.length ? " (busy)" : ""}` : "gpu: n/a";
  } catch (_) { $("gpu").textContent = "gpu: ?"; }
  setTimeout(pollGpu, 10000);
}
setTab("runs"); pollGpu();
