"use strict";
// bnk web UI: chat with streaming + live engine telemetry. No dependencies.

const $ = (s) => document.querySelector(s);
const store = {
  get(k, d) { try { const v = localStorage.getItem("bnk." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("bnk." + k, JSON.stringify(v)); } catch {} },
};

// ------------------------------------------------------------------ state
let convs = store.get("convs", []);
let current = store.get("current", null);
let settings = Object.assign({ system: "", temperature: 0.7, top_p: 0.95, top_k: 20, max_tokens: 8192,
  presence_penalty: 0, seed: 0, think: "low", key: "" }, store.get("settings", {}));
let busy = false, abortCtl = null;

function save() { store.set("convs", convs); store.set("current", current); }
function conv() { return convs.find((c) => c.id === current); }
function newConv() {
  const c = { id: Math.random().toString(36).slice(2, 10), title: "New chat", messages: [], created: Date.now() };
  convs.unshift(c); current = c.id; save(); renderConvs(); renderThread();
}

// ------------------------------------------------------------------ markdown
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const KW = /\b(def|class|return|if|elif|else|for|while|in|import|from|as|with|try|except|finally|raise|yield|lambda|and|or|not|is|None|True|False|const|let|var|function|new|this|async|await|switch|case|break|continue|default|struct|enum|template|typename|public|private|protected|static|void|int|float|double|char|bool|auto|unsigned|namespace|using|include|fn|pub|mut|impl|trait|match|go|func|package|type|interface|nullptr|true|false|null|undefined|export|extends|implements|throw|catch|do|sizeof|constexpr|inline|virtual|override|self)\b/;
function highlight(code) {
  const re = /(\/\/[^\n]*|#(?!include)[^\n]*|\/\*[\s\S]*?\*\/|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]\w*(?=\s*\())|[A-Za-z_]\w*/g;
  let out = "", last = 0, m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    const t = m[0];
    let cls = null;
    if (t.startsWith("//") || t.startsWith("/*") || (t.startsWith("#") && !/^#include/.test(t))) cls = "tk-c";
    else if (/^["'`]/.test(t)) cls = "tk-s";
    else if (/^\d/.test(t)) cls = "tk-n";
    else if (KW.test(t) && t.match(KW)[0] === t) cls = "tk-k";
    else if (m[1] && /^[A-Za-z_]/.test(t)) cls = "tk-f";
    out += cls ? `<span class="${cls}">${esc(t)}</span>` : esc(t);
    last = m.index + t.length;
  }
  return out + esc(code.slice(last));
}
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${esc(codes[+i])}</code>`);
}
function markdown(src, streaming) {
  const lines = src.replace(/\r/g, "").split("\n");
  let html = "", i = 0;
  while (i < lines.length) {
    const ln = lines[i];
    const fence = ln.match(/^\s*```\s*([\w+#.-]*)/);
    if (fence) {
      const lang = fence[1] || "code";
      const body = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
      i++;
      const code = body.join("\n");
      html += `<div class="code"><div class="code-head"><span>${esc(lang)}</span><button data-copy>copy</button></div><pre><code>${highlight(code)}</code></pre></div>`;
      continue;
    }
    if (/^\s*$/.test(ln)) { i++; continue; }
    const h = ln.match(/^(#{1,6})\s+(.*)/);
    if (h) { html += `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`; i++; continue; }
    if (/^\s*(---|\*\*\*|___)\s*$/.test(ln)) { html += "<hr>"; i++; continue; }
    if (/^\s*\|.*\|\s*$/.test(ln) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const row = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const head = row(ln);
      i += 2;
      let t = "<table><thead><tr>" + head.map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>";
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) t += "<tr>" + row(lines[i++]).map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>";
      html += t + "</tbody></table>";
      continue;
    }
    if (/^\s*>/.test(ln)) {
      const q = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) q.push(lines[i++].replace(/^\s*>\s?/, ""));
      html += `<blockquote>${markdown(q.join("\n"))}</blockquote>`;
      continue;
    }
    const li = ln.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)/);
    if (li) {
      const ordered = /\d/.test(li[2]);
      let t = ordered ? "<ol>" : "<ul>";
      while (i < lines.length) {
        const m = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)/);
        if (!m) {
          if (/^\s{2,}\S/.test(lines[i])) { t = t.replace(/<\/li>$/, " " + inline(lines[i].trim()) + "</li>"); i++; continue; }
          break;
        }
        t += `<li>${inline(m[3])}</li>`;
        i++;
      }
      html += t + (ordered ? "</ol>" : "</ul>");
      continue;
    }
    const para = [];
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !/^\s*(```|#{1,6}\s|>|[-*+]\s|\d+[.)]\s|\|)/.test(lines[i])) para.push(lines[i++]);
    if (!para.length) para.push(lines[i++]);
    html += `<p>${inline(para.join("\n")).replace(/\n/g, "<br>")}</p>`;
  }
  if (streaming) html += '<span class="cursor"></span>';
  return html;
}

// ------------------------------------------------------------------ rendering
function renderConvs() {
  const list = $("#conv-list");
  list.innerHTML = "";
  for (const c of convs) {
    const d = document.createElement("div");
    d.className = "conv" + (c.id === current ? " active" : "");
    d.textContent = c.title;
    d.title = c.title;
    d.onclick = () => { if (busy) return; current = c.id; save(); renderConvs(); renderThread(); };
    const del = document.createElement("button");
    del.className = "del"; del.textContent = "✕"; del.title = "Delete";
    del.onclick = (e) => {
      e.stopPropagation();
      if (busy && c.id === current) return;
      convs = convs.filter((x) => x.id !== c.id);
      if (current === c.id) current = convs[0] ? convs[0].id : null;
      save(); renderConvs(); renderThread();
    };
    d.appendChild(del);
    list.appendChild(d);
  }
}

function msgEl(m) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + m.role;
  if (m.role === "user") {
    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = m.content;
    wrap.appendChild(b);
    return wrap;
  }
  wrap.innerHTML = `<div class="avatar">bn</div><div class="body"><div class="think" hidden><div class="think-head"><span class="chev">▸</span><span class="label">Reasoning</span><span class="secs"></span></div><div class="think-body"></div></div><div class="content"></div><div class="meta"></div></div>`;
  wrap.querySelector(".think-head").onclick = () => wrap.querySelector(".think").classList.toggle("open");
  updateMsg(wrap, m, false);
  return wrap;
}

function updateMsg(el, m, streaming) {
  const th = el.querySelector(".think");
  if (m.reasoning) {
    th.hidden = false;
    th.classList.toggle("live", streaming && !m.content);
    el.querySelector(".think-body").textContent = m.reasoning;
    const secs = m.think_secs ? ` · ${m.think_secs.toFixed(1)}s` : "";
    el.querySelector(".secs").textContent = secs;
    el.querySelector(".label").textContent = streaming && !m.content ? "Reasoning…" : "Reasoning";
  }
  el.querySelector(".content").innerHTML = m.error ? `<div class="err">${esc(m.error)}</div>` :
    markdown(m.content || "", streaming && (!m.reasoning || m.content));
  const meta = el.querySelector(".meta");
  meta.innerHTML = "";
  const s = m.stats;
  if (s) {
    const chip = (k, v, hot) => { const c = document.createElement("span"); c.className = "chip" + (hot ? " hot" : ""); c.innerHTML = `${k} <b>${v}</b>`; meta.appendChild(c); };
    if (s.tps) chip("decode", `${s.tps.toFixed(1)} t/s`, true);
    if (s.completion_tokens != null) chip("tokens", s.completion_tokens);
    if (s.prefill_tps) chip("prefill", `${Math.round(s.prefill_tps)} t/s`);
    if (s.reused) chip("cached", `${s.reused} tok`);
    if (s.tokens_per_round) chip("MTP", `${s.tokens_per_round.toFixed(2)} t/round`);
    if (s.expert_miss_rate != null) chip("miss", `${(s.expert_miss_rate * 100).toFixed(1)}%`);
  }
}

function renderThread() {
  const th = $("#thread");
  th.querySelectorAll(".msg").forEach((n) => n.remove());
  const c = conv();
  $("#empty").hidden = !!(c && c.messages.length);
  if (!c) return;
  for (const m of c.messages) th.appendChild(msgEl(m));
  th.scrollTop = th.scrollHeight;
}

// ------------------------------------------------------------------ sending
async function send(text) {
  if (busy || !text.trim()) return;
  if (!conv()) newConv();
  const c = conv();
  c.messages.push({ role: "user", content: text });
  if (c.title === "New chat") c.title = text.slice(0, 60);
  const am = { role: "assistant", content: "", reasoning: "" };
  c.messages.push(am);
  save(); renderConvs();
  $("#empty").hidden = true;
  const th = $("#thread");
  th.appendChild(msgEl(c.messages[c.messages.length - 2]));
  const el = msgEl(am);
  th.appendChild(el);
  th.scrollTop = th.scrollHeight;

  const msgs = [];
  if (settings.system.trim()) msgs.push({ role: "system", content: settings.system });
  for (const m of c.messages.slice(0, -1)) {
    if (m.error) continue;
    const o = { role: m.role, content: m.content };
    if (m.role === "assistant" && m.reasoning) o.reasoning_content = m.reasoning;
    msgs.push(o);
  }
  const body = { model: "bnk", messages: msgs, stream: true, temperature: +settings.temperature, top_p: +settings.top_p,
    top_k: +settings.top_k, max_tokens: +settings.max_tokens, presence_penalty: +settings.presence_penalty };
  if (+settings.seed) body.seed = +settings.seed;
  if (settings.think === "off") body.chat_template_kwargs = { enable_thinking: false };
  else body.reasoning_effort = settings.think;

  busy = true; setBusy(true);
  abortCtl = new AbortController();
  const t0 = performance.now();
  let tThinkEnd = null, raf = 0;
  const repaint = () => { raf = 0; updateMsg(el, am, true); if (nearBottom()) th.scrollTop = th.scrollHeight; };
  try {
    const headers = { "Content-Type": "application/json" };
    if (settings.key) headers.Authorization = "Bearer " + settings.key;
    const res = await fetch("/v1/chat/completions", { method: "POST", headers, body: JSON.stringify(body), signal: abortCtl.signal });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error?.message || res.statusText);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, k); buf = buf.slice(k + 2);
        const line = chunk.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const data = line.slice(6);
        if (data === "[DONE]") continue;
        const j = JSON.parse(data);
        const ch = j.choices && j.choices[0];
        if (ch && ch.delta) {
          if (ch.delta.reasoning_content) am.reasoning += ch.delta.reasoning_content;
          if (ch.delta.content) { if (!tThinkEnd && am.reasoning) tThinkEnd = performance.now(); am.content += ch.delta.content; }
          if (ch.delta.tool_calls) am.content += "\n\n```json\n" + JSON.stringify(ch.delta.tool_calls, null, 2) + "\n```";
        }
        if (j.timings) am.stats = Object.assign({}, j.timings);
        if (!raf) raf = requestAnimationFrame(repaint);
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") am.error = String(e.message || e);
  } finally {
    if (am.reasoning) am.think_secs = ((tThinkEnd || performance.now()) - t0) / 1000;
    if (am.stats) am.stats.completion_tokens = am.stats.completion_tokens ?? undefined;
    busy = false; setBusy(false);
    updateMsg(el, am, false);
    save();
  }
}
function nearBottom() { const t = $("#thread"); return t.scrollHeight - t.scrollTop - t.clientHeight < 140; }
function setBusy(b) { $("#send").hidden = b; $("#stop").hidden = !b; }

// ------------------------------------------------------------------ telemetry
const spark = { dec: [], pf: [], max: 60 };
let lastTok = null;
function setGauge(id, frac, text) {
  const g = $(id);
  g.querySelector(".g-fg").style.strokeDashoffset = String(314 * (1 - Math.max(0, Math.min(1, frac))));
  g.querySelector("b").textContent = text;
}
function drawSpark() {
  const cv = $("#spark"), dpr = devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
  const x = cv.getContext("2d");
  x.setTransform(dpr, 0, 0, dpr, 0, 0);
  x.clearRect(0, 0, w, h);
  const all = spark.dec.concat(spark.pf.map((v) => v / 10));
  const mx = Math.max(20, ...all) * 1.15;
  x.strokeStyle = "rgba(124,146,255,0.08)"; x.lineWidth = 1;
  for (let i = 1; i < 4; i++) { x.beginPath(); x.moveTo(0, (h * i) / 4); x.lineTo(w, (h * i) / 4); x.stroke(); }
  const line = (arr, col, fill, scale) => {
    if (arr.length < 2) return;
    const step = w / (spark.max - 1), off = spark.max - arr.length;
    x.beginPath();
    arr.forEach((v, i) => { const px = (off + i) * step, py = h - ((v * scale) / mx) * h; i ? x.lineTo(px, py) : x.moveTo(px, py); });
    x.strokeStyle = col; x.lineWidth = 2; x.shadowColor = col; x.shadowBlur = 10; x.stroke(); x.shadowBlur = 0;
    if (fill) {
      x.lineTo(w, h); x.lineTo(off * step, h); x.closePath();
      const g = x.createLinearGradient(0, 0, 0, h); g.addColorStop(0, fill); g.addColorStop(1, "transparent");
      x.fillStyle = g; x.fill();
    }
  };
  line(spark.pf, "#ff4fd2", null, 0.1);
  line(spark.dec, "#38e8ff", "rgba(56,232,255,0.18)", 1);
}
async function poll() {
  try {
    const r = await fetch("/api/stats");
    const s = await r.json();
    const e = s.engine || {};
    $("#model-name").textContent = s.model + (e.mtp ? "  ·  MTP" : "");
    const dot = $("#status-dot");
    dot.className = "dot " + (!s.alive ? "down" : s.active ? "busy" : "idle");
    // live decode rate from the active request, else the last one
    let live = 0;
    if (s.active && s.active.gen_tokens != null) {
      const now = performance.now();
      if (lastTok && lastTok.id === s.active.id) {
        const dt = (now - lastTok.t) / 1000, dn = s.active.gen_tokens - lastTok.n;
        if (dt > 0.2) live = dn / dt;
      }
      lastTok = { id: s.active.id, t: now, n: s.active.gen_tokens };
    } else lastTok = null;
    const tps = live || e.last_tps || 0;
    spark.dec.push(live || 0); spark.pf.push(s.active && !s.active.gen_tokens ? (e.last_prefill_tps || 0) : 0);
    if (spark.dec.length > spark.max) { spark.dec.shift(); spark.pf.shift(); }
    drawSpark();
    const f1 = (v) => (v ? v.toFixed(1) : "–");
    $("#tk-tps").textContent = f1(tps); $("#t-tps").firstChild.nodeValue = f1(tps);
    $("#tk-pf").textContent = e.last_prefill_tps ? Math.round(e.last_prefill_tps) : "–";
    $("#tk-tpr").textContent = e.last_tokens_per_round ? e.last_tokens_per_round.toFixed(2) : "–";
    $("#t-pf").textContent = e.last_prefill_tps ? Math.round(e.last_prefill_tps) : "–";
    $("#t-tpr").textContent = e.last_tokens_per_round ? e.last_tokens_per_round.toFixed(2) : "–";
    $("#t-acc").textContent = e.last_accept ? (e.last_accept * 100).toFixed(0) : "–";
    $("#t-miss").textContent = e.last_miss_rate != null ? (e.last_miss_rate * 100).toFixed(1) : "–";
    if (e.experts_total) {
      const g = e.experts_resident / e.experts_total;
      $("#tier-gpu").style.width = (g * 100).toFixed(1) + "%";
      $("#tier-ram").style.width = ((1 - g) * 100).toFixed(1) + "%";
      $("#t-res").textContent = `${e.experts_resident} (${(g * 100).toFixed(0)}%)`;
      $("#t-nres").textContent = `${e.experts_total - e.experts_resident}`;
      $("#t-swaps").textContent = `${e.cache_swaps} swaps · ${e.expert_cache_gb.toFixed(1)} GB`;
    }
    if (e.vram_total_mb) setGauge("#g-vram", e.vram_used_mb / e.vram_total_mb, (e.vram_used_mb / 1024).toFixed(1) + "G");
    if (e.ram_total_mb) setGauge("#g-ram", 1 - e.ram_free_mb / e.ram_total_mb, ((e.ram_total_mb - e.ram_free_mb) / 1024).toFixed(0) + "G");
    if (e.n_ctx) {
      setGauge("#g-ctx", e.pos / e.n_ctx, e.pos > 999 ? (e.pos / 1000).toFixed(1) + "k" : String(e.pos));
      $("#ctx-hint").textContent = `${e.pos.toLocaleString()} / ${e.n_ctx.toLocaleString()} ctx`;
    }
    const rows = (s.recent || []).slice(-10).reverse().map((r) =>
      `<tr><td>${r.prompt_tokens}</td><td>${r.gen_tokens}</td><td>${r.prefill_tps ? Math.round(r.prefill_tps) : "–"}</td><td>${r.tps ? r.tps.toFixed(1) : "–"}</td><td>${r.tokens_per_round ? r.tokens_per_round.toFixed(2) : "–"}</td></tr>`).join("");
    $("#req-rows").innerHTML = rows;
    $("#tel-foot").innerHTML = `${e.requests || 0} requests · ${(e.gen_tokens || 0).toLocaleString()} tokens generated<br>` +
      `${e.cpu_threads || "?"} CPU expert threads · uptime ${Math.round(s.uptime / 60)} min` + (s.waiting ? ` · ${s.waiting} queued` : "");
  } catch (err) {
    $("#status-dot").className = "dot down";
  }
}

// ------------------------------------------------------------------ settings + wiring
function bindSettings() {
  const map = { "#s-system": "system", "#s-temp": "temperature", "#s-topp": "top_p", "#s-topk": "top_k", "#s-max": "max_tokens",
    "#s-pres": "presence_penalty", "#s-seed": "seed", "#s-key": "key" };
  for (const [sel, key] of Object.entries(map)) {
    const el = $(sel);
    el.value = settings[key];
    el.oninput = () => { settings[key] = el.type === "number" || el.type === "range" ? +el.value : el.value; store.set("settings", settings); labels(); };
  }
  const labels = () => { $("#s-temp-v").textContent = (+settings.temperature).toFixed(2); $("#s-topp-v").textContent = (+settings.top_p).toFixed(2); };
  labels();
  const seg = $("#think-seg");
  const paint = () => seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === settings.think));
  seg.querySelectorAll("button").forEach((b) => b.onclick = () => { settings.think = b.dataset.v; store.set("settings", settings); paint(); });
  paint();
  const open = (o) => { $("#drawer").hidden = !o; $("#scrim").hidden = !o; };
  $("#open-settings").onclick = () => open(true);
  $("#close-settings").onclick = $("#scrim").onclick = () => open(false);
}

function init() {
  bindSettings();
  const input = $("#input");
  const grow = () => { input.style.height = "auto"; input.style.height = Math.min(220, input.scrollHeight) + "px"; };
  input.oninput = grow;
  input.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); } };
  $("#composer").onsubmit = (e) => { e.preventDefault(); const t = input.value; input.value = ""; grow(); send(t); };
  $("#stop").onclick = () => abortCtl && abortCtl.abort();
  $("#new-chat").onclick = () => { if (!busy) newConv(); };
  $("#clear-all").onclick = () => { if (busy) return; convs = []; current = null; save(); renderConvs(); renderThread(); };
  $("#toggle-tel").onclick = () => { $("#layout").classList.toggle("no-tel"); store.set("tel", !$("#layout").classList.contains("no-tel")); };
  if (store.get("tel", true) === false) $("#layout").classList.add("no-tel");
  $("#suggests").querySelectorAll("button").forEach((b) => b.onclick = () => send(b.textContent));
  $("#thread").addEventListener("click", (e) => {
    const b = e.target.closest("[data-copy]");
    if (!b) return;
    navigator.clipboard?.writeText(b.closest(".code").querySelector("code").innerText);
    b.textContent = "copied"; setTimeout(() => (b.textContent = "copy"), 1200);
  });
  if (!conv() && convs.length) current = convs[0].id;
  renderConvs(); renderThread();
  poll(); setInterval(poll, 1000);
  addEventListener("resize", drawSpark);
  input.focus();
}
init();
