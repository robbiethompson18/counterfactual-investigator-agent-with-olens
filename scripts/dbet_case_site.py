"""Static build of the lens case page (transcript-viewer/web/case.html) for the Donation Bet forensics run.

    ../olens/.venv/bin/python scripts/dbet_case_site.py RUN_DIR OUT_DIR [--expview URL]

One case per transcript in RUN_DIR (scripts/dbet_forensics.py): the case page's token view, O-lens per layer with
per-bullet and joint FVE, NLA (L42) and J++, read from the cached reads instead of the viewer server. O-lens and NLA
reads with FVE < MIN_FVE are dropped (tokens left with none are drawn faded, as unread tokens are on the case page).
J++ keeps the layers the case page shows by default (L % 4 == 3 or L >= 56), top 8; --lite keeps 8 layers, for Artifacts.
Live tools that need the GPU (your bullets, residual writes, vectors, steering) are hidden. Case data ships as OUT_DIR/cases/<id>.gz.txt
(base64 gzip, fetched on demand), so the page can be published as an Artifact with those files.
"""

import argparse
import base64
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

MIN_FVE = 0.0  # default display cutoff (adjustable on the page)
FLOOR = {"v": 0.0}  # reads below this are not shipped at all (--floor)
LAYERS = [32, 40, 48]
WEB = Path(__file__).resolve().parents[1] / "transcript-viewer/web"
COMPARISON = {
    "good_below": "good vs bad",
    "good_above": "good vs bad",
    "beers_good_below": "good vs beers",
    "beers_good_above": "good vs beers",
    "nothing_good_below": "good vs nothing",
    "nothing_good_above": "good vs nothing",
}


def load(run):
    trs = [json.loads(x) for x in (run / "transcripts.jsonl").open()]
    reads, nla, jpp = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for p in sorted(run.glob("reads*.jsonl")):
        for x in map(json.loads, p.open()):
            reads[x["tid"]][(x["pos"], x["layer"])] = x
    for p in sorted(run.glob("nla*.jsonl")):
        for x in map(json.loads, p.open()):
            nla[x["tid"]][x["pos"]] = x["nla"]
    for p in sorted(run.glob("jpp*.jsonl")):
        for x in map(json.loads, p.open()):
            jpp[x["tid"]][x["pos"]] = x["jpp"]
    layers = json.loads((run / "jpp_layers.json").read_text())
    return trs, reads, nla, jpp, layers


def case(t, reads, nla, jpp, jl, dec, jpp_keep, jpp_k):
    toks, ids = t["tokens"], t["ids"]
    special = [bool(re.fullmatch(r"<\|[^|]+\|>|</?think>", s)) for s in toks]
    sp, sr, sy = t["sections"]["prompt"], t["sections"]["reasoning"], t["sections"]["reply"]
    u0 = next(i for i in range(sp[1]) if toks[i] == "user") + 2  # after "<|im_start|>user\n"
    u1 = max(i for i in range(sp[1]) if toks[i] == "<|im_end|>")
    y1 = sy[1] - 1 if toks[sy[1] - 1] == "<|im_end|>" else sy[1]
    msgs = [[u0, u1], [sr[0], sr[1]], [sy[0], y1]]
    starts = [m[0] for m in msgs] + [len(ids)]
    zones = [[i, min(s for s in starts if s > i)] for i, s in enumerate(toks) if s == "<|im_end|>" and any(x > i for x in starts)]
    rd = []
    for (pos, L), x in sorted(reads.items()):
        if x.get("fve", 0) >= FLOOR["v"]:
            rd.append(
                {
                    "pos": pos,
                    "layer": L,
                    "temperature": 0,
                    "n": 1,
                    "norm": None,
                    "samples": [{"text": x.get("text", ""), "bullets": x["bullets"], "bfve": x.get("bfve"), "fve": x["fve"]}],
                }
            )
    keep = [i for i, L in enumerate(jl) if jpp_keep(L)]
    ex = {}
    for pos in set(nla) | set(jpp):
        e = {}
        if pos in nla and nla[pos]["fve"] >= FLOOR["v"]:
            e["nla"] = nla[pos]
        if pos in jpp:
            e["jpp"] = {"layers": [jl[i] for i in keep], "top": [[[dec(tk), round(q, 3)] for tk, q in jpp[pos][i][:jpp_k]] for i in keep]}
        if e:
            ex[str(pos)] = e
    j = t["judge"]
    good = "≤" if t["arm"].endswith("below") else ">"
    other = "beers" if t["arm"].startswith("beers") else "nothing" if t["arm"].startswith("nothing") else "a bad cause"
    sub = (
        f"{COMPARISON[t['arm']]} · good cause wins if estimate {good} {t['meta']['threshold']:,} (other side: {other}) · "
        f"estimate {j['estimate']:,.0f} → {'good' if t['on_good_side'] else 'other'} side · "
        f"CoT acknowledges bet: {'yes' if j['acknowledges'] else 'no'}"
    )
    tr = {
        "tid": t["tid"],
        "tokens": toks,
        "special": special,
        "msgs": msgs,
        "roles": ["user", "thinking", "assistant"],
        "zones": zones,
        "reads": rd,
        "match": False,
        "sub": sub,
    }
    return {"tr": tr, "ex": ex}


SHIM = r"""const CASES = __CASES__;
const EXPVIEW = __EXPVIEW__;
const CACHE = {};
async function caseData(tid){
  const c = CASES.find(x => x.tid === tid) || CASES[0];
  if (!CACHE[c.tid]) { const t = await (await fetch('cases/' + c.file)).text(), bin = Uint8Array.from(atob(t.trim()), ch => ch.charCodeAt(0));
    CACHE[c.tid] = JSON.parse(await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).text()); }
  return CACHE[c.tid];
}
// the viewer server's endpoints, answered from the cached reads; anything live needs the GPU server
const api = async (p, body) => {
  const u = new URL(p, 'http://x'), tid = u.searchParams.get('tid');
  if (u.pathname === '/api/transcript') return (await caseData(tid)).tr;
  if (u.pathname === '/api/extras_all') return (await caseData(tid)).ex;
  if (u.pathname === '/api/status') return {pod: {state: 'static'}};
  if (u.pathname === '/api/vectors') return {vectors: [], judge: false};
  if (u.pathname === '/api/steer') return {runs: [], calib: null};
  if (u.pathname === '/api/locate') return {pattern: ''};
  throw new Error('static build: live tools need the GPU viewer server');
};"""  # noqa: E501


def page(cases, expview):
    h = (WEB / "case.html").read_text()
    idx = (WEB / "index.html").read_text()
    css = idx[idx.index("<style>") + 7 : idx.index("</style>")]

    def sub(a, b):
        nonlocal h
        assert a in h, a[:60]
        h = h.replace(a, b, 1)

    sub('<link rel="stylesheet" href="/style.css">', "<style>" + css + "</style>")
    sub("<title>Lens case study</title>", "<title>Donation Bet lens cases</title>")
    sub(
        "<h1>WeirdChat <span>×</span> lens case study</h1>",
        '<h1>Donation Bet <span>×</span> lens cases</h1><select id="casesel" style="width:auto;max-width:420px;padding:3px 8px;font-size:12.5px"></select><label class="muted" style="font-size:12px"><input type="checkbox" id="cstar" style="width:auto"> ★ full reads only</label>',  # noqa: E501
    )
    sub(
        '<button class="btn" id="fill"',
        '<label class="muted" style="font-size:12px">min FVE <input id="minfve" type="number" step="0.05" style="width:62px;padding:2px 4px"></label><button class="btn" id="fill" hidden',  # noqa: E501
    )
    sub(
        '<a class="btn" id="back" href="/">viewer</a>',
        '<a class="btn" href="pairs.html">matched pairs</a><a class="btn" id="back" target="_blank" rel="noopener">transcripts + judge (expview)</a>',  # noqa: E501
    )
    sub('<div class="card" id="vcard">', '<div class="card" id="vcard" hidden>')
    sub('<div class="sec"><h3>Residual writes', '<div class="sec" hidden><h3>Residual writes')
    sub('<div class="sec"><h3>Your bullets', '<div class="sec" hidden><h3>Your bullets')
    sub(
        "const api = async (p, body) => { const r = await fetch(p, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)} : {}); const j = await r.json(); if (j.error) throw new Error(j.error); return j; };",  # noqa: E501
        SHIM.replace("__CASES__", json.dumps(cases)).replace("__EXPVIEW__", json.dumps(expview)),
    )
    sub(
        "const LAYERS = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60];",
        f"const LAYERS = {json.dumps(LAYERS)};\nlet TH = (() => {{ try {{ const v = parseFloat(localStorage.getItem('dbet_minfve0')); return isNaN(v) ? {MIN_FVE} : v; }} catch (e) {{ return {MIN_FVE}; }} }})();",  # noqa: E501
    )
    sub(
        "st.reads = {}; for (const r of tr.reads) if (r.temperature === 0 && r.n === 1) st.reads[r.pos + '|' + r.layer] = r;",
        "st.reads = {}; for (const r of tr.reads) if (r.temperature === 0 && r.n === 1 && r.samples[0].fve >= TH) st.reads[r.pos + '|' + r.layer] = r;\n  st.nlaOk = p => (st.ex[p] || {}).nla && st.ex[p].nla.fve >= TH;",  # noqa: E501
    )
    sub(
        "const tid = hp.t || '64946fe603c30451478e522376ef4c0fae1fd52f53579b049678a3a6f742e256a1/t035';",
        "const tid = st.tid || hp.t || CASES[0].tid; st.tid = tid;",
    )
    sub("st.b = +(hp.b || am[0] + 19)", "st.b = +(hp.b || tr.tokens.length - 1)")
    sub(
        "$('#sub').textContent = `${tr.tid.split('/').pop()} · ${tr.match ? 'judge: match' : 'judge: no match'}`;",
        "$('#sub').textContent = tr.sub;",
    )
    sub(
        "api('/api/locate?tid=' + encodeURIComponent(tid)).then(l => $('#back').href = `/#p=${l.pattern}&t=${encodeURIComponent(tid)}`);",
        "$('#back').href = EXPVIEW; $('#back').title = 'expview item forensics/' + tid;",
    )
    sub("(user turn + first ${st.b - st.asst0 + 1} reply tokens)", "(whole transcript; faded = no O-lens/NLA read with FVE ≥ ${TH})")
    sub(
        "const done = (st.ex[p] || {}).jpp && (st.ex[p] || {}).nla && LAYERS.every(L => st.reads[p + '|' + L]);",
        "const done = LAYERS.some(L => st.reads[p + '|' + L]) || st.nlaOk(p);",
    )
    sub("$('#nla').innerHTML = x.nla ?", "$('#nla').innerHTML = st.nlaOk(p) ?")
    sub("'<div class=\"muted\">Not read yet.</div>'", '`<div class="muted">No O-lens read with FVE ≥ ${TH} at this token (yet).</div>`')
    sub("<span>‖h‖ ${Math.round(r.norm)}</span>", "${r.norm == null ? '' : `<span>‖h‖ ${Math.round(r.norm)}</span>`}")
    sub(
        "load(); setInterval(() => { if (!document.hidden) load(); }, 30000);",
        """{ const sel = $('#casesel');
  const fill = () => { const groups = {}, only = $('#cstar').checked;
    for (const c of CASES) if (!only || c.star) (groups[c.group] = groups[c.group] || []).push(c);
    sel.innerHTML = Object.entries(groups).map(([g, cs]) => `<optgroup label="${esc(g)}">` + cs.map(c => `<option value="${esc(c.tid)}" ${c.tid === st.tid ? 'selected' : ''}>${esc(c.label)}</option>`).join('') + '</optgroup>').join(''); };
  fill(); $('#cstar').onchange = () => { fill(); if (sel.value && sel.value !== st.tid) sel.onchange(); };
  sel.onchange = () => { st.tid = sel.value; st.sel = null; st.contrib = {}; try { location.hash = 't=' + encodeURIComponent(sel.value); } catch (e) {} load(); };
  if (hp.t && CASES.some(c => c.tid === hp.t)) sel.value = hp.t; }
{ const m = $('#minfve'); m.value = TH; m.onchange = () => { const v = parseFloat(m.value); if (!isNaN(v)) { TH = v; try { localStorage.setItem('dbet_minfve0', v); } catch (e) {} load(); } }; }
load();""",  # noqa: E501
    )
    return h


PAIRS_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Donation Bet matched pairs</title>
<style>__CSS__</style>
<style>
.pmain{padding:12px 16px 40px}
.pgrid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}
@media (max-width:1000px){.pgrid{grid-template-columns:1fr}}
.pcol{min-width:0}
.ptoks{max-height:46vh;overflow:auto;border:1px solid var(--line);border-radius:10px;padding:8px 10px;background:var(--panel);font-size:13px;line-height:1.7;white-space:pre-wrap;word-break:break-word}
.ptoks h4{font:600 11px var(--sans);text-transform:uppercase;letter-spacing:.06em;color:var(--ink3);margin:8px 0 2px;white-space:normal}
.ptoks h4:first-child{margin-top:0}
.prompt{white-space:pre-wrap;font-size:12.5px;border:1px solid var(--line);border-radius:10px;padding:8px 10px;background:var(--panel2);margin-bottom:8px;max-height:160px;overflow:auto}
.prompt mark{background:color-mix(in srgb,var(--accent) 35%,transparent);color:inherit;border-radius:3px;padding:0 1px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 8px;font-size:12px}
.chips span{border:1px solid var(--line);border-radius:9px;padding:0 8px}
.chips .y{background:var(--blue-soft)}
.tk.miss{opacity:.45}
.lens h3{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink3);margin:12px 0 6px;font-weight:650}
.jt td{padding:1px 3px;font-family:var(--mono);font-size:11.5px;vertical-align:top}
.jt span{padding:0 3px;margin-right:2px;border-radius:3px;display:inline-block}
.rates{font-size:13px;margin:6px 0 12px}
.ctx{font-family:var(--mono);font-size:12px;color:var(--ink2);white-space:pre-wrap;margin-bottom:6px}
</style></head><body>
<header><h1>Donation Bet <span>×</span> matched pairs</h1>
<label class="muted" style="font-size:12px">good wins if <select id="fdir" style="width:auto;padding:2px 6px;font-size:12px"><option value="">any</option><option value="above">estimate &gt; line</option><option value="below">estimate ≤ line</option></select></label>
<label class="muted" style="font-size:12px">estimates <select id="fpos" style="width:auto;padding:2px 6px;font-size:12px"><option value="">any</option><option value="aa">both above line</option><option value="bb">both below/at line</option><option value="split">one each side</option></select></label>
<label class="muted" style="font-size:12px">CoT acknowledges bet <select id="fack" style="width:auto;padding:2px 6px;font-size:12px"><option value="">any</option><option value="both">both</option><option value="neither">neither</option><option value="a">nothing side only</option><option value="b">beers side only</option></select></label>
<label class="muted" style="font-size:12px"><input type="checkbox" id="fstar" style="width:auto"> ★ full reads only</label>
<select id="psel" style="width:auto;max-width:420px;padding:3px 8px;font-size:12.5px"></select><span class="muted" id="pcount" style="font-size:12px"></span>
<label class="muted" style="font-size:12px">min FVE <input id="minfve" type="number" step="0.05" style="width:62px;padding:2px 4px"></label><div class="grow"></div><a class="btn" href="index.html">single cases</a><a class="btn" id="ev" target="_blank" rel="noopener">transcripts + judge (expview)</a></header>
<div class="pmain"><div id="phead"></div><div class="pgrid"><div class="pcol" id="colA"></div><div class="pcol" id="colB"></div></div></div>
<script>
const PAIRS = __PAIRS__, EXPVIEW = __EXPVIEW__;
let MIN_FVE = (() => { try { const v = parseFloat(localStorage.getItem('dbet_minfve0')); return isNaN(v) ? __MIN__ : v; } catch (e) { return __MIN__; } })();
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const tokStr = t => JSON.stringify(t).slice(1, -1);
const CACHE = {};
async function caseData(f){
  if (!CACHE[f]) { const t = await (await fetch('cases/' + f)).text(), bin = Uint8Array.from(atob(t.trim()), ch => ch.charCodeAt(0));
    CACHE[f] = JSON.parse(await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
  }
  const tr = CACHE[f].tr; tr.rd = {}; for (const r of tr.reads) if (r.samples[0].fve >= MIN_FVE) (tr.rd[r.pos] = tr.rd[r.pos] || []).push(r);
  return CACHE[f];
}
const S = {};
function secOf(tr, p){ for (let k = 0; k < tr.msgs.length; k++) { const [a, b] = tr.msgs[k]; if (p >= a && p < b) return [k, (p - a) / Math.max(1, b - a)]; }
  return [-1, p / tr.tokens.length]; }
function aligned(from, to, p){
  const [k, f] = secOf(from, p);
  if (k < 0) { // template tokens: the same landmark (k-th occurrence of this token string) on the other side
    const t = from.tokens[p]; let n = 0; for (let i = 0; i < p; i++) if (from.tokens[i] === t) n++;
    let m = 0; for (let i = 0; i < to.tokens.length; i++) if (to.tokens[i] === t) { if (m === n) return i; m++; }
    return Math.min(to.tokens.length - 1, Math.round(f * to.tokens.length)); }
  const [a, b] = to.msgs[k]; return Math.min(b - 1, a + Math.round(f * (b - a)));
}
function col(side){
  const P = S.pair, x = P[side], d = S[side], tr = d.tr, ex = d.ex;
  let h = `<h3 style="font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink3);margin:0 0 6px">${esc(x.label)}</h3>
    <div class="prompt">${x.prompt_html}</div>
    <div class="chips"><span>estimate <b>${Number(x.estimate).toLocaleString()}</b></span><span class="${x.good_side ? 'y' : ''}">${x.good_side ? 'good-cause side' : 'other side'}</span>
    <span class="${x.ack ? 'y' : ''}">CoT acknowledges bet: ${x.ack ? 'yes' : 'no'}</span><span>${esc(x.tid)}</span></div>
    <div class="ptoks" id="tk${side}">`;
  const names = ['prompt (user)', 'reasoning (CoT)', 'reply'];
  let at = 0;
  tr.msgs.forEach(([a, b], k) => {
    if (a > at) { h += '<h4>template</h4>' + spans(side, tr, ex, at, a); }
    h += `<h4>${names[k]}</h4>` + spans(side, tr, ex, a, b); at = b; });
  if (at < tr.tokens.length) h += '<h4>template</h4>' + spans(side, tr, ex, at, tr.tokens.length);
  h += `</div><div class="lens" id="ln${side}"><div class="muted" style="margin-top:8px">Click a token: its reads show here. The two sides select independently.</div></div>`;
  return h;
}
function spans(side, tr, ex, a, b){
  let h = ''; for (let p = a; p < b; p++) { const t = tr.tokens[p], cls = ['tk'];
    if (tr.special[p]) cls.push('sp');
    if (!(tr.rd[p] || ((ex[p] || {}).nla && ex[p].nla.fve >= MIN_FVE))) cls.push('miss');
    if (/^\s*\n\s*$/.test(t)) { cls.push('nlg'); h += `<span class="${cls.join(' ')}" data-s="${side}" data-p="${p}">↵</span>${esc(t)}`; continue; }
    h += `<span class="${cls.join(' ')}" data-s="${side}" data-p="${p}">${esc(t)}</span>`; }
  return h;
}
function lens(side, p){
  const d = S[side], tr = d.tr, x = d.ex[p] || {}, rs = (tr.rd[p] || []).slice().sort((u, v) => u.layer - v.layer);
  let h = `<h3>${esc(JSON.stringify(tr.tokens[p]))} · token ${p}</h3><div class="ctx">…${esc(tr.tokens.slice(Math.max(0, p - 24), p).join(''))}<b>${esc(tr.tokens[p])}</b></div>`;
  h += '<h3>NLA · L42</h3>' + (x.nla && x.nla.fve >= MIN_FVE ? `<div class="lr"><div class="hd"><span>reconstruction</span><span class="fve">cos <b>${x.nla.cos.toFixed(2)}</b> · FVE <b>${x.nla.fve.toFixed(2)}</b></span></div><div class="nla" style="white-space:pre-wrap;font-size:13px">${esc(x.nla.text)}</div></div>` : `<div class="muted">No NLA read with FVE ≥ ${MIN_FVE} (yet).</div>`);
  h += '<h3>O-lens</h3>' + (rs.length ? rs.map(r => { const s = r.samples[0];
    return `<div class="lr"><div class="hd"><b>L${r.layer}</b><span class="fve">joint FVE <b>${s.fve.toFixed(2)}</b></span></div>` + s.bullets.map((b, j) => { const f = s.bfve ? s.bfve[j] : null;
      return `<div class="bl"><span>${esc(b)}</span><div class="w"><i style="width:${f == null ? 0 : Math.min(100, 100 * f / Math.max(1e-9, s.fve))}%"></i></div><span class="n">${f == null ? '' : f.toFixed(2)}</span></div>`; }).join('') + '</div>'; }).join('')
    : `<div class="muted">No O-lens read with FVE ≥ ${MIN_FVE} (yet).</div>`);
  h += '<h3>J++ lens</h3>' + (x.jpp ? '<table class="jt">' + x.jpp.layers.map((L, i) => [L, x.jpp.top[i]]).reverse().map(([L, top]) => `<tr><td class="muted">L${L}</td><td>` +
    top.map(([t, q]) => `<span title="p ${q.toFixed(3)}" style="background:color-mix(in srgb, var(--zone-ink) ${Math.round(Math.min(1, q * 2) * 45)}%, transparent)">${esc(tokStr(t))}</span>`).join('') + '</td></tr>').join('') + '</table>' : '<div class="muted">No J++ read.</div>');
  $('#ln' + side).innerHTML = h;
}
function pick(side, p, follow){
  document.querySelectorAll(`#tk${side} .tk.sel`).forEach(e => e.classList.remove('sel'));
  const el = document.querySelector(`#tk${side} .tk[data-p="${p}"]`); if (el) { el.classList.add('sel'); if (!follow) {} else { const box = $('#tk' + side); box.scrollTop = el.offsetTop - box.offsetTop - box.clientHeight / 3; } }
  lens(side, p);
}
async function show(i){
  const P = PAIRS[i]; S.pair = P;
  [S.a, S.b] = await Promise.all([caseData(P.a.file), caseData(P.b.file)]);
  const r = P.rates;
  $('#phead').innerHTML = `<div class="card"><div style="font-size:15px;font-weight:600">${esc(P.question_text)}</div>
    <div class="rates">${esc(P.comparison)} · line <b>${P.line.toLocaleString()}</b> · % of the question's ~100 rollouts per arm with estimate above the line:
    <b>${(100 * r.a).toFixed(0)}%</b> ${esc(P.a.short)} (n ${r.na}; its mirror ${(100 * r.am).toFixed(0)}%, Δ ${(100 * (r.am - r.a)).toFixed(0)} pp) vs
    <b>${(100 * r.b).toFixed(0)}%</b> ${esc(P.b.short)} (n ${r.nb}; its mirror ${(100 * r.bm).toFixed(0)}%, Δ ${(100 * (r.bm - r.b)).toFixed(0)} pp).
    ${P.star ? '<b>★ full reads:</b> O-lens L40 + NLA on every token of both transcripts (L32/L48 on punctuation). ' : 'Punctuation, newline and template tokens have O-lens L40 + NLA; every token has J++. '}Each pair: same question, good cause on the same side of the line; only the other outcome differs (nothing vs beers). Matched on question + direction (the rollouts had no per-request seed). Reads with FVE &lt; ${MIN_FVE} are hidden (box in the header); faded tokens have none.</div></div>`;
  $('#colA').innerHTML = col('a'); $('#colB').innerHTML = col('b');
  for (const el of document.querySelectorAll('.ptoks .tk')) el.onclick = () => pick(el.dataset.s, +el.dataset.p, false);  // each side selects on its own
  for (const s of ['a', 'b']) { const th = S[s].tr.tokens.indexOf('</think>'); if (th >= 0) pick(s, th, true); }
}
const sel = $('#psel');
const pos = p => p.a.above && p.b.above ? 'aa' : !p.a.above && !p.b.above ? 'bb' : 'split';
const posTxt = {aa: 'both above', bb: 'both below', split: 'split'};
function fillSel(){
  const fd = $('#fdir').value, fp = $('#fpos').value, fa = $('#fack').value, fs = $('#fstar').checked, groups = {};
  const ack = p => p.a.ack && p.b.ack ? 'both' : !p.a.ack && !p.b.ack ? 'neither' : p.a.ack ? 'a' : 'b';
  PAIRS.forEach((p, i) => { if ((!fd || p.dir === fd) && (!fp || pos(p) === fp) && (!fa || ack(p) === fa) && (!fs || p.star)) (groups[p.comparison] = groups[p.comparison] || []).push([p, i]); });
  const n = Object.values(groups).reduce((k, v) => k + v.length, 0);
  sel.innerHTML = Object.entries(groups).map(([g, ps]) => `<optgroup label="${esc(g)}">` + ps.map(([p, i]) => `<option value="${i}">${esc(p.label)} · ${posTxt[pos(p)]}</option>`).join('') + '</optgroup>').join('');
  $('#pcount').textContent = `${n} of ${PAIRS.length} pairs`;
  if (n) show(+sel.value); else { $('#phead').innerHTML = '<div class="card muted">No pair matches these filters.</div>'; $('#colA').innerHTML = $('#colB').innerHTML = ''; }
}
$('#fdir').onchange = fillSel; $('#fpos').onchange = fillSel; $('#fack').onchange = fillSel; $('#fstar').onchange = fillSel;
sel.onchange = () => show(+sel.value); $('#ev').href = EXPVIEW;
{ const m = $('#minfve'); m.value = MIN_FVE; m.onchange = () => { const v = parseFloat(m.value); if (!isNaN(v)) { MIN_FVE = v; try { localStorage.setItem('dbet_minfve0', v); } catch (e) {} show(+sel.value); } }; }
fillSel();
</script></body></html>"""  # noqa: E501


def pairs_page(run, cases_by_tid, trs_by_tid, expview):
    import difflib

    pf = run / "pairs_nothing_beers.json"
    pairs = json.loads(pf.read_text()) if pf.exists() else []
    judged, cnt = {}, defaultdict(lambda: [0, 0])
    src = Path("data/runs/donation_bet_goodbad")
    for s in ["2026-10-22_bets", "2026-10-22_beers-nothing"]:
        judged |= {(j["arm"], j["item"], j["sample"]): j for j in map(json.loads, (src / s / "judged.jsonl").open())}
        for r in map(json.loads, (src / s / "rows.jsonl").open()):
            e = judged.get((r["arm"], r["item"], r["sample"]), {}).get("estimate") if "error" not in r else None
            if e is not None:
                c = cnt[(r["arm"], r["item"])]
                c[0] += e > r["meta"]["threshold"]
                c[1] += 1

    def diff(x, y):
        a, b = re.split(r"(\s+)", x), re.split(r"(\s+)", y)
        out = []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            seg = "".join(a[i1:i2]).replace("&", "&amp;").replace("<", "&lt;")
            out.append(seg if op == "equal" else f"<mark>{seg}</mark>")
        return "".join(out)

    sf = run / "showcase.json"
    star = {(p["a"], p["b"]) for p in json.loads(sf.read_text())["pairs"]} if sf.exists() else set()
    out = []
    for p in pairs:
        ta, tb = trs_by_tid[p["a"]], trs_by_tid[p["b"]]
        pa, pb = ta["messages"][0]["content"], tb["messages"][0]["content"]
        q = re.search(r"Question: (.*)", pa).group(1)
        other = lambda t: (
            "good vs nothing" if t["arm"].startswith("nothing") else "good vs beers" if t["arm"].startswith("beers") else "good vs bad"
        )
        side = lambda t, other=other: (
            f"{other(t)} · good wins if the estimate exceeds the line"
            if t["arm"].endswith("above")
            else f"{other(t)} · good wins if it does not exceed"
        )
        mirror = lambda arm: arm[: -len("above")] + "below" if arm.endswith("above") else arm[: -len("below")] + "above"
        ent = lambda t, pr, alt, side=side, other=other: {
            "tid": t["tid"],
            "file": cases_by_tid[t["tid"]]["file"],
            "label": side(t),
            "short": other(t),
            "prompt_html": diff(pr, alt),
            "estimate": t["judge"]["estimate"],
            "above": t["judge"]["estimate"] > t["meta"]["threshold"],
            "good_side": t["on_good_side"],
            "ack": t["judge"]["acknowledges"],
        }
        ka, na = cnt[(ta["arm"], ta["item"])]
        kb, nb = cnt[(tb["arm"], tb["item"])]
        kam, nam = cnt[(mirror(ta["arm"]), ta["item"])]
        kbm, nbm = cnt[(mirror(tb["arm"]), tb["item"])]
        out.append(
            {
                "comparison": p["comparison"],
                "question": p["question"],
                "dir": "above" if ta["arm"].endswith("above") else "below",
                "question_text": q,
                "line": ta["meta"]["threshold"],
                "star": (p["a"], p["b"]) in star,
                "label": f"{'★ ' if (p['a'], p['b']) in star else ''}{p['question']} · "
                f"{'ack' if ta['judge']['acknowledges'] else 'no ack'} / {'ack' if tb['judge']['acknowledges'] else 'no ack'}",
                "a": ent(ta, pa, pb),
                "b": ent(tb, pb, pa),
                "rates": {"a": ka / na, "na": na, "b": kb / nb, "nb": nb, "am": kam / nam, "bm": kbm / nbm},
            }
        )
    idx = (WEB / "index.html").read_text()
    css = idx[idx.index("<style>") + 7 : idx.index("</style>")]
    return (
        PAIRS_HTML.replace("__CSS__", css)
        .replace("__PAIRS__", json.dumps(out))
        .replace("__EXPVIEW__", json.dumps(expview))
        .replace("__MIN__", str(MIN_FVE))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--lite", action="store_true", help="J++ at 8 layers (L % 8 == 7, and 62), for an Artifact-sized build")
    ap.add_argument("--jpp-k", type=int, default=8)
    ap.add_argument("--pairs-only", action="store_true", help="rewrite pairs.html (and index.html) only; case files as they are")
    ap.add_argument("--floor", type=float, default=0.0, help="drop O-lens/NLA reads below this FVE from the data")
    ap.add_argument("--single-cases-url", default="index.html", help="link from the pairs page to the single-case page")
    ap.add_argument("--expview", default="https://claude.ai/artifact/Qb8tg4N6vu9iv4tP93e2VJ")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    FLOOR["v"] = a.floor
    (out / "cases").mkdir(parents=True, exist_ok=True)
    trs, reads, nla, jpp, jl = load(run)
    keep = (lambda L: L % 8 == 7 or L == 62) if a.lite else (lambda L: L % 4 == 3 or L >= 56)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    memo = {}
    dec = lambda i: memo.setdefault(i, tok.decode([i]))
    order = ["good_below", "good_above", "beers_good_below", "beers_good_above", "nothing_good_below", "nothing_good_above"]
    sf = run / "showcase.json"
    stars = set(json.loads(sf.read_text())["tids"]) if sf.exists() else set()
    cases = []
    for t in sorted(trs, key=lambda t: (order.index(t["arm"]), not t["judge"]["acknowledges"], not t["on_good_side"], t["item"])):
        f = hashlib.sha1(t["tid"].encode()).hexdigest()[:16] + ".gz.txt"
        if not a.pairs_only:
            c = case(t, reads[t["tid"]], nla[t["tid"]], jpp[t["tid"]], jl, dec, keep, a.jpp_k)
            (out / "cases" / f).write_text(base64.b64encode(gzip.compress(json.dumps(c, separators=(",", ":")).encode(), 9)).decode())
        good = "≤" if t["arm"].endswith("below") else ">"
        cases.append(
            {
                "tid": t["tid"],
                "file": f,
                "group": f"{COMPARISON[t['arm']]} · good wins if {good} line",
                "star": t["tid"] in stars,
                "label": f"{'★ ' if t['tid'] in stars else ''}{t['item']} s{t['sample']} · "
                f"ack {'yes' if t['judge']['acknowledges'] else 'no'} · {'good' if t['on_good_side'] else 'other'} side",
            }
        )
    (out / "index.html").write_text(page(cases, a.expview))
    (out / "pairs.html").write_text(
        pairs_page(run, {c["tid"]: c for c in cases}, {t["tid"]: t for t in trs}, a.expview).replace(
            'href="index.html">single cases', f'href="{a.single_cases_url}" target="_blank" rel="noopener">single cases'
        )
    )
    sizes = sum(p.stat().st_size for p in (out / "cases").glob("*.gz.txt"))
    print(f"{out}/index.html + {len(cases)} cases ({sizes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
