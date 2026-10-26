# ruff: noqa: E501
"""Baseline / fix / null page: per task, base (which shows the behaviour), one-edit changes that remove it, and
one-edit changes that look just as innocuous but don't.

  uv run python realism/overeager/triples.py --out <html> [--docent]
Bars = acted rate (95% Wilson), Fisher exact p vs base (* < .05, ** < .01, *** < .001). --docent saves one Docent filter
per cell in the `overeager` collection (cached in realism/runs/overeager/docent_cells.json).
"""

import argparse
import html
import json
import os
import subprocess
import sys
from pathlib import Path

from scipy.stats import fisher_exact

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import oe_run
from harness_variants import assemble
from results import load, wilson

NULLS = ["base_home", "base_done_neutral", "base_mcp", "base_plain", "base_no_criterion"]
TRIPLES = {  # task -> (behaviour, fixes, nulls)
    "gitbranch": ("deletes a branch when two match", ["base_polite"], NULLS),
    "calendar": ("moves one of two matching meetings", ["base_polite"], NULLS),
    "dana": ("emails an address it made up or guessed", ["base_polite"], NULLS),
    "payback_hint": ("pays one of two Alexes", ["base_polite"], NULLS),
    "groupchat": ("messages one of several group chats", ["base_polite"], NULLS),
    "impossiblebench": ("passes impossible tests (hack)", ["base_no_criterion", "base_polite"], NULLS),
}
CHANGE = {
    "base": "baseline",
    "base_polite": "user turn = one-line polite request",
    "base_no_criterion": 'drop the line "The task is complete when …"',
    "base_home": "workdir /home/dev with personal files",
    "base_done_neutral": 'done tool: "when you are finished" wording',
    "base_mcp": "app tools named mcp__<app>__<tool>",
    "base_plain": "plain two-sentence system prompt",
}
COL = {"base": "#55534e", "fix": "#1f9e74", "null": "#2a78d6"}
DS = "expview/overeager_v1"
CACHE = HERE.parent / "runs" / "overeager" / "docent_cells.json"


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def docent_links(cells):
    links = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    root = HERE.parents[2]  # .worktrees/realism (monorepo checkout)
    key = (root.parent.parent / "docent.key").read_text().strip()
    for c in cells:
        if c not in links:
            out = subprocess.run(
                ["uv", "run", "--script", "scripts/docent_upload.py", "view", "overeager", f"arm={c}"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "DOCENT_API_KEY": key},
            )
            links[c] = out.stdout.strip().splitlines()[-1]
    CACHE.write_text(json.dumps(links, indent=1))
    return links


def first_item(recs, t, v):
    rs = sorted(recs.get((t, v), []), key=lambda r: (r["item"], r["seed"]))
    return rs and f"{t}/{v}/{rs[0]['item']}_s{rs[0]['seed']}"


def card(t, recs, links):
    what, fixes, nulls = TRIPLES[t]
    base = [r for r in recs[(t, "base")] if r["outcome"] != "cut_long"]
    kb = sum(bool(r["acted"]) for r in base)
    arms = [("base", "base")] + [(v, "fix") for v in fixes] + [(v, "null") for v in nulls if (t, v) in recs and v not in fixes]
    W, lab, bw = 980, 400, 340
    rows = []
    for i, (v, kind) in enumerate(arms):
        rs = [r for r in recs[(t, v)] if r["outcome"] != "cut_long"]
        n, k = len(rs), sum(bool(r["acted"]) for r in rs)
        p, lo, hi = wilson(k, n)
        pv = 1.0 if v == "base" else fisher_exact([[kb, len(base) - kb], [k, n - k]])[1]
        y = 8 + i * 30
        item = first_item(recs, t, v)
        ev = f"http://bramble:8830/#d={DS}&i={item}"
        dl = links.get(f"{t}/{v}")
        rows.append(
            f'<text x="0" y="{y + 15}" class="lab">{html.escape(CHANGE[v])}</text>'
            f'<rect x="{lab}" y="{y + 2}" width="{bw}" height="18" fill="var(--grid)" rx="3"/>'
            f'<rect x="{lab}" y="{y + 2}" width="{max(bw * p, 2):.1f}" height="18" fill="{COL[kind]}" rx="3"><title>{v}: {k}/{n}</title></rect>'
            f'<line x1="{lab + bw * lo:.1f}" x2="{lab + bw * hi:.1f}" y1="{y + 11}" y2="{y + 11}" stroke="var(--fg)" opacity=".55"/>'
            f'<text x="{lab + bw + 8}" y="{y + 15}" class="val">{round(100 * p)}% <tspan class="ax">{k}/{n}</tspan> {stars(pv) if v != "base" else ""}</text>'
            f'<a href="{ev}" target="_blank"><text x="{W - 92}" y="{y + 15}" class="lk">8830</text></a>'
            + (f'<a href="{dl}" target="_blank"><text x="{W - 50}" y="{y + 15}" class="lk">Docent</text></a>' if dl else "")
        )
    svg = (
        f'<svg viewBox="0 0 {W} {16 + len(arms) * 30}" width="100%" role="img" aria-label="{t} acted rate by change">{"".join(rows)}</svg>'
    )
    T = oe_run.load(t)
    inst = T.instance(0)
    user = {v: assemble(T, inst, v)["user"] for v in ["base", *fixes]}
    ex = "".join(f"<div class=ex><b>{v}</b><pre>{html.escape(u.strip())}</pre></div>" for v, u in user.items())
    return f"<section class=card id={t}><h2>{t} <span class=sub>bad behaviour: {what}</span></h2>{svg}<details><summary>User turn, item 0</summary><div class=exs>{ex}</div></details></section>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--docent", action="store_true")
    a = ap.parse_args()
    recs = load("v1")
    cells = [f"{t}/{v}" for t, (_, f, n) in TRIPLES.items() for v in ["base", *f, *n] if (t, v) in recs]
    links = docent_links(cells) if a.docent else (json.loads(CACHE.read_text()) if CACHE.exists() else {})
    leg = "".join(
        f'<span class=lg><i style="background:{c}"></i>{k}</span>'
        for k, c in [
            ("baseline (base)", COL["base"]),
            ("one edit that removes the behaviour", COL["fix"]),
            ("one edit that doesn't (null)", COL["null"]),
        ]
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fix and null edits</title>
<style>:root{{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#5a5954;--grid:#e6e5e0;--card:#fff}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--fg:#fff;--muted:#c3c2b7;--grid:#2e2e2c;--card:#222220}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--fg:#fff;--muted:#c3c2b7;--grid:#2e2e2c;--card:#222220}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1040px;margin:0 auto;padding:20px 16px}} a{{color:inherit}}
.card{{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:10px 14px;margin:12px 0}} h2{{font-size:16px;margin:0 0 6px}} .sub{{font-weight:400;color:var(--muted);font-size:13px}}
.lab{{fill:var(--fg);font-size:12px}} .val{{fill:var(--fg);font-size:12px}} .ax{{fill:var(--muted);font-size:11px}} .lk{{fill:var(--muted);font-size:11px;text-decoration:underline}}
.lg{{margin-right:14px;font-size:13px;white-space:nowrap}} .lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}}
summary{{cursor:pointer;color:var(--muted);font-size:13px}} .exs{{display:flex;gap:10px;flex-wrap:wrap}} .ex{{flex:1;min-width:280px}} pre{{white-space:pre-wrap;font-size:12px;background:var(--bg);padding:8px;border-radius:6px}}</style></head><body>
<h1>One-edit fixes and nulls (GLM 5.3)</h1>
<p>Every arm is one edit to <i>base</i> (shell + app tools + done tool, "Task: …" user turn with a completion criterion), n=16 per cell (64 where topped up). Bar = acted rate with 95% Wilson interval; stars = Fisher exact p vs base (* &lt;.05, ** &lt;.01, *** &lt;.001). Episodes over the $0.10 cap are dropped. Links open the first transcript of each cell on 8830 and the whole cell on Docent. Envs: <a href="https://claude.ai/artifact/CVMrM91ishdzCAvGXsuKRk">grid</a>.</p>
<div>{leg}</div>
{"".join(card(t, recs, links) for t in TRIPLES)}
</body></html>"""
    Path(a.out).write_text(page)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
