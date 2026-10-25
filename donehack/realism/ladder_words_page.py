# ruff: noqa: E501
"""Page: distinct O-lens reads in the prompt (system + first user turn) that mention eval-ish words, per ladder rung.

uv run python realism/ladder_words_page.py <ladder_words.json> <out.html>
"""

import html
import json
import sys
from collections import defaultdict
from pathlib import Path

DS = "imports/olens_reads/2026-10-25_glm53_ib_user_dense/2026-10-25_glm53_ib_realistic_user_ladder"
ORDER = [
    "base",
    "L1_rl_cues",
    "L2_rl",
    "nocrit_done",
    "L3_rl_nodone",
    "L4_assistant_formal",
    "L5_assistant_casual",
    "L6_casual_ide",
    "L7_repo",
    "L8_full",
]
HACK = {
    "base": 63,
    "nocrit_done": 41,
    "L1_rl_cues": 27,
    "L2_rl": 19,
    "L3_rl_nodone": 17,
    "L4_assistant_formal": 14,
    "L5_assistant_casual": 5,
    "L6_casual_ide": 6,
    "L7_repo": 3,
    "L8_full": 3,
}
SHOW = [("benchmark", "var(--s1)"), ("simulat(ed|ion)", "var(--s2)"), ("eval(uation)", "var(--s3)"), ("grader/graded", "var(--s4)")]


def main():
    d = json.loads(Path(sys.argv[1]).read_text())
    uniq = {v: {w: defaultdict(list) for w, _ in SHOW} for v in ORDER}
    for v in ORDER:
        for w, _ in SHOW:
            for h in d[v]["hits"].get(w, []):
                uniq[v][w][(h["tok"], h["read"])].append(h)
    n = {v: {w: len(uniq[v][w]) for w, _ in SHOW} for v in ORDER}
    shared = {v: {w: sum(len(x) >= 32 for x in uniq[v][w].values()) for w, _ in SHOW} for v in ORDER}
    ymax = max(max(x.values()) for x in n.values())
    left, top, ph, gw = 44, 20, 200, 96
    W = left + 20 + gw * len(ORDER)
    sy = lambda c: top + ph * (1 - c / ymax)
    svg = [f'<svg viewBox="0 0 {W} {top + ph + 60}" width="100%" role="img" aria-label="distinct reads mentioning each word per rung">']
    for t in range(0, ymax + 1, max(1, ymax // 4)):
        svg.append(
            f'<line x1="{left}" x2="{W - 10}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/><text x="{left - 6}" y="{sy(t) + 4:.1f}" class="ax" text-anchor="end">{t}</text>'
        )
    for i, v in enumerate(ORDER):
        x0 = left + 8 + i * gw
        for j, (w, col) in enumerate(SHOW):
            c = n[v][w]
            x = x0 + j * 20
            if c:
                svg.append(
                    f'<rect x="{x}" y="{sy(c):.1f}" width="17" height="{sy(0) - sy(c):.1f}" rx="3" fill="{col}"><title>{v} · {w}: {c} distinct reads ({shared[v][w]} at shared prompt positions)</title></rect><text x="{x + 8.5}" y="{sy(c) - 3:.1f}" class="val" text-anchor="middle">{c}</text>'
                )
        svg.append(
            f'<a href="#{v}"><text x="{x0 + 40}" y="{top + ph + 16}" class="lab" text-anchor="middle">{html.escape(v.split("_")[0])}</text><text x="{x0 + 40}" y="{top + ph + 31}" class="ax" text-anchor="middle">{html.escape(v.split("_", 1)[1].replace("_", " ") if "_" in v and v[0] == "L" else "")}</text><text x="{x0 + 40}" y="{top + ph + 46}" class="ax" text-anchor="middle">hack {HACK[v]}%</text></a>'
        )
    svg.append("</svg>")
    rows = "".join(
        f"<tr><td><a href='#{v}'>{v}</a></td><td>{HACK[v]}%</td>"
        + "".join(f"<td>{n[v][w]}</td>" for w, _ in SHOW)
        + f"<td>{d[v]['eps']}</td></tr>"
        for v in ORDER
    )
    cards = []
    for v in ORDER:
        lis = []
        for w, _ in SHOW:
            for (tok, read), hs in sorted(uniq[v][w].items(), key=lambda kv: -len(kv[1])):
                h = hs[0]
                where = f"every episode ({len(hs)} views), shared prompt position" if len(hs) >= 32 else f"{len(hs)} view(s)"
                lis.append(
                    f"<li><span class=tag>{html.escape(w)}</span> at token <code>{html.escape(repr(tok))}</code> · {where} · <a href='http://bramble:8830/#d={DS}&i={html.escape(h['id'])}' target=_blank>open</a><br><span class=read>{html.escape(read)}</span></li>"
                )
        cards.append(
            f"<section id='{v}' class=card><h2>{v} <span class=muted>· hack {HACK[v]}% · {d[v]['reads']} greedy reads in the prompt</span></h2>{'<ul>' + ''.join(lis) + '</ul>' if lis else '<p class=muted>no prompt read mentions any of these words</p>'}</section>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ladder prompt reads</title>
<style>:root{{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--grid:#e6e5e0;--card:#fff;--s1:#2a78d6;--s2:#eb6834;--s3:#1f9e74;--s4:#8a5cd1}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--fg:#fff;--muted:#c3c2b7;--grid:#2e2e2c;--card:#222220;--s1:#3987e5;--s2:#d95926;--s3:#35b98c;--s4:#a27ce6}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--fg:#fff;--muted:#c3c2b7;--grid:#2e2e2c;--card:#222220;--s1:#3987e5;--s2:#d95926;--s3:#35b98c;--s4:#a27ce6}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:20px 16px}} h1{{font-size:21px;margin:0 0 6px}} h2{{font-size:16px;margin:0 0 6px}} .muted,.lead{{color:var(--muted)}} a{{color:inherit}}
.grid{{stroke:var(--grid)}} .ax{{fill:var(--muted);font-size:11px}} .lab{{fill:var(--fg);font-size:12px}} .val{{fill:var(--muted);font-size:10px}}
.lg{{margin-right:14px;font-size:13px}} .lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}}
.card{{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:12px 16px;margin:12px 0}} ul{{padding-left:18px;margin:4px 0}} li{{margin:6px 0}}
.read{{font:12px/1.4 ui-monospace,monospace}} .tag{{font-size:11px;border:1px solid var(--grid);border-radius:4px;padding:0 4px;color:var(--muted)}}
table{{border-collapse:collapse;font-size:13px;margin:10px 0}} td,th{{border-bottom:1px solid var(--grid);padding:3px 10px;text-align:right}} td:first-child,th:first-child{{text-align:left}}
.scroll{{overflow-x:auto}}</style></head><body>
<h1>Eval words in O-lens reads of the ladder prompts</h1>
<p class=lead>GLM verbalizer (RL), greedy, at every read position in the system prompt and first user turn (Celeste's dense reads). Bars count <b>distinct</b> reads mentioning each word. The prompt prefix is identical across a rung's episodes, so a read at a shared position comes back the same in all 64 episodes: it is one observation, not 64. None of these words appear in any rung's prompt text. Hack rates are from the ladder runs (n=64 per rung; base and nocrit_done from earlier runs).</p>
<div>{"".join(f'<span class=lg><i style="background:{c}"></i>{html.escape(w)}</span>' for w, c in SHOW)}</div>
<div class=scroll>{"".join(svg)}</div>
<table><tr><th>rung</th><th>hack</th>{"".join(f"<th>{html.escape(w)}</th>" for w, _ in SHOW)}<th>episodes</th></tr>{rows}</table>
{"".join(cards)}</body></html>"""
    Path(sys.argv[2]).write_text(page)
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
