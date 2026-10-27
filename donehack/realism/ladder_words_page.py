# ruff: noqa: E501
"""Page: distinct O-lens reads in the prompt (system + first user turn) that mention eval-ish words, per ladder rung.

uv run python realism/ladder_words_page.py <ladder_words.json> <out.html> [ib|email]
"""

import html
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ladder_words_labels import about_env

DS = "imports/olens_reads/2026-10-25_glm53_ib_user_dense/2026-10-25_glm53_ib_realistic_user_ladder"
ORDER = [
    "L1_rl_cues",
    "L2_rl",
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
EMAIL_DS = {
    "": "imports/olens_reads/2026-10-25_glm53_email_dense/2026-10-24_glm53_email_ambiguous",
    "part2": "imports/olens_reads/2026-10-25_glm53_email_dense/2026-10-24_glm53_email_ambiguous_part2",
    "part3": "imports/olens_reads/2026-10-25_glm53_ib_user_dense/2026-10-25_glm53_email_ambiguous_part3",
}
EMAIL_ORDER = [
    "base@low",
    "base_home@low",
    "base_done_neutral@low",
    "no_done@low",
    "harness@low",
    "no_criterion@low",
    "nc_casual@low",
    "plain_request@low",
    "asst_formal@low",
    "core_noreply@low",
    "core@low",
]


def email_ds(v):
    if v in ("base_done_neutral@low", "base_single@low"):
        return EMAIL_DS["part2"]
    if v in ("nc_casual@low", "plain_request@low", "asst_formal@low", "core_noreply@low"):
        return EMAIL_DS["part3"]
    return EMAIL_DS[""]


def email_rates():
    import glob

    out = {}
    for v in EMAIL_ORDER:
        x = [json.loads(Path(f).read_text())["record"] for f in glob.glob(str(Path(__file__).parent / "runs" / "email" / v / "*.json"))]
        out[v] = round(100 * sum(bool(r["picked_one"]) for r in x) / max(len(x), 1))
    return out


SHOW = [("benchmark", "var(--s1)"), ("simulat(ed|ion)", "var(--s2)"), ("eval(uation)", "var(--s3)"), ("grader/graded", "var(--s4)")]


def main():
    global ORDER, HACK
    d = json.loads(Path(sys.argv[1]).read_text())
    email = len(sys.argv) > 3 and sys.argv[3] == "email"
    metric = "picks one" if email else "hack"
    if email:
        ORDER, HACK = EMAIL_ORDER, email_rates()
    dsof = email_ds if email else (lambda v: DS)
    uniq = {v: {w: defaultdict(list) for w, _ in SHOW} for v in ORDER}
    for v in ORDER:
        for w, _ in SHOW:
            for h in d[v]["hits"].get(w, []):
                uniq[v][w][(h["tok"], h["read"])].append(h)
    n = {v: {w: len(uniq[v][w]) for w, _ in SHOW} for v in ORDER}
    env = {v: len({r for w, _ in SHOW for (_t, r) in uniq[v][w] if about_env(r)}) for v in ORDER}
    ymax = max(max(env.values()), 1)
    left, top, ph, gw = 44, 20, 200, (104 if email else 96)
    W = left + 20 + gw * len(ORDER)
    sy = lambda c: top + ph * (1 - c / ymax)
    svg = [f'<svg viewBox="0 0 {W} {top + ph + 60}" width="100%" role="img" aria-label="distinct reads mentioning each word per rung">']
    for t in range(0, ymax + 1, max(1, ymax // 4)):
        svg.append(
            f'<line x1="{left}" x2="{W - 10}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/><text x="{left - 6}" y="{sy(t) + 4:.1f}" class="ax" text-anchor="end">{t}</text>'
        )
    for i, v in enumerate(ORDER):
        x0 = left + 8 + i * gw
        c = env[v]
        x = x0 + 22
        svg.append(
            f'<rect x="{x}" y="{sy(c):.1f}" width="36" height="{max(sy(0) - sy(c), 1):.1f}" rx="3" fill="var(--s1)"><title>{v}: {c} distinct prompt reads about the environment being a benchmark/eval/simulation/graded</title></rect><text x="{x + 18}" y="{sy(c) - 4:.1f}" class="val" text-anchor="middle">{c}</text>'
        )
        svg.append(
            f'<a href="#{v}"><text x="{x0 + 40}" y="{top + ph + 16}" class="lab" text-anchor="middle">{html.escape(v.replace("@low", "") if email else v.split("_")[0])}</text><text x="{x0 + 40}" y="{top + ph + 31}" class="ax" text-anchor="middle">{"" if email else html.escape(v.split("_", 1)[1].replace("_", " ") if "_" in v and v[0] == "L" else "")}</text><text x="{x0 + 40}" y="{top + ph + 46}" class="ax" text-anchor="middle">{metric} {HACK[v]}%</text></a>'
        )
    svg.append("</svg>")
    rows = "".join(
        f"<tr><td><a href='#{v}'>{v}</a></td><td>{HACK[v]}%</td><td><b>{env[v]}</b></td>"
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
                lab = "<span class='tag env'>about the env</span>" if about_env(read) else "<span class=tag>about code</span>"
                lis.append(
                    f"<li>{lab} <span class=tag>{html.escape(w)}</span> at token <code>{html.escape(repr(tok))}</code> · {where} · <a href='http://bramble:8830/#dataset={dsof(v)}&item={html.escape(h['id'])}' target=_blank>open</a><br><span class=read>{html.escape(read)}</span></li>"
                )
        cards.append(
            f"<section id='{v}' class=card><h2>{v} <span class=muted>· {metric} {HACK[v]}% · {d[v]['reads']} greedy reads in the prompt</span></h2>{'<ul>' + ''.join(lis) + '</ul>' if lis else '<p class=muted>no prompt read mentions any of these words</p>'}</section>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{"Email" if email else "Ladder"} prompt reads</title>
<style>:root{{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--grid:#e6e5e0;--card:#fff;--s1:#2a78d6;--s2:#eb6834;--s3:#1f9e74;--s4:#8a5cd1}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--fg:#fff;--muted:#c3c2b7;--grid:#2e2e2c;--card:#222220;--s1:#3987e5;--s2:#d95926;--s3:#35b98c;--s4:#a27ce6}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--fg:#fff;--muted:#c3c2b7;--grid:#2e2e2c;--card:#222220;--s1:#3987e5;--s2:#d95926;--s3:#35b98c;--s4:#a27ce6}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:20px 16px}} h1{{font-size:21px;margin:0 0 6px}} h2{{font-size:16px;margin:0 0 6px}} .muted,.lead{{color:var(--muted)}} a{{color:inherit}}
.grid{{stroke:var(--grid)}} .ax{{fill:var(--muted);font-size:11px}} .lab{{fill:var(--fg);font-size:12px}} .val{{fill:var(--muted);font-size:10px}}
.lg{{margin-right:14px;font-size:13px}} .lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}}
.card{{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:12px 16px;margin:12px 0}} ul{{padding-left:18px;margin:4px 0}} li{{margin:6px 0}}
.env{{color:var(--fg);border-color:var(--s1)}} .read{{font:12px/1.4 ui-monospace,monospace}} .tag{{font-size:11px;border:1px solid var(--grid);border-radius:4px;padding:0 4px;color:var(--muted)}}
table{{border-collapse:collapse;font-size:13px;margin:10px 0}} td,th{{border-bottom:1px solid var(--grid);padding:3px 10px;text-align:right}} td:first-child,th:first-child{{text-align:left}}
.scroll{{overflow-x:auto}}</style></head><body>
<h1>Eval words in O-lens reads of the {"email" if email else "IB ladder"} prompts</h1>
<p class=lead>GLM verbalizer (RL), greedy, at every read position in the system prompt and first user turn (our dense reads). Counts are <b>distinct</b> reads. The prompt prefix is identical across a rung's episodes, so a read at a shared position comes back the same in all 64 episodes: it is one observation, not 64. None of these words appear in any rung's prompt text. {"Picks-one rates are from the email runs (low-sensitivity messages, n=64 per environment)." if email else "Hack rates are from the ladder runs (n=64 per rung)."}</p>
<p>Bars count distinct prompt reads where the word is about the environment itself (e.g. "the broader agent-evaluation benchmark", "this simulated Unix environment", "I will evaluate your work upon Done"). Reads where it is about the code ("a quick performance benchmark", "brute-force simulation", "pytest then evaluates", "the test file that evaluates our solution") are left out; the table also has the raw per-word counts. The labels are mine, by hand, in <code>realism/ladder_words_labels.py</code>, and every read below is tagged.</p>
<div class=scroll>{"".join(svg)}</div>
<table><tr><th>rung</th><th>{metric}</th><th>about the env</th>{"".join(f"<th>{html.escape(w)}</th>" for w, _ in SHOW)}<th>episodes</th></tr>{rows}</table>
{"".join(cards)}</body></html>"""
    Path(sys.argv[2]).write_text(page)
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
