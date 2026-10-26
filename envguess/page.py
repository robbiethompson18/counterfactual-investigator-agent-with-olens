# ruff: noqa: E501
"""Results page (single HTML, inline SVG charts) from score.py's summary.

~/workspace/donehack/.venv/bin/python envguess/page.py --summary data/envguess/summary_v1.json --links data/envguess/links_v1.json --out <html>
"""

import argparse
import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SETTINGS = REPO / "data" / "envguess" / "settings"
ARMS = ["env", "env_tx", "tx"]
ARM_NAME = {"env": "Env only", "env_tx": "Env + 1 transcript", "tx": "Transcript only"}
FAM_NAME = {
    "overeager": "Over-eagerness (GLM 5.3, today)",
    "realism": "ImpossibleBench / email realism (GLM 5.3)",
    "robbie/scheming": "Robbie: scheming (Qwen)",
    "robbie/hiring-eval": "Robbie: hiring eval (Qwen)",
    "robbie/hiring-resume": "Robbie: hiring resume (Qwen)",
    "robbie/dumb-user-report": "Robbie: hiding own mistakes (Qwen)",
    "robbie/dumb-user-exploit": "Robbie: exploiting naive users (Qwen)",
    "robbie/shutdown": "Robbie: shutdown (Qwen)",
    "robbie/whistleblow": "Robbie: whistleblowing (Qwen)",
}
E = html.escape


def pct(x):
    return f"{100 * x:.0f}%"


def bars(groups, metrics, colors, width=640, row=22, label_w=170, title=""):
    """groups: [(label, {metric: [mean, lo, hi]})]; horizontal bars 0..100% with CI whiskers."""
    plot_w = width - label_w - 60
    h = 28 + len(groups) * (len(metrics) * row + 14)
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" aria-label="{E(title)}">']
    for t in (0, 0.25, 0.5, 0.75, 1):
        x = label_w + t * plot_w
        out.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="18" y2="{h - 4}" class="grid"/><text x="{x:.1f}" y="12" class="ax" text-anchor="middle">{int(t * 100)}%</text>'
        )
    y = 24
    for glabel, vals in groups:
        out.append(f'<text x="0" y="{y + len(metrics) * row / 2 + 4:.1f}" class="lab">{E(glabel)}</text>')
        for m, c in zip(metrics, colors, strict=True):
            mean, lo, hi = vals[m]
            out.append(f'<rect x="{label_w}" y="{y + 3}" width="{max(1, mean * plot_w):.1f}" height="{row - 7}" rx="2" fill="{c}"/>')
            out.append(
                f'<line x1="{label_w + lo * plot_w:.1f}" x2="{label_w + hi * plot_w:.1f}" y1="{y + row / 2 - 0.5:.1f}" y2="{y + row / 2 - 0.5:.1f}" class="ci"/>'
            )
            out.append(f'<text x="{label_w + hi * plot_w + 6:.1f}" y="{y + row / 2 + 4:.1f}" class="val">{pct(mean)}</text>')
            y += row
        y += 14
    out.append("</svg>")
    return "".join(out)


def build(summary, links):
    S = json.loads(Path(summary).read_text())
    L = json.loads(Path(links).read_text()) if links else {}
    P = S["pooled"]
    D = S["diffs"]
    fams = list(S["by_family"])
    head = bars(
        [(ARM_NAME[a], P[a]) for a in ARMS],
        ["real_named", "null_named"],
        ["var(--real)", "var(--null)"],
        title="share of real and null edits named, per arm",
    )
    top = bars(
        [(ARM_NAME[a], P[a]) for a in ARMS],
        ["real_top1", "null_top1"],
        ["var(--real)", "var(--null)"],
        title="share named in the first hypothesis, per arm",
    )
    pair = bars([(ARM_NAME[a], P[a]) for a in ARMS], ["pairwise"], ["var(--acc)"], title="pairwise real-over-null score")
    fam_rows = []
    for f in fams:
        cells = "".join(
            f'<td class="num">{pct(S["by_family"][f][a]["real_named"][0])} / {pct(S["by_family"][f][a]["null_named"][0])}<br><span class="sub">pair {S["by_family"][f][a]["pairwise"][0]:.2f}</span></td>'
            for a in ARMS
        )
        n = sum(1 for s in S["settings"] if S["per_setting"].get(f"env|{s}", {}).get("family") == f)
        fam_rows.append(f"<tr><td>{E(FAM_NAME.get(f, f))}<br><span class='sub'>{n} settings</span></td>{cells}</tr>")
    set_rows = []
    for s in S["settings"]:
        st = json.loads((SETTINGS / s / "setting.json").read_text())
        b = st["base"]["rate"]
        ed = "<br>".join(
            f"<span class='{e['role']}'>{e['role']}</span> {E(e['arm'])} {e['rate'][0]}/{e['rate'][1]}"
            for _, e in sorted(st["edits"].items(), key=lambda kv: kv[1]["role"] != "real")
        )
        cells = ""
        for a in ARMS:
            v = S["per_setting"].get(f"{a}|{s}")
            if not v:
                cells += "<td></td>"
                continue
            lk = L.get(f"{a}|{s}", {})
            ll = " ".join(f'<a href="{E(u)}">{E(k)}</a>' for k, u in lk.items())
            cells += f'<td class="num">{pct(v["real_named"])} / {pct(v["null_named"])}<br><span class="sub">pair {v["pairwise"]:.2f} · n={v["n"]}</span><br><span class="lk">{ll}</span></td>'
        set_rows.append(
            f"<tr><td><b>{E(s)}</b><br><span class='sub'>{E(st['model'])} · base {b[0]}/{b[1]}</span><br><span class='sub'>{ed}</span></td>{cells}</tr>"
        )

    def d(k, m):
        x = D[k][m]
        return f"{100 * x[0]:+.0f} pp [{100 * x[1]:+.0f}, {100 * x[2]:+.0f}]"

    def dp(k):
        x = D[k]["pairwise"]
        return f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]"

    return TEMPLATE.format(
        n=S["n_investigations"],
        ns=len(S["settings"]),
        head=head,
        top=top,
        pair=pair,
        fam_rows="".join(fam_rows),
        set_rows="".join(set_rows),
        d_envtx_real=d("env_tx-env", "real_named"),
        d_envtx_null=d("env_tx-env", "null_named"),
        d_envtx_pair=dp("env_tx-env"),
        d_tx_real=d("tx-env", "real_named"),
        d_tx_null=d("tx-env", "null_named"),
        d_tx_pair=dp("tx-env"),
        d_envtx_top=d("env_tx-env", "real_top1"),
        expview=L.get("_expview", "#"),
        expview8830=L.get("_expview8830", "#"),
        docent=L.get("_docent", "#"),
        findings=L.get("_findings", ""),
    )


TEMPLATE = """<title>Env vs transcript investigators</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,600&display=swap">
<style>
/* Layout: one reading column; headline charts first, then per-family table, then every setting with links. */
:root {{
  --bg: #f7f8f6; --fg: #15191c; --muted: #5b6268; --line: #dfe2de; --card: #ffffff;
  --real: #1c7c5c; --null: #c1773a; --acc: #3b5ba9;
  --display: "Fraunces", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #141719; --fg: #ecefec; --muted: #a2a9ae; --line: #2c3236; --card: #1b1f22; --real: #4cc49a; --null: #e49a5c; --acc: #8ea8ec; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #141719; --fg: #ecefec; --muted: #a2a9ae; --line: #2c3236; --card: #1b1f22; --real: #4cc49a; --null: #e49a5c; --acc: #8ea8ec; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); }}
.wrap {{ max-width: 1040px; margin: 0 auto; padding-block: 28px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 18px; }}
h1 {{ font-family: var(--display); font-size: 1.9rem; line-height: 1.2; margin: 0; text-wrap: balance; }}
h2 {{ font-family: var(--display); font-size: 1.25rem; margin: 18px 0 0; text-wrap: balance; }}
p, li {{ max-width: 75ch; margin: 0; }}
ul {{ margin: 0; padding-left: 1.2rem; display: grid; gap: 6px; }}
a {{ color: var(--acc); }}
.sub {{ color: var(--muted); font-size: 0.8rem; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; display: grid; gap: 8px; min-width: 0; }}
.charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.85rem; color: var(--muted); }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }}
svg text {{ font-family: var(--body); }}
.lab {{ fill: var(--fg); font-size: 13px; }} .val {{ fill: var(--fg); font-size: 12px; font-family: var(--mono); }} .ax {{ fill: var(--muted); font-size: 11px; }}
.grid {{ stroke: var(--line); }} .ci {{ stroke: var(--fg); opacity: .55; stroke-width: 1.5; }}
.tw {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--card); }}
table {{ border-collapse: collapse; width: 100%; min-width: 760px; font-size: 0.86rem; }}
th, td {{ text-align: left; vertical-align: top; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
th {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600; }}
td.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }}
.real {{ color: var(--real); font-weight: 600; }} .null {{ color: var(--null); font-weight: 600; }}
.lk a {{ font-family: var(--body); margin-right: 6px; }}
code {{ font-family: var(--mono); font-size: .88em; }}
</style>
<div class="wrap">
<h1>Does a transcript help an investigator find the edit that matters?</h1>
<p>{ns} settings where one edit to the inputs moves a behaviour a lot and at least one measured null edit does not. For each setting, 4 runs where the behaviour happened; DeepSeek V4.1 Flash investigates each run twice in three arms: the run's exact inputs only, the inputs plus that run's transcript, or the transcript only. Each investigation submits ranked hypotheses with proposed tests. Claude Sonnet 5.5 then judges, blind to which is which, whether the hypotheses name the feature each candidate edit changes. {n} investigations.</p>
{findings}
<div class="legend"><span><i style="background:var(--real)"></i>real edit (changes the rate)</span><span><i style="background:var(--null)"></i>null edit (measured, no change)</span><span><i style="background:var(--acc)"></i>pairwise: real named before null (0.5 = can't tell)</span><span>Whiskers: 95% CI, bootstrap over settings.</span></div>
<div class="charts">
<div class="card"><b>Named anywhere in the hypotheses or tests</b>{head}</div>
<div class="card"><b>Named in the first hypothesis</b>{top}</div>
<div class="card"><b>Pairwise: real edit named before a null</b>{pair}</div>
</div>
<div class="card"><b>Paired differences vs env only (per setting, then averaged)</b>
<ul>
<li>Env + transcript: real named {d_envtx_real}; null named {d_envtx_null}; real in first hypothesis {d_envtx_top}; pairwise {d_envtx_pair}.</li>
<li>Transcript only: real named {d_tx_real}; null named {d_tx_null}; pairwise {d_tx_pair}.</li>
</ul></div>
<h2>By family</h2>
<p class="sub">Cells: real named / null named, and the pairwise score.</p>
<div class="tw"><table><thead><tr><th>family</th><th>env only</th><th>env + transcript</th><th>transcript only</th></tr></thead><tbody>{fam_rows}</tbody></table></div>
<h2>Every setting</h2>
<p class="sub">Rates are the subject model's behaviour rate (k/n) in the baseline and under each edit. Links open the investigations (8830 needs the tailnet) and their Docent transcripts.</p>
<div class="tw"><table><thead><tr><th>setting, edits</th><th>env only</th><th>env + transcript</th><th>transcript only</th></tr></thead><tbody>{set_rows}</tbody></table></div>
<h2>Where everything is</h2>
<ul>
<li>Every investigation with its evidence folder and each judge call (exact prompt and reply): <a href="{expview}">expview page</a>, also on <a href="{expview8830}">bramble:8830</a> (tailnet).</li>
<li>Investigator transcripts on <a href="{docent}">Docent</a> (collection <code>envguess</code>, metadata arm / setting / family / real_named / null_named).</li>
<li>Code: <code>envguess/</code> on branch <code>claude/env-vs-transcript</code> (build.py, investigate.py, judge.py, score.py).</li>
</ul>
</div>
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--links", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).write_text(build(a.summary, a.links))
    print(a.out)
