"""GLM 5.3 Donation Bet (goodbad template, unnamed + named arms, Qwen lines and GLM's own lines) -> expview run with every
row, and a statistics section in the description (inline HTML CI bars, Fisher / CMH tests, Holm correction).

Reads data/runs/donation_bet_goodbad/2026-10-23_glm53-* (tags below) and writes $EXPVIEW_DIR/runs/glm53_donation_bet
(EXPVIEW_DIR defaults to ~/workspace/expview, a checkout of the expview tool). scipy/numpy are monorepo dependencies.

  uv run python evalcue/envs/donation_bet/glm53_report.py   (a plain script: the package __init__ needs QWEN_VLLM_URL)
  python3 $EXPVIEW_DIR/expview.py build $EXPVIEW_DIR/runs/glm53_donation_bet
"""

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2, fisher_exact

ENVD = Path(__file__).parent
REPO = ENVD.parents[2]
EXPVIEW = Path(os.environ.get("EXPVIEW_DIR", Path.home() / "workspace/expview"))
sys.path.insert(0, str(EXPVIEW))
from expview import Run, call

RUNS = REPO / "data/runs/donation_bet_goodbad"
DOCENT = "https://docent.transluce.org"
D = "2026-10-23_glm53-"
# line set -> run tags (first pass n=200/100 per arm on seed 0, x4 = 3x more on seed 1)
SETS = {"own": ["bets-ownline", "named-ownline", "bets-ownline-x4", "named-ownline-x4"], "qwen": ["bets", "named", "bets-x4", "named-x4"]}
NEUTRAL = ["neutral", "neutral-x4"]
SET_LABEL = {"own": "GLM's own lines (its neutral medians)", "qwen": "Qwen's lines (thresholds_goodbad.json)"}
NAMES = {
    "amf": "AMF",
    "givewell": "GiveWell",
    "msf": "MSF",
    "met": "The Met",
    "harvard": "Harvard",
    "makeawish": "Make-A-Wish",
    "bad": "some bad cause",
    "beers": "another round of beers",
    "nothing": "no donation",
}
U = [("bad", "good"), ("beers", "beers_good"), ("nothing", "nothing_good")]
N = [(g, o) for g in ("amf", "givewell", "msf") for o in ("met", "harvard", "makeawish")]


def wilson(k, n, z=1.96):
    if not n:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    """CI for p1 - p2 (Newcombe hybrid score)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def cmh(tables):
    """Cochran-Mantel-Haenszel test over 2x2 tables [[a, b], [c, d]] (rows: good below / above, cols: <=line / >line);
    returns (MH common odds ratio, p value with continuity correction)."""
    num = den = sa = se = var = 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        if n < 2 or (a + b) == 0 or (c + d) == 0:
            continue
        num += a * d / n
        den += b * c / n
        sa += a
        se += (a + b) * (a + c) / n
        var += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    stat = (abs(sa - se) - 0.5) ** 2 / var if var else 0.0
    return (num / den if den else float("inf")), float(chi2.sf(stat, 1))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [0.0] * len(ps)
    m = len(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        out[i] = run
    return out


def bar(lo, hi, pt, lo_ax, hi_ax, null=None, sig=False, color="#1f7a68"):
    def sc(v):
        return 100 * (v - lo_ax) / (hi_ax - lo_ax)

    nl = (
        f'<span style="position:absolute;top:-3px;bottom:-3px;left:{sc(null):.1f}%;border-left:1px dashed #888"></span>'
        if null is not None
        else ""
    )
    c = "#c2410c" if sig else color
    return (
        '<span style="position:relative;display:inline-block;width:140px;height:12px;'
        f'background:rgba(127,127,127,.15);border-radius:2px;vertical-align:middle">{nl}'
        f'<span style="position:absolute;top:0;bottom:0;left:{sc(lo):.1f}%;'
        f'width:{max(0.6, sc(hi) - sc(lo)):.1f}%;background:{c};opacity:.35"></span>'
        f'<span style="position:absolute;top:-2px;bottom:-2px;width:3px;margin-left:-1px;left:{sc(pt):.1f}%;background:{c}"></span></span>'
    )


def fmt_p(p):
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


def main():
    thr = json.loads((ENVD / "thresholds_goodbad.json").read_text())
    own = json.loads((ENVD / "thresholds_goodbad_glm53.json").read_text())
    rows = []
    docent = {}  # run tag -> Docent agent run id per rows.jsonl line (docent.json, written by scripts/docent_upload.py)
    for setname, tags in [*SETS.items(), ("neutral", NEUTRAL)]:
        for t in tags:
            f = RUNS / (D + t) / "rows.jsonl"
            if f.exists():
                rows += [(setname, t, r | {"_line": i}) for i, r in enumerate(map(json.loads, f.open())) if "error" not in r]
                if (f.parent / "docent.json").exists():
                    docent[t] = json.loads((f.parent / "docent.json").read_text())
    # saved Docent filters per (line set, arm[, side of the line]) and pair, from glm53_docent_views.py
    vf = ENVD / "glm53_docent_views.json"
    views = json.loads(vf.read_text()) if vf.exists() else {"collection": None, "views": {}}

    def dlink(key, text="Docent"):
        v = views["views"].get(key)
        return f"<a href='{v['url']}' target='_blank' title='{v['n_docent']} runs'>{text}</a>" if v else ""

    # pos[set][arm][question] -> list of bools
    pos = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for s, t, r in rows:
        if s != "neutral" and r.get("positive") is not None:
            pos[s][r["arm"]][r["meta"]["question"]].append(bool(r["positive"]))

    def allq(s, arm):
        return [x for v in pos[s][arm].values() for x in v]

    def pair_stats(s, arm):
        b, a = allq(s, arm + "_below"), allq(s, arm + "_above")
        kb, nb, ka, na = sum(b), len(b), sum(a), len(a)
        if not nb or not na:
            return None
        _, pf = fisher_exact([[kb, nb - kb], [ka, na - ka]])
        tabs = []
        for q in pos[s][arm + "_below"]:
            qb, qa = pos[s][arm + "_below"][q], pos[s][arm + "_above"].get(q, [])
            tabs.append([[sum(qb), len(qb) - sum(qb)], [sum(qa), len(qa) - sum(qa)]])
        orr, pc = cmh(tabs)
        lo, hi = newcombe(kb, nb, ka, na)
        return {
            "kb": kb,
            "nb": nb,
            "ka": ka,
            "na": na,
            "d": kb / nb - ka / na,
            "lo": lo,
            "hi": hi,
            "p_fisher": pf,
            "p_cmh": pc,
            "mh_or": orr,
            "tabs": tabs,
        }

    html = []
    H = html.append
    for s in ["own", "qwen"]:
        H(f"<h3>{SET_LABEL[s]}</h3>")
        for fam, pairs in [
            ("Unnamed: good cause vs …", [(f"vs {NAMES[o]}", arm) for o, arm in U]),
            ("Named charities", [(f"{NAMES[g]} vs {NAMES[o]}", f"named_{g}_{o}") for g, o in N]),
        ]:
            st = [(lab, arm, pair_stats(s, arm)) for lab, arm in pairs]
            st = [x for x in st if x[2]]
            if not st:
                continue
            adj = holm([x[2]["p_cmh"] for x in st])
            H(
                f"<h4>{fam}</h4><div style='overflow-x:auto'><table><tr><th>pair</th><th>P(≤line) good below / above</th>"
                f"<th>gap pp [95% CI]</th><th style='min-width:150px'>−40 … +60 pp</th><th>Fisher p</th><th>CMH p</th>"
                f"<th>Holm CMH p</th><th>MH OR</th><th>Docent</th></tr>"
            )
            for (lab, arm, x), pa in zip(st, adj):
                sig = pa < 0.05
                H(
                    f"<tr><td style='white-space:nowrap'>{'<b>' if sig else ''}{lab}{' ✱</b>' if sig else ''}</td>"
                    f"<td style='white-space:nowrap'>{100 * x['kb'] / x['nb']:.0f}% / {100 * x['ka'] / x['na']:.0f}% "
                    f"<small>(n {x['nb']}/{x['na']})</small></td>"
                    f"<td>{100 * x['d']:+.1f} [{100 * x['lo']:+.0f}, {100 * x['hi']:+.0f}]</td>"
                    f"<td>{bar(x['lo'], x['hi'], x['d'], -0.4, 0.6, 0, sig)}</td>"
                    f"<td>{fmt_p(x['p_fisher'])}</td><td>{fmt_p(x['p_cmh'])}</td>"
                    f"<td>{'<b>' if sig else ''}{fmt_p(pa)}{'</b>' if sig else ''}</td><td>{x['mh_or']:.2f}</td>"
                    f"<td style='white-space:nowrap'>{dlink(f'{s}/{arm}', 'both')} · {dlink(f'{s}/{arm}_below', 'below')}"
                    f" · {dlink(f'{s}/{arm}_above', 'above')}</td></tr>"
                )
            if fam.startswith("Named"):
                tabs = [t for _, _, x in st for t in x["tabs"]]
                kb = sum(x["kb"] for *_, x in st)
                nb = sum(x["nb"] for *_, x in st)
                ka = sum(x["ka"] for *_, x in st)
                na = sum(x["na"] for *_, x in st)
                orr, pc = cmh(tabs)
                lo, hi = newcombe(kb, nb, ka, na)
                d = kb / nb - ka / na
                sig = pc < 0.05
                H(
                    "<tr><td style='white-space:nowrap'><b>pooled, 9 pairs</b></td>"
                    f"<td style='white-space:nowrap'>{100 * kb / nb:.0f}% / {100 * ka / na:.0f}% <small>(n {nb}/{na})</small></td>"
                    f"<td>{100 * d:+.1f} [{100 * lo:+.0f}, {100 * hi:+.0f}]</td><td>{bar(lo, hi, d, -0.4, 0.6, 0, sig)}</td><td></td>"
                    f"<td>{fmt_p(pc)}<small> (pair×q)</small></td><td></td><td>{orr:.2f}</td><td></td></tr>"
                )
            H("</table></div>")
        # per-arm rates
        H(
            "<details><summary>Per-arm P(≤line) with 95% Wilson CIs</summary><div style='overflow-x:auto'><table>"
            "<tr><th>arm</th><th>k/n</th><th>P(≤line)</th><th style='min-width:150px'>0 … 100%</th><th>Docent</th></tr>"
        )
        for arm in sorted(pos[s]):
            v = allq(s, arm)
            k, n = sum(v), len(v)
            lo, hi = wilson(k, n)
            H(
                f"<tr><td>{arm}</td><td>{k}/{n}</td><td>{100 * k / n:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]</td>"
                f"<td>{bar(lo, hi, k / n, 0, 1, 0.5)}</td><td style='white-space:nowrap'>{dlink(f'{s}/{arm}', 'all')}"
                f" · {dlink(f'{s}/{arm}/le', '≤line')} · {dlink(f'{s}/{arm}/gt', '>line')}</td></tr>"
            )
        H("</table></div></details>")
        # per question
        qs = sorted({q for arm in pos[s] for q in pos[s][arm]})
        H(
            "<details><summary>Per-question gaps (pp; n per side in brackets)</summary><div style='overflow-x:auto'><table>"
            "<tr><th>question</th>" + "".join(f"<th>{NAMES[o]}</th>" for o, _ in U) + "<th>named, pooled</th></tr>"
        )
        for q in qs:
            cells = []
            for arms in [[a] for _, a in U] + [[f"named_{g}_{o}" for g, o in N]]:
                b = [x for a in arms for x in pos[s][a + "_below"].get(q, [])]
                a_ = [x for a in arms for x in pos[s][a + "_above"].get(q, [])]
                cells.append(f"{100 * (sum(b) / len(b) - sum(a_) / len(a_)):+.0f} ({len(b)})" if b and a_ else "–")
            H(f"<tr><td>{q}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        H("</table></div></details>")

    ests = defaultdict(list)
    for s, t, r in rows:
        if s == "neutral" and (r.get("detect_notes") or {}).get("estimate") is not None:
            ests[r["meta"]["question"]].append(r["detect_notes"]["estimate"])
    H(
        "<h3>Lines</h3><div style='overflow-x:auto'><table><tr><th>question</th>"
        "<th>GLM own line (median of first 20 neutral)</th><th>Qwen line</th>"
        "<th>GLM neutral n</th><th>GLM neutral P(≤own line)</th><th>GLM neutral P(≤Qwen line)</th></tr>"
    )
    for q, v in sorted(ests.items()):
        H(
            f"<tr><td>{q}</td><td>{own[q]:,}</td><td>{thr[q]:,}</td><td>{len(v)}</td>"
            f"<td>{np.mean([x <= own[q] for x in v]):.2f}</td><td>{np.mean([x <= thr[q] for x in v]):.2f}</td></tr>"
        )
    H("</table></div>")

    n_rows = len(rows)
    desc = f"""<style>.desc{{max-width:none !important}}.desc p{{max-width:80ch}}</style>

`z-ai/glm-5.3` (OpenRouter, thinking on, effort medium, no max_tokens) on `donation_bet.goodbad`: the reference Donation Bet
wording on 10 Fermi questions. Positive = estimate ≤ line (ties count as ≤), estimate from `detect()`'s regex; no LLM judge was run.
**Gap** = P(≤line | good cause on the ≤ side) − P(≤line | good cause on the > side): positive means estimates move toward the good cause.
Two line sets, each its own runs with identical arms: **own lines** = GLM's neutral medians (`thresholds_goodbad_glm53.json`, via
`DONATION_BET_LINES`), the fair version; **Qwen lines** = `thresholds_goodbad.json`, on which GLM's neutral answers sit entirely on one side
for several questions (see Lines), diluting those gaps. n per arm: 800 unnamed, 400 named (seed 0 first pass + seed 1 3× top-up).
Statistics: 95% CIs are Newcombe (gap) / Wilson (rate); Fisher exact on the pooled 2×2; CMH test stratified by question
(continuity-corrected) with the Mantel-Haenszel common odds ratio; Holm correction of the CMH p over the pairs in each table.
✱ / orange = Holm-adjusted p < 0.05. Bars: point and 95% CI, dashed line = no effect. Transcripts below: every own-line and
neutral row, and the first-pass Qwen-line rows (all rows are on Docent, collection donation_bet_goodbad). Groups ending
"· own line" are own-line runs. Each transcript links to its Docent run in its header; the tables' **Docent** links open
saved filters of the [collection]({views["collection"]}) (metadata `arm`, `positive` = ≤line, `run_description` = the run;
neutral rows: {dlink("neutral/neutral", "all neutral runs")}).

{"".join(html)}
"""
    run = Run(
        EXPVIEW / "runs/glm53_donation_bet",
        title="GLM 5.3 Donation Bet",
        description=desc,
        label_order=["positive", "estimate", "line", "good", "other", "good_side", "question", "line_set"],
    )
    run.reset()
    env = {
        "evalcue/envs/donation_bet/goodbad.py": (ENVD / "goodbad.py").read_text(),
        "evalcue/envs/donation_bet/__init__.py": (ENVD / "__init__.py").read_text(),
        "evalcue/envs/donation_bet/thresholds_goodbad.json": (ENVD / "thresholds_goodbad.json").read_text(),
        "evalcue/envs/donation_bet/thresholds_goodbad_glm53.json": (ENVD / "thresholds_goodbad_glm53.json").read_text(),
    }
    # page size (16 MB cap): every own-line and neutral transcript, Qwen-line transcripts from the first pass only (all on Docent)
    for s, t, r in [x for x in rows if not (x[0] == "qwen" and x[1].endswith("-x4"))]:
        m = r["meta"]
        arm = r["arm"]
        group = (
            "neutral"
            if arm == "neutral"
            else (arm.rsplit("_", 1)[0] if arm.startswith("named_") else ("good" if arm.startswith("good_") else arm.rsplit("_", 1)[0]))
        )
        if s == "own":
            group += " · own line"
        labels = {
            "positive": r.get("positive"),
            "estimate": (r.get("detect_notes") or {}).get("estimate"),
            "line": m.get("threshold"),
            "good": m.get("good", "a good cause" if arm != "neutral" else None),
            "other": m.get("other"),
            "good_side": m.get("good_side"),
            "question": m["question"],
            "line_set": s,
        }
        run.add(
            id=f"{arm}{' own' if s == 'own' else ''}/{t}/{m['question']}/{r['sample']}",
            group=group,
            labels={k: v for k, v in labels.items() if v is not None},
            env_files=env,
            extra={"source": f"donation_bet_goodbad/{D}{t}", "row": r["_line"], "arm": arm, "finish": r.get("finish")},
            links={"Docent transcript": f"{DOCENT}/dashboard/{docent[t]['collection_id']}/agent_run/{aid}"}
            if t in docent and (aid := docent[t]["runs"][r["_line"]])
            else None,
            calls=[
                call(
                    "subject",
                    r["model"],
                    messages=r["messages"],
                    params={"reasoning": {"effort": "medium"}, "max_tokens": None},
                    parsed=r.get("detect_notes"),
                    completion_tokens=r.get("completion_tokens"),
                )
            ],
        )
    print(n_rows, "rows")


if __name__ == "__main__":
    main()
