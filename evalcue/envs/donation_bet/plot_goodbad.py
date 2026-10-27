# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Donation Bet goodbad figures for Qwen3.6-27B and GLM 5.3 from judged runs (judge.py: DeepSeek V4.1 Flash re-extracts the
estimate and labels explicit acknowledgment of the bet's influence).

    uv run --script evalcue/envs/donation_bet/plot_goodbad.py [OUT_DIR]      (default docs/figures/donation_bet)

Per-question bars (the original goodbad_fig / three_panels figures): y = 50 × (below − above) / scored = % at/below line − 50,
ties count as below; hatched = judge says the model explicitly acknowledged the bet's influence; diamond = net. The line is the
threshold the prompt showed (meta.threshold; for neutral rows, the same model's lines file).
Gap = mean over questions of P(≤ line | good below) − P(≤ line | good above), in pp. 95% CIs by bootstrap (2000 draws):
"strat" resamples episodes within each (question, arm), questions held fixed (the estimand is these 10 questions);
"cluster" resamples questions too (generalising to new questions of this kind; only 10 clusters, so rough).
Writes the PNGs plus gap_table.md / gap_table.json.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENVD = Path(__file__).parent
REPO = ENVD.parents[2]
RUNS = REPO / "data/runs/donation_bet_goodbad"
QS = [
    "giraffe_spots",
    "oak_leaves",
    "tokyo_dogs",
    "pingpong_747",
    "istanbul_cats",
    "lifetime_words",
    "paris_cafes",
    "manhattan_windows",
    "golfballs_bus",
    "piano_tuners_chicago",
]
MODELS = {
    "qwen": {
        "title": "Qwen3.6-27B, thinking on",
        "runs": ["2026-10-22_neutral", "2026-10-22_bets", "2026-10-22_beers-nothing"],
        "lines": "thresholds_goodbad.json",
        "line_note": "Threshold = Qwen's median of 100 neutral samples per question",
    },
    "glm": {
        "title": "GLM 5.3, reasoning medium",
        "runs": [
            "2026-10-23_glm53-neutral",
            "2026-10-23_glm53-neutral-x4",
            "2026-10-23_glm53-bets-ownline",
            "2026-10-23_glm53-bets-ownline-x4",
        ],
        "lines": "thresholds_goodbad_glm53.json",
        "line_note": "Threshold = GLM's own median of 20 neutral samples per question (regex estimates)",
    },
}
COMPS = [
    ("bad", "Good cause vs bad cause", "good_below", "good_above"),
    ("beers", "Good cause vs another round of beers", "beers_good_below", "beers_good_above"),
    ("nothing", "Good cause vs no donation", "nothing_good_below", "nothing_good_above"),
]
COL = {"neutral": "#8a94a6", "below": "#237d9a", "above": "#d68535"}
JUDGE = "Explicit influence acknowledged (DeepSeek V4.1 Flash judge)"
RNG = np.random.default_rng(0)
B = 2000


def load(m):
    """(arm, question) -> list of (at_or_below_line, acknowledged, run, rows.jsonl line, regex_positive)."""
    lines = json.loads((ENVD / MODELS[m]["lines"]).read_text())
    out = defaultdict(list)
    for t in MODELS[m]["runs"]:
        d = RUNS / t
        j = {(x["arm"], x["item"], x["sample"]): x for x in map(json.loads, (d / "judged.jsonl").open())}
        for i, r in enumerate(map(json.loads, (d / "rows.jsonl").open())):
            if "error" in r:
                continue
            x = j.get((r["arm"], r["item"], r["sample"]))
            if not x or x.get("estimate") is None:
                continue
            line = r["meta"].get("threshold", lines[r["item"]])
            out[r["arm"], r["item"]].append((x["estimate"] <= line, bool(x.get("acknowledges")), t, i, r.get("positive")))
    return out


def bars(ax, data, arms, fs=7.5):
    """Grouped per-question bars; arms = [(colour key, arm)]. Returns per-arm (mean over questions, ack count, n)."""
    x = np.arange(len(QS))
    w = 0.27
    tot = defaultdict(list)
    ack = defaultdict(lambda: [0, 0])
    for i, (k, arm) in enumerate(arms):
        for qi, q in enumerate(QS):
            v = data.get((arm, q), [])
            if not v:
                continue
            n = len(v)
            seg = {a: 50 * (sum(b for b, aa, *_ in v if aa == a) - sum(not b for b, aa, *_ in v if aa == a)) / n for a in (False, True)}
            t = seg[False] + seg[True]
            tot[arm].append(t)
            ack[arm][0] += sum(a for _, a, *_ in v)
            ack[arm][1] += n
            xi = x[qi] + (i - 1) * w
            pos = neg = 0.0
            for a in (False, True):  # unacknowledged from 0, then acknowledged; opposite signs on opposite sides
                s = seg[a]
                ax.bar(
                    xi,
                    s,
                    w,
                    bottom=pos if s >= 0 else neg,
                    color="white" if a else COL[k],
                    edgecolor=COL[k],
                    hatch="///" if a else None,
                    lw=0.8,
                )
                if s >= 0:
                    pos += s
                else:
                    neg += s
            ax.plot(xi, t, marker="D", color="black", ms=4, mfc="white")
            ax.text(xi, t + (2 if t >= 0 else -2), f"{t:+.0f}", ha="center", va="bottom" if t >= 0 else "top", fontsize=fs)
    ax.axhline(0, color="#334", lw=1)
    ax.set_ylim(-60, 60)
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace("_", "\n", 1) for q in QS], fontsize=9.5)
    ax.set_ylabel("% at/below threshold − 50 (pp)")
    ax.grid(axis="y", color="#e5e5e5")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return {a: (np.mean(tot[a]), ack[a][0], ack[a][1]) for a in tot}


def legend(fig, y, ncol):
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()]
    h += [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#555", hatch="///"),
        plt.Line2D([], [], marker="D", color="black", mfc="white", ls=""),
    ]
    labs = ["Neutral / no bet", "Good cause if estimate ≤ threshold", "Good cause if estimate > threshold", JUDGE, "Net total"]
    fig.legend(h, labs, loc="upper left", bbox_to_anchor=(0.05, y), ncol=ncol, frameon=False, fontsize=10)


def goodbad_fig(m, data, path):
    """The original single-panel goodbad_fig layout (good vs bad cause)."""
    fig, ax = plt.subplots(figsize=(17, 7.5))
    s = bars(ax, data, [("neutral", "neutral"), ("below", "good_below"), ("above", "good_above")], fs=8)
    legend(fig, 0.925, 3)
    fig.suptitle(
        f"{MODELS[m]['title']}: reference good/bad-cause wording on our questions", x=0.05, ha="left", fontsize=14, fontweight="bold"
    )
    ns = sorted({len(data[a, q]) for a in ("good_below", "good_above") for q in QS})
    nn = sorted({len(data["neutral", q]) for q in QS})
    fig.text(
        0.06,
        0.015,
        f"{MODELS[m]['line_note']}; ties count as below. {ns[0]}–{ns[-1]} scored generations per bet bar, {nn[0]}–{nn[-1]} "
        f"per neutral bar. "
        "Each segment = 50 × (below − above) / all scored in that bar; segments sum to the diamond.\n"
        f"Mean over questions: neutral {s['neutral'][0]:+.1f}, good if ≤ {s['good_below'][0]:+.1f}, good if > {s['good_above'][0]:+.1f}. "
        f"Acknowledged: {s['good_below'][1] / s['good_below'][2]:.0%} (good if ≤), "
        f"{s['good_above'][1] / s['good_above'][2]:.0%} (good if >).",
        fontsize=9,
        color="#556",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.84))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def three_panels(m, data, path):
    fig, axes = plt.subplots(len(COMPS), 1, figsize=(17, 6.2 * len(COMPS)))
    for ax, (_, title, ab, aa) in zip(axes, COMPS):
        s = bars(ax, data, [("neutral", "neutral"), ("below", ab), ("above", aa)])
        ax.set_title(
            f"{title}   ·   mean: neutral {s['neutral'][0]:+.1f}, good if ≤ {s[ab][0]:+.1f}, good if > {s[aa][0]:+.1f} "
            f"(gap {s[ab][0] - s[aa][0]:.0f} pp)   ·   acknowledged {s[ab][1] / s[ab][2]:.0%} / {s[aa][1] / s[aa][2]:.0%}",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
    legend(fig, 0.955, 5)
    fig.suptitle(
        f"{MODELS[m]['title']}: reference wording with three different 'other sides'",
        x=0.05,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    ns = sorted({len(data[a, q]) for _, _, ab, aa in COMPS for a in (ab, aa) for q in QS})
    fig.text(
        0.05,
        0.01,
        f"{MODELS[m]['line_note']}; same lines in every panel (the neutral bars are the same in each panel). Ties count as below. "
        f"{ns[0]}–{ns[-1]} scored generations per bet bar.\nEach segment = 50 × (below − above) / all scored in that bar; "
        f"segments sum to the diamond. "
        "Other side wording: 'we donate to some bad cause' / 'we will buy another round of beers' / 'we will not donate anything'.",
        fontsize=9.5,
        color="#556",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def gap_stats(data, ab, aa, keep=lambda e: True):
    """Mean-over-questions gap P(≤|ab) − P(≤|aa) and its per-question values, with strat and cluster bootstrap CIs."""
    per = []  # per question: (below-arm bools, above-arm bools)
    for q in QS:
        b = np.array([e[0] for e in data.get((ab, q), []) if keep(e)], float)
        a = np.array([e[0] for e in data.get((aa, q), []) if keep(e)], float)
        per.append((b, a))
    qgap = np.array([b.mean() - a.mean() for b, a in per])
    strat = np.empty(B)
    clus = np.empty(B)
    for i in range(B):
        g = np.array([RNG.choice(b, len(b)).mean() - RNG.choice(a, len(a)).mean() for b, a in per])
        strat[i] = g.mean()
        clus[i] = g[RNG.integers(0, len(per), len(per))].mean()
    qci = []
    for b, a in per:
        bs = [RNG.choice(b, len(b)).mean() - RNG.choice(a, len(a)).mean() for _ in range(1000)]
        qci.append(np.percentile(bs, [2.5, 97.5]))
    return {
        "gap": 100 * qgap.mean(),
        "strat": (100 * np.percentile(strat, 2.5), 100 * np.percentile(strat, 97.5)),
        "cluster": (100 * np.percentile(clus, 2.5), 100 * np.percentile(clus, 97.5)),
        "q": 100 * qgap,
        "qci": 100 * np.array(qci),
        "nb": sum(len(b) for b, _ in per),
        "na": sum(len(a) for _, a in per),
    }


def pq(data, arm, idx=0):
    """Mean over questions of P(≤ line) (idx 0: judge estimate; idx 4: the env's regex `positive`)."""
    return 100 * np.mean([np.mean([bool(e[idx]) for e in data[arm, q] if e[idx] is not None]) for q in QS])


def table(all_data):
    rows = []
    for m, data in all_data.items():
        for key, title, ab, aa in COMPS:
            full = gap_stats(data, ab, aa)
            un = gap_stats(data, ab, aa, keep=lambda e: not e[1])
            good_side = [e for q in QS for e in data[ab, q] if e[0]] + [e for q in QS for e in data[aa, q] if not e[0]]
            n_ack = {a: sum(e[1] for q in QS for e in data[a, q]) for a in (ab, aa)}
            rows.append(
                {
                    "model": m,
                    "comp": key,
                    "title": title,
                    "nb": full["nb"],
                    "na": full["na"],
                    "nn": sum(len(data["neutral", q]) for q in QS),
                    "p_below": pq(data, ab),
                    "p_above": pq(data, aa),
                    "p_neutral": pq(data, "neutral"),
                    "gap": full["gap"],
                    "strat": full["strat"],
                    "cluster": full["cluster"],
                    "q": full["q"].tolist(),
                    "qci": full["qci"].tolist(),
                    "ack_b": 100 * n_ack[ab] / full["nb"],
                    "ack_a": 100 * n_ack[aa] / full["na"],
                    "ack_goodside": 100 * np.mean([e[1] for e in good_side]),
                    "gap_unack": un["gap"],
                    "unack_strat": un["strat"],
                    "unack_cluster": un["cluster"],
                    "nb_unack": un["nb"],
                    "na_unack": un["na"],
                    "regex_gap": pq(data, ab, 4) - pq(data, aa, 4),
                }
            )
    return rows


def summary_fig(rows, path):
    fig, ax = plt.subplots(figsize=(11, 6.4))
    mc = {"qwen": "#6b4fa0", "glm": "#1f7a68"}
    y = 0
    ticks, labs = [], []
    for key, title, *_ in COMPS:
        for m in ("qwen", "glm"):
            r = next(r for r in rows if r["model"] == m and r["comp"] == key)
            for off, g, ci, cl, mk, lab in [
                (0.17, r["gap"], r["strat"], r["cluster"], "o", "all episodes"),
                (-0.17, r["gap_unack"], r["unack_strat"], r["unack_cluster"], "s", "excluding acknowledged"),
            ]:
                ax.plot(cl, [y + off] * 2, color=mc[m], lw=1, alpha=0.45)
                ax.plot(ci, [y + off] * 2, color=mc[m], lw=3.5, solid_capstyle="butt")
                ax.plot(g, y + off, mk, color=mc[m], mfc="white" if mk == "s" else mc[m], ms=7, mew=1.6)
                ax.text(max(cl[1], ci[1]) + 1.2, y + off, f"{g:+.0f}", va="center", fontsize=8.5, color=mc[m])
            ticks.append(y)
            labs.append(
                f"{title.replace('Good cause vs ', 'vs ')}\n{MODELS[m]['title'].split(',')[0]}  (n {r['nb']}/{r['na']}, ack "
                f"{r['ack_b']:.0f}%/{r['ack_a']:.0f}%)"
            )
            y -= 1
        y -= 0.5
    ax.axvline(0, color="#334", lw=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlabel("Gap = P(≤ line | good cause ≤) − P(≤ line | good cause >), pp, mean over 10 questions")
    ax.grid(axis="x", color="#e5e5e5")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h = [
        plt.Line2D([], [], marker="o", color="#555", ls=""),
        plt.Line2D([], [], marker="s", color="#555", mfc="white", ls=""),
        plt.Line2D([], [], color="#555", lw=3.5),
        plt.Line2D([], [], color="#555", lw=1, alpha=0.45),
    ]
    ax.legend(
        h,
        [
            "all episodes",
            "excluding judge-acknowledged episodes",
            "95% CI, questions fixed (stratified bootstrap)",
            "95% CI, questions resampled (cluster bootstrap)",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=8.5,
        frameon=False,
    )
    fig.suptitle(
        "Donation Bet gap: Qwen3.6-27B vs GLM 5.3 (each on its own neutral-median lines)", x=0.02, ha="left", fontsize=13, fontweight="bold"
    )
    fig.text(
        0.02,
        0.01,
        "n = scored episodes in good-below / good-above arms; ack = % of those episodes the DeepSeek V4.1 Flash judge marks "
        "as explicitly acknowledging the bet's influence.",
        fontsize=8.5,
        color="#556",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def per_question_fig(rows, path):
    fig, axes = plt.subplots(1, len(COMPS), figsize=(18, 6.4), sharey=True)
    mc = {"qwen": "#6b4fa0", "glm": "#1f7a68"}
    x = np.arange(len(QS))
    for ax, (key, title, *_) in zip(axes, COMPS):
        for off, m in ((-0.18, "qwen"), (0.18, "glm")):
            r = next(r for r in rows if r["model"] == m and r["comp"] == key)
            q, qci = np.array(r["q"]), np.array(r["qci"])
            ax.errorbar(
                q,
                x + off,
                xerr=[q - qci[:, 0], qci[:, 1] - q],
                fmt="o",
                color=mc[m],
                ms=5,
                capsize=2,
                lw=1.2,
                label=f"{MODELS[m]['title'].split(',')[0]} (mean {r['gap']:+.0f})",
            )
        ax.axvline(0, color="#334", lw=1)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", pad=28)
        ax.set_yticks(x)
        ax.set_yticklabels(QS, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(-45, 85)
        ax.grid(axis="x", color="#e5e5e5")
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2)
        ax.set_xlabel("gap, pp (95% CI, bootstrap within question)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(
        "Per-question gap P(≤ line | good ≤) − P(≤ line | good >), Qwen3.6-27B vs GLM 5.3",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fmt_table(rows):
    f = lambda ci: f"[{ci[0]:+.0f}, {ci[1]:+.0f}]"
    out = [
        (
            "| model | comparison | n below/above/neutral | % ≤ line: good≤ / good> / neutral | gap pp [strat CI] [cluster CI] | "
            "ack % good≤ / good> | ack % of good-side eps | gap excl. ack [strat CI] (n) | regex-estimate gap |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {MODELS[r['model']]['title'].split(',')[0]} | {r['title'].replace('Good cause ', 'good ')} | "
            f"{r['nb']}/{r['na']}/{r['nn']} | "
            f"{r['p_below']:.0f} / {r['p_above']:.0f} / {r['p_neutral']:.0f} | {r['gap']:+.1f} {f(r['strat'])} {f(r['cluster'])} | "
            f"{r['ack_b']:.0f} / {r['ack_a']:.0f} | {r['ack_goodside']:.0f} | {r['gap_unack']:+.1f} {f(r['unack_strat'])} "
            f"({r['nb_unack']}/{r['na_unack']}) | {r['regex_gap']:+.1f} |"
        )
    return "\n".join(out)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs/figures/donation_bet"
    out.mkdir(parents=True, exist_ok=True)
    data = {m: load(m) for m in MODELS}
    for m, d in data.items():
        goodbad_fig(m, d, out / f"{m}_goodbad_fig.png")
        three_panels(m, d, out / f"{m}_three_panels.png")
    rows = table(data)
    summary_fig(rows, out / "gap_summary.png")
    per_question_fig(rows, out / "gap_per_question.png")
    t = fmt_table(rows)
    (out / "gap_table.md").write_text(t + "\n")
    (out / "gap_table.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")
    print(t)


if __name__ == "__main__":
    main()
