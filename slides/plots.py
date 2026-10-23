# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "scipy"]
# ///
"""Figures for slides/neel.md, from the runs on HF (pull `data/` first, see CLAUDE.md). Carries its own deps (PEP 723):

uv run --script slides/plots.py   # writes slides/img/*.png
"""

import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]  (a PEP 723 dep, not a project one)
from scipy.stats import binom

RUNS, OUT = Path("data/runs"), Path("slides/img")
SETS = {"n30_nothink": "cue", "who2_n30_nothink": "attribution", "heldout_n30_nothink": "held out"}
TS, OL, PL = "#2a78d6", "#eb6834", "#8f8e88"  # TS agent, O-lens agent, placebo (gray: the control, always direct-labeled)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
    }
)


def rates():
    """Dumbbell: harsh-grade rate on the base arm vs each attribution arm."""
    arms = {}
    for run in ("who2_n30_nothink", "heldout_n30_nothink"):
        rows = [json.loads(line) for line in (RUNS / f"hidden_goal_grade/2026-10-22_{run}/rows.jsonl").read_text().splitlines()]
        k = collections.Counter(r["arm"] for r in rows if r["positive"])
        n = collections.Counter(r["arm"] for r in rows)
        prompt = {r["arm"]: r["messages"][0]["content"] for r in rows}
        for arm in n:
            if arm != "base":
                arms[arm] = (k[arm] / n[arm], run == "heldout_n30_nothink", prompt[arm].split(".")[0])
    order = sorted(arms, key=lambda a: arms[a][0])
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for i, arm in enumerate(order):
        rate = arms[arm][0]
        ax.plot([0.03, rate], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter([rate], [i], s=90, color=OL if rate >= 0.5 else TS, zorder=2, edgecolor="#fcfcfb", linewidth=2)
        ax.text(1.03, i, f"{rate * 30:.0f}/30", va="center", color=INK, fontsize=12)
    ax.scatter([0.03] * len(order), range(len(order)), s=50, color=MUTED, zorder=2, label="base: “A student in my class” (0–2/30)")
    short = lambda t: (
        t.replace(" wrote this essay", "")
        .replace(" asked me to pre-screen this essay", " asked me to pre-screen it")
        .replace("“I”", "“I wrote it”")
    )
    ax.set_yticks(range(len(order)), [("* " if arms[a][1] else "") + short(f"“{arms[a][2]}”") for a in order], fontsize=11.5)
    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of replies grading ≤ 4/10   (n = 30 per arm; * = held out)")
    ax.grid(axis="x", color=GRID)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.08), frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "rates.png", dpi=160)


def predictions():
    files = [d / "predictions.jsonl" for d in RUNS.glob("cfbench/2026-10-23_grade-big*")]
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines()]
    clip = lambda p: min(0.99, max(0.01, p))
    ll = lambda k, n, p: binom.logpmf(k, n, clip(p))
    by = collections.defaultdict(list)
    for r in rows:
        by[(SETS[r["env"].split("2026-10-22_")[1]], r["base_arm"], r["cf_arm"], r["condition"])].append(r)
    gain, orac = collections.defaultdict(dict), {}
    for (s, b, c, cond), rs in by.items():
        r = rs[0]
        null = (r["k_base"] + 0.5) / (r["n_base"] + 1)
        gain[(s, b, c)][cond] = sum(ll(r["k_cf"], r["n_cf"], x["p"]) for x in rs) / len(rs) - ll(r["k_cf"], r["n_cf"], null)
        orac[(s, b, c)] = ll(r["k_cf"], r["n_cf"], r["k_cf"] / r["n_cf"]) - ll(r["k_cf"], r["n_cf"], null)
    skill = lambda cs, k: sum(gain[c][k] for c in cs) / sum(orac[c] for c in cs)

    # skill by case set
    by_set = {s: [c for c in gain if c[0] == s] for s in ("held out", "attribution", "cue")}
    groups = [("all 38", list(gain))] + [(f"{s} ({len(cs)})", cs) for s, cs in by_set.items()]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    w = 0.26
    conditions = [("agentic", TS, "TS agent"), ("olens", OL, "O-lens agent"), ("olens_placebo", PL, "O-lens placebo")]
    for j, (cond, color, name) in enumerate(conditions):
        for i, (_, cs) in enumerate(groups):
            v = skill(cs, cond)
            ax.bar(i + (j - 1) * w, v, w - 0.03, color=color, label=name if i == 0 else None, zorder=2)
            ax.text(i + (j - 1) * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=11, color=INK)
    ax.set_xticks(range(len(groups)), [g for g, _ in groups])
    ax.set_ylabel("cfbench skill (1 = oracle, 0 = “edit does nothing”)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(OUT / "skill.png", dpi=160)

    # per-case nats over the TS agent
    cases = sorted(gain, key=lambda c: gain[c]["olens"] - gain[c]["agentic"])
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for i, c in enumerate(cases):
        ax.barh(i, gain[c]["olens"] - gain[c]["agentic"], 0.7, color=OL, zorder=2)
        ax.scatter(gain[c]["olens_placebo"] - gain[c]["agentic"], i, s=36, facecolor="none", edgecolor=MUTED, linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(cases)), [f"{c[1]} → {c[2]}  ({c[0]})" for c in cases], fontsize=9)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("log-likelihood gain over the TS agent (nats, mean of 5 draws)")
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.text(30, 6, "bars: O-lens agent\nhollow dots: O-lens placebo", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "case_gains.png", dpi=160)


OUT.mkdir(exist_ok=True)
rates()
predictions()
