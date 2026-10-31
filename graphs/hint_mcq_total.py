"""Running total of the hint_mcq answer task: every Flash run on the original 200 items and the 2,000-question scale-up,
scored only on items where Qwen followed the hint on all 8 base samples, with no text-inversion gate. Two charts: one bar
per condition over all items, and the same split by hint type (gold letter vs another wrong letter), 95% item bootstrap.

    uv run python -m graphs.hint_mcq_total [--out docs/img/hint-mcq-total]

Runs are picked by name (original header and gate-killed runs, every `*_answer-mcq-scale2k-d1*`/`-d2*` run and fill), so
fills still in progress count. A (item, condition, draw) seen in two runs is counted once. Other predictors' runs
(sol50, astra10) and conditions other than the three charted (e.g. jlens_agentic) are skipped.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from graphs.style import INK, MUTED, bar_kw

RUNS = Path("data/runs/cfbench")
ORIG = {"d1": ["2026-10-27_answer-mcq-olens-header", "2026-10-27_answer-mcq-orig-killed-d1"],
        "d2": ["2026-10-27_answer-mcq-olens-header-d2", "2026-10-27_answer-mcq-orig-killed-d2"]}  # fmt: skip
OTHER_PREDICTORS = ("sol50", "astra10")
TITLE = {"d1": "hint is the correct answer", "d2": "hint is the wrong answer"}
CANON = {"agentic": "transcript_agentic", "olens": "olens_agentic", "olens_placebo": "olens_placebo_agentic"}
CONDS = ["transcript_agentic", "olens_agentic", "olens_placebo_agentic"]
LABEL = {"transcript_agentic": "Transcript-only", "olens_agentic": "O-lens", "olens_placebo_agentic": "O-lens placebo"}
YLABEL = "% draws naming Qwen's no-hint answer"
N_BOOT = 4000


def runs(slot: str) -> list[Path]:
    scale = [p for p in sorted(RUNS.glob(f"*_answer-mcq-scale2k-{slot}*")) if not p.name.endswith(OTHER_PREDICTORS)]
    return [RUNS / name for name in ORIG[slot]] + scale


def load(slot: str) -> dict:
    """condition -> item -> [correct per draw], 8/8 hint followers with every condition present."""
    by = {c: defaultdict(list) for c in CONDS}
    seen = set()
    for run in runs(slot):
        for line in run.joinpath("predictions.jsonl").read_text().splitlines():
            r = json.loads(line)
            c = CANON.get(r["condition"], r["condition"])
            item = r["env"].rsplit("@", 1)[1]
            hint = re.search(r"thinks the answer is ([A-J])", json.dumps(r["shown"]) + json.dumps(r["messages"]))
            if c not in by or not hint or r["answers_base"].get(hint.group(1), 0) < sum(r["answers_base"].values()):
                continue
            key = (item, c, r["draw"], "or_nothink2" in r["env"])  # original and scale-up items share ids
            if key not in seen:
                seen.add(key)
                by[c][item].append(bool(r["correct"]))
    keep = [k for k in by["olens_agentic"] if all(by[c][k] for c in CONDS)]
    return {c: {k: by[c][k] for k in keep} for c in CONDS}


def summarize(by: dict, rng: np.random.Generator) -> dict:
    keys = sorted(by["olens_agentic"])
    means = {c: np.array([np.mean(by[c][k]) for k in keys]) for c in CONDS}
    means["olens - placebo"] = means["olens_agentic"] - means["olens_placebo_agentic"]
    out = {}
    for name, m in means.items():
        boot = [np.mean(m[rng.integers(0, len(keys), len(keys))]) for _ in range(N_BOOT)]
        out[name] = (m.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5))
    return out


def panel(ax: plt.Axes, st: dict) -> None:
    for x, c in enumerate(CONDS):
        m, lo, hi = st[c]
        ax.bar(x, 100 * m, width=0.55, **bar_kw(c))
        ax.errorbar(x, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="none", ecolor=MUTED, elinewidth=1.2, capsize=4)
        ax.text(x, 100 * hi + 1, f"{100 * m:.1f}%", ha="center", va="bottom", color=INK)
    ax.axhline(10, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([LABEL[c] for c in CONDS], fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 50)


def report(name: str, n: int, st: dict) -> None:
    print(f"== {name}: {n} items")
    for c, (m, lo, hi) in st.items():
        print(f"   {LABEL.get(c, c):16s} {100 * m:+5.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("docs/img/hint-mcq-total"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    slots = {slot: load(slot) for slot in ("d1", "d2")}
    note = "Qwen followed the hint 8/8, no text-inversion gate. Flash medium, harness v7, 3-5 draws per item. 95% item bootstrap."

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (slot, by) in zip(axes, slots.items(), strict=True):
        st = summarize(by, rng)
        n = len(by["olens_agentic"])
        report(TITLE[slot], n, st)
        panel(ax, st)
        ax.set_title(f"{TITLE[slot]} (n={n})")
    axes[0].set_ylabel(YLABEL)
    axes[0].text(-0.45, 10.8, "chance", fontsize=8, color=MUTED)
    fig.suptitle("Answer task on hint_mcq, by hint type (running total)", x=0.01, ha="left")
    fig.text(0.01, -0.03, note, color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "by_hint.png", bbox_inches="tight")

    pooled = {c: {(slot, k): v for slot, by in slots.items() for k, v in by[c].items()} for c in CONDS}
    st = summarize(pooled, rng)
    n, n_q = len(pooled["olens_agentic"]), len({k for _, k in pooled["olens_agentic"]})
    report("both hint types", n, st)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    panel(ax, st)
    ax.set_ylabel(YLABEL)
    ax.text(-0.45, 10.8, "chance", fontsize=8, color=MUTED)
    ax.set_title(f"Answer task on hint_mcq: all {n} items, both hint types")
    n_d1, n_d2 = (len(by["olens_agentic"]) for by in slots.values())
    caption = f"{n_d1} correct-hint + {n_d2} wrong-hint items ({n_q} distinct questions, bootstrapped as independent).\n"
    fig.text(0.01, -0.06, caption + note.replace(" Flash", "\nFlash"), color=MUTED, fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(args.out / "combined.png", bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
