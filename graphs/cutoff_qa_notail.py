"""cutoff_qa_tool_inline without the tail reads (Kylie, 2026-10-30): Haiku's error rate naming Qwen's answer with the O-lens block cut
to the question's punctuation, that plus <|im_end|>, or the full set (punctuation + the last 10 tokens, the v3 run), each next to
its placebo, with transcript-only for reference; one panel each for stale / not-stale / all items. Bars are the mean over items of
the per-item error over 3 draws, whiskers 95% bootstrap CIs over items. A second figure shows the paired differences (per item,
then the mean) with 95% bootstrap CIs, stale items. docs/cutoff-qa-inline.md.

    uv run python -m graphs.cutoff_qa_notail [--out docs/img/cutoff-qa-notail.png] [--out-diff docs/img/cutoff-qa-notail-diff.png]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from cfbench.catalog import CUTOFF_QA_TOOL_CUTS, CUTOFF_QA_TOOL_V3
from graphs.cutoff_qa_inline import ci
from graphs.style import OLENS, bar_kw, ci_bar, diff_rows

RUN, CUTS = CUTOFF_QA_TOOL_V3, CUTOFF_QA_TOOL_CUTS  # transcript-only and the full set, then the cuts
READS = {"qpunct": "question\npunctuation", "qpunct_imend": "punctuation\n+ <|im_end|>", "full": "full set\n(+ last 10)"}
PANELS = {"stale (Qwen's answer is out of date)": "stable_wrong", "not stale": "stable_right", "all items": None}


def per_item() -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """'transcript' / '<reads>' / '<reads>_placebo' -> item -> mean error over draws; item -> gate_class."""
    err: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cls = {}
    keep = {"transcript_inline": "transcript", "olens_inline": "full", "olens_placebo_inline": "full_placebo"}
    keep |= {f"olens_{c}{p}_inline": f"{c}{p}" for c in CUTS for p in ("", "_placebo")}
    for run in [RUN, *CUTS.values()]:
        for line in (run / "predictions.jsonl").open():
            cond = line[line.index('"condition": "') + 14 :].split('"', 1)[0]  # skip json-parsing the 1.6 GB of other conditions
            if cond not in keep:
                continue
            r = json.loads(line)
            err[keep[cond]][r["item"]].append(0.0 if r["correct"] else 1.0)
            cls[r["item"]] = r["gate_class"]
    return {c: {i: sum(v) / len(v) for i, v in items.items()} for c, items in err.items()}, cls


def bars(data, cls, out: Path) -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 4.2), sharey=True)
    width = 0.36
    for ax, (title, value) in zip(axes, PANELS.items(), strict=True):
        items = [i for i in data["transcript"] if value is None or cls[i] == value]

        def bar(x, key, cond, ax=ax, items=items):
            xs = [data[key][i] for i in items]
            m, (lo, hi) = sum(xs) / len(xs), ci(xs)
            ci_bar(ax, x, cond, 100 * m, 100 * lo, 100 * hi, width=width)

        bar(0, "transcript", "transcript_inline")
        for k, reads in enumerate(READS, start=1):
            bar(k - width / 2, reads, "olens_inline")
            bar(k + width / 2, f"{reads}_placebo", "olens_placebo_inline")
        ax.set_xticks(range(len(READS) + 1))
        ax.set_xticklabels(["no reads", *READS.values()], fontsize=8)
        ax.set_title(f"{title}, n = {len(items)}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of items where Haiku misses Qwen's answer")
    axes[0].set_ylim(0, 60)
    handles = [plt.Rectangle((0, 0), 1, 1, **bar_kw(c)) for c in ("transcript_inline", "olens_inline", "olens_placebo_inline")]
    fig.legend(handles, ["transcript only", OLENS, f"{OLENS} placebo (another item's reads)"], loc="upper left", frameon=False,
               fontsize=8.5, ncol=3, bbox_to_anchor=(0.06, 1.04))  # fmt: skip
    fig.suptitle(f"Predicting Qwen3.6-27B's (tool-call) answer: {OLENS} reads at the question's punctuation vs the full set, error rate",
                 x=0.06, y=1.1, ha="left")  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


def diffs(data, cls, out: Path) -> None:
    """Rows a − b in error rate (pp, lower = better for a), stale items, grouped by read set."""
    items = [i for i in data["transcript"] if cls[i] == "stable_wrong"]
    rows: list = []
    for reads, name in READS.items():
        rows.append(name.replace("\n", " ").replace("punctuation +", "question punctuation +"))
        for a, b, ca, cb in ((reads, f"{reads}_placebo", "olens_inline", "olens_placebo_inline"),
                             (reads, "transcript", "olens_inline", "transcript_inline"),
                             (f"{reads}_placebo", "transcript", "olens_placebo_inline", "transcript_inline")):  # fmt: skip
            ds = [100 * (data[a][i] - data[b][i]) for i in items]
            rows.append((ca, cb, sum(ds) / len(ds), *ci(ds)))
    for cut, name in list(READS.items())[:2]:
        rows.append(f"{name.replace(chr(10), ' ').replace('punctuation +', 'question punctuation +')} − full set (+ last 10), both {OLENS}")
        ds = [100 * (data[cut][i] - data["full"][i]) for i in items]
        rows.append(("olens_inline", "olens_inline", sum(ds) / len(ds), *ci(ds)))
    fig, ax = plt.subplots(figsize=(6.4, 0.35 + 0.34 * len(rows)))
    ax.set_xlim(-30, 10)
    diff_rows(ax, rows)
    ax.set_xlabel("difference in error rate, pp, paired by item (95% bootstrap CI; filled = excludes 0)", fontsize=8.5)
    ax.set_title(f"Stale items, n = {len(items)} (negative = the first condition misses less)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/img/cutoff-qa-notail.png"))
    p.add_argument("--out-diff", type=Path, default=Path("docs/img/cutoff-qa-notail-diff.png"))
    a = p.parse_args()
    data, cls = per_item()
    bars(data, cls, a.out)
    diffs(data, cls, a.out_diff)
