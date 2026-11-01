"""How many tokens each lens's reads add to the investigator's (Haiku 5.5) prompt, for the write-up: does one lens just give it more to
think about? Same runs (menus_inline, `all` menu) and weighting as graphs.headline_inline: MCQ pools MMLU-Pro and SuperGPQA cases, and
pooled weights MCQ, knowledge cutoff and preferences equally. Added tokens = the condition's mean `tokens.prompt` (Anthropic's count
of the whole opening call, tool schemas included) minus prompt only's on the same cases, averaged over draws. Rows with calls !=
1 are dropped (a second call's prompt also holds the first): a handful of retries. No whiskers: this describes what was shown.

    uv run python -m graphs.headline_tokens [--out-dir docs/img]

Writes headline-tokens.png and headline-thinking.png (mean thinking tokens per condition, pooled) and prints added prompt and completion
tokens and % right per set, placebos included.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfbench.scoring import prediction_fields
from graphs import menus_inline
from graphs.headline_inline import LENSES, PLACEBOS, PLAIN_COND, TS
from graphs.style import bar_kw, label, positions

SETS = {  # the canonical menus_inline runs (graphs/CLAUDE.md)
    "MMLU-Pro": menus_inline.runs("mcq_unselected"),
    "SuperGPQA": menus_inline.runs("supergpqa"),
    "Knowledge Cutoff": menus_inline.runs("cutoff"),
    "Idiosyncratic Preferences": menus_inline.runs("favorites"),
}
MCQ = ["MMLU-Pro", "SuperGPQA"]
METRICS = ("prompt", "completion", "reasoning", "correct")


def load(runs: list[Path]) -> dict:
    """condition -> item -> metric -> mean over draws."""
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in (r for run in runs for r in prediction_fields(run)):
        if r["condition"] in PLAIN_COND and r["tokens"].get("calls") == 1:
            case = rows[PLAIN_COND[r["condition"]]][r["item"]]
            case["prompt"].append(r["tokens"]["prompt"])
            case["completion"].append(r["tokens"]["completion"])
            case["reasoning"].append(r["tokens"]["reasoning"])
            case["correct"].append(100 * r["correct"])
    return {c: {i: {m: float(np.mean(case[m])) for m in METRICS} for i, case in items.items()} for c, items in rows.items()}


def added(data: dict, c: str, metric: str) -> list[float]:
    """Condition c minus prompt only, case by case."""
    return [case[metric] - data[TS][i][metric] for i, case in data[c].items() if i in data[TS]]


def summary(sets: dict[str, dict], c: str, metric: str) -> dict[str, float]:
    """Each set's mean added `metric`, plus MCQ (pooled by case) and the three tasks' equal-weight mean."""
    out = {name: float(np.mean(added(data, c, metric))) for name, data in sets.items()}
    out["MCQ"] = float(np.mean([v for name in MCQ for v in added(sets[name], c, metric)]))
    out["pooled"] = float(np.mean([out["MCQ"], out["Knowledge Cutoff"], out["Idiosyncratic Preferences"]]))
    return out


def chart_ax(ax, tokens: dict[str, dict[str, float]], s: float = 1.0) -> None:
    """Prompt tokens each lens adds, pooled and per set, on ax; `s` scales the text (the deck passes graphs.style.SLIDE_S)."""
    groups = ["pooled", *SETS]
    width = 0.27
    for k, c in enumerate(LENSES):
        xs = [g + (k - 1) * width for g in range(len(groups))]
        ys = [tokens[c][g] / 1000 for g in groups]
        ax.bar(xs, ys, width * 0.95, zorder=3, label=label(c), **bar_kw(c))
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=7 * s)
    ax.set_xticks(range(len(groups)), [g.replace(" ", "\n") for g in groups], fontsize=8.5 * s)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("tokens added to Haiku's prompt (thousands)")
    ax.legend(loc="upper right")


def chart(tokens: dict[str, dict[str, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
    chart_ax(ax, tokens)
    ax.set_title("Prompt tokens each lens adds over prompt only")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


THINKING_GROUPS = [("prompt\nonly", [TS]), *((label(c), [c, PLACEBOS[c]]) for c in LENSES)]


def thinking_means(sets: dict[str, dict]) -> dict[str, float]:
    """Mean thinking tokens per condition, with the three tasks weighted equally."""

    def mean(c: str, names: list[str]) -> float:
        return float(np.mean([case["reasoning"] for name in names for case in sets[name][c].values()]))

    conditions = [c for _, cs in THINKING_GROUPS for c in cs]
    return {c: float(np.mean([mean(c, MCQ), mean(c, ["Knowledge Cutoff"]), mean(c, ["Idiosyncratic Preferences"])])) for c in conditions}


def thinking_ax(ax, means: dict[str, float], s: float = 1.0) -> None:
    xs, ticks = positions(means, THINKING_GROUPS)
    for c, v in means.items():
        ax.bar(xs[c], v, 0.8, zorder=3, **bar_kw(c))
        ax.annotate(f"{v:.0f}", (xs[c], v), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=7.5 * s)
    ax.set_xticks(ticks, [name for name, _ in THINKING_GROUPS], fontsize=8.5 * s)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Haiku's thinking tokens per prediction")


def thinking(sets: dict[str, dict], path: Path) -> None:
    means = thinking_means(sets)
    print("thinking tokens, pooled:", {c: round(v) for c, v in means.items()})
    fig, ax = plt.subplots(figsize=(4.8, 3.2), layout="constrained")
    thinking_ax(ax, means)
    ax.set_title("Thinking tokens, pooled over the three tasks")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main(out: Path) -> None:
    sets = {name: load(runs) for name, runs in SETS.items()}
    print("prompt-only prompt tokens:", {name: round(float(np.mean([v["prompt"] for v in d[TS].values()]))) for name, d in sets.items()})
    for metric in METRICS:
        print(f"== added {metric} vs prompt only")
        for c in (x for lens in LENSES for x in (lens, PLACEBOS[lens])):
            print(f"{c:24}", {k: round(v, 1) for k, v in summary(sets, c, metric).items()})
    chart({c: summary(sets, c, "prompt") for c in LENSES}, out / "headline-tokens.png")
    thinking(sets, out / "headline-thinking.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("docs/img"))
    main(ap.parse_args().out_dir)
