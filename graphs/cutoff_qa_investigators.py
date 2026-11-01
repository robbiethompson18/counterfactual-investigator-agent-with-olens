"""Knowledge cutoff by investigator: is the lenses' gain on stale items just Haiku's later knowledge cutoff? Claude Haiku 5.5 (cutoff
after Qwen's) beside Gemini 2.5 Flash-Lite (OpenRouter: 2025-01-31, about Qwen's), each predicting Qwen3.6-27B's answer from the
question alone and, where run, with the item's O-lens, J++ and NLA reads, each beside its placebo; plus each model's census (asked
the question itself, no investigator prompt: what its own knowledge says). Panels: stale items (Qwen's answer is out of date), not
stale, all. Exact match to Qwen's modal answer (post-2026-10-31 labels), per-item mean over draws then the mean over items, 95%
bootstrap CIs over items; a run that isn't finished (meta.json status done) is left out.

Runs (Robbie, 2026-10-31): the menus_inline cutoff runs, canonical menu (cfbench.catalog MENUS / CANONICAL_MENU), Haiku at 3
draws (draw 0 plus the draws 1-2 top-up); Flash-Lite has prompt-only only there, at 1 draw. Plus one bar from the old prompt: Haiku
told the model is Qwen3.6-27B with a ~January 2025 cutoff (the only told run; no menus_inline told run exists). A second figure is
Haiku's paired differences on all items. gate_class (stale vs not) and the label (qwen_modal) come from the cutoff_qa item bank
(environments/cutoff_qa/env.py), since the menus_inline runs' rows don't carry gate_class.

    uv run python -m graphs.cutoff_qa_investigators [--out PNG] [--out-diff PNG]
"""

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt

from cfbench.catalog import CANONICAL_MENU, CUTOFF_CENSUS, CUTOFF_FLASHLITE, CUTOFF_TOLD
from environments.cutoff_qa.env import _items as bank_items
from environments.cutoff_qa.env import norm
from graphs import menus_inline
from graphs.style import MUTED, bar_kw, ci_bar, departure, diff_rows, label

CENSUS = CUTOFF_CENSUS
DEPARTURE = 'Flash-Lite rows are one draw per item; Haiku\'s "told" bar (model name and cutoff given) is on an earlier prompt.'


MENU_RUNS = menus_inline.runs("cutoff")  # every finished menus_inline cutoff run; each bar keeps its own condition


def menus_bars(lens: str) -> list:
    return [(f"{lens}{p}_inline", MENU_RUNS, f"{lens}_{CANONICAL_MENU}{p}_inline") for p in ("", "_placebo")]


# investigator -> [(bar, run, condition in the run)]; each investigator's census bar is added in load(). One set, `canonical`,
# so graphs.talk can name it
SETS = {
    "canonical": {
        "Claude Haiku 5.5": [
            ("transcript_inline", MENU_RUNS, "transcript_inline"),
            ("transcript_told_oldprompt_inline", [CUTOFF_TOLD], "transcript_inline"),
            *menus_bars("olens"),
            *menus_bars("jlens"),
            *menus_bars("nla"),
        ],
        "Gemini 2.5 Flash-Lite": [("transcript_inline", [CUTOFF_FLASHLITE], "transcript_inline")],
    },
}
OUT = Path("docs/img/cutoff-qa-investigators.png")
OUT_DIFF = Path("docs/img/cutoff-qa-investigators-diff.png")
GROUPS = {"stale (Qwen's answer is out of date)": "stable_wrong", "not stale": "stable_right", "all items": None}


def per_item(runs: list[Path], by: str | Callable[[dict], object] = "gate_class") -> dict[str, dict[str, tuple]]:
    """condition -> item -> (the row's `by` field, or by(row); mean accuracy over draws), pooled over the runs."""
    acc: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    cls = {}
    for line in (ln for run in runs for ln in (run / "predictions.jsonl").open()):
        r = json.loads(line)
        acc[r["condition"]][r["item"]].append(bool(r["correct"]))
        cls[r["item"]] = by(r) if callable(by) else r[by]
    return {c: {i: (cls[i], sum(v) / len(v)) for i, v in items.items()} for c, items in acc.items()}


def ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


# the difference chart: (question, [(a, b), ...]) groups, each row a − b
PAIRS = [
    ("Does the lens beat its placebo?", [(f"{lens}_inline", f"{lens}_placebo_inline") for lens in ("olens", "jlens", "nla")]),
    ("Does the placebo alone help?", [(f"{lens}_placebo_inline", "transcript_inline") for lens in ("olens", "jlens", "nla")]),
    ("Lens vs transcript only", [(f"{lens}_inline", "transcript_inline") for lens in ("olens", "jlens", "nla")]),
]


def diffs(data: dict, out: Path, xlim: tuple[float, float]) -> None:
    """graphs.style.diff_rows over every item, a row per PAIRS pair grouped by the question it answers: the mean paired difference
    in per-item accuracy (pp) with its 95% bootstrap CI over items; filled when the CI excludes 0."""
    rows: list = []
    for q, group in PAIRS:
        rows.append(q)
        for a, b in group:
            ds = [100 * (data[a][i][1] - data[b][i][1]) for i in data[a]]
            rows.append((a, b, sum(ds) / len(ds), *ci(ds)))
    fig, ax = plt.subplots(figsize=(6.2, 0.35 + 0.34 * len(rows)))
    ax.set_xlim(*xlim)
    diff_rows(ax, rows)
    ax.set_xlabel("difference in % of items right, paired by item (95% bootstrap CI; filled = excludes 0)", fontsize=8.5)
    ax.set_title(f"All {len(data['transcript_inline'])} items")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


def done(run: Path) -> bool:
    meta = run / "meta.json"
    return meta.exists() and json.loads(meta.read_text()).get("status") == "done"


def load(which: str) -> dict[str, dict[str, dict]]:
    """investigator -> bar -> item -> (gate_class, accuracy), for the finished runs."""
    bank = bank_items()  # gate_class and the label (qwen_modal: Qwen's gated modal answer) per item
    gold, cls = {i: it["qwen_modal"] for i, it in bank.items()}, {i: it["gate_class"] for i, it in bank.items()}
    out: dict[str, dict[str, dict]] = {}
    for who, bars in SETS[which].items():
        rows = [json.loads(line) for line in CENSUS[who].open()]
        out[who] = {"self_answer": {r["item"]: (cls[r["item"]], float(norm(r["reply"]) == gold[r["item"]])) for r in rows}}
        loaded: dict[tuple, dict] = {}  # the canonical bars share one run list: read it once
        for bar, runs, cond in bars:
            if all(map(done, runs)):
                if tuple(runs) not in loaded:
                    loaded[tuple(runs)] = per_item(runs, by=lambda r: cls[r["item"]])
                out[who][bar] = loaded[tuple(runs)][cond]
    return out


def draw(axes, data: dict, s: float = 1.0) -> list:
    """One panel per GROUPS entry on `axes`, an investigator's bars side by side; returns the legend handles. `s` scales the
    text (the deck passes graphs.style.SLIDE_S)."""
    for ax, (group, value) in zip(axes, GROUPS.items(), strict=True):
        x, ticks, n = 0.0, [], 0
        for bars in data.values():
            start = x
            for c, items in bars.items():
                vals = [a for g, a in items.values() if value is None or g == value]
                lo, hi = ci(vals)
                ci_bar(ax, x, c, 100 * sum(vals) / len(vals), 100 * lo, 100 * hi, fontsize=7.5 * s)
                x, n = x + 0.85, max(n, len(vals))
            ticks.append((start + x - 0.85) / 2)
            x += 0.9
        ax.set_xticks(ticks, list(data))
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{group}, n = {n}", fontsize=9.5 * s)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% naming Qwen's answer")
    seen = {c: None for bars in data.values() for c in bars}
    return [plt.Rectangle((0, 0), 1, 1, label=label(c), **bar_kw(c)) for c in seen]


def main(out: Path, out_diff: Path) -> None:
    data = load("canonical")
    diffs(data["Claude Haiku 5.5"], out_diff, xlim=(-36, 36))  # Haiku, all items
    n_bars = sum(len(bars) for bars in data.values())
    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(max(6.5, 0.62 * n_bars + 1.5), 2.5 * len(GROUPS)), sharex=True, layout="constrained")
    handles = draw(axes, data)
    axes[-1].legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8.5
    )  # not an outside figure legend: see departure
    departure(fig, DEPARTURE)
    fig.suptitle("Knowledge Cutoff: a later-cutoff vs a same-cutoff investigator", x=0.01, ha="left")
    fig.text(0.995, 0.975, "exact match, 95% CI over items", ha="right", fontsize=8, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--out-diff", type=Path, default=OUT_DIFF)
    a = p.parse_args()
    main(a.out, a.out_diff)
