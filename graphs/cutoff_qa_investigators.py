"""Knowledge cutoff by investigator: is the lenses' gain on stale items just Haiku's later knowledge cutoff? Claude Haiku 5.5 (cutoff
after Qwen's) beside Gemini 2.5 Flash-Lite (OpenRouter: 2025-01-31, about Qwen's), each predicting Qwen3.6-27B's answer from the
question alone and, where run, with the item's O-lens, J++ and NLA reads, each beside its placebo; plus each model's census (asked
the question itself, no investigator prompt: what its own knowledge says). Panels: stale items (Qwen's answer is out of date), not
stale, all. Exact match to Qwen's modal answer (post-2026-10-31 labels), per-item mean over draws then the mean over items, 95%
bootstrap CIs over items; a run that isn't finished (meta.json status done) is left out.

Two sets (Robbie, 2026-10-31):
- `canonical` (default, the final write-up): the menus_inline cutoff runs, `all` menu (graphs/menus_inline.py SETS), Haiku at 3
  draws (draw 0 plus the draws 1-2 top-up); Flash-Lite has prompt-only only there, at 1 draw.
- `cutoff_qa_inline`: the older eval's prompt and end-of-prompt reads, with every Flash-Lite condition and the told prompt (the
  model is Qwen3.6-27B, cutoff ~January 2025); docs/cutoff-qa-inline.md. Not for the write-up.

    uv run python -m graphs.cutoff_qa_investigators [--set canonical|cutoff_qa_inline] [--out PNG]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from environments.cutoff_qa.env import norm
from graphs.cutoff_qa_inline import ci, per_item
from graphs.style import MUTED, bar_kw, ci_bar, label

CF = Path("data/runs/cfbench")
CENSUS = {
    "Claude Haiku 5.5": Path("data/runs/cutoff_qa_census/2026-10-31_haiku-1k/census.jsonl"),
    "Gemini 2.5 Flash-Lite": Path("data/runs/cutoff_qa_census/2026-10-31_flashlite-1k/census_flashlite.jsonl"),
}
REF = CF / "2026-10-29_cutoff-qa-1k-transcript/predictions.jsonl"  # gold labels and gate_class per item


def old_lens_runs(prefix: str) -> list:
    """(bar, run, condition in the run) for the cutoff_qa_inline lens runs, one run per condition."""
    slugs = [("olens", "olens-placebo"), ("jlens", "jlens-placebo"), ("nla", "nla-placebo-mmlu")]
    return [(f"{c}{p}_inline", [CF / f"{prefix}-{s}"], f"{c}{p}_inline") for c, ps in slugs for p, s in (("", c), ("_placebo", ps))]


TOPUP = CF / "2026-10-31_menus-cutoff-all-d12b"  # draws 1-2 of prompt only and every `all`-menu condition


def menus_runs(run: str, lens: str) -> list:
    return [(f"{lens}{p}_inline", [CF / run, TOPUP], f"{lens}_all{p}_inline") for p in ("", "_placebo")]


# set -> investigator -> [(bar, run, condition in the run)]; each investigator's census bar is added in load()
SETS = {
    "canonical": {
        "Claude Haiku 5.5": [
            ("transcript_inline", [CF / "2026-10-30_menus-cutoff-olens-jlens", TOPUP], "transcript_inline"),
            *menus_runs("2026-10-30_menus-cutoff-olens-jlens", "olens"),
            *menus_runs("2026-10-30_menus-cutoff-olens-jlens", "jlens"),
            *menus_runs("2026-10-31_menus-cutoff-nla-batch", "nla"),
        ],
        "Gemini 2.5 Flash-Lite": [("transcript_inline", [CF / "2026-10-31_menus-cutoff-flashlite-transcript"], "transcript_inline")],
    },
    "cutoff_qa_inline": {
        "Claude Haiku 5.5": [
            ("transcript_inline", [CF / "2026-10-29_cutoff-qa-1k-transcript"], "transcript_inline"),
            ("transcript_told_inline", [CF / "2026-10-31_cutoff-qa-1k-told-transcript"], "transcript_inline"),
            *old_lens_runs("2026-10-29_cutoff-qa-1k"),
        ],
        "Gemini 2.5 Flash-Lite": [
            ("transcript_inline", [CF / "2026-10-31_cutoff-qa-1k-flashlite-transcript"], "transcript_inline"),
            ("transcript_told_inline", [CF / "2026-10-31_cutoff-qa-1k-flashlite-told-transcript"], "transcript_inline"),
            *old_lens_runs("2026-10-31_cutoff-qa-1k-flashlite"),
        ],
    },
}
OUT = {"canonical": "docs/img/cutoff-qa-investigators.png", "cutoff_qa_inline": "docs/img/cutoff-qa-investigators-old-prompt.png"}
GROUPS = {"stale (Qwen's answer is out of date)": "stable_wrong", "not stale": "stable_right", "all items": None}


def done(run: Path) -> bool:
    meta = run / "meta.json"
    return meta.exists() and json.loads(meta.read_text()).get("status") == "done"


def load(which: str) -> dict[str, dict[str, dict]]:
    """investigator -> bar -> item -> (gate_class, accuracy), for the finished runs."""
    ref = [json.loads(line) for line in REF.open()]
    gold, cls = {r["item"]: r["gold"] for r in ref}, {r["item"]: r["gate_class"] for r in ref}
    out: dict[str, dict[str, dict]] = {}
    for who, bars in SETS[which].items():
        rows = [json.loads(line) for line in CENSUS[who].open()]
        out[who] = {"self_answer": {r["item"]: (cls[r["item"]], float(norm(r["reply"]) == gold[r["item"]])) for r in rows}}
        for bar, runs, cond in bars:
            if all(map(done, runs)):
                out[who][bar] = per_item(runs, by=lambda r: cls[r["item"]])[cond]
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


def main(which: str, out: Path) -> None:
    data = load(which)
    n_bars = sum(len(bars) for bars in data.values())
    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(max(6.5, 0.62 * n_bars + 1.5), 2.5 * len(GROUPS)), sharex=True, layout="constrained")
    handles = draw(axes, data)
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8.5)
    fig.suptitle("Knowledge cutoff: a later-cutoff vs a same-cutoff investigator", x=0.01, ha="left")
    fig.text(0.995, 0.975, "exact match, 95% CI over items", ha="right", fontsize=8, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=list(SETS), default="canonical")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    main(a.set, a.out or Path(OUT[a.set]))
