"""cutoff_qa_inline by investigator: is the O-lens's gain on stale items just Haiku's later knowledge cutoff? Haiku 5.5 (cutoff
after Qwen's) beside Gemini 2.5 Flash-Lite (OpenRouter: 2025-01-31, about Qwen's), each predicting Qwen3.6-27B's answer from the
question alone and with the item's O-lens, J++ and NLA reads, each beside its placebo. Both also get the told prompt (the model
is Qwen3.6-27B, cutoff ~January 2025); Flash-Lite also gets the census (asked the question itself, no investigator prompt: what a
same-cutoff model answers). Panels: stale items (Qwen's answer is out of date), not stale, all. Exact match to Qwen's modal answer
(post-2026-10-31 labels), per-item mean over draws then the mean over items, 95% bootstrap CIs over items; a run that
isn't finished (meta.json status done) is left out (Robbie, 2026-10-31: chart Haiku now, add Flash-Lite as its runs land).

    uv run python -m graphs.cutoff_qa_investigators [--out docs/img/cutoff-qa-investigators.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from environments.cutoff_qa.env import norm
from graphs.cutoff_qa_inline import ci, per_item
from graphs.style import MUTED, bar_kw, ci_bar, label

CF = Path("data/runs/cfbench")
CENSUS = Path("data/runs/cutoff_qa_census/2026-10-31_flashlite-1k/census_flashlite.jsonl")
# investigator -> [(condition, run)]; "self_answer" is the census, the rest cfbench runs on the 999-item inputs
LENSES = [
    ("olens", "olens", "olens-placebo"),
    ("jlens", "jlens", "jlens-placebo"),
    ("nla", "nla", "nla-placebo-mmlu"),
]  # (cond, slug, placebo slug)


def lens_runs(prefix: str) -> list:
    return [(f"{c}{p}_inline", CF / f"{prefix}-{slug}") for c, s, ps in LENSES for p, slug in (("", s), ("_placebo", ps))]


# investigator -> [(condition, run)]; "self_answer" is the census, the rest cfbench runs on the 999-item inputs
INVESTIGATORS = {
    "Claude Haiku 5.5": [
        ("transcript_inline", CF / "2026-10-29_cutoff-qa-1k-transcript"),
        ("transcript_told_inline", CF / "2026-10-31_cutoff-qa-1k-told-transcript"),
        *lens_runs("2026-10-29_cutoff-qa-1k"),
    ],
    "Gemini 2.5 Flash-Lite": [
        ("self_answer", CENSUS),
        ("transcript_inline", CF / "2026-10-31_cutoff-qa-1k-flashlite-transcript"),
        ("transcript_told_inline", CF / "2026-10-31_cutoff-qa-1k-flashlite-told-transcript"),
        *lens_runs("2026-10-31_cutoff-qa-1k-flashlite"),
    ],
}
GROUPS = {"stale (Qwen's answer is out of date)": "stable_wrong", "not stale": "stable_right", "all items": None}


def census(path: Path, gold: dict[str, str], cls: dict[str, str]) -> dict[str, tuple]:
    """item -> (gate_class, 1.0 if the reply's norm equals Qwen's modal answer): the per_item shape, one draw."""
    rows = [json.loads(line) for line in path.open()]
    return {r["item"]: (cls[r["item"]], float(norm(r["reply"]) == gold[r["item"]])) for r in rows}


def load() -> dict[str, dict[str, dict]]:
    """investigator -> condition -> item -> (gate_class, accuracy), for the runs that exist."""
    out: dict[str, dict[str, dict]] = {}
    ref = [json.loads(line) for line in (INVESTIGATORS["Claude Haiku 5.5"][0][1] / "predictions.jsonl").open()]
    gold, cls = {r["item"]: r["gold"] for r in ref}, {r["item"]: r["gate_class"] for r in ref}
    for who, conds in INVESTIGATORS.items():
        for c, path in conds:
            run = path.parent if path.is_file() else path
            if not (run / "meta.json").exists() or json.loads((run / "meta.json").read_text()).get("status") != "done":
                continue  # not started, or still running
            if c == "self_answer":
                out.setdefault(who, {})[c] = census(path, gold, cls)
            else:
                data = per_item([path])
                out.setdefault(who, {})[c] = data["transcript_inline" if c == "transcript_told_inline" else c]
    return out


def main(out: Path) -> None:
    data = load()
    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(11, 2.5 * len(GROUPS)), sharex=True, layout="constrained")
    for ax, (group, value) in zip(axes, GROUPS.items(), strict=True):
        x, ticks, n = 0.0, [], 0
        for conds in data.values():
            start = x
            for c, items in conds.items():
                vals = [a for g, a in items.values() if value is None or g == value]
                m = sum(vals) / len(vals)
                lo, hi = ci(vals)
                ci_bar(ax, x, c, 100 * m, 100 * lo, 100 * hi)
                x, n = x + 0.85, max(n, len(vals))
            ticks.append((start + x - 0.85) / 2)
            x += 0.9
        ax.set_xticks(ticks, list(data))
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{group}, n = {n}", fontsize=9.5)
    for ax in axes:
        ax.set_ylim(0, 105)
        ax.set_ylabel("% naming Qwen's answer")
    seen = {c: None for conds in data.values() for c in conds}
    handles = [plt.Rectangle((0, 0), 1, 1, **bar_kw(c)) for c in seen]
    fig.legend(handles, [label(c) for c in seen], loc="outside lower center", ncol=5, fontsize=8.5)
    fig.suptitle("Predicting Qwen3.6-27B's answer to a factual question: a later-cutoff vs a same-cutoff investigator", x=0.01, ha="left")
    fig.text(0.995, 0.97, "exact match, 95% bootstrap CI over items", ha="right", fontsize=8, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/img/cutoff-qa-investigators.png"))
    main(p.parse_args().out)
