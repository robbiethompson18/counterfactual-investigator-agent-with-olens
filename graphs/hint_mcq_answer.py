"""Charts for the hint_mcq answer task (docs/envs/hint_cf.md, "Assistant-header reads" section): % of predictor draws
naming Qwen's no-hint letter, per condition, for the gold-letter hint, the other-wrong-letter hint and both pooled, with
95% item-bootstrap CIs; plus one dot grid of correct draws per item for each hint type. Same colors as charts.py.

    uv run python -m graphs.hint_mcq_answer [--out docs/img/hint-mcq-answer] [--no-filter]

Runs: data/runs/cfbench/2026-10-27_answer-mcq-olens-header (hint+d1 -> nohint, 18 survivors) and
2026-10-27_answer-mcq-olens-header-d2 (hint+d2 -> nohint, 7 survivors), 5 draws each. By default only items whose hinted
arm followed the hint on all 8 samples are scored (data/runs/hint_cf_wrong_mcq/2026-10-26_or_nothink2/rows.jsonl):
a base run that ignored the hint hands the predictor the own answer (2509, 1386 on d1; 1656, 1699 on d2).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfbench.catalog import HINT_MCQ_ANSWER_RUNS, HINT_MCQ_ROLLOUTS
from graphs.style import BG, INK, MUTED, bar_kw, color

ROLLOUTS, RUNS = HINT_MCQ_ROLLOUTS, HINT_MCQ_ANSWER_RUNS
CONDS = ["agentic", "olens", "olens_placebo"]
LABEL = {"agentic": "Transcript-only", "olens": "O-lens", "olens_placebo": "O-lens placebo"}
COLOR = {c: color(c) for c in LABEL}  # graphs.style: a placebo is its tool's colour, hatched
N_BOOT = 4000


def followed_all(rollouts: list[dict], item: str, arm: str) -> bool:
    """Every hinted sample of the item answered the hint letter."""
    rows = [r for r in rollouts if r["item"] == item and r["arm"] == arm]
    hint = rows[0]["meta"]["d1" if arm == "hint+d1" else "d2"]
    return all(r["detect_notes"]["answer"] == hint for r in rows)


def load(run: Path, rollouts: list[dict], filter_followers: bool) -> tuple[dict, dict]:
    """condition -> (arm, item) -> [correct per draw], and (arm, item) -> (own letter, hint letter)."""
    by = {c: defaultdict(list) for c in CONDS}
    meta = {}
    for line in run.joinpath("predictions.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = (r["base_arm"], r["env"].rsplit("@", 1)[1])
        if filter_followers and not followed_all(rollouts, key[1], key[0]):
            continue
        if r["condition"] in by:
            by[r["condition"]][key].append(bool(r["correct"]))
        if key not in meta:
            m = re.search(r"thinks the answer is ([A-J])", json.dumps(r["shown"]) + json.dumps(r["messages"]))
            meta[key] = (r["modal_cf"], m.group(1) if m else "?")
    return by, meta


def summarize(by: dict, rng: np.random.Generator) -> tuple[list, dict]:
    keys = sorted(by["olens"])
    means = {c: np.array([np.mean(by[c][k]) for k in keys]) for c in CONDS}
    out = {}
    for c in CONDS:
        boot = [np.mean(means[c][rng.integers(0, len(keys), len(keys))]) for _ in range(N_BOOT)]
        out[c] = (means[c].mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), int(np.sum(means[c] > 0)))
    diff = means["olens"] - means["olens_placebo"]
    boot = [np.mean(diff[rng.integers(0, len(keys), len(keys))]) for _ in range(N_BOOT)]
    out["diff"] = (diff.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), int(np.sum(diff > 0)), int(np.sum(diff < 0)))
    return keys, out


def bars(panels: dict, out: Path, filtered: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (name, by) in zip(axes, panels.items(), strict=True):
        keys, st = summarize(by, rng)
        print(f"== {name}: {len(keys)} pairs")
        for c in CONDS:
            m, lo, hi, n_hit = st[c]
            print(f"   {LABEL[c]:16s} {100 * m:5.1f}% [{100 * lo:.0f}, {100 * hi:.0f}]  items with any hit {n_hit}/{len(keys)}")
        m, lo, hi, wins, losses = st["diff"]
        ties = len(keys) - wins - losses
        print(f"   olens - placebo {100 * m:+.1f}pp [{100 * lo:+.0f}, {100 * hi:+.0f}]  wins {wins} losses {losses} ties {ties}")
        for x, c in enumerate(CONDS):
            m, lo, hi, _ = st[c]
            ax.bar(x, 100 * m, width=0.55, **bar_kw(c))
            ax.errorbar(x, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="none", ecolor=MUTED, elinewidth=1.2, capsize=4)
            ax.text(x, 100 * hi + 1.5, f"{100 * m:.1f}%", ha="center", va="bottom", color=INK)
        ax.axhline(10, color=MUTED, linestyle=":", linewidth=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels([LABEL[c] for c in CONDS], fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{name}: {len(keys)} pairs")
    axes[0].set_ylabel("% draws naming Qwen's no-hint answer")
    axes[0].set_ylim(0, 60)
    axes[0].text(-0.45, 10.8, "chance", fontsize=8, color=MUTED)
    which = "items where Qwen followed the hint 8/8" if filtered else "all survivors"
    title = f"Answer task on hint_mcq: name Qwen's letter with the hint removed ({which}; Flash, 5 draws per pair)"
    fig.suptitle("Legacy selected cohort · " + title, x=0.01, ha="left")
    note = "95% item-level bootstrap. The wrong-hint items are a subset of the correct-hint items, so the pooled panel counts them twice."
    fig.text(0.01, -0.03, note, color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", bbox_inches="tight")


def per_item(by: dict, meta: dict, title: str, out: Path) -> None:
    keys = sorted(by["olens"])
    fig, ax = plt.subplots(figsize=(1.2 + 0.55 * len(keys), 3.2))
    for yi, c in enumerate(CONDS):
        for xi, k in enumerate(keys):
            hits, n = sum(by[c][k]), len(by[c][k])
            size = 70 + 220 * hits / max(n, 1)
            hatch = (
                bar_kw(c)["hatch"] if hits else None
            )  # hatching takes the edge colour, so a hatched marker gets a background-coloured edge
            ax.scatter(xi, yi, s=size, facecolor=COLOR[c] if hits else "none", edgecolors=BG if hatch else COLOR[c], linewidths=1.5,
                       zorder=3, hatch=hatch)  # fmt: skip
            if hits:
                ax.text(xi, yi, f"{hits}/{n}", ha="center", va="center", color="white" if c == "agentic" else INK, fontsize=7, zorder=4)
    ax.set_yticks(range(3))
    ax.set_yticklabels([LABEL[c] for c in CONDS])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k[1]}\n{meta[k][0]}/{meta[k][1]}" for k in keys], fontsize=7)
    ax.set_ylim(-0.7, 2.7)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Legacy · {title}: correct draws per item (own letter / hint letter)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("docs/img/hint-mcq-answer"))
    ap.add_argument("--no-filter", action="store_true", help="score every survivor, including items with a hint-ignoring base run")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rollouts = [json.loads(line) for line in ROLLOUTS.read_text().splitlines()]
    panels, metas = {}, {}
    for name, run in RUNS.items():
        panels[name], metas[name] = load(run, rollouts, not args.no_filter)
    pooled = {c: {} for c in CONDS}
    for by in panels.values():
        for c in CONDS:
            pooled[c].update(by[c])
    panels["both hint types"] = pooled
    bars(panels, args.out, not args.no_filter)
    gold, wrong = "hint is the correct answer", "hint is a wrong answer"
    per_item(panels[gold], metas[gold], "Correct-answer hint", args.out / "per_item_gold_hint.png")
    per_item(panels[wrong], metas[wrong], "Wrong-answer hint", args.out / "per_item_wrong_hint.png")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
