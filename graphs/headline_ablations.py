"""The write-up's ablation charts on graphs.headline_inline's runs (the canonical menus_inline menu, `prompt`): split by whether Qwen
is right, lens and placebo differences.

- headline-qwen-right.png / headline-qwen-wrong.png: % of cases the investigator (Haiku 5.5) gets right, transcript only vs
  O-lens, J++ and NLA, one panel per set with a notion of right: MMLU-Pro, SuperGPQA (Qwen's modal letter is the gold one) and
  knowledge cutoff (Qwen's answer is still true vs stale). Preferences have no truth, so they are left out.
- headline-lens-diffs.png: each lens minus its own placebo, pooled over the three tasks (MCQ, cutoff, preferences), three rows,
  paired CIs from graphs.headline_inline.Boot (Robbie, 2026-10-31: just those three); headline-placebo-diffs.png: each lens's
  placebo minus transcript only, per task and pooled.
- headline-essay-correctness.png: essay grading's AUROC per condition on the micro-essays Qwen grades "wrong" and the rest. Qwen is
  wrong on a micro-essay when it grades a top-half one (PERSUADE holistic score 4-6) below its own median grade over the 204, or a
  bottom-half one (1-3) at or above it; Qwen's grade is its mean on the plain base prompt. CIs resample micro-essays.
- headline-blog-correctness.png: blog sentence deletion's AUROC per condition, the qwen_wrong and qwen_right versions (Qwen's
  answer about the author vs the truth), CIs resampling bloggers. Every draw is one prediction.

    uv run python -m graphs.headline_ablations [--out-dir docs/img]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfbench.scoring import author_cases, grouped_auc_ci, grouped_mean_ci, standard_essay_comparison
from graphs import cutoff_qa_inline, menus_inline
from graphs.headline_inline import CANON, LENSES, PLACEBOS, POOLED, TITLES, TS, Boot, load
from graphs.style import MUTED, ci_bar, diff_rows, label, positions

FOUR = [(label(c), [c]) for c in (TS, *LENSES)]
SEVEN = [("transcript\nonly", [TS]), *((label(c), [c, PLACEBOS[c]]) for c in LENSES)]
ESSAY_INPUTS = Path("data/runs/cfbench_evals/2026-10-30_essay-grading-inline-inputs/cases.jsonl")
ESSAY_BANK = Path("data/banks/essay_grading/persuade_v1.json")


def by_correctness() -> dict[str, dict[str, dict]]:
    """set -> condition -> case -> (qwen_right, accuracy) on the canonical menus_inline runs and menu; the case is its own
    bootstrap group. MCQ: Qwen's modal letter is the gold one; cutoff: Qwen's answer is still true (gate_class stable_right,
    from the cutoff eval's cases)."""
    right: dict = defaultdict(dict)
    for s in ("mcq_unselected", "supergpqa"):
        for run in menus_inline.runs(s):
            for r in map(json.loads, (run / "predictions.jsonl").open()):
                right[s][r["item"]] = r["modal_cf"] == r["gold"]
    right["cutoff"] = {i: g == "stable_right" for i, (g, _) in cutoff_qa_inline.per_item(cutoff_qa_inline.RUNS)[TS].items()}
    menus = menus_inline.load()
    sets = {"MMLU-Pro": "mcq_unselected", "SuperGPQA": "supergpqa", "Knowledge cutoff": "cutoff"}
    menu_name = {c: m for m, c in CANON.items()}
    return {name: {c: {i: (right[s][i], a) for i, a in menus[s][menu_name[c]].items()} for c in (TS, *LENSES)} for name, s in sets.items()}


def accuracy_panels(sets: dict, right: bool, path: Path) -> None:
    fig, axes = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 3.4), sharey=True, layout="constrained", squeeze=False)
    for ax, (name, conds) in zip(axes[0], sets.items(), strict=True):
        xs, ticks = positions(conds, FOUR)
        n = 0
        for c, x in xs.items():
            vals = [a for ok, a in conds[c].values() if ok == right]
            m, lo, hi = grouped_mean_ci(vals, list(range(len(vals))))
            ci_bar(ax, x, c, 100 * m, 100 * lo, 100 * hi)
            n = len(vals)
        ax.set_xticks(ticks, [name for name, _ in FOUR], fontsize=8.5)
        ax.set_title(f"{name}, n = {n}", fontsize=9.5)
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("% of cases the investigator gets right")
    fig.suptitle(f"Cases where Qwen is {'right' if right else 'wrong'}", x=0.01, ha="left")
    fig.text(0.995, 0.97, "95% bootstrap CI over cases", ha="right", fontsize=8, color=MUTED)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def lens_diffs(boot: Boot, path: Path, placebo: bool = False) -> None:
    """Each lens (or its placebo) minus transcript only, per task and pooled."""
    rows: list = []
    for task in [*POOLED, "pooled"]:
        rows.append("pooled over the three tasks" if task == "pooled" else TITLES[task])
        tasks = POOLED if task == "pooled" else [task]
        conds = [PLACEBOS[lens] if placebo else lens for lens in LENSES]
        rows += [(c, TS, *boot.diff_ci(tasks, c, TS)) for c in conds]
    print(path.name, [(r[0][:5], f"{r[2]:+.1f} [{r[3]:+.1f}, {r[4]:+.1f}]") for r in rows if not isinstance(r, str)])
    fig, ax = plt.subplots(figsize=(6.2, 0.35 + 0.34 * len(rows)))
    ax.set_xlim((-32, 8) if placebo else (-6, 20))
    diff_rows(ax, rows)
    ax.set_xlabel("difference in % of cases right, paired by case (95% bootstrap CI; filled = excludes 0)", fontsize=8.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def lens_placebo_diffs(boot: Boot, path: Path) -> None:
    """Each lens minus its own placebo, pooled over the three tasks: three rows."""
    rows = [(lens, PLACEBOS[lens], *boot.diff_ci(POOLED, lens, PLACEBOS[lens])) for lens in LENSES]
    print(path.name, [(a[:5], f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]") for a, _, m, lo, hi in rows])
    fig, ax = plt.subplots(figsize=(6.2, 0.35 + 0.34 * len(rows)))
    ax.set_xlim(-6, 34)  # J++'s placebo hurts so much that J++ minus it is near +30
    diff_rows(ax, rows)
    ax.set_title("Each lens minus its placebo, pooled over the three tasks", loc="left", fontsize=9.5)
    ax.set_xlabel("difference in % of cases right, paired by case (95% bootstrap CI; filled = excludes 0)", fontsize=8.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def essay_qwen_wrong() -> dict[str, bool]:
    """essay -> Qwen grades it on the wrong side of the median."""
    bank = json.loads(ESSAY_BANK.read_text())
    human = {i["id"]: i["holistic_score"] for i in (bank["items"] if isinstance(bank, dict) else bank)}
    grade = {}
    for r in map(json.loads, ESSAY_INPUTS.open()):
        base = r["base"] if isinstance(r["base"], dict) else json.loads(r["base"])
        grade[r["item"]] = sum(int(k) * v for k, v in base.items()) / sum(base.values())
    median = float(np.median(list(grade.values())))
    return {i: (human[i] >= 4) != (grade[i] >= median) for i in grade}


def auroc_panels(groups: dict[str, dict], bars: list, path: Path, title: str, note: str) -> None:
    """groups: panel title -> condition -> (scores, truth, bootstrap group) over every draw."""
    fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 3.4), sharey=True, layout="constrained", squeeze=False)
    for ax, (name, conds) in zip(axes[0], groups.items(), strict=True):
        xs, ticks = positions(conds, bars)
        for c, x in xs.items():
            scores, truth, cluster = conds[c]
            ci_bar(ax, x, c, *grouped_auc_ci(scores, truth, cluster), fmt="{:.3f}")
        ax.set_xticks(ticks, [name for name, cs in bars if any(c in conds for c in cs)], fontsize=8.5)
        ax.set_title(name, fontsize=9.5)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--", zorder=4)
    axes[0][0].set_ylabel("AUROC")
    fig.suptitle(title, x=0.01, ha="left")
    fig.text(0.995, 0.97, f"hatched = placebo; {note}", ha="right", fontsize=8, color=MUTED)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def essay_groups() -> dict[str, dict]:
    scores, reference = standard_essay_comparison()
    wrong = essay_qwen_wrong()
    groups = {}
    for name, flag in (("Qwen grades the micro-essay wrong", True), ("Qwen grades the micro-essay right", False)):
        keys = [k for k in reference if wrong[k[1]] == flag]
        groups[f"{name}, {len({k[1] for k in keys})} of 204 micro-essays"] = {
            c: (
                [p for k in keys for p in scores[c][k]],
                [reference[k]["truth"] for k in keys for _ in scores[c][k]],
                [k[1] for k in keys for _ in scores[c][k]],
            )
            for c in (TS, *(c for lens in LENSES for c in (lens, PLACEBOS[lens])))
        }
    return groups


def blog_groups() -> dict[str, dict]:
    """The deck's blog cases (cfbench.scoring.author_cases) split by Qwen's answer about the author: version ->
    condition -> (scores, truth, blogger)."""
    _, y, version, blogger, xs = author_cases()
    groups = {}
    for v, name in (("qwen_wrong", "Qwen is wrong about the author"), ("qwen_right", "Qwen is right about the author")):
        m = version == v
        groups[f"{name}, {int(m.sum())} cases, {len(set(blogger[m]))} bloggers"] = {
            c: (x[m].tolist(), y[m].astype(int).tolist(), blogger[m].tolist()) for c, x in xs.items()
        }
    return groups


def main(out: Path) -> None:
    sets = by_correctness()
    for right in (True, False):
        accuracy_panels(sets, right, out / f"headline-qwen-{'right' if right else 'wrong'}.png")
        for name, conds in sets.items():
            means = {c: 100 * np.mean([a for ok, a in conds[c].values() if ok == right]) for c in conds}
            tag = "right" if right else "wrong"
            print(f"Qwen {tag} {name}:", "  ".join(f"{c.removesuffix('_inline')} {m:.1f}" for c, m in means.items()))
    boot = Boot(load())
    lens_placebo_diffs(boot, out / "headline-lens-diffs.png")
    lens_diffs(boot, out / "headline-placebo-diffs.png", placebo=True)
    auroc_panels(
        essay_groups(),
        SEVEN,
        out / "headline-essay-correctness.png",
        "Essay grading, by whether Qwen grades the micro-essay like the human did",
        "95% bootstrap CI over micro-essays",
    )
    auroc_panels(blog_groups(), [("transcript\nonly", [TS]), ("O-lens", ["olens_inline", "olens_placebo_inline"])],
                 out / "headline-blog-correctness.png", "Blog sentence deletion, by whether Qwen is right about the author",
                 "95% bootstrap CI over bloggers")  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("docs/img"))
    main(ap.parse_args().out_dir)
