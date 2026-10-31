"""The write-up's Results headline: the three non-counterfactual tasks (MCQ, knowledge cutoff, idiosyncratic preferences), % of cases
the investigator (Haiku 5.5) gets right, transcript-only vs O-lens, J++ and NLA with their placebos, and the standard probe on MCQ.
Essay grading, a counterfactual, gets its own charts with its probe and stays out of the pooled numbers. MCQ, cutoff and preferences
come only from the canonical menus_inline runs, `all` menu (graphs.menus_inline SETS: one prompt, the same read positions and
placebo rule on every set, up to 3 draws); MCQ pools MMLU-Pro and SuperGPQA cases; an essay case is right when
p >= 0.5 matches whether the grade goes up, or the probe's predicted shift > 0 does. Pooled numbers weight the three tasks equally. CIs
are 95% bootstraps resampling questions (essays for essay grading) within each task, paired across conditions.

    uv run python -m graphs.headline_inline [--out-dir docs/img]

Writes headline-metamodels, headline-placebos, headline-probes, headline-probes-error, headline-<task> (y axis from 50) and
headline-essay-diff.png (paired differences in % right and AUROC, the latter from cfbench.scoring.essay_auroc_bars), and prints every
paired difference the write-up quotes.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfbench.scoring import essay_auroc_bars, standard_essay_comparison, standard_probe_rows
from graphs import menus_inline
from graphs.style import MUTED, ci_bar, diff_rows, label, positions

LENSES = ["olens_inline", "jlens_inline", "nla_inline"]
PLACEBOS = {c: c.replace("_inline", "_placebo_inline") for c in LENSES}
TS, PROBE = "transcript_inline", "probe_response_start"
SEVEN = [TS, *(c for lens in LENSES for c in (lens, PLACEBOS[lens]))]
TITLES = {"mcq": "Multiple choice", "cutoff": "Knowledge cutoff", "preferences": "Idiosyncratic preferences", "essay": "Essay grading"}
POOLED = ["mcq", "cutoff", "preferences"]
# task -> its menus_inline sets; the `all` menu's conditions under the names the charts use
MENU_SETS = {"mcq": ["mcq_unselected", "supergpqa"], "cutoff": ["cutoff"], "preferences": ["favorites"]}
CANON = {TS: TS, **{c.replace("_", "_all_", 1): c for c in SEVEN[1:]}}  # olens_all_placebo_inline -> olens_placebo_inline
YLABEL = "% of cases the investigator gets right"
N_BOOT = 4000


def load() -> dict[str, tuple[dict, dict]]:
    """task -> (condition -> case -> accuracy (share of its draws right), case -> bootstrap group). MCQ, cutoff and preferences
    come only from the canonical menus_inline runs, `all` menu (CLAUDE.local.md); MCQ adds the standardized probe, and its
    bootstrap resamples the probe's question groups."""
    menus = menus_inline.load()
    out: dict[str, tuple[dict, dict]] = {}
    for task, sets in MENU_SETS.items():
        acc = {CANON[c]: {(s, i): v for s in sets for i, v in menus[s][c].items()} for c in CANON}
        out[task] = acc, {k: k for k in acc[TS]}
    acc, groups = out["mcq"]
    acc[PROBE] = {}
    for s, target in (("mcq_unselected", "mmlu"), ("supergpqa", "supergpqa")):
        for row in standard_probe_rows(target, "response_start", target):
            if (s, row["item"]) in acc[TS]:
                acc[PROBE][s, row["item"]] = float(row["predicted_label"] == row["label"])
                groups[s, row["item"]] = (s, row["group"])
    assert acc[PROBE].keys() == acc[TS].keys(), "the probe must cover every MCQ case"
    scores, reference = standard_essay_comparison()

    def right(c: str, k: tuple, p: float) -> bool:
        return (p >= 0.5 if c.endswith("_inline") else p > 0) == bool(reference[k]["truth"])

    acc = {c: {k: float(np.mean([right(c, k, p) for p in ps])) for k, ps in cases.items()} for c, cases in scores.items()}
    out["essay"] = acc, {k: k[1] for k in reference}
    return out


class Boot:
    """Paired bootstrap: the same resampled groups for every condition of a task."""

    def __init__(self, tasks: dict):
        self.tasks, self.draws = tasks, {}
        rng = np.random.default_rng(0)
        for task, (acc, groups) in tasks.items():
            ids = sorted({str(g) for g in groups.values()})
            self.draws[task] = (ids, rng.multinomial(len(ids), [1 / len(ids)] * len(ids), size=N_BOOT))

    def samples(self, task: str, c: str) -> tuple[float, np.ndarray]:
        """Condition c's accuracy (%) and its bootstrap draws."""
        acc, groups = self.tasks[task]
        ids, counts = self.draws[task]
        where = {g: i for i, g in enumerate(ids)}
        total, n = np.zeros(len(ids)), np.zeros(len(ids))
        for k, v in acc[c].items():
            i = where[str(groups[k])]
            total[i] += v
            n[i] += 1
        return 100 * total.sum() / n.sum(), 100 * (counts @ total) / (counts @ n)

    def stat(self, tasks: list[str], c: str) -> tuple[float, np.ndarray]:
        """The mean over tasks, each weighted equally."""
        pairs = [self.samples(t, c) for t in tasks]
        return float(np.mean([m for m, _ in pairs])), np.mean([b for _, b in pairs], axis=0)

    def bar(self, tasks: list[str], c: str) -> tuple[float, float, float]:
        m, b = self.stat(tasks, c)
        lo, hi = np.percentile(b, [2.5, 97.5])
        return m, float(lo), float(hi)

    def diff_ci(self, tasks: list[str], a: str, b: str) -> tuple[float, float, float]:
        ma, ba = self.stat(tasks, a)
        mb, bb = self.stat(tasks, b)
        lo, hi = np.percentile(ba - bb, [2.5, 97.5])
        return ma - mb, float(lo), float(hi)

    def diff(self, tasks: list[str], a: str, b: str) -> str:
        m, lo, hi = self.diff_ci(tasks, a, b)
        return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def bars(ax, boot: Boot, tasks: list[str], groups: list, error: bool = False) -> None:
    conditions = {c: boot.bar(tasks, c) for _, cs in groups for c in cs}
    xs, ticks = positions(conditions, groups)
    for c, (m, lo, hi) in conditions.items():
        if error:
            m, lo, hi = 100 - m, 100 - hi, 100 - lo
        ci_bar(ax, xs[c], c, m, lo, hi)
    ax.set_xticks(ticks, [name for name, _ in groups], fontsize=8.5)
    ax.grid(axis="x", visible=False)


def figure(path: Path, panels: list, ylabel: str = YLABEL, ylim: float = 100, floor: float = 0) -> None:
    """panels: (title, draw(ax)). `floor`: where the y axis starts (the per-task charts start at 50)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 3.4), sharey=True, layout="constrained", squeeze=False)
    for ax, (title, draw) in zip(axes[0], panels, strict=True):
        draw(ax)
        ax.set_title(title)
        ax.set_ylim(floor, ylim)
    axes[0][0].set_ylabel(ylabel)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def essay_diffs(boot: Boot, path: Path) -> None:
    """Essay grading's paired differences: each lens vs transcript-only and vs its placebo, and the probe vs transcript-only."""
    pairs = [(lens, TS) for lens in LENSES] + [(lens, PLACEBOS[lens]) for lens in LENSES] + [(PROBE, TS)]
    auroc = essay_auroc_bars()[3]
    print(
        "essay AUROC:",
        {f"{a[:5]}-{b[:5]}": (round(100 * auroc[a, b][0], 1), [round(100 * x, 1) for x in auroc[a, b][1]]) for a, b in pairs},
    )
    right = [(a, b, *boot.diff_ci(["essay"], a, b)) for a, b in pairs]
    points = [(a, b, 100 * auroc[a, b][0], 100 * auroc[a, b][1][0], 100 * auroc[a, b][1][1]) for a, b in pairs]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.6), layout="constrained")
    for ax, rows, title in ((axes[0], right, "difference in % of cases right"), (axes[1], points, "difference in AUROC x 100")):
        diff_rows(ax, rows)
        ax.set_title(title)
        ax.set_xlim(-6, 26)
    fig.supxlabel("Essay grading, paired by essay (95% bootstrap CI; filled = excludes 0)", fontsize=8.5, color=MUTED)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main(out: Path) -> None:
    tasks = load()
    boot = Boot(tasks)
    for task in [*TITLES, "pooled"]:
        ts = POOLED if task == "pooled" else [task]
        print(f"== {task}: " + "  ".join(f"{c.removesuffix('_inline')} {boot.bar(ts, c)[0]:.1f}" for c in SEVEN))
        for lens in LENSES:
            placebo = PLACEBOS[lens]
            print(f"   {lens[:5]}: vs transcript {boot.diff(ts, lens, TS)}  vs placebo {boot.diff(ts, lens, placebo)}", end="")
            print(f"  placebo vs transcript {boot.diff(ts, placebo, TS)}")
    print(
        "between lenses:",
        {f"{a[:5]}-{b[:5]}": boot.diff(POOLED, a, b) for a, b in ((LENSES[0], LENSES[1]), (LENSES[0], LENSES[2]), (LENSES[1], LENSES[2]))},
    )
    for task in ("mcq", "essay"):
        print(
            f"{task} probe {boot.bar([task], PROBE)[0]:.1f}: " + "  ".join(f"probe-{c[:5]} {boot.diff([task], PROBE, c)}" for c in LENSES)
        )

    lens_groups = [(label(c), [c]) for c in LENSES]
    pair_groups = [(label(c), [c, PLACEBOS[c]]) for c in LENSES]
    seven = [("transcript\nonly", [TS]), *pair_groups]
    probe_groups = [*lens_groups, ("probe", [PROBE])]
    pooled = {"pooled over the three tasks": POOLED}
    figure(
        out / "headline-metamodels.png",
        [(t, lambda ax, ts=ts: bars(ax, boot, ts, [("transcript\nonly", [TS]), *lens_groups])) for t, ts in pooled.items()],
    )
    figure(out / "headline-placebos.png", [(t, lambda ax, ts=ts: bars(ax, boot, ts, pair_groups)) for t, ts in pooled.items()])
    probes = [(TITLES["mcq"], ["mcq"])]
    figure(out / "headline-probes.png", [(t, lambda ax, ts=ts: bars(ax, boot, ts, probe_groups)) for t, ts in probes])
    figure(out / "headline-probes-error.png", [(t, lambda ax, ts=ts: bars(ax, boot, ts, probe_groups, error=True)) for t, ts in probes],
           ylabel="% of cases the investigator gets wrong", ylim=60)  # fmt: skip
    for task, title in TITLES.items():
        groups = [*seven, ("probe", [PROBE])] if task == "essay" else seven
        figure(out / f"headline-{task}.png", [(title, lambda ax, t=task, g=groups: bars(ax, boot, [t], g))], floor=50)
    essay_diffs(boot, out / "headline-essay-diff.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("docs/img"))
    main(ap.parse_args().out_dir)
