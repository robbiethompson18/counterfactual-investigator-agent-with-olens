"""The write-up's Results headline: the four single-turn tasks (MCQ, knowledge cutoff, idiosyncratic preferences, essay grading),
% of cases the investigator (Haiku 5.5) gets right, transcript-only vs O-lens, J++ and NLA with their placebos, and the standard
probes where they exist. Reuses each eval's own runs and scoring: MCQ pools MMLU-Pro and SuperGPQA cases; an essay case is right when
p >= 0.5 matches whether the grade goes up, or the probe's predicted shift > 0 does. Pooled numbers weight the four tasks equally. CIs
are 95% bootstraps resampling questions (essays for essay grading) within each task, paired across conditions.

    uv run python -m graphs.headline_inline [--out-dir docs/img]

Writes headline-metamodels, headline-placebos, headline-probes, headline-probes-error and headline-<task>.png, and prints every
paired difference the write-up quotes.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfbench.scoring import standard_essay_comparison, standard_mcq_comparison
from graphs import cutoff_qa_inline, favorites_qa_inline, mcq_unselected, supergpqa_inline
from graphs.style import ci_bar, label, positions

LENSES = ["olens_inline", "jlens_inline", "nla_inline"]
PLACEBOS = {c: c.replace("_inline", "_placebo_inline") for c in LENSES}
TS, PROBE = "transcript_inline", "probe_response_start"
SEVEN = [TS, *(c for lens in LENSES for c in (lens, PLACEBOS[lens]))]
TITLES = {"mcq": "Multiple choice", "cutoff": "Knowledge cutoff", "preferences": "Idiosyncratic preferences", "essay": "Essay grading"}
YLABEL = "% of cases the investigator gets right"
N_BOOT = 4000


def load() -> dict[str, tuple[dict, dict]]:
    """task -> (condition -> case -> accuracy, case -> bootstrap group)."""
    out: dict[str, tuple[dict, dict]] = {}
    acc: dict = {}
    groups: dict = {}
    for target, runs in (("mmlu", mcq_unselected.RUNS), ("supergpqa", supergpqa_inline.RUNS)):
        data, clusters = standard_mcq_comparison(runs, target)
        for c, items in data.items():
            acc.setdefault(c, {}).update({(target, i): v for i, (_, v) in items.items()})
        groups |= {(target, i): (target, g) for i, g in clusters.items()}
    out["mcq"] = acc, groups
    for task, runs in (("cutoff", cutoff_qa_inline.RUNS), ("preferences", favorites_qa_inline.RUNS)):
        data = {c: {i: v for i, (_, v) in items.items()} for c, items in cutoff_qa_inline.per_item(runs).items()}
        out[task] = data, {i: i for i in data[TS]}
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

    def diff(self, tasks: list[str], a: str, b: str) -> str:
        ma, ba = self.stat(tasks, a)
        mb, bb = self.stat(tasks, b)
        lo, hi = np.percentile(ba - bb, [2.5, 97.5])
        return f"{ma - mb:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def bars(ax, boot: Boot, tasks: list[str], groups: list, error: bool = False) -> None:
    conditions = {c: boot.bar(tasks, c) for _, cs in groups for c in cs}
    xs, ticks = positions(conditions, groups)
    for c, (m, lo, hi) in conditions.items():
        if error:
            m, lo, hi = 100 - m, 100 - hi, 100 - lo
        ci_bar(ax, xs[c], c, m, lo, hi)
    ax.set_xticks(ticks, [name for name, _ in groups], fontsize=8.5)
    ax.grid(axis="x", visible=False)


def figure(path: Path, panels: list, ylabel: str = YLABEL, ylim: float = 100) -> None:
    """panels: (title, draw(ax))."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 3.4), sharey=True, layout="constrained", squeeze=False)
    for ax, (title, draw) in zip(axes[0], panels, strict=True):
        draw(ax)
        ax.set_title(title)
        ax.set_ylim(0, ylim)
    axes[0][0].set_ylabel(ylabel)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main(out: Path) -> None:
    tasks = load()
    boot = Boot(tasks)
    names = list(TITLES)
    for task in [*names, "pooled"]:
        ts = names if task == "pooled" else [task]
        print(f"== {task}: " + "  ".join(f"{c.removesuffix('_inline')} {boot.bar(ts, c)[0]:.1f}" for c in SEVEN))
        for lens in LENSES:
            placebo = PLACEBOS[lens]
            print(f"   {lens[:5]}: vs transcript {boot.diff(ts, lens, TS)}  vs placebo {boot.diff(ts, lens, placebo)}", end="")
            print(f"  placebo vs transcript {boot.diff(ts, placebo, TS)}")
    print(
        "between lenses:",
        {f"{a[:5]}-{b[:5]}": boot.diff(names, a, b) for a, b in ((LENSES[0], LENSES[1]), (LENSES[0], LENSES[2]), (LENSES[1], LENSES[2]))},
    )
    for task in ("mcq", "essay"):
        print(
            f"{task} probe {boot.bar([task], PROBE)[0]:.1f}: " + "  ".join(f"probe-{c[:5]} {boot.diff([task], PROBE, c)}" for c in LENSES)
        )

    lens_groups = [(label(c), [c]) for c in LENSES]
    pair_groups = [(label(c), [c, PLACEBOS[c]]) for c in LENSES]
    seven = [("transcript\nonly", [TS]), *pair_groups]
    probe_groups = [*lens_groups, ("probe", [PROBE])]
    four = {"pooled over the four tasks": names}
    figure(
        out / "headline-metamodels.png",
        [(t, lambda ax, ts=ts: bars(ax, boot, ts, [("transcript\nonly", [TS]), *lens_groups])) for t, ts in four.items()],
    )
    figure(out / "headline-placebos.png", [(t, lambda ax, ts=ts: bars(ax, boot, ts, pair_groups)) for t, ts in four.items()])
    probes = [(TITLES[t], [t]) for t in ("mcq", "essay")]
    figure(out / "headline-probes.png", [(t, lambda ax, ts=ts: bars(ax, boot, ts, probe_groups)) for t, ts in probes])
    figure(out / "headline-probes-error.png", [(t, lambda ax, ts=ts: bars(ax, boot, ts, probe_groups, error=True)) for t, ts in probes],
           ylabel="% of cases the investigator gets wrong", ylim=60)  # fmt: skip
    for task, title in TITLES.items():
        figure(out / f"headline-{task}.png", [(title, lambda ax, t=task: bars(ax, boot, [t], seven))])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("docs/img"))
    main(ap.parse_args().out_dir)
