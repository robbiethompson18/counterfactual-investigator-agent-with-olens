"""essay_intro pilot summary (2026-10-29): the blind plain screen and the motive arms, from the runs' measure labels.

PYTHONPATH=. uv run python scripts/essay_intro_pilot.py data/envs/essay_intro/<blind run> data/envs/essay_intro/<motive run>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from environments.essay_intro.env import bank


def scores(run: Path) -> dict[tuple[str, str], list[float | None]]:
    """(arm, item) -> scores, from labels/measure@*.jsonl (episode ids are arm~item~sample)."""
    out = defaultdict(list)
    for r in map(json.loads, next((run / "labels").glob("measure@*.jsonl")).open()):
        arm, item, _ = r["episode"].split("~")
        out[arm, item].append(r["values"]["score"])
    return out


def main(blind: Path, motive: Path):
    b, h = scores(blind), bank()
    rows = [(item, h[item]["human_score"], [s for s in v if s is not None], len(v)) for (_, item), v in sorted(b.items())]
    means = np.array([np.mean(s) for _, _, s, _ in rows if s])
    human = np.array([hs for _, hs, s, _ in rows if s])
    n_eps = sum(n for *_, n in rows)
    print(f"blind screen: {len(rows)} paragraphs, {n_eps} episodes, no Score line {sum(n - len(s) for _, _, s, n in rows)}")
    print("histogram of per-paragraph mean score (rounded):", {k: int((np.round(means) == k).sum()) for k in range(1, 11)})
    r = np.corrcoef(means, human)[0, 1]
    print(f"mean {means.mean():.2f}, sd across paragraphs {means.std():.2f}, Pearson r with human score {r:.2f}")
    print("mean blind score by human score:", {hs: round(float(means[human == hs].mean()), 2) for hs in sorted(set(human))})
    sds = [np.std(s) for _, _, s, _ in rows if len(s) > 1]
    print(f"within-paragraph SD: median {np.median(sds):.2f}, mean {np.mean(sds):.2f}, max {np.max(sds):.2f}")
    print("per-paragraph mean scores:", sorted(round(float(m), 1) for m in means))
    for t in range(4, 9):  # fraction of a paragraph's runs at or above t, per paragraph: how evenly they spread over 0-1
        rates = np.array([np.mean([x >= t for x in s]) for _, _, s, _ in rows if s])
        print(
            f"  plain rate of score >= {t}: per-paragraph rates histogram (0-.2,.2-.4,...,.8-1) "
            f"{np.histogram(rates, bins=5, range=(0, 1))[0].tolist()}, mean {rates.mean():.2f}"
        )

    m = scores(motive)
    items = sorted({i for _, i in m})
    print(f"\nmotive run: {len(items)} paragraphs, arms {sorted({a for a, _ in m})}")
    mean = lambda a, i: np.mean([s for s in m[a, i] if s is not None])
    for arm in sorted({a for a, _ in m}):
        allv = [s for i in items for s in m[arm, i]]
        got = [s for s in allv if s is not None]
        hi, lo = np.mean([s >= 7 for s in got]), np.mean([s <= 5 for s in got])
        line = f"  {arm:10} n={len(allv)} none={len(allv) - len(got)} mean {np.mean(got):.2f}  >=7 {hi:.2f}  <=5 {lo:.2f}"
        if arm != "plain":
            d = np.array([mean(arm, i) - mean("plain", i) for i in items])
            moved = (np.abs(d) >= 1).sum()
            line += f"  shift vs plain {d.mean():+.2f} (per paragraph {np.round(d, 1).tolist()}), |shift|>=1 in {moved}/{len(items)}"
        print(line)
    blind_mean = lambda i: np.mean([s for s in b["plain", i] if s is not None])
    pairs = [(i[:6], round(float(mean("plain", i)), 1), round(float(blind_mean(i)), 1)) for i in items]
    print("  per paragraph plain mean (this run) vs blind mean:", pairs)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
