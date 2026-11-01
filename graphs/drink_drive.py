"""drink_drive: Qwen is a bad friend. The WeirdChat designated-driver prompt (Transluce t053) with "a couple of drinks" replaced by
one to five drinks and nothing else changed; the share of 50 Qwen3.6-27B replies (thinking off, T 1) per arm that tell the user to
drive, with or without a caveat (Haiku 5.5 judge), Wilson 95% CIs. One series, so ink. The deck draws the same bars under @slide
(graphs.talk.drink_drive).

    uv run python -m graphs.drink_drive [--out docs/img/drink-drive.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download

from cfbench.catalog import DRINK_RUN
from cfbench.scoring import HF_REPO
from graphs.cutoff_grader_mentions import wilson
from graphs.style import INK, MUTED

ARMS = {"one": "1", "two": "2", "couple": "a couple\n(original)", "three": "3", "four": "4", "five": "5"}


def rates() -> dict[str, tuple[float, float, float]]:
    """arm -> (share of replies labelled drive or mixed, Wilson 95% CI)."""
    path = Path("data") / DRINK_RUN / "labels/judge@haiku55-v1.jsonl"
    if not path.exists():
        hf_hub_download(HF_REPO, f"{DRINK_RUN}/labels/judge@haiku55-v1.jsonl", repo_type="dataset", local_dir="data")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    out = {}
    for arm in ARMS:
        ys = [r["values"]["drive_or_mixed"] for r in rows if r["episode"].split("~")[0] == arm]
        out[arm] = (sum(ys) / len(ys), *wilson(sum(ys), len(ys)))
    return out


def main(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.6), layout="constrained")
    for x, (arm, (v, lo, hi)) in enumerate(rates().items()):
        ax.bar(x, v, 0.8, color=INK, zorder=3)
        ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9, zorder=4)
        ax.annotate(f"{v:.0%}", (x, hi), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", color=INK)
        print(f"{arm}: {v:.0%} [{lo:.0%}, {hi:.0%}]")
    ax.set_xticks(range(len(ARMS)), list(ARMS.values()))
    ax.set_xlabel("drinks the user says they had")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.25), [f"{t:.0%}" for t in np.arange(0, 1.01, 0.25)])
    ax.set_ylabel("tells the user to drive")
    ax.grid(axis="x", visible=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/img/drink-drive.png"))
    main(ap.parse_args().out)
