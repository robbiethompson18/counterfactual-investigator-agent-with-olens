"""Blog sentence deletion (cfbench/evals/author_blog_drop_inline): AUROC of P(the deletion is significant) per condition, the
deck's bars (graphs.talk author()) at write-up size: transcript only, O-lens and NLA each beside its placebo, the standalone L2
probe and the nested probe + transcript combiner, computed by cfbench.scoring.author_bars (95% CIs resampling bloggers).

    uv run python -m graphs.author_blog_drop   # -> docs/img/author-blog-drop.png
"""

from pathlib import Path

import matplotlib.pyplot as plt

from cfbench.scoring import AUTHOR_GROUPS, author_bars, author_cases
from graphs.style import MUTED, ci_bar, departure, positions

DEPARTURE = ("O-lens at layers 32/40/48, NLA at layer 42 only; positions are the question's and post's punctuation and newlines, "
             "no tail tokens; Haiku at effort low, one draw per case; AUROC.")  # fmt: skip


def main() -> None:
    bars, _, n = author_bars()
    bloggers = len(set(author_cases()[3]))
    pooled = {c: bars["pooled", c] for _, cs in AUTHOR_GROUPS for c in cs}
    xs, ticks = positions(pooled, AUTHOR_GROUPS)
    fig, ax = plt.subplots(figsize=(6.4, 4.1), layout="constrained")
    for c, x in xs.items():
        v, (lo, hi), _ = pooled[c]
        ci_bar(ax, x, c, v, lo, hi, fmt="{:.3f}")
    ax.set_xticks(ticks, [g for g, _ in AUTHOR_GROUPS], fontsize=9)
    ax.set_xlim(-0.6, ticks[-1] + 0.6)
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("AUROC of P(significant)")
    ax.set_title("Blog sentence deletion: predicting a substantial drop in Qwen's confidence")
    ax.grid(axis="x", visible=False)
    fig.supxlabel(f"{n:,} cases · {bloggers} bloggers · 95% bootstrap CIs over bloggers · hatched = placebo", fontsize=8, color=MUTED)
    departure(fig, DEPARTURE)
    out = Path("docs/img/author-blog-drop.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, {c: round(v, 3) for c, (v, _, _) in pooled.items()})


if __name__ == "__main__":
    main()
