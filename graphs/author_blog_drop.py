"""Blog sentence deletion (cfbench/evals/author_blog_drop_inline): AUROC of P(the deletion is significant) per condition. main()
is the write-up's chart: transcript only, O-lens and NLA each beside its placebo, the standalone L2
probe and the nested probe + transcript combiner, computed by cfbench.scoring.author_bars (95% CIs resampling bloggers).
slide() is the deck's (graphs.talk author(), Kylie 2026-11-01): all cases with Qwen right and wrong weighted 50:50 beside the
Qwen-wrong cases, no probe + transcript bar.

    uv run python -m graphs.author_blog_drop   # -> docs/img/author-blog-drop.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfbench.scoring import AUTHOR_GROUPS, author_bars, author_cases
from graphs.style import INK, MUTED, OLENS, SLIDE_S, bar_kw, ci_bar, departure, positions, slide_figure, slide_save

DEPARTURE = (f"{OLENS} at layers 32/40/48, NLA at layer 42 only; positions are the question's and post's punctuation and newlines, "
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
    ax.set_xticks(ticks, [OLENS if g == "olens" else g for g, _ in AUTHOR_GROUPS], fontsize=9)
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


def slide(out: Path) -> None:
    """AUROC of P(the deletion is significant) per condition (Kylie, 2026-11-01): all cases with Qwen right and wrong
    weighted 50:50, and the Qwen-wrong cases; no probe + transcript bar. Also the paired differences table
    (fig/author_pairs.tex, not on the slide)."""
    titles = {"pooled": "all cases (Qwen right : wrong weighted 50:50)", "qwen_wrong": "Qwen wrong"}
    bars, pairs, n = author_bars(versions=tuple(titles), balanced=True)
    groups = [(g, cs) for g, cs in AUTHOR_GROUPS if "transcript+probe" not in cs]
    one = {c: 0 for _, cs in groups for c in cs}
    width = positions(one, groups)[1][-1] + 1.2
    fig, axes = slide_figure(130, 58, ncols=len(titles), sharey=True)  # the deck's full figure height (graphs.style FULL_H)
    base = 0.5
    top = max(hi for (_, (_, hi), _) in bars.values()) + 0.06
    for ax, (ver, title) in zip(axes, titles.items(), strict=True):
        xs, ticks = positions(one, groups)
        for c, x in xs.items():
            val, (lo, hi), st = bars[ver, c]
            ax.bar(x, val - base, 0.8, bottom=base, **bar_kw(c))
            ax.errorbar(x, val, [[val - lo], [hi - val]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
            for text, dy in ((f"{val:.3f}", 1.5), (st, 6.5)):
                ax.annotate(
                    text, (x, hi), xytext=(0, dy), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * SLIDE_S, color=INK
                )
        ax.set_xticks(ticks, [OLENS if g == "olens" else g for g, _ in groups])
        ax.set_xlim(-0.6, width - 0.6)
        ax.tick_params(axis="x", labelsize=7 * SLIDE_S)
        ax.set_title(title, fontsize=7 * SLIDE_S, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    yt = np.arange(base, top, 0.1)
    axes[0].set_ylim(base, top)
    axes[0].set_yticks(yt, [f"{t:.1f}" for t in yt])
    axes[0].set_ylabel("AUROC of P(significant)")
    axes[-1].text(1.0, 1.12, "hatched = placebo", transform=axes[-1].transAxes, fontsize=6.5 * SLIDE_S, color=MUTED, ha="right")
    departure(fig, "Haiku at effort low, one draw per case; AUROC.")  # the slide leaves out the read settings (Kylie, 2026-11-01)
    slide_save(fig, out / "author.pdf")
    name = {"transcript_inline": r"\sw{transcript}\ transcript", "olens_inline": r"\sw{olens}\ \olensname",
            "olens_placebo_inline": r"\swp{olens}\ \olensname{} placebo", "nla_inline": r"\sw{nla}\ NLA",
            "nla_placebo_inline": r"\swp{nla}\ NLA placebo", "transcript+probe": r"\sw{probe}\ probe + transcript"}  # fmt: skip
    pf = lambda p: "< .001" if p < 1e-3 else f"{p:.3f}".lstrip("0")
    rows = [
        f"{name[a]} − {name[c].split(' ', 1)[1]} & {d:+.3f} & [{lo:+.3f}, {hi:+.3f}] & {pf(p)}\\\\"
        for (a, c), (d, (lo, hi), p) in pairs.items()
    ]
    head = [r" & $\Delta$ AUROC & 95\% CI & p\\"]
    (out / "author_pairs.tex").write_text("\\begin{tabular}{@{}lrrr@{}}\n" + "\n".join([*head, "\\hline", *rows]) + "\n\\end{tabular}\n")
    (out / "author_n.tex").write_text(f"{n:,}")
    _, _, v, b, _ = author_cases()
    (out / "author_bloggers.tex").write_text(f"{len(set(b)):,}")
    (out / "author_wrong_n.tex").write_text(f"{(v == 'qwen_wrong').sum():,}")


if __name__ == "__main__":
    main()
