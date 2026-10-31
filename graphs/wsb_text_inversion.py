"""Two bar charts for the "text inversion vs the O-lens on WorkspaceBench" write-up (docs/wsb-flash-prompt-only.md): the four
contested families, one bar for the best published metamodel ("WSB SOTA") and one for Sonnet 5.5 reading the prompt text (TI).

    uv run python -m graphs.wsb_text_inversion [--out docs/img/wsb-text-inversion]

  <out>-benchmark.png  WSB's own scoring: the lens passes if ANY readout over its read grid passes (55 to ~230 per item);
                       Sonnet is one answer per item (pass@1, mean of 3-5 draws).
  <out>-pass1.png      both at pass@1: the lens's per-readout pass rate from the public per-cell rows; Sonnet unchanged.

Numbers are constants below, each with its source; they are not recomputed from runs. Lens headline = the post's bar chart
(Gemini 3.8 Flash judge; Sonnet 5 for jailbreak), best of Oracle Lens RL / SFT. Lens per-readout = agu18dec/olens_ablations
`rl.base.off_k0.own.i300` rows (Opus 5.5 judge), the only public per-cell data; jailbreak_recognition has none. Sonnet = our
wsb/ runs on data/runs/wsb_*/ (Opus 5.5 judges, their prompts).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from graphs.style import INK, MUTED, bar_kw

FAMILIES = ["user_modeling", "basic_readout", "moral_rationale", "jailbreak_recognition"]
LABEL = {
    "user_modeling": "User modeling\n(100 items)",
    "basic_readout": "Basic readout\n(100 items)",
    "moral_rationale": "Moral rationale\n(200 items)",
    "jailbreak_recognition": "Jailbreak recognition\n(86 items)",
}
# Post's bar chart, best of RL / SFT: user_modeling SFT 0.97, basic_readout RL 0.99, moral RL 0.76, jailbreak SFT 0.96
LENS_HEADLINE = {"user_modeling": 0.97, "basic_readout": 0.99, "moral_rationale": 0.76, "jailbreak_recognition": 0.96}
# Per-readout gold rate in rl.base.off_k0.own.i300: user_modeling inferred-basis gold, selfdescribe 0.203 (10,834 cells) and
# synthsys 0.105 (12,375), item-weighted 47/53; basic_readout 805/1100 layer cells expressed; moral 1,872/12,869 cells gold
LENS_PER_READOUT = {"user_modeling": 0.151, "basic_readout": 0.732, "moral_rationale": 0.145, "jailbreak_recognition": None}
# Sonnet 5.5 pass@1: user_modeling plain answers, selfdescribe 0.809 (47) and synthsys 0.214 (53) item-weighted
# (2026-10-28_sonnet-selfdescribe47, _sonnet-synthsys53; "shape" elicit lifts them to 0.844 / 0.478 -> 0.65);
# basic_readout 2026-10-28_sonnet-as-qwen judge pass@1 (x5); moral 2026-10-28_sonnet-open-model-x3 dropped-refusals pass@1;
# jailbreak 2026-10-28_sonnet-open-model-x3 dropped-refusals pass@1
TEXT_PASS1 = {"user_modeling": 0.494, "basic_readout": 0.85, "moral_rationale": 0.693, "jailbreak_recognition": 0.833}
TEXT_NOTE = {"user_modeling": "0.65 with an\nelicit sentence"}


def chart(lens: dict, title: str, lens_label: str, out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = range(len(FAMILIES))
    w = 0.36
    for i, fam in enumerate(FAMILIES):
        v = lens[fam]
        if v is None:
            ax.text(i - w / 2, 0.03, "no public\nper-readout\nrows", ha="center", va="bottom", fontsize=8, color=MUTED)
        else:
            ax.bar(i - w / 2, v, w, label=lens_label if i == 0 else None, **bar_kw("olens"))
            ax.text(i - w / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
        t = TEXT_PASS1[fam]
        ax.bar(i + w / 2, t, w, label="Sonnet 5.5 text inversion, pass@1" if i == 0 else None, **bar_kw("transcript"))
        ax.text(i + w / 2, t + 0.015, f"{t:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
        if fam in TEXT_NOTE:
            ax.text(i + w / 2, t + 0.07, TEXT_NOTE[fam], ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(list(x), [LABEL[f] for f in FAMILIES])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("pass rate")
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, -0.14), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/img/wsb-text-inversion")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chart(
        LENS_HEADLINE,
        "WorkspaceBench's own scoring: the lens passes if any readout over its grid passes (55 to ~230 per item)",
        "O-lens, WSB headline (any-of-grid)",
        out.with_name(out.name + "-benchmark.png"),
    )
    chart(
        LENS_PER_READOUT,
        "Both at pass@1: the lens scored per readout (one cell), Sonnet one answer per item",
        "O-lens, per readout",
        out.with_name(out.name + "-pass1.png"),
    )


if __name__ == "__main__":
    main()
