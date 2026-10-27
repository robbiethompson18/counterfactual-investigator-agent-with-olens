# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Donation Bet goodbad with thinking off (empty think block prefilled) vs the thinking-on runs, Qwen3.6-27B and GLM 5.3.

    uv run --script evalcue/envs/donation_bet/nothink_compare.py [OUT_DIR]      (default docs/figures/donation_bet/nothink)

Thinking-off runs (2026-10-25, thinking-on sampling so only the think block differs):
  qwen-nothink-cap8k     Qwen bf16 vLLM, the chat template's enable_thinking=False ("<think>\\n\\n</think>\\n\\n"), Qwen lines
  glm53-nothink-ownline  GLM 5.3 via OpenRouter raw completions, "<|assistant|><think></think>" prefilled, GLM's own lines
Same estimand, CIs and judge (DeepSeek V4.1 Flash, judge.py) as plot_goodbad.py. Writes per-model bar figures for the off runs,
onoff_gaps.png (gap per comparison, on vs off) and onoff_table.md / .json.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import plot_goodbad as pg  # ty: ignore[unresolved-import]

matplotlib.use("Agg")
import matplotlib.pyplot as plt

pg.MODELS["qwen"]["title"] = "Qwen3.6-27B thinking on, own lines"
pg.MODELS["glm"]["title"] = "GLM 5.3 thinking on (medium), own lines"
pg.MODELS["qwen_off"] = {
    "title": "Qwen3.6-27B thinking off, empty think block",
    "runs": ["2026-10-25_qwen-nothink-cap8k"],
    "lines": "thresholds_goodbad.json",
    "line_note": "Threshold = Qwen's thinking-on median of 100 neutral samples per question",
}
pg.MODELS["glm_off"] = {
    "title": "GLM 5.3 thinking off, empty think block",
    "runs": ["2026-10-25_glm53-nothink-ownline"],
    "lines": "thresholds_goodbad_glm53.json",
    "line_note": "Threshold = GLM's thinking-on median of 20 neutral samples per question (regex estimates)",
}
PAIRS = [("qwen", "qwen_off", "Qwen3.6-27B"), ("glm", "glm_off", "GLM 5.3")]
C = {"on": "#6b4fa0", "off": "#d68535"}


def onoff_fig(rows, path):
    by = {(r["model"], r["comp"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (on, off, name) in zip(axes, PAIRS):
        x = np.arange(len(pg.COMPS))
        for k, (m, lab, dx) in enumerate([(on, "thinking on", -0.18), (off, "thinking off (empty think)", 0.18)]):
            rs = [by[m, c[0]] for c in pg.COMPS]
            g = np.array([r["gap"] for r in rs])
            lo = g - np.array([r["strat"][0] for r in rs])
            hi = np.array([r["strat"][1] for r in rs]) - g
            col = C["on" if k == 0 else "off"]
            ax.bar(x + dx, g, 0.34, color=col, label=lab, yerr=[lo, hi], capsize=3, error_kw={"lw": 1})
            for xi, gi, h, l in zip(x + dx, g, hi, lo):
                ax.text(xi, gi + h + 1 if gi >= 0 else gi - l - 1, f"{gi:+.0f}", ha="center", va="bottom" if gi >= 0 else "top", fontsize=8)
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_xticks(x, ["good vs bad", "good vs beers", "good vs nothing"])
        ax.set_title(name)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("leakage gap, pp")
    axes[0].set_ylim(-10, 56)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="lower center", ncol=2)
    fig.suptitle(
        "Donation Bet gap = P(≤ line | good cause below) − P(≤ line | good cause above), thinking on vs an empty think block\n"
        "judge estimates, mean over 10 questions, 95% CI resampling within question",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else pg.REPO / "docs/figures/donation_bet/nothink"
    out.mkdir(parents=True, exist_ok=True)
    data = {m: pg.load(m) for m in ["qwen", "qwen_off", "glm", "glm_off"]}
    for m in ["qwen_off", "glm_off"]:
        pg.goodbad_fig(m, data[m], out / f"{m}_goodbad_fig.png")
        pg.three_panels(m, data[m], out / f"{m}_three_panels.png")
    rows = pg.table(data)
    onoff_fig(rows, out / "onoff_gaps.png")
    t = pg.fmt_table(rows)
    (out / "onoff_table.md").write_text(t + "\n")
    (out / "onoff_table.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")
    print(t)


if __name__ == "__main__":
    main()
