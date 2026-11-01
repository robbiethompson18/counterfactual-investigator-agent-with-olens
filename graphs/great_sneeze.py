"""great_sneeze: CHIVE's Qwen3.5-397B incident prompt_003654 (adamkarvonen/chive-data), where Qwen invents "The Great Sneeze" in a
made-up country's history: the share of 30 samples with a sneeze-based event in Vardania's history (CHIVE's grader), the original
conversation on top in ink, then each of CHIVE's edits by rate, Wilson 95% CIs. Shown in docs/final-writeup.md's appendix; the deck
draws the same bars under @slide (graphs.talk.sneeze). Not a condition, so a colour outside the conditions'.

    uv run python -m graphs.great_sneeze [--out docs/img/great-sneeze.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download

from cfbench.catalog import SNEEZE_RUN
from cfbench.scoring import HF_REPO
from graphs.cutoff_grader_mentions import wilson
from graphs.style import INK, MUTED

# The edit labels paraphrase Adam's experiment summaries.
SNEEZE_EDITS = {
    1: "coastal tropical, not landlocked",
    2: "formerly in the Serbanian Empire",
    3: "one turn, country + “make it absurd…”",
    4: "fun fact: spice bazaars, not mountains",
    5: "coastal + formerly in an empire",
    6: "Landlocked: No, mountains kept",
    7: "renamed Tuvalia, new culture",
    8: "fun fact: landmarks, not mountains",
    9: "independence + colony lines removed",
    10: "landlocked, fun fact about beaches",
    11: "one turn, no “absurd and funny and wacky”",
    12: "“absurd and funny and wacky” removed",
}
SNEEZE_C = "#5b6c8f"


def rows() -> list[tuple[str, int, int]]:
    """(label, samples with a sneeze event, samples): the original conversation, then each edit by rate, highest first."""
    path = Path("data") / SNEEZE_RUN
    if not path.exists():
        hf_hub_download(HF_REPO, str(SNEEZE_RUN), repo_type="dataset", local_dir="data")
    r = next(r for r in map(json.loads, path.open()) if r["prompt_id"] == "prompt_003654")
    edits = sorted(((SNEEZE_EDITS[e["exp"]], e["k"], e["n"]) for e in r["exps"]), key=lambda x: -x[1] / x[2])
    return [("original conversation", r["k0"], r["n0"]), *edits]


def main(out: Path) -> None:
    rs = rows()
    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    for y, (label, k, n) in enumerate(rs):
        lo, hi = wilson(k, n)
        ax.barh(y, k / n, 0.7, color=INK if y == 0 else SNEEZE_C, zorder=3)
        ax.errorbar(k / n, y, xerr=[[k / n - lo], [hi - k / n]], color=MUTED, lw=0.9, capsize=2, capthick=0.9, zorder=4)
        ax.annotate(f"{k}/{n}", (hi, y), xytext=(2, 0), textcoords="offset points", va="center", color=INK)
        print(f"{label}: {k}/{n}")
    ax.set_yticks(range(len(rs)), [x[0] for x in rs])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 0.88)
    ax.set_xticks(np.arange(0, 0.76, 0.25), [f"{t:.0%}" for t in np.arange(0, 0.76, 0.25)])
    ax.set_xlabel("samples with a sneeze-based event")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/img/great-sneeze.png"))
    main(ap.parse_args().out)
