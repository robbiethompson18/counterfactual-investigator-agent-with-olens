"""Score hint_qwen_dsv4_letter runs: % of draws naming Qwen's no-hint letter.

    uv run python -m cfbench.evals.hint_qwen_dsv4_letter.score data/runs/cfbench/<date>_<slug> [...]

Per (condition, version), never pooled: % right with a 95% bootstrap CI over items, % right per class ("knew": the truth is
the hinted letter; "didn't": a different one), and the paired O-lens comparisons (hint_qwen_dsv4/score.py's). Baselines
printed beside it: always naming the hinted letter (50%: right exactly on the "knew" half) and uniform chance (mean over
items of 1 / number of options).
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

from cfbench.core import canon_condition
from cfbench.evals.hint_qwen_dsv4.score import boot_mean, paired, right_by_item
from cfbench.evals.hint_qwen_dsv4_letter.eval import VERSIONS
from environments._lib.core import load_env

OPTION = re.compile(r"^\(?[A-P][.)] ", re.MULTILINE)  # "A. ..." (hint_mcq) or "(A) ..." (hinted_truthfulqa)


def n_options(version: str, item: str) -> int:
    msgs = load_env(VERSIONS[version]["env"]).build(VERSIONS[version]["arms"][1], item).messages
    return len(OPTION.findall(msgs[-1]["content"]))


def main():
    rows = [json.loads(line) for path in sys.argv[1:] for line in (Path(path) / "predictions.jsonl").read_text().splitlines()]
    right = right_by_item(rows, lambda r: r.get("correct"))
    meta = {(r["version"], r["item"]): r for r in rows}
    rng = np.random.default_rng(0)
    for (cond, version), items in sorted(right.items()):
        per_item = {i: float(np.mean(v)) for i, v in items.items()}
        knew, didnt = (np.mean([x for i, x in per_item.items() if meta[version, i]["class"] == c]) for c in ("knew", "didn't"))
        lo, hi = boot_mean(list(per_item.values()), rng)
        chance = np.mean([1 / n_options(meta[version, i]["source"], i) for i in items])
        invalid = sum(1 for r in rows if canon_condition(r["condition"]) == cond and r["version"] == version and not r["valid"])
        print(f"{cond + ' ' + version:36} items {len(items):>3}  right {np.mean(list(per_item.values())):.0%} [{lo:.0%}, {hi:.0%}]  "
              f"knew {knew:.0%} / didn't {didnt:.0%}  (hinted letter 50%, chance {chance:.0%})  invalid {invalid}")  # fmt: skip
    for x in paired(right, np.random.default_rng(1)):
        lo, hi = x["ci"]
        print(f"{x['version']:11} {x['a']} − {x['b']}: {x['diff']:+.0%} [{lo:+.0%}, {hi:+.0%}] over {x['n']} items; "
              f"sign {x['better']}/{x['worse']} p = {x['p_sign']:.3g}")  # fmt: skip


if __name__ == "__main__":
    main()
