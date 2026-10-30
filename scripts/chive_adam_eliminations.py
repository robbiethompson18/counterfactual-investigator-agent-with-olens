"""Eliminations CHIVE's 50pp rule can never test, counted on Adam Karvonen's released data (adamkarvonen/chive-data), the
three main-body WildChat eval runs (Gemma 3 27B, Qwen3-8B, Qwen3.5-397B). docs/chive-dataset-audit.md §4.

    uv run python scripts/chive_adam_eliminations.py extract RUN_DIR   # streams each investigations.jsonl (~1.5 GB), keeps
                                                                        # only prompt_id + per-experiment k/n -> RUN_DIR/adam_rates/
    uv run python scripts/chive_adam_eliminations.py count RUN_DIR     # the table
"""

import collections
import json
import sys
from pathlib import Path

import requests

RUNS = ["gemma3_27b_it_wildchat_eval", "qwen3_8b_wildchat_eval", "qwen3p5_397b_wildchat_eval"]
HF = "https://huggingface.co/datasets/adamkarvonen/chive-data/resolve/main/datasets"


def extract(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        url = f"{HF}/{run}/investigations.jsonl"
        with requests.get(url, stream=True, timeout=600) as r, open(out_dir / f"rates_{run}.jsonl", "w") as f:
            r.raise_for_status()
            for line in r.iter_lines(chunk_size=1 << 20):
                if not line:
                    continue
                d = json.loads(line)
                exps = [{"exp": e["exp_num"], "k": e["grader_yes"], "n": e["n"], "t": e["temperature"]} for e in d["experiment_log"]]
                row = {"prompt_id": d["prompt_id"], "k0": d["baseline_grader_yes"], "n0": len(d["baseline_grader_verdicts"]), "exps": exps}
                f.write(json.dumps(row) + "\n")
        print("extracted", run)


def band(b, r):
    pp = abs(r - b) * 100
    return "true" if pp >= 50 else ("false" if pp <= 15 else "excluded")


def count(out_dir: Path):
    tot = collections.Counter()
    for run in RUNS:
        inv = [json.loads(line) for line in (out_dir / f"rates_{run}.jsonl").read_text().splitlines()]
        ex = [(d["prompt_id"], d["k0"] / d["n0"], e["k"] / e["n"], e["k"]) for d in inv for e in d["exps"] if e["k"] is not None and e["n"]]
        bands = collections.Counter(band(b, r) for _, b, r, _ in ex)
        stuck = [(p, b) for p, b, r, k in ex if k == 0 and 0.10 < b < 0.50]  # behaviour gone, but the baseline was under 50%
        has_true = {p for p, b, r, _ in ex if band(b, r) == "true"}
        inv_stuck = {p for p, _ in stuck}
        row = {"inv": len(inv), "ex": len(ex), "excluded": bands["excluded"], "stuck": len(stuck), "inv_stuck": len(inv_stuck)}
        row["inv_stuck_no_true"] = len(inv_stuck - has_true)
        tot.update(row)
        print(run, row, f"stuck = {100 * row['stuck'] / row['ex']:.1f}% of experiments")
    print("TOTAL", dict(tot))
    print(
        f"stuck = {100 * tot['stuck'] / tot['ex']:.1f}% of experiments, in {100 * tot['inv_stuck'] / tot['inv']:.0f}% of investigations;",
        f"{100 * tot['inv_stuck_no_true'] / tot['inv']:.0f}% of investigations have one and no >=50pp experiment;",
        f"excluded band {100 * tot['excluded'] / tot['ex']:.0f}%",
    )


if __name__ == "__main__":
    out = Path(sys.argv[2]) / "adam_rates"
    extract(out) if sys.argv[1] == "extract" else count(out)
