"""Two audits of CHIVE on Adam Karvonen's released data (adamkarvonen/chive-data), the three main-body WildChat eval runs
(Gemma 3 27B, Qwen3-8B, Qwen3.5-397B). docs/chive-dataset-audit.md §1 and §4.

    uv run python scripts/chive_adam_eliminations.py extract RUN_DIR    # stream each investigations.jsonl (~1.5 GB), keep only
                                                                         # prompt_id + per-experiment k/n -> RUN_DIR/adam_rates/
    uv run python scripts/chive_adam_eliminations.py count RUN_DIR      # §4: eliminations the 50pp rule can't test, claim-bank cost
    uv run python scripts/chive_adam_eliminations.py offpolicy RUN_DIR  # §1: prompts with a prior assistant turn, and each tool arm's
                                                                         # accuracy minus transcript-only on them (~700 MB to data/.cache)
"""

import collections
import json
import statistics
import sys
from pathlib import Path

import requests

RUNS = ["gemma3_27b_it_wildchat_eval", "qwen3_8b_wildchat_eval", "qwen3p5_397b_wildchat_eval"]
HF = "https://huggingface.co/datasets/adamkarvonen/chive-data/resolve/main/datasets"
CACHE = Path("data/.cache/chive_adam")
ARMS = {  # binary_evals arm files per run (answerer Opus 4.8); the 397B run has no tool arms
    "gemma3_27b_it_wildchat_eval": {
        "plain": "plain",
        "cf": "cf",
        "ao": "ao_latentqa_cls_past_lens_gemma-3-27b-it",
        "nla": "nla_gemma3-27b-L41",
        "sae": "sae_gemma-scope-2-27b-it_layer_40_width_65k_l0_medium",
    },
    "qwen3_8b_wildchat_eval": {
        "plain": "plain",
        "cf": "cf",
        "ao": "ao_latentqa_cls_past_lens_addition_Qwen3-8B",
        "nla": "nla_qwen3-8b-baseline-nosalvage",
        "sae": "sae_qwen3-8b-saes_trainer_2",
    },
    "qwen3p5_397b_wildchat_eval": {"plain": "plain", "cf": "cf"},
}


def cached(rel: str, name: str) -> list[dict]:
    """Download datasets/<rel> once into data/.cache/chive_adam/<name>, return its rows."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        path.write_bytes(requests.get(f"{HF}/{rel}", timeout=600).content)
    return [json.loads(line) for line in path.read_text().splitlines()]


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


def rates(out_dir: Path, run: str) -> list[dict]:
    return [json.loads(line) for line in (out_dir / f"rates_{run}.jsonl").read_text().splitlines()]


def count(out_dir: Path):
    tot = collections.Counter()
    for run in RUNS:
        inv = rates(out_dir, run)
        ex = [(d["prompt_id"], d["k0"] / d["n0"], e["k"] / e["n"], e["k"]) for d in inv for e in d["exps"] if e["k"] is not None and e["n"]]
        bands = collections.Counter(band(b, r) for _, b, r, _ in ex)
        stuck = [(p, b) for p, b, r, k in ex if k == 0 and 0.10 < b < 0.50]  # behaviour gone, but the baseline was under 50%
        has_true = {p for p, b, r, _ in ex if band(b, r) == "true"}
        n_stuck = collections.Counter(p for p, _ in stuck)
        row = {"inv": len(inv), "ex": len(ex), "excluded": bands["excluded"], "stuck": len(stuck), "inv_stuck": len(n_stuck)}
        row["inv_stuck_no_true"] = len(set(n_stuck) - has_true)
        # what the claim bank loses: verified investigations that have only false claims but hold a stuck elimination
        bank: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for c in cached(f"{run}/binary_counterfactual_questions.jsonl", f"bank_{run}.jsonl"):
            bank[c["prompt_id"]][c["is_true"]] += 1
        positive = {p for p, c in bank.items() if c[True]}
        gain = {p for p in bank if not bank[p][True] and n_stuck[p]}
        row |= {"bank_positive": len(positive), "bank_gain": len(gain), "true_claims": sum(c[True] for c in bank.values())}
        row["true_gain"] = sum(min(2, n_stuck[p]) for p in gain)
        tot.update(row)
        print(run, row, f"stuck = {100 * row['stuck'] / row['ex']:.1f}% of experiments")
    print("TOTAL", dict(tot))
    print(
        f"stuck = {100 * tot['stuck'] / tot['ex']:.1f}% of experiments, in {100 * tot['inv_stuck'] / tot['inv']:.0f}% of investigations;",
        f"{100 * tot['inv_stuck_no_true'] / tot['inv']:.0f}% of investigations have one and no >=50pp experiment;",
        f"excluded band {100 * tot['excluded'] / tot['ex']:.0f}%",
    )
    print(
        f"claim bank: investigations with a true claim +{100 * tot['bank_gain'] / tot['bank_positive']:.0f}%,",
        f"true claims +{100 * tot['true_gain'] / tot['true_claims']:.0f}% if each gained investigation gives up to 2",
    )


def offpolicy(out_dir: Path):
    prompts = cached("gemma3_27b_it_wildchat_eval/prompts.jsonl", "prompts.jsonl")  # the same 4,714 prompts in every WildChat eval run
    prior = {r["id"]: sum(m["role"] == "assistant" for m in r["messages"]) for r in prompts}
    off = {p for p, n in prior.items() if n >= 1}
    print(f"prompt pool {len(prior)}: {len(off)} ({100 * len(off) / len(prior):.0f}%) have a prior assistant turn")
    for run in RUNS:
        inv = [d["prompt_id"] for d in rates(out_dir, run)]
        ev = {c["claim_id"]: c for c in cached(f"{run}/eval_set_conc3_n2000.jsonl", f"evalset_{run}.jsonl")}
        arms = {}
        for arm, f in ARMS[run].items():
            rows = cached(f"{run}/binary_evals/{f}_claude-opus-4-8_high_conc3.jsonl", f"{run}__{arm}.jsonl")
            arms[arm] = {c["claim_id"]: c["correct"] for c in rows}
        quality = [c for c in ev if arms["cf"].get(c) and ev[c]["edit_cleanliness"] != "confounded" and c in arms["plain"]]
        off_inv = 100 * sum(p in off for p in inv) / len(inv)
        off_claims = 100 * sum(ev[c]["prompt_id"] in off for c in ev) / len(ev)
        print(f"== {run}: investigations {len(inv)}, off-policy {off_inv:.0f}%;")
        print(f"   eval claims {len(ev)}, off-policy {off_claims:.0f}%; quality slice {len(quality)}")
        for label, want in [("off-policy", True), ("on-policy", False)]:
            ids = [c for c in quality if (ev[c]["prompt_id"] in off) == want]
            line = f"   {label:10s} n={len(ids):4d} transcript-only {statistics.mean(arms['plain'][c] for c in ids):.3f}"
            for arm in ("ao", "nla", "sae"):
                if arm in arms:
                    d = [arms[arm][c] - arms["plain"][c] for c in ids]
                    line += f" | {arm}-plain {statistics.mean(d):+.3f} (SE {statistics.stdev(d) / len(d) ** 0.5:.3f})"
            print(line)


if __name__ == "__main__":
    out = Path(sys.argv[2]) / "adam_rates"
    {"extract": extract, "count": count, "offpolicy": offpolicy}[sys.argv[1]](out)
