"""Label every stage-3 experiment with the odds-ratio rule agreed for WeirdChat (20 Oct 2026).

CHIVE's own gate (build_claims.py: moved >= 50 points, inert <= 15) can't fire from WeirdChat's
30-50% baselines, so each experiment is compared with its investigation's baseline instead:

- mover: odds ratio >= 3 in either direction AND Fisher's exact p < 0.01
- inert: odds ratio <= 1.5 (point estimate)
- dead zone: everything else; no claim either way

Rates are the grader's counts (baseline = the 64 stage-1 samples, experiment = its own samples).
A zero cell gets the Haldane-Anscombe +0.5 correction for the odds ratio only.

    uv run python claims.py data/runs/wc30_pilot          # writes claims.jsonl, prints a summary
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import fisher_exact

MOVER_OR = 3.0
MOVER_P = 0.01
INERT_OR = 1.5


def odds_ratio(k0: int, n0: int, k1: int, n1: int) -> float:
    a, b, c, d = k1, n1 - k1, k0, n0 - k0
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (a / b) / (c / d)


def label(k0: int, n0: int, k1: int, n1: int) -> dict:
    ratio = odds_ratio(k0, n0, k1, n1)
    symmetric = max(ratio, 1 / ratio)  # how far from 1 in either direction
    p = fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])[1]
    if symmetric >= MOVER_OR and p < MOVER_P:
        kind = "mover_up" if ratio > 1 else "mover_down"
    elif symmetric <= INERT_OR:
        kind = "inert"
    else:
        kind = "dead_zone"
    return {"odds_ratio": round(ratio, 3), "log_odds_ratio": round(math.log(ratio), 3), "fisher_p": p, "label": kind}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main() -> None:
    run_dir = Path(sys.argv[1])
    meta = json.loads((run_dir / "wc_meta.json").read_text())
    scores = {v["prompt_id"]: v["score"] for v in load_jsonl(run_dir / "verification.jsonl")}
    rows = []
    for inv in load_jsonl(run_dir / "investigations.jsonl"):
        pid = inv["prompt_id"]
        k0, n0 = inv["baseline_grader_yes"], len(inv["baseline_grader_verdicts"])
        for exp in inv["experiment_log"]:
            k1, n1 = exp["grader_yes"], len(exp["grader_verdicts"])
            rows.append(
                {
                    "prompt_id": pid,
                    "behavior_id": meta.get(pid, {}).get("behavior_id"),
                    "exp_num": exp["exp_num"],
                    "verification_score": scores.get(pid),
                    "k_baseline": k0,
                    "n_baseline": n0,
                    "k_counterfactual": k1,
                    "n_counterfactual": n1,
                    "rate_baseline": round(k0 / n0, 4),
                    "rate_counterfactual": round(k1 / n1, 4),
                    **label(k0, n0, k1, n1),
                    "edit_record": exp.get("edit_record"),
                    "messages": exp["messages"],
                }
            )
    (run_dir / "claims.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    by_prompt: dict[str, Counter] = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], Counter())[r["label"]] += 1
    print(f"{'behaviour':30s} {'base':>6s} {'verif':>5s} {'exps':>4s} {'up':>3s} {'down':>4s} {'inert':>5s} {'dead':>4s}")
    for inv in load_jsonl(run_dir / "investigations.jsonl"):
        pid = inv["prompt_id"]
        c = by_prompt.get(pid, Counter())
        base = inv["baseline_grader_yes"] / len(inv["baseline_grader_verdicts"])
        score = scores.get(pid)
        print(
            f"{meta.get(pid, {}).get('behavior_id', pid)[:30]:30s} {base:6.0%} {score if score is not None else '-':>5} "
            f"{sum(c.values()):4d} {c['mover_up']:3d} {c['mover_down']:4d} {c['inert']:5d} {c['dead_zone']:4d}"
        )
    total = Counter(r["label"] for r in rows)
    with_mover = sum(1 for c in by_prompt.values() if c["mover_up"] + c["mover_down"])
    print(
        f"\n{len(rows)} experiments over {len(by_prompt)} investigations: {dict(total)}; "
        f"{with_mover} investigations have at least one mover -> {run_dir / 'claims.jsonl'}"
    )


if __name__ == "__main__":
    main()
