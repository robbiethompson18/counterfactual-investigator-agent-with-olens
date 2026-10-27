"""The answer task: predict the model's answer in the counterfactual arm, scored as % correct (cfbench/CLAUDE.md).

The rate task (`submit_prediction`, RMSE, `cfbench.score`) and this one never share a grader: a run is one task, every row
says which, and each scorer refuses the other's rows. A case is an ordered arm pair on one item; the truth is the
counterfactual arm's most common canonical answer (`env.canon_answer` of each run's final reply) and a draw is correct when
its submission canonicalises to it. The null is "the edit does nothing": the base arm's most common answer.

    uv run python -m cfbench.answer data/runs/cfbench/<run dir> [...]
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from openai.types.chat import ChatCompletionFunctionToolParam

SYSTEM = (
    "You are an investigator predicting how an AI model would behave if its environment were changed. "
    "You get transcripts of the model's runs and the exact change, and you predict the answer it gives."
)
SUBMIT: ChatCompletionFunctionToolParam = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit your final answer: what a fresh run in the counterfactual arm most likely answers, in the model's format.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "the predicted answer, in the format the model was told to use"},
                "rationale": {"type": "string", "description": "one or two sentences: the main reason for this answer"},
            },
            "required": ["answer", "rationale"],
        },
    },
}
NUDGE = "Now call submit_answer with your answer."
EMPTY = {"answer": None}


def check(arguments: str) -> tuple[dict | None, str]:
    """Parsed submission if valid, else None and the error to send back to the predictor."""
    try:
        args = json.loads(arguments)
        answer = str(args["answer"]).strip()
    except (ValueError, KeyError, TypeError) as e:
        return None, f"Invalid submission ({e!r}). Call submit_answer with a string answer."
    if not answer:
        return None, "answer is empty. Call submit_answer again."
    return {"answer": answer, "rationale": str(args.get("rationale", ""))}, "Answer recorded."


Group = tuple[str, str]  # (condition, predictor)
CaseKey = tuple[str, str, str, str]  # (env incl. item, target model, base_arm, cf_arm)


def load(paths: list[Path]) -> tuple[dict[Group, dict[CaseKey, dict]], Counter]:
    """Per (condition, predictor) and case: the case fields plus `canon` (each valid draw's canonical answer, draw order);
    and the invalid-row count per (condition, predictor, *case)."""
    cases: dict[Group, dict[CaseKey, dict]] = defaultdict(dict)
    invalid: Counter = Counter()
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    for r in sorted(rows, key=lambda r: r["draw"]):
        assert r.get("task") == "answer", f"{r.get('task')!r} rows are scored by cfbench.score, not cfbench.answer"
        group, key = (r["condition"], r["predictor"]), (r["env"], r["model"], r["base_arm"], r["cf_arm"])
        if not r["valid"]:
            invalid[group + key] += 1
            continue
        case = cases[group].setdefault(key, {k: r[k] for k in ("answers_base", "answers_cf", "modal_cf", "k_cf", "n_cf")} | {"canon": []})
        case["canon"].append(r["canon"])
    return cases, invalid


def modal(counts: dict[str, int]) -> str | None:
    return max(counts, key=counts.__getitem__) if counts else None


def main():
    paths = [p / "predictions.jsonl" if p.is_dir() else p for p in map(Path, sys.argv[1:])]
    cases, invalid = load(paths)
    print(
        f"{'condition':13} {'predictor':30} {'case':60} {'base modal':>14} {'cf modal':>14} {'answers (each draw)':40} "
        f"{'invalid':>7} {'correct':>7} {'null':>4}"
    )
    for (condition, predictor), group in sorted(cases.items()):
        for (env, model, base_arm, cf_arm), c in sorted(group.items()):
            base_modal, correct = modal(c["answers_base"]), [a == c["modal_cf"] for a in c["canon"]]
            name = f"{env.split(':')[-1]}/{model.split('/')[-1]} {base_arm}->{cf_arm}"
            null = "yes" if base_modal == c["modal_cf"] else "no"
            print(
                f"{condition:13} {predictor:30} {name:60} {base_modal!s:>14} {c['modal_cf']:>9} {c['k_cf']:>2}/{c['n_cf']:<2} "
                f"{' '.join(map(str, c['canon'])):40} {invalid[condition, predictor, env, model, base_arm, cf_arm]:>7} "
                f"{sum(correct) / len(correct) * 100:>6.0f}% {null:>4}"
            )
    print()
    for (condition, predictor), group in sorted(cases.items()):
        per_case = [sum(a == c["modal_cf"] for a in c["canon"]) / len(c["canon"]) for c in group.values()]
        null = [modal(c["answers_base"]) == c["modal_cf"] for c in group.values()]
        print(
            f"{condition:13} {predictor:30} correct {sum(per_case) / len(per_case) * 100:.0f}% "
            f"(edit does nothing: {sum(null) / len(null) * 100:.0f}%), {len(group)} cases"
        )


if __name__ == "__main__":
    main()
