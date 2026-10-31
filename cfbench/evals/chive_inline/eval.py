"""chive_inline: CHIVE's own yes/no benchmark, transcript vs edit only, single-turn (Robbie, 2026-10-31).

The claims are the official release's eval sets (adamkarvonen/chive-data, `eval_set_conc3_n2000`, pinned in
chive_critique/data.py): one version per target (Qwen3-8B 1,870, Gemma 3 27B 1,637, Qwen3.5-397B 1,309; balanced true /
false). Each claim says an intervention moves the behavior's rate by >= 30pp; truth is CHIVE's label. The prompt is CHIVE's
own plain-arm prompt (`binary_evals/plain_*`, their transcript-only Opus 4.8 arm), byte for byte, with two cuts:

- both conditions drop its "Question:" line (a "why did you ..." written from the reply; the BEHAVIOR line says the same);
- `transcript_inline` keeps the conversation and the target's one reply; `no_transcript_inline` (edit only) drops the reply,
  keeping the conversation up to it (earlier assistant turns are part of the input), BEHAVIOR, BASELINE, INTERVENTION and
  PREDICTION.

The closing sentence names our submit tool in place of their `answer` tool. Scored as AUROC (score.py) beside CHIVE's Opus
plain / cf AUROCs on the same claims.

    uv run python -m cfbench.eval chive_inline --prep
    uv run python -m cfbench.eval chive_inline --show --conditions transcript_inline,no_transcript_inline
    uv run python -m cfbench.eval chive_inline --slug chive-haiku --draws 3 --predictors anthropic-batch/claude-haiku-5-5 --batch submit
    uv run python -m cfbench.evals.chive_inline.score data/runs/cfbench/<date>_chive-haiku
"""

import json
from functools import cache
from pathlib import Path

from chive_critique.data import ADAM_LABELS, ADAM_RUNS, adam

ENVS: list[str] = []
MODEL_NAME = None
TASK = "claim"
VERSIONS = {run: {} for run in ADAM_RUNS}
CONDITIONS = ["transcript_inline", "no_transcript_inline"]
INLINE_EFFORT = "low"
THEIR_END = "Then call the `answer` tool with your calibrated probability that the claim is true."
OUR_END = "Then call `submit_prediction` once with your calibrated probability that the claim is true."


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None, strict: bool):
    """cases.jsonl: per claim CHIVE's plain prompt and the offsets of the reply, the Question line and the claim. The reply is
    the last "Assistant:" turn before the Question line; it equals the eval set's chosen_completion on all but 2 Qwen3.5-397B
    claims, where it differs by a few words (CHIVE's prompt is what we cut)."""
    off = 0
    with (out / "cases.jsonl").open("w") as f:
        for run in versions:
            reply = {r["claim_id"]: str(r["chosen_completion"]).strip() for r in adam(f"{run}/eval_set_conc3_n2000.jsonl")}
            for r in adam(f"{run}/binary_evals/plain_claude-opus-4-8_high_conc3.jsonl"):
                p = r["prompt"]
                c0 = p.rindex("\n\nClaim:\nBEHAVIOR")
                q0 = p.rindex("\n\nQuestion: ", 0, c0)
                a0 = p.rindex("\n\nAssistant: ", 0, q0)
                off += p[a0 + 13 : q0].strip()[:300] != reply[r["claim_id"]][:300]
                assert p.endswith(THEIR_END), r["claim_id"]
                row = {"version": run, "claim_id": r["claim_id"], "is_true": str(r["is_true"]) == "True", "prompt": p}
                f.write(json.dumps(row | {"a0": a0, "q0": q0, "c0": c0, "opus_plain_p": r["prob_true"]}) + "\n")
    assert off <= 2, f"{off} prompts whose last reply isn't the eval set's chosen_completion"


def reads(inputs: Path) -> list[str]:
    return []


def cases(inputs: Path, versions: list[str]):
    for c in map(json.loads, (inputs / "cases.jsonl").read_text().splitlines()):
        if c["version"] not in versions:
            continue
        row = {"version": c["version"], "item": c["claim_id"], "model": ADAM_LABELS[c["version"]], "env_name": "chive",
               "env": f"eval:chive_inline/{c['version']}", "task": TASK, "base_arm": "original", "cf_arm": "intervention",
               "k_base": 1, "n_base": 2, "k_cf": int(c["is_true"]), "n_cf": 1, "opus_plain_p": c["opus_plain_p"],
               }  # fmt: skip
        yield c["claim_id"], row, []


@cache
def _claims(inputs: str) -> dict[str, dict]:
    return {c["claim_id"]: c for c in map(json.loads, (Path(inputs) / "cases.jsonl").read_text().splitlines())}


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    c = _claims(case["inputs"])[case["item"]]
    p, a0, q0, c0 = c["prompt"], c["a0"], c["q0"], c["c0"]
    keep = q0 if condition == "transcript_inline" else a0
    assert condition in CONDITIONS, condition
    return (p[:keep] + p[c0:]).replace(THEIR_END, OUR_END), {}
