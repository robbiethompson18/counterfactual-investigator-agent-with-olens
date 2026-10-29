"""knowledge_cutoff: predict Qwen3.6-27B's reply to a post-cutoff factual question from the prompt alone (no reply shown).

Robbie, 2026-10-29. WorkspaceBench's knowledge_cutoff family with an investigator at pass@1: the label is Qwen's own stable
(usually stale) answer, the investigator sees the question and predicts the reply in one turn (inline.py, Robbie's prompt) with
only submit_answer. There is no counterfactual: base arm = cf arm = the env arm, and the answer task's truth is the modal
answer of Qwen's samples (`modal_cf`), so cfbench.answer's "edit does nothing" column is always yes here. Decisions and results:
docs/envs/knowledge_cutoff.md.

Investigator: Claude Haiku 5.5 at effort medium (`--predictors anthropic/claude-haiku-5.5`; INLINE_EFFORT), not Gemini 3.8 Flash:
Flash holds Qwen's stale answer itself on a quarter of the items (the census in the doc). The system prompt never names the model
(MODEL_NAME; the `*_told` versions add one sentence naming Qwen and its cutoff to the user turn, kept for an ablation).

Versions: `recall` (the env's arm; default) and `recall_told`. No MCQ version: a multiple-choice render changes Qwen's own
answer (docs/envs/knowledge_cutoff.md). Conditions (all single-turn): `transcript_inline` (the prompt),
`olens_placebo_inline` (plus another prompt's end-of-prompt reads from the inputs' placebo file), `olens_inline` (plus the
item's own reads from the inputs' reads file, once a read job over the env run exists: scripts/olens_requests_from_runs.py
envs:knowledge_cutoff/RUN, then prep with --reads).

    uv run python -m cfbench.eval knowledge_cutoff --prep \\
        --placebo data/runs/olens_reads/2026-10-29_hint-qwen-dsv4-tail/reads.placebo-head.jsonl
    uv run python -m cfbench.eval knowledge_cutoff --show --limit 1
    uv run python -m cfbench.eval knowledge_cutoff --slug cutoff-haiku-dev109 --predictors anthropic/claude-haiku-5.5 --draws 5 \\
        --conditions transcript_inline,olens_placebo_inline
    uv run python -m cfbench.answer data/runs/cfbench/<date>_cutoff-haiku-dev109

Cases: items of the env runs in VERSIONS whose Qwen answer is stable (the modal `answer` measure in >= 7/8 of samples) with a
non-numeric gold. Items from WorkspaceBench's bank are dev only (they passed WSB's Opus/Sonnet gate; never in a significance test).
"""

import json
import re
from collections import Counter
from pathlib import Path

from cfbench.core import Rollout, render
from cfbench.evals.knowledge_cutoff import inline
from environments._lib.core import load_env
from environments._lib.index import latest_labels
from environments._lib.runfolder import LOCAL, read_episodes

ENV = "knowledge_cutoff"
ENVS = [ENV]
QWEN = "Qwen/Qwen3.6-27B"
MODEL_NAME = "the model"  # the runner asserts the id never reaches the investigator
TASK = "answer"
CONDITIONS = ("transcript_inline", "olens_placebo_inline")
INLINE_EFFORT = "medium"
SYSTEM = inline.SYSTEM  # cfbench.eval passes it to predict() in place of the answer task's system prompt
STABLE = 7 / 8
NUMERIC = re.compile(r"[\d.,k]+")
RUNS: dict[str, list[str]] = {"recall": ["2026-10-29_dev109-recall"]}  # Qwen env runs per arm (data/envs/knowledge_cutoff/)
VERSIONS: dict[str, dict] = {"recall": {"arm": "recall", "told": False}, "recall_told": {"arm": "recall", "told": True}}
DEFAULT_VERSIONS = ["recall"]
ARM_LABELS = {"recall": "free-text answer"}
PAIR_LABELS = {"recall -> recall": "free text (no counterfactual)"}


def _mod():
    return load_env(ENV)


def collect(version: str) -> list[dict]:
    """Per stable item of the version's arm: Qwen's answers over the env runs, the modal one, one episode's prompt."""
    arm, mod = str(VERSIONS[version]["arm"]), _mod()
    answers: dict[str, Counter] = {}
    prompts: dict[str, tuple[str, list[dict]]] = {}
    for run in RUNS[arm]:
        _, labels = latest_labels(LOCAL / ENV / run)
        for ep in read_episodes(LOCAL / ENV / run):
            if ep.get("error") or ep["arm"] != arm or ep["target"].split("@")[0].lower() != QWEN.lower():
                continue
            answers.setdefault(ep["item"], Counter())[(labels.get(ep["id"]) or {}).get("answer")] += 1
            prompts.setdefault(ep["item"], (f"{ENV}/{run}:{ep['id']}", ep["messages"]))
    out = []
    for item, c in sorted(answers.items()):
        modal, k = c.most_common(1)[0]
        n = sum(c.values())
        if modal is None or k / n < STABLE or NUMERIC.fullmatch(modal):
            continue
        src, messages = prompts[item]
        out.append({"version": version, "item": item, "modal": modal, "answers": {str(a): v for a, v in c.items()}, "n": n,
                    "source": src, "messages": messages, "meta": mod.build(arm, item).meta})  # fmt: skip
    return out


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None = None, strict: bool = False):
    """cases.jsonl plus placebo/reads.jsonl (any end-of-prompt reads file: a file, or a dir holding reads.jsonl) and, when
    `--reads` has been copied in by hand, reads/reads.jsonl (the item's own tail reads, ids = the env episode source)."""
    (out / "placebo").mkdir(parents=True)
    src = placebo if placebo.is_file() else placebo / "reads.jsonl"
    if src.exists():
        (out / "placebo/reads.jsonl").write_text(src.read_text())
    cs = [c for v in versions for c in collect(v)[: n_items or None]]
    (out / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cs))
    print({v: sum(c["version"] == v for c in cs) for v in versions}, "stable non-numeric items")


def reads(inputs: Path) -> list[str]:
    return [str(p) for p in (inputs / "reads/reads.jsonl", inputs / "placebo/reads.jsonl") if p.exists()]


class Prompt:
    """The item's question as the starting context (no edit: there is no counterfactual) and the answer-task hooks."""

    BEHAVIOR = "Positive if the reply names Qwen3.6-27B's own stable answer to the question (the bank's gold)."
    ANSWER_FORMAT = "just the answer, as the question asks (a name, a product, a number)"
    ANSWER_MATCH = "the bank's alias regex: the one bank option the answer names, case- and accent-insensitive"

    def __init__(self, meta: dict, messages: list[dict]):
        self.meta, self.start = meta, render(messages[:1])

    def view(self, base_arm: str, cf_arm: str, rollout: Rollout) -> tuple[str, str]:
        return self.start, "No change: the counterfactual arm is the same prompt."

    def canon_answer(self, text: str) -> str | None:
        return _mod().named_option(self.meta, text or "")


def cases(inputs: Path, versions: list[str]):
    for line in (inputs / "cases.jsonl").read_text().splitlines():
        c = json.loads(line)
        if c["version"] not in versions:
            continue
        arm, env = VERSIONS[c["version"]]["arm"], Prompt(c["meta"], c["messages"])
        base = [Rollout(source=c["source"], messages=c["messages"][:1], label=True, answer=c["modal"])]
        row = {"version": c["version"], "item": c["item"], "model": QWEN, "env_name": ENV, "task": TASK,
               "env": f"eval:knowledge_cutoff/{c['version']}@{c['item']}", "base_arm": arm, "cf_arm": arm,
               "k_base": 1, "n_base": 1, "answers_base": c["answers"], "answers_cf": c["answers"],
               "modal_cf": c["modal"], "k_cf": c["answers"][c["modal"]], "n_cf": c["n"],
               "gold": c["meta"]["gold"], "truth": c["meta"]["truth"], "label_type": c["meta"]["label_type"]}  # fmt: skip
        yield env, row, base


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    question = base[0].messages[0]["content"]
    inputs = Path(case["inputs"]) if "inputs" in case else None  # cfbench.eval sets it on the case before calling INLINE
    rows, extra = None, {}
    if condition == "olens_placebo_inline":
        assert inputs is not None
        path = inputs / "placebo/reads.jsonl"
        src = inline.placebo_source(path, f"{case['version']}:{case['item']}", seed)
        rows, extra = inline.reads_by_source(path)[src], {"placebo_sources": [src]}
    elif condition == "olens_inline":
        assert inputs is not None
        rows = inline.reads_by_source(inputs / "reads/reads.jsonl")[base[0].source]
    told = bool(VERSIONS[case["version"]]["told"])
    return inline.prompt(question, condition, told, "Qwen3.6-27B" if told else "the model", rows), extra


def LINKS(row: dict) -> dict[str, dict]:
    env_run, _, ep = row["shown"][0].partition(":")
    env, run = env_run.split("/")
    return {"transcript": {"env": env, "run": run, "episode": ep}}
