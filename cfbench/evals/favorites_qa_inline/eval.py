"""favorites_qa_inline: predict Qwen3.6-27B's own pick ("What is your favorite constellation? Answer in one word.") from the
question alone: cutoff_qa_inline's setup on the favorites_qa env (WorkspaceBench basic_readout's implicit items, scaled up).

Robbie, 2026-10-29. The investigator sees the question Qwen was asked (not the reply) and predicts the reply in one turn with one
submit_answer call (cutoff_qa_inline's prompt, its inline.py). The label is Qwen's modal answer from the bank's stability gate
(`qwen_modal`, >= 7/8 samples agree); there is no truth. Cases: the episodes of the Qwen env run RUN, in bank order.

Conditions (single-turn, no tools): `transcript_inline` (the question), `olens_inline` (plus the item's own end-of-prompt reads:
the "." ending the question, <|im_end|>, "\\n", <|im_start|>, "assistant", "\\n", layers 32-60, from READS; Qwen's thinking-off
render then has "<think>\\n\\n</think>\\n\\n" before the reply, so the last read is 4 tokens before the answer),
`olens_placebo_inline` (another item's reads at the same positions, presented as this item's). The model is never named.
Investigator: Claude Haiku 5.5 at effort medium, as in cutoff_qa_inline.

    uv run python -m cfbench.eval favorites_qa_inline --prep
    uv run python -m cfbench.eval favorites_qa_inline --slug favorites-haiku-pilot62 --predictors anthropic/claude-haiku-5.5 --draws 3
    uv run python -m cfbench.evals.cutoff_qa_inline.report data/runs/cfbench/<date>_favorites-haiku-pilot62 --by shape

Read requests for RUN's episodes (the six end-of-prompt positions, computed with the Qwen tokenizer): `requests` below.
"""

import json
import random
from pathlib import Path

from cfbench.core import Rollout, render
from cfbench.evals.cutoff_qa_inline import inline
from environments._lib.core import load_env
from environments._lib.runfolder import LOCAL, read_episodes

ERROR_RATE = True  # the env viewer shows this eval as error rate, lower is better (Kylie, 2026-10-30)
ENV = "favorites_qa"
ENVS = [ENV]
RUN = "2026-10-29_pilot62-nohint"  # the Qwen env run whose prompts (and O-lens reads) the cases are
READS = Path("data/runs/olens_reads/2026-10-29_favorites-pilot62-tail/reads.jsonl")  # end-of-prompt reads of RUN's episodes
QWEN = "Qwen/Qwen3.6-27B"
MODEL_NAME = "the model"
TASK = "answer"
CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
INLINE_EFFORT = "medium"
SYSTEM = inline.SYSTEM
VERSIONS: dict[str, dict] = {"nohint": {"arm": "nohint"}}
ARM_LABELS = {"nohint": "the question, verbatim"}
PAIR_LABELS = {"nohint -> nohint": "no counterfactual"}


def _mod():
    return load_env(ENV)


def collect() -> list[dict]:
    """One case per episode of RUN, in bank order: the prompt, Qwen's gated modal answer (the label) and the bank metadata."""
    mod = _mod()
    v1 = {it["id"]: it for it in mod.bank("v1")["items"]}  # RUN's bank: v2 re-gated the pilot, so its labels can differ
    order = {item: k for k, item in enumerate(v1)}
    out = []
    for ep in read_episodes(LOCAL / ENV / RUN):
        assert not ep.get("error") and ep["target"].split("@")[0].lower() == QWEN.lower() and ep["meta"]["gate_class"] == "stable"
        modal = v1[ep["item"]]["qwen_modal"]  # the bank's (a raw sample), not the episode's older gate-normalized copy
        out.append({"version": "nohint", "item": ep["item"], "modal": mod.norm(modal), "source": f"{ENV}/{RUN}:{ep['id']}",
                    "messages": ep["messages"][:1], "meta": ep["meta"] | {"qwen_modal": modal}})  # fmt: skip
    return sorted(out, key=lambda c: order[c["item"]])


def requests(out: Path):
    """olens_read_requests.py rows for RUN's episodes at the six end-of-prompt tokens (cutoff200-tail's positions)."""
    from transformers import AutoTokenizer  # uv run --with transformers --with jinja2

    tok = AutoTokenizer.from_pretrained(QWEN)
    with out.open("w") as f:
        for c in collect():
            text = tok.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text)["input_ids"]
            pos = list(range(len(ids) - 10, len(ids) - 4))  # ".", <|im_end|>, \n, <|im_start|>, assistant, \n; then the empty think block
            assert [tok.decode([ids[p]]) for p in pos][1:] == ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"], c["item"]
            row = {"id": c["source"], "item": c["item"], "version": "nohint", "messages": c["messages"], "tools": None}
            f.write(json.dumps(row | {"generation_prompt": True, "priority": 0, "positions": pos}, ensure_ascii=False) + "\n")
    print("wrote", out)


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None = None, strict: bool = False):
    """cases.jsonl plus reads/reads.jsonl (READS, if it exists yet); the placebo is drawn from the same file (another item)."""
    (out / "reads").mkdir(parents=True)
    if READS.exists():
        (out / "reads/reads.jsonl").write_text(READS.read_text())
    cs = collect()[: n_items or None]
    (out / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cs))
    print(len(cs), "cases;", "reads copied" if READS.exists() else f"no reads yet ({READS})")


def reads(inputs: Path) -> list[str]:
    return [str(p) for p in [inputs / "reads/reads.jsonl"] if p.exists()]


class Prompt:
    BEHAVIOR = "Positive if the reply names Qwen3.6-27B's own stable answer to the question (the bank's qwen_modal)."
    ANSWER_FORMAT = "just the answer, as the question asks (one word or a short name)"
    ANSWER_MATCH = "the answer normalized (lowercase ASCII, punctuation and a leading article stripped) equals Qwen's modal answer"

    def __init__(self, meta: dict, messages: list[dict]):
        self.meta, self.start = meta, render(messages[:1])

    def view(self, base_arm: str, cf_arm: str, rollout: Rollout) -> tuple[str, str]:
        return self.start, "No change: there is no counterfactual arm."

    def canon_answer(self, text: str) -> str | None:
        return _mod().norm(text or "") or None


def cases(inputs: Path, versions: list[str]):
    for line in (inputs / "cases.jsonl").read_text().splitlines():
        c = json.loads(line)
        env, m = Prompt(c["meta"], c["messages"]), c["meta"]
        base = [Rollout(source=c["source"], messages=c["messages"], label=True, answer=c["modal"])]
        row = {"version": c["version"], "item": c["item"], "model": QWEN, "env_name": ENV, "task": TASK,
               "env": f"eval:favorites_qa_inline/{c['version']}@{c['item']}", "base_arm": "nohint", "cf_arm": "nohint",
               "k_base": 1, "n_base": 1, "answers_base": {c["modal"]: 1}, "answers_cf": {c["modal"]: 1},
               "modal_cf": c["modal"], "k_cf": m["qwen_modal_count"], "n_cf": 8, "gold": m["qwen_modal"],
               "gate_class": m["gate_class"], "domain": m["domain"], "category": m["category"], "shape": m["shape"]}  # fmt: skip
        yield env, row, base


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    question = base[0].messages[0]["content"]
    rows, extra = None, {}
    if condition != "transcript_inline":
        by_source = inline.reads_by_source(Path(case["inputs"]) / "reads/reads.jsonl")
        src = base[0].source
        if condition == "olens_placebo_inline":  # another item's reads, seeded per (draw, case), never its own
            src = random.Random(f"{seed}:{case['version']}:{case['item']}").choice(sorted(set(by_source) - {src}))
            extra = {"placebo_sources": [src]}
        rows = by_source[src]
    return inline.prompt(question, condition, False, MODEL_NAME, rows), extra


def LINKS(row: dict) -> dict[str, dict]:
    env_run, _, ep = row["shown"][0].partition(":")
    env, run = env_run.split("/")
    return {"transcript": {"env": env, "run": run, "episode": ep}}


if __name__ == "__main__":
    requests(Path(__import__("sys").argv[1]))
