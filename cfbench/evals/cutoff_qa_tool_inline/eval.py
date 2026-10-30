"""cutoff_qa_tool_inline: cutoff_qa_inline with Qwen answering through a submit_answer tool call, so the end-of-prompt reads
are no longer next-token reads (Robbie, 2026-10-29).

Qwen saw each bank v2 question with a `submit_answer(answer)` tool (scripts/cutoff_gate.py SUBMIT_TOOL, tool_choice
"required"): the chat template puts the tool schema in a system block before the question, and the reply opens with the
tool-call wrapper, so in Qwen's native format the answer is 20 tokens past the last read instead of 5 (the empty think
block, then `<tool_call>\n<function=submit_answer>\n<parameter=answer>\n`). Labels are that gate's
(runs/cutoff_gate/2026-10-29_v2-tool-required): Qwen's modal tool-call answer, stable = >= 7/8. Cases: the first 200
non-numeric items of bank v2, in bank order, stable under that gate. No env run: the gate's prompts are the transcripts.

Everything else is cutoff_qa_inline's (its Prompt, INLINE and report.py): the investigator sees the question only (not the
tool schema, which says nothing about the answer), Claude Haiku 5.5 at effort medium, conditions transcript_inline /
olens_inline / olens_placebo_inline, reads at the same six end-of-prompt tokens and layers 32-60.

    uv run --with transformers --with jinja2 python -m cfbench.evals.cutoff_qa_tool_inline.eval REQUESTS.jsonl   # read requests
    uv run python -m cfbench.eval cutoff_qa_tool_inline --prep
    uv run python -m cfbench.eval cutoff_qa_tool_inline --slug cutoff-qa-tool-haiku-200 --predictors anthropic/claude-haiku-5.5 --draws 3
    uv run python -m cfbench.evals.cutoff_qa_inline.report data/runs/cfbench/<date>_cutoff-qa-tool-haiku-200
"""

import json
from pathlib import Path

from cfbench.core import Rollout
from cfbench.evals.cutoff_qa_inline import inline
from cfbench.evals.cutoff_qa_inline.eval import INLINE, Prompt  # noqa: F401  (the runner reads INLINE off this module)
from environments._lib.core import load_env

ENV = "cutoff_qa"
ENVS = [ENV]
GATE = "cutoff_gate/2026-10-29_v2-tool-required"  # the tool-call gate whose labels and prompts the cases are
READS = Path("data/runs/olens_reads/2026-10-29_cutoff-tool200-tail/reads.jsonl")  # end-of-prompt reads of the tool prompts
QWEN = "Qwen/Qwen3.6-27B"
MODEL_NAME = "the model"
TASK = "answer"
CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
INLINE_EFFORT = "medium"
SYSTEM = inline.SYSTEM
VERSIONS: dict[str, dict] = {"nohint": {"arm": "nohint"}}
ARM_LABELS = {"nohint": "the question, verbatim (asked with a submit_answer tool)"}
PAIR_LABELS = {"nohint -> nohint": "no counterfactual"}
N = 200


def tool() -> dict:
    return json.loads((Path("data/runs") / GATE / "meta.json").read_text())["params"]["tools"][0]


def collect() -> list[dict]:
    """The first N non-numeric bank v2 items stable under GATE, in bank order, labelled with GATE's modal answer."""
    mod = load_env(ENV)
    gate = {q["id"]: q for q in json.loads((Path("data/runs") / GATE / "summary.json").read_text())["questions"]}
    out = []
    for it in mod.bank("v2")["items"]:
        g = gate[it["id"]]
        if it["numeric"] or not g["class"].startswith("stable"):
            continue
        meta = it | {"qwen_modal": g["modal"], "qwen_modal_count": g["modal_count"], "gate_class": g["class"], "gate_run": GATE}
        out.append({"version": "nohint", "item": it["id"], "modal": mod.norm(g["modal"]), "source": f"{GATE}:{it['id']}",
                    "messages": [{"role": "user", "content": it["question"]}], "meta": meta})  # fmt: skip
    return out[:N]


def requests(out: Path):
    """olens_read_requests.py rows: the tool-schema render, read at its six end-of-prompt tokens (cutoff200-tail's positions)."""
    from transformers import AutoTokenizer  # uv run --with transformers --with jinja2

    tok, tools = AutoTokenizer.from_pretrained(QWEN), [tool()]
    with out.open("w") as f:
        for c in collect():
            text = tok.apply_chat_template(c["messages"], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text)["input_ids"]
            pos = list(range(len(ids) - 10, len(ids) - 4))  # ".", <|im_end|>, \n, <|im_start|>, assistant, \n; then the empty think block
            assert [tok.decode([ids[p]]) for p in pos][1:] == ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"], c["item"]
            row = {"id": c["source"], "item": c["item"], "version": "nohint", "messages": c["messages"], "tools": tools}
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


def cases(inputs: Path, versions: list[str]):
    for line in (inputs / "cases.jsonl").read_text().splitlines():
        c = json.loads(line)
        env, m = Prompt(c["meta"], c["messages"]), c["meta"]
        base = [Rollout(source=c["source"], messages=c["messages"], label=True, answer=c["modal"])]
        row = {"version": c["version"], "item": c["item"], "model": QWEN, "env_name": ENV, "task": TASK,
               "env": f"eval:cutoff_qa_tool_inline/{c['version']}@{c['item']}", "base_arm": "nohint", "cf_arm": "nohint",
               "k_base": 1, "n_base": 1, "answers_base": {c["modal"]: 1}, "answers_cf": {c["modal"]: 1},
               "modal_cf": c["modal"], "k_cf": m["qwen_modal_count"], "n_cf": 8,
               "gold": m["qwen_modal"], "truth": m["truth"], "truth_now": m.get("truth_now"), "gate_class": m["gate_class"],
               "domain": m["domain"], "answer_type": m["answer_type"]}  # fmt: skip
        yield env, row, base


if __name__ == "__main__":
    requests(Path(__import__("sys").argv[1]))
