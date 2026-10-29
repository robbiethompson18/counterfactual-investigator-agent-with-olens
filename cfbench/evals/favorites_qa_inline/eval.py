"""favorites_qa_inline: predict Qwen3.6-27B's own pick ("What is your favorite constellation? Answer in one word.") from the
question alone: cutoff_qa_inline's setup on the favorites_qa env (WorkspaceBench basic_readout's implicit items, scaled up).

Robbie, 2026-10-29. The investigator sees the question Qwen was asked (not the reply) and predicts the reply in one turn with one
submit_answer call (cutoff_qa_inline's prompt and reads blocks, its inline.py). The label is Qwen's modal answer from the bank's
stability gate (`qwen_modal`, >= 7/8 samples agree); there is no truth. Cases: the episodes of the Qwen env run RUN (every
stable item of bank BANK), in bank order. The pilot (RUN 2026-10-29_pilot62-nohint, bank v1, O-lens only) is this file at
that commit in git history.

Conditions (single-turn, no tools; the model is never named; Claude Haiku 5.5 at effort medium):
- `transcript_inline`: the question.
- `olens_inline`: plus the item's own O-lens reads (space lens + AR, T 1) at the six end-of-prompt tokens (the "." ending the
  question, <|im_end|>, "\\n", <|im_start|>, "assistant", "\\n"), layers 32-60, from READS. Qwen's thinking-off render then has
  "<think>\\n\\n</think>\\n\\n" before the reply, so the last read is 4 tokens before the answer.
- `nla_inline`: the all-layer NLA's readouts at the same tokens and layers (NLA_READS, critic FVE shown).
- `jlens_inline`: J++ top-3 at every token of the rendered prompt, the same layers (JREADS).
- `*_placebo_inline`: another favorites item's reads of the same kind, presented as this item's; the three placebos use the
  same other item for a given (draw, case). (cutoff_qa_inline drew its NLA/J++ placebos from unrelated pools; here every
  prompt has the same shape, so another item is the clean null.)

    uv run --with transformers --with jinja2 python -m cfbench.evals.favorites_qa_inline.eval DIR   # DIR/tail.jsonl, DIR/all.jsonl
    uv run python -m cfbench.eval favorites_qa_inline --prep
    uv run python -m cfbench.eval favorites_qa_inline --slug favorites-haiku-v3 --predictors anthropic/claude-haiku-5.5 --draws 3
    uv run python -m cfbench.evals.cutoff_qa_inline.report data/runs/cfbench/<date>_favorites-haiku-v3 --by shape

tail.jsonl is the O-lens and NLA readers' request file, all.jsonl the J++ reader's (rendered text, positions all).
"""

import json
import random
from pathlib import Path

from cfbench.core import Rollout, render
from cfbench.evals.cutoff_qa_inline import inline
from cfbench.evals.cutoff_qa_inline.eval import jpp_rows
from environments._lib.core import load_env
from environments._lib.runfolder import LOCAL, read_episodes

ENV = "favorites_qa"
ENVS = [ENV]
BANK = "v3"
RUN = "2026-10-29_v3-1500-nohint"  # the Qwen env run whose prompts (and reads) the cases are
READS = Path("data/runs/olens_reads/2026-10-29_favorites-v3-tail/reads.jsonl")  # O-lens, end-of-prompt tokens of RUN's episodes
NLA_READS = Path("data/runs/nla_reads/2026-10-29_favorites-v3-tail/reads.jsonl")  # NLA, the same tokens and layers
JREADS = Path("data/runs/jpp_reads/2026-10-29_favorites-v3-all/reads.jsonl")  # J++, every token of RUN's prompts
QWEN = "Qwen/Qwen3.6-27B"
MODEL_NAME = "the model"
TASK = "answer"
CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline", "nla_inline", "nla_placebo_inline", "jlens_inline",
              "jlens_placebo_inline")  # fmt: skip
INLINE_EFFORT = "medium"
SYSTEM = inline.SYSTEM
VERSIONS: dict[str, dict] = {"nohint": {"arm": "nohint"}}
ARM_LABELS = {"nohint": "the question, verbatim"}
PAIR_LABELS = {"nohint -> nohint": "no counterfactual"}
READ_FILES = {"olens": "reads.jsonl", "nla": "nla.jsonl", "jlens": "jreads.jsonl"}  # condition prefix -> file under inputs/reads/


def _mod():
    return load_env(ENV)


def collect() -> list[dict]:
    """One case per episode of RUN, in bank order: the prompt, Qwen's gated modal answer (the label) and the bank metadata."""
    mod = _mod()
    bank = {it["id"]: it for it in mod.bank(BANK)["items"]}
    order = {item: k for k, item in enumerate(bank)}
    out = []
    for ep in read_episodes(LOCAL / ENV / RUN):
        assert not ep.get("error") and ep["target"].split("@")[0].lower() == QWEN.lower() and ep["meta"]["gate_class"] == "stable"
        modal = bank[ep["item"]]["qwen_modal"]  # a raw sample (the pilot's episodes carry an older gate-normalized copy)
        out.append({"version": "nohint", "item": ep["item"], "modal": mod.norm(modal), "source": f"{ENV}/{RUN}:{ep['id']}",
                    "messages": ep["messages"][:1], "meta": ep["meta"] | {"qwen_modal": modal}})  # fmt: skip
    return sorted(out, key=lambda c: order[c["item"]])


def requests(out: Path):
    """The read requests for RUN's episodes: tail.jsonl (O-lens and NLA: the six end-of-prompt tokens, cutoff200-tail's
    positions) and all.jsonl (J++: the rendered prompt, every token)."""
    from transformers import AutoTokenizer  # uv run --with transformers --with jinja2

    tok = AutoTokenizer.from_pretrained(QWEN)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "tail.jsonl").open("w") as tail, (out / "all.jsonl").open("w") as every:
        for c in collect():
            text = tok.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text)["input_ids"]
            pos = list(range(len(ids) - 10, len(ids) - 4))  # ".", <|im_end|>, \n, <|im_start|>, assistant, \n; then the empty think block
            assert [tok.decode([ids[p]]) for p in pos][1:] == ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"], c["item"]
            row = {"id": c["source"], "env": ENV, "arm": "nohint", "item": c["item"], "priority": 0}
            tail.write(json.dumps(row | {"messages": c["messages"], "tools": None, "generation_prompt": True, "positions": pos}) + "\n")
            every.write(json.dumps(row | {"text": text, "positions": "all"}) + "\n")
    print("wrote", out / "tail.jsonl", out / "all.jsonl")


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None = None, strict: bool = False):
    """cases.jsonl plus whichever reads exist yet (O-lens, NLA, J++ restricted to JLENS_LAYERS); placebos come from the same files."""
    (out / "reads").mkdir(parents=True)
    for src, name in ((READS, "reads.jsonl"), (NLA_READS, "nla.jsonl")):
        if src.exists():
            (out / "reads" / name).write_text(src.read_text())
        print(src, "copied" if src.exists() else "missing")
    if JREADS.exists():
        rows = [r for rs in jpp_rows(JREADS).values() for r in rs]
        (out / "reads/jreads.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(JREADS, "copied" if JREADS.exists() else "missing")
    cs = collect()[: n_items or None]
    (out / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cs))
    print(len(cs), "cases")


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
    question, src, extra = base[0].messages[0]["content"], base[0].source, {}
    if condition == "transcript_inline":
        return inline.prompt(question, condition, False, MODEL_NAME, None), extra
    tool = condition.split("_")[0]
    by_source = inline.reads_by_source(Path(case["inputs"]) / "reads" / READ_FILES[tool])
    if condition.endswith("placebo_inline"):  # another item, seeded per (draw, case), the same item for all three tools
        src = random.Random(f"{seed}:{case['version']}:{case['item']}").choice(sorted(set(by_source) - {src}))
        extra = {"placebo_sources": [src]}
    rows = by_source[src]
    if tool == "nla":
        return inline.nla_prompt(question, False, MODEL_NAME, rows), extra
    if tool == "jlens":
        return inline.jlens_prompt(question, False, MODEL_NAME, sorted(rows, key=lambda r: r["token_index"])), extra
    return inline.prompt(question, "olens_inline", False, MODEL_NAME, rows), extra


def LINKS(row: dict) -> dict[str, dict]:
    env_run, _, ep = row["shown"][0].partition(":")
    env, run = env_run.split("/")
    return {"transcript": {"env": env, "run": run, "episode": ep}}


if __name__ == "__main__":
    requests(Path(__import__("sys").argv[1]))
