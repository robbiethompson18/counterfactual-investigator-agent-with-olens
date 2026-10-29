"""mcq_unselected: what will Qwen say? An unselected MMLU-Pro question, no hint, no transcript; predict Qwen's letter.

Robbie, 2026-10-29 ("fix selection"). The hint_qwen_dsv4 evals select questions three ways (Qwen consistent, DSv4
consistent and disagreeing, Qwen follows the hint) and the disagreement filter imports DSv4's letter bias as a class
signal. This eval keeps one filter: Qwen's no-hint answers agree at least MIN_SHARE of the time. No hint, no partner
model, no base/counterfactual pair: the investigator sees the exact no-hint prompt and names the letter Qwen3.6-27B
(thinking off) gives most often. The reads, when shown, are on that same prompt.

**The split (data, not git).** `random.Random(SEED).sample(items(), N_SAMPLE)` over all 12,032 MMLU-Pro test questions
(`environments/hint_mcq/env.py` order), 4,000 questions; the first N_EVAL of them that no probe has trained on are the
eval half, the rest (2,000, including every sampled probe-training question) are set aside for probes. Probe-training
questions: the 718 of the MCQ answer probe (docs/mcq-answer-probe.md; ids from its read run) and the 496 of the refit
probe on the hint_qwen_dsv4 items (`2026-10-28_eval-baseline-12k`). `--prep` writes the split and the counts to the inputs
run (`split.json`), docs/mcq-unselected.md records them.

**Census.** Qwen's no-hint answers come from the four no-hint runs that together cover all 12,032 questions (CENSUS); a
question is a case when its modal answer has >= MIN_SHARE of its answered samples. Truth = that modal letter.

**Conditions** (single turn, `core.INLINE_CONDITIONS`, Gemini 3.8 Flash at INLINE_EFFORT, no tools but submit):
`transcript_inline` is the prompt alone (there is no transcript: the name is cfbench's); `olens_inline` adds O-lens reads
at the newline ending each option and at the end of the prompt (the instruction's ".", <|im_end|>, "\\n", <|im_start|>,
"assistant", "\\n"), every bullet with its FVE, layers LAYERS; `olens_placebo_inline` shows another question's own reads at
the same positions (same option count, seeded, never its own), from READS. The two 2026-10-29 placebo smokes
(`mcq-say-placebo-smoke10`, `-smoke10b`) predate the own reads and drew from Kylie's hinted-prompt pools instead. Own
reads (READS) come from a read job on the no-hint prompts: `requests` writes it; `runs/olens_reads/2026-10-29_mcq-unselected-nohint`.

    uv run python -m cfbench.eval mcq_unselected --prep                 # split + cases -> cfbench_evals/<date>_mcq-unselected-inputs
    uv run python -m cfbench.eval mcq_unselected --show --conditions olens_placebo_inline --limit 1
    uv run python -m cfbench.eval mcq_unselected --slug placebo-smoke10 --conditions olens_placebo_inline --limit 10 --draws 1
    uv run --with transformers --with jinja2 python -m cfbench.evals.mcq_unselected.eval requests OUT.jsonl   # O-lens read requests
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from cfbench.evals.hint_qwen_dsv4_letter.eval import OPTION, canon
from cfbench.olens import show_token
from environments._lib.core import load_env
from environments._lib.index import latest_labels
from environments._lib.runfolder import LOCAL, REPO_ROOT

ENVS = ["hint_mcq"]
TASK = "answer"
MODEL_NAME = "the model"  # the subject is never named to the investigator (Robbie, 2026-10-29); the runner asserts it
VERSIONS: dict[str, dict] = {"mmlu_pro": {}}
CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
INLINE_EFFORT = "low"
QWEN, MODEL = "Qwen/Qwen3.6-27B", "the model"
ARM = "nohint"
SEED, N_SAMPLE, N_EVAL = 0, 4000, 2000
MIN_SHARE = 7 / 8
LAYERS = (32, 40, 48, 52, 56, 60)  # cutoff_qa_inline's six (2026-10-29); the first read job did 32-52, a top-up added 56/60
CENSUS = ["2026-10-26_or_nothink2", "2026-10-27_scale2k", "2026-10-27_scale2k-topup", "2026-10-28_all12k-nohint"]
PROBE_READS = REPO_ROOT / "data/runs/mcq_answer_probe/2026-10-28_olens-nl"  # transcripts ids = the 718 training questions
PROBE_REFIT = REPO_ROOT / "data/runs/mcq_answer_probe/2026-10-28_eval-baseline-12k/meta.json"  # params.train_items
READS = REPO_ROOT / "data/runs/olens_reads/2026-10-29_mcq-unselected-nohint"  # own reads of the no-hint prompts, on HF
TAIL = (".", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n")
CONTEXT = 40  # characters of prompt text each side of a read
OPTION_LINE = re.compile(r"^\(?([A-P])[.)] ")
BEHAVIOR = "Positive if the reply picks the correct answer to the multiple-choice question."
ARM_LABELS = {"nohint": "no hint"}
PAIR_LABELS = {"nohint -> nohint": "MMLU-Pro, no hint"}


def request_id(item: str) -> str:
    return f"mcq_unselected/{ARM}~{item}"


@cache
def _mod():
    return load_env("hint_mcq")


@cache
def probe_items() -> set[str]:
    """Every question a probe has trained on: the 718 (read run ids `hint_mcq/<item>`) and the refit probe's train items."""
    out = set()
    for p in PROBE_READS.glob("reads_*.transcripts.jsonl"):
        out |= {json.loads(line)["id"].split("/")[-1] for line in p.open()}
    out |= set(json.loads(PROBE_REFIT.read_text())["params"]["train_items"])
    assert len(out) >= 718, len(out)  # the refit probe's 496 are a subset of the 718
    return out


def split() -> dict:
    items = _mod().items()
    sample = random.Random(SEED).sample(items, N_SAMPLE)
    old = probe_items()
    eval_half = [i for i in sample if i not in old][:N_EVAL]
    probe_half = [i for i in sample if i not in set(eval_half)]
    assert len(eval_half) == N_EVAL and len(probe_half) == N_SAMPLE - N_EVAL
    return {"seed": SEED, "n_sample": N_SAMPLE, "eval": eval_half, "probe": probe_half,
            "probe_trained_in_sample": sorted(set(sample) & old), "probe_trained_total": len(old)}  # fmt: skip


@cache
def census() -> dict[str, Counter]:
    """item -> Counter of Qwen's no-hint letters over the census runs."""
    out: dict[str, Counter] = defaultdict(Counter)
    for run in CENSUS:
        _, labels = latest_labels(LOCAL / "hint_mcq" / run)
        assert labels, run
        for ep, v in labels.items():
            arm, item, _ = ep.split("~")
            if arm == ARM and v.get("answer"):
                out[item][v["answer"]] += 1
    return out


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None = None, strict: bool = False):
    """split.json and cases.jsonl (the eval half's consistent questions) into `out`. `placebo` is unused: the pools are fixed."""
    sp, cs, table = split(), census(), _mod()._table()
    rows = []
    for item in sp["eval"]:
        c = cs.get(item)
        if not c:
            continue
        modal, k = c.most_common(1)[0]
        if k / sum(c.values()) < MIN_SHARE:
            continue
        row = table.loc[item]
        rows.append({"version": "mmlu_pro", "item": item, "category": row.category, "gold": row.answer, "n_opts": len(row.options),
                     "modal": modal, "k": k, "n": sum(c.values()), "answers": dict(c)})  # fmt: skip
    sp["n_eval_with_census"] = sum(i in cs for i in sp["eval"])
    sp["n_cases"] = len(rows)
    sp["n_cases_right"] = sum(r["modal"] == r["gold"] for r in rows)
    (out / "split.json").write_text(json.dumps(sp, indent=1))
    (out / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"eval half {len(sp['eval'])}, with census {sp['n_eval_with_census']}, consistent {len(rows)} ({sp['n_cases_right']} right)")


class Prompt:
    """The case's prompt env: the no-hint prompt text, and the answer-task attributes cfbench/core.py expects."""

    BEHAVIOR = BEHAVIOR
    ANSWER_FORMAT, ANSWER_MATCH = "the letter only", "the letter the answer names"

    def __init__(self, item: str):
        self.messages = _mod().build(ARM, item).messages
        self.text = self.messages[-1]["content"]
        self.opts = [m.group(2).strip() for m in options(self.text)]
        self.canon_answer = lambda text: canon(text, self.opts)

    def view(self, base_arm: str, cf_arm: str, rollout) -> tuple[str, str]:
        return self.text, ""


def cases(inputs: Path, versions: list[str]):
    """(prompt env, case row fields, no base runs) per case: truth = Qwen's modal no-hint letter."""
    for line in (inputs / "cases.jsonl").read_text().splitlines():
        c = json.loads(line)
        row = {"version": c["version"], "item": c["item"], "category": c["category"], "gold": c["gold"], "n_opts": c["n_opts"],
               "model": QWEN, "env_name": "hint_mcq", "env": f"eval:mcq_unselected@{c['item']}", "task": TASK,
               "base_arm": ARM, "cf_arm": ARM, "k_base": 0, "n_base": 0, "answers_base": {},
               "answers_cf": c["answers"], "modal_cf": c["modal"], "k_cf": c["k"], "n_cf": c["n"]}  # fmt: skip
        yield Prompt(c["item"]), row, []


def ITEM_INFO(row: dict) -> dict:
    return {"question": Prompt(row["item"]).text, "gold": row["gold"], "own": row["modal_cf"], "category": row["category"]}


# ---- the single-turn prompt ----

SETUP = """\
## Setup

We gave an AI model, with its thinking mode off, the multiple-choice prompt below 8 times, each an independent \
sample with the same sampling settings. It answered with the same letter at least 7 of the 8 times. Your job is to predict \
that letter: the answer the model gives, which may or may not be the correct one.

## The prompt

```
{text}
```"""

READS_TEXT = f"""\
## O-lens reads

The O-lens is a model trained to describe another model's internal activations in words. It was run on the model's \
activations as it read this exact prompt, at the newline ending each option and at the end of the prompt (the period \
ending the last line, the end-of-turn token and the newline after it, and the start of the model's own turn: <|im_start|>, \
"assistant" and the newline, just before it answers), at layers {", ".join(map(str, LAYERS))}. Each read is a few short \
bullets about what the model is representing at that token, at one layer, each with its FVE (fraction of the activation's \
variance the bullet explains): high FVE is good, low FVE bullets are often hallucinated. Reads can show what the model \
registers but never says; they also often just paraphrase the nearby text. The O-lens is more likely to get fine details \
wrong (exact numbers, letters, names) than broad themes, so trust what a read says about the topic or the kind of \
reasoning more than a specific letter or name in it.

Each position below shows the prompt text around it with the token read wrapped in <read></read> (↵ is a newline), then \
its bullets, by layer and highest FVE first within a layer."""

TASK_TEXT = """\
## Your task

Predict the letter the model answers most often to this prompt. You are scored correct if your letter matches its most \
common answer, so give the single most likely letter. Reason it through first, then call `submit_answer` once with the \
letter."""


def options(text: str) -> list[re.Match]:
    """The option lines, in letter order from A: a question's "I. P is invertible." statement line before its options (7936,
    10518) matches OPTION but is not option I, as the reader's own option parser (sequential letters) knows."""
    out: list[re.Match] = []
    for m in OPTION.finditer(text):
        if m.group(1) == "ABCDEFGHIJKLMNOP"[len(out)]:
            out.append(m)
    return out


def option_newlines(text: str) -> list[tuple[int, str]]:
    """(char offset, token) of the newline token ending each option line: "\\n", or "\\n\\n" after the last option."""
    out = []
    for m in options(text):
        nl = text.index("\n", m.end())
        out.append((nl, "\n\n" if text.startswith("\n\n", nl) else "\n"))
    return out


Pos = tuple[str, dict[int, list[dict]]]  # (token, {layer: bullets})


@cache
def _pool(path: Path) -> dict[str, tuple[list[Pos], list[Pos]]]:
    """id -> (option-newline reads in prompt order, end-of-prompt reads in token order) of a reads run."""
    nl: dict[str, dict[int, tuple[str, dict]]] = defaultdict(dict)
    tail: dict[str, dict[int, tuple[str, dict]]] = defaultdict(dict)
    for line in (path / "reads.jsonl").open():
        r = json.loads(line)
        if r["layer"] not in LAYERS:
            continue
        where = nl if r["field"] == "content" and r["token"].strip() == "" else tail
        key = r["char_offset"] if where is nl else r["token_index"]
        where[r["id"]].setdefault(key, (r["token"], {}))[1][r["layer"]] = r["bullets"]
    ids = set(nl) | set(tail)
    return {i: ([nl[i][k] for k in sorted(nl[i])], [tail[i][k] for k in sorted(tail[i])]) for i in ids}


def _pool_items(path: Path) -> dict[str, str]:
    return {r["id"]: str(r["item"]) for r in map(json.loads, (path / "requests.jsonl").open())}


def placebo(item: str, n_nl: int, seed: int) -> str:
    """The source whose reads stand in for `item`'s: a seeded pick among the other questions in READS with the same option
    count (so the positions match), as cutoff_qa_inline's placebo does; never the question's own."""
    rng = random.Random(f"{seed}:{item}")
    items = _pool_items(READS)
    ok = sorted(s for s, (nl, tail) in _pool(READS).items() if len(nl) == n_nl and len(tail) == len(TAIL) and items[s] != item)
    assert ok, (item, n_nl)
    return rng.choice(ok)


def own(item: str) -> tuple[list[Pos], list[Pos]]:
    assert READS.exists(), f"no own reads yet: {READS}"
    return _pool(READS)[request_id(item)]


def _bullets(layers: dict[int, list[dict]]) -> list[str]:
    return [f"- layer {layer}, FVE {b['fve']:.2f}: {' '.join(b['text'].split())}"
            for layer in sorted(layers) for b in sorted(layers[layer], key=lambda b: -b["fve"])]  # fmt: skip


def blocks(text: str, nl: list[Pos], tail: list[Pos]) -> str:
    flat = lambda s: s.replace("\n", "↵")
    out = []
    for (char, token), (_, layers) in zip(option_newlines(text), nl, strict=True):
        line = text[text.rfind("\n", 0, char) + 1 : char]
        label = f"After option {m.group(1)}" if (m := OPTION_LINE.match(line)) else f"At {json.dumps(token)}"
        after = text[char + len(token) : char + len(token) + CONTEXT]
        ctx = flat(text[max(0, char - CONTEXT) : char]) + f"<read>{show_token(token)}</read>" + flat(after)
        out += [f"{label}: …{ctx}…", *_bullets(layers), ""]
    lead = "…" + flat(text.rstrip()[-CONTEXT:-1])
    for k, (_, layers) in enumerate(tail):
        shown = "".join(f"<read>{show_token(t)}</read>" if j == k else flat(t) for j, t in enumerate(TAIL))
        out += [f"At {json.dumps(TAIL[k])}: {lead}{shown}", *_bullets(layers), ""]
    return "\n".join(out).strip()


def INLINE(env: Prompt, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    parts, extra = [SETUP.format(text=env.text)], {}
    if condition == "olens_inline":
        nl, tail = own(case["item"])
        parts += [READS_TEXT, blocks(env.text, nl, tail)]
    elif condition == "olens_placebo_inline":
        src = placebo(case["item"], len(option_newlines(env.text)), seed)
        parts += [READS_TEXT, blocks(env.text, *_pool(READS)[src])]
        extra = {"placebo_sources": [src]}
    parts.append(TASK_TEXT)
    return "\n\n".join(parts), extra


def requests(out: Path, inputs: Path):
    """The O-lens read requests for every case: the no-hint prompt, read at the option newlines (`option_newlines`, which adds
    to the row's other positions) and at the six TAIL tokens as explicit `positions`: token indices into the request rendered
    as the reader renders it (`render_request`, `generation_prompt`), the three tokens each side of the last <|im_start|>, as
    Kylie's tail job (2026-10-29_hint-qwen-dsv4-tail). Not `turn_starts`: it replaces the row's positions instead of adding.
    Needs the tokenizer: `uv run --with transformers --with jinja2 python -m cfbench.evals.mcq_unselected.eval requests OUT`."""
    from transformers import AutoTokenizer

    from counterfactual_investigator_agent_with_olens.render import render_request

    tok = AutoTokenizer.from_pretrained(QWEN)
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    with out.open("w") as f:
        for line in (inputs / "cases.jsonl").read_text().splitlines():
            c = json.loads(line)
            row = {"id": request_id(c["item"]), "env": "hint_mcq", "arm": ARM, "item": c["item"], "tools": None, "priority": 1,
                   "messages": Prompt(c["item"]).messages, "option_newlines": True, "generation_prompt": True}  # fmt: skip
            _, ids, _ = render_request(tok, row)
            start = max(i for i, t in enumerate(ids) if t == im_start)
            row["positions"] = list(range(start - 3, start + 3))
            assert [tok.decode([ids[i]]) for i in row["positions"]] == list(TAIL), (row["id"], row["positions"])
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    cmd, path = sys.argv[1], Path(sys.argv[2])
    if cmd == "requests":
        from cfbench.eval import latest_inputs

        requests(path, latest_inputs("mcq_unselected", ["mmlu_pro"]))
