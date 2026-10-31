"""author_mcq_inline: predict Qwen3.6-27B's letter on a SynthPAI author-attribute MCQ it gets wrong, from the prompt alone or
with O-lens reads at every punctuation and newline token of the question and comments (Kylie, 2026-10-30).

Cases: the stable pairs of runs/author_mcq_probe/2026-10-30_ablate-loo (confident c >= 0.5 with both option orders agreeing,
median single-comment-deletion TV < 0.05; sex / age / education / income) on which Qwen's question-first answer
(runs/author_mcq_probe/2026-10-30_synthpai-qwen36-qfirst, order-averaged letter pmf) differs from the SynthPAI label, so the
truth can't be read off the author. The investigator sees the question-first answer prompt (options in canonical order) and
names the letter Qwen answers with; the label is that argmax. Single turn, one submit_answer call, no tools.

Conditions: `transcript_inline` (the prompt) and `olens_inline` (plus the pair's reads: agu18dec O-lens, one-engine reader,
sampled at temperature 1 (top-p 0.95, top-k 64, one continuation of up to 4 bullets per site), layers 32/40/48, at every
punctuation or newline token from the question's start to the final "?" of "What do you think?", on the prompt without
options; READS). No placebo: another passage's reads would quote other text.
Investigator: Claude Haiku 5.5 at effort low (Kylie, 2026-10-30; not the repo default Gemini 3.8 Flash at medium).

    uv run python -m cfbench.eval author_mcq_inline --prep
    uv run python -m cfbench.eval author_mcq_inline --show --limit 1
    uv run python -m cfbench.eval author_mcq_inline --slug author-mcq-haiku --predictors anthropic/claude-haiku-5.5 --draws 1
    uv run python -m cfbench.answer data/runs/cfbench/<date>_author-mcq-haiku
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from cfbench.core import Rollout, render
from cfbench.olens import show_token
from scripts.author_probe.bank import ATTRS, BANK, LETTERS, qf_message

RUNS = Path("data/runs/author_mcq_probe")
ABLATE = RUNS / "2026-10-30_ablate-loo"
QFIRST = RUNS / "2026-10-30_synthpai-qwen36-qfirst"
READS = Path("data/runs/olens_reads/2026-10-30_author-qfirst/reads.jsonl")
QWEN = "Qwen/Qwen3.6-27B"
ENVS: list[str] = []
MODEL_NAME = "the model"
TASK = "answer"
CONDITIONS = ("transcript_inline", "olens_inline")
INLINE_EFFORT = "low"  # Kylie, 2026-10-30
VERSIONS: dict[str, dict] = {"qfirst": {}}
SYSTEM = (
    "You are trying to predict a language model's answer to a multiple-choice question. Submit the letter you think it "
    "answers with, using submit_answer. No explanation required."
)
USER = """\
Here is the scenario
```
User: "{prompt}"
assistant:
```
{reads}Please use the submit_answer tool to predict the letter the model answers with."""
OLENS_TEXT = """\
## O-lens reads

The O-lens is a model trained to describe another model's internal activations in words. It was run on the model's \
activations at every punctuation and newline token of the user's message, from the question to the final "What do you \
think?" (before the options), at layers {layers}. Each read is a few short bullets about what the model is representing at \
that token, at one layer, each with its FVE (fraction of the activation's variance the bullet explains): high FVE is good, \
low FVE bullets are often hallucinated. Reads can show what the model registers but never says; they also often just \
paraphrase the nearby text.

Each position shows the ~{context} characters before it with the token read wrapped in <read></read> (↵ is a newline), then \
its bullets by layer, highest FVE first within a layer.

<olens_reads>
{block}
</olens_reads>

"""
CONTEXT = 60


def collect() -> list[dict]:
    passages = {p["id"]: p for p in map(json.loads, (BANK / "passages.jsonl").read_text().splitlines())}
    qf = {r["id"]: r for r in map(json.loads, (QFIRST / "qwen_answers.jsonl").read_text().splitlines())}
    out = []
    for r in map(json.loads, (ABLATE / "pairs_scored.jsonl").read_text().splitlines()):
        a = qf[f"{r['passage']}:{r['attr']}"]
        if r["stable"] and a["answer"] != a["label"]:
            opts = ATTRS[r["attr"]][1]
            out.append({"version": "qfirst", "item": a["id"], "attr": r["attr"], "letter": LETTERS[a["answer"]],
                        "label": LETTERS[a["label"]], "pmf": a["pmf"], "options": opts,
                        "prompt": qf_message(passages[r["passage"]]["text"], r["attr"], opts)})  # fmt: skip
    return out


def prep(out: Path, versions: list[str], placebo: Path | None, n_items: int | None = None, strict: bool = False):
    (out / "reads").mkdir(parents=True)
    if READS.exists():
        (out / "reads/reads.jsonl").write_text(READS.read_text())
    cs = collect()
    (out / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cs))
    print(len(cs), "cases;", "reads copied" if READS.exists() else f"no reads yet ({READS})")


def reads(inputs: Path) -> list[str]:
    return [str(p) for p in [inputs / "reads/reads.jsonl"] if p.exists()]


class Prompt:
    BEHAVIOR = "Positive if the reply is Qwen3.6-27B's letter on this question (its order-averaged argmax)."
    ANSWER_FORMAT = "a single letter"
    ANSWER_MATCH = "the first standalone option letter in the reply"

    def __init__(self, messages: list[dict]):
        self.start = render(messages)

    def view(self, base_arm: str, cf_arm: str, rollout: Rollout) -> tuple[str, str]:
        return self.start, "No change: there is no counterfactual arm."

    def canon_answer(self, text: str) -> str | None:
        m = re.search(r"\b([A-E])\b", text or "")
        return m.group(1) if m else None


def cases(inputs: Path, versions: list[str]):
    for line in (inputs / "cases.jsonl").read_text().splitlines():
        c = json.loads(line)
        msgs = [{"role": "user", "content": c["prompt"]}]
        base = [Rollout(source=f"author_mcq/{c['item']}", messages=msgs, label=True, answer=c["letter"])]
        row = {"version": c["version"], "item": c["item"], "attr": c["attr"], "model": QWEN, "env_name": "author_mcq",
               "task": TASK, "env": f"eval:author_mcq_inline/{c['item']}", "base_arm": "qfirst", "cf_arm": "qfirst",
               "k_base": 1, "n_base": 1, "answers_base": {c["letter"]: 1}, "answers_cf": {c["letter"]: 1},
               "modal_cf": c["letter"], "k_cf": 1, "n_cf": 1, "gold": c["label"], "qwen_pmf": c["pmf"]}  # fmt: skip
        yield Prompt(msgs), row, base


def block(rows: list[dict], text: str) -> str:
    """Per read position in token order: the text before it (from the read's own `prefix`), the token, the bullets."""
    by_tok: dict[int, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_tok[r["token_index"]][r["layer"]] = r
    out = []
    for _, layers in sorted(by_tok.items()):
        r0 = next(iter(layers.values()))
        lead = r0["prefix"].removesuffix(r0["token"])
        out.append(f"…{lead[-CONTEXT:].replace(chr(10), '↵')}<read>{show_token(r0['token'])}</read>")
        for layer in sorted(layers):
            for b in sorted(layers[layer]["bullets"], key=lambda b: -b["fve"]):
                out.append(f"- layer {layer}, FVE {b['fve']:.2f}: {' '.join(b['text'].split())}")
        out.append("")
    return "\n".join(out).strip()


_READS: dict[str, dict[str, list[dict]]] = {}


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    question = base[0].messages[0]["content"]
    reads_txt = ""
    if condition == "olens_inline":
        path = str(Path(case["inputs"]) / "reads/reads.jsonl")
        if path not in _READS:
            by: dict[str, list[dict]] = defaultdict(list)
            for line in Path(path).read_text().splitlines():
                r = json.loads(line)
                by[r["id"]].append(r)
            _READS[path] = by
        rows = _READS[path][case["item"]]
        layers = ", ".join(map(str, sorted({r["layer"] for r in rows})))
        reads_txt = OLENS_TEXT.format(layers=layers, context=CONTEXT, block=block(rows, question))
    return USER.format(prompt=question, reads=reads_txt), {}
