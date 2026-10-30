"""author_mcq_drop_inline: if we remove this one comment, does Qwen3.6-27B become much less sure of its answer? (Kylie,
2026-10-30: "find P(argmax) decrease significantly ... sure to unsure is fine".)

Truth from runs/author_mcq_probe/2026-10-30_ablate-loo, the leave-one-comment-out run, which used the ORIGINAL format (the
passage, then the first-person question, options and "Answer with the letter only." in one user turn; both option orders,
pmf averaged). With p0 = Qwen's order-averaged probability on its answer and p1 the same without the comment, drop = p0 -
p1; a case is positive when drop > max(0.15, 5 x the pair's median drop over its comments). Cases: the stable pairs
(confident c >= 0.5 with both orders agreeing, median single-comment TV < 0.05) on which Qwen's answer differs from the
SynthPAI label; every positive comment, plus as many negative comments from the same pair (seeded sample). TASK "claim":
P(the drop is significant), truth 0/1; p0, p1, drop and TV kept as fields.

The investigator sees the answer prompt (canonical option order), Qwen's answer with p0, and the comment to be removed;
model name hidden. `olens_inline` adds O-lens reads of the original-format prompt (READS: every punctuation and newline
token of the comments and the question, up to its "?"; requests from `requests` below; not collected yet). Investigator:
Claude Haiku 5.5 at effort low (Kylie, 2026-10-30).

    uv run python -m cfbench.eval author_mcq_drop_inline --prep
    uv run python -m cfbench.eval author_mcq_drop_inline --show
    uv run --with transformers --with jinja2 python -m cfbench.evals.author_mcq_drop_inline.eval OUT.jsonl   # read requests
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from cfbench.core import Rollout, render
from cfbench.olens import show_token
from scripts.author_probe.bank import ATTRS, BANK, LETTERS, MODEL, answer_message

ABLATE = Path("data/runs/author_mcq_probe/2026-10-30_ablate-loo")
READS = Path("data/runs/olens_reads/2026-10-30_author-origfmt/reads.jsonl")
QWEN = "Qwen/Qwen3.6-27B"
ENVS: list[str] = []
MODEL_NAME = "the model"
TASK = "claim"
CONDITIONS = ("transcript_inline", "olens_inline")
INLINE_EFFORT = "low"  # Kylie, 2026-10-30
VERSIONS: dict[str, dict] = {"loo": {}}
SEED = 0
CONTEXT = 60
USER = """\
Here is the scenario. A model was given this message and answered with a single letter:
```
User: "{prompt}"
assistant: {answer}
```
The model's probability on {answer} was {p0:.2f} (averaged over this option order and the reversed one).

Now suppose this one comment is deleted from the message, with everything else (the other comments, the question, the \
options and their order) unchanged:
```
{comment}
```

{reads}## Your task

Call the deletion significant if the model's probability on {answer} falls by more than max(0.15, 5 × the median fall \
from deleting any one comment of this message), i.e. by more than 0.15 for most messages. Deleting a typical comment barely \
moves it. What is the probability that deleting this comment is significant? Submit it with submit_prediction."""
OLENS_TEXT = """\
## O-lens reads

The O-lens is a model trained to describe another model's internal activations in words. It was run on the model's \
activations on the original message (with the comment) at every punctuation and newline token of the comments and of the \
question up to its "?" (before the options), at layers {layers}. Each read is a few short bullets about what the model is \
representing at that token, at one layer, each with its FVE (fraction of the activation's variance the bullet explains): \
high FVE is good, low FVE bullets are often hallucinated. Reads can show what the model registers but never says; they also \
often just paraphrase the nearby text. Positions inside the comment to be deleted are marked [IN THE COMMENT].

Each position shows the ~{context} characters before it with the token read wrapped in <read></read> (↵ is a newline), then \
its bullets by layer, highest FVE first within a layer.

<olens_reads>
{block}
</olens_reads>

"""


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines()]


def collect() -> list[dict]:
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    rng, out = random.Random(SEED), []
    for r in load(ABLATE / "pairs_scored.jsonl"):
        if not r["stable"] or r["answer"] == r["label"]:
            continue
        drops = [-d["qwen_shift"] for d in r["deletions"]]
        thr = max(0.15, 5 * sorted(drops)[len(drops) // 2])
        pos = [d for d, x in zip(r["deletions"], drops, strict=True) if x > thr]
        neg = [d for d, x in zip(r["deletions"], drops, strict=True) if x <= thr]
        opts, q = ATTRS[r["attr"]][1], ATTRS[r["attr"]][0]
        p0 = r["pmf"][r["answer"]]
        for d in pos + rng.sample(neg, min(len(pos), len(neg))):
            out.append({"version": "loo", "item": f"{r['passage']}:{r['attr']}#{d['i']}", "pair": f"{r['passage']}:{r['attr']}",
                        "attr": r["attr"], "comment": d["text"], "positive": -d["qwen_shift"] > thr, "threshold": thr,
                        "answer": LETTERS[r["answer"]], "label": LETTERS[r["label"]], "p0": p0, "p1": d["pmf"][r["answer"]],
                        "drop": -d["qwen_shift"], "tv": d["tv"], "flip": not d["argmax_kept"], "pmf": r["pmf"],
                        "pmf_without": d["pmf"], "prompt": answer_message(passages[r["passage"]]["text"], q, opts)})  # fmt: skip
    return out


def prep(out: Path, versions: list[str], placebo: Path | None, n_items: int | None = None, strict: bool = False):
    (out / "reads").mkdir(parents=True)
    if READS.exists():
        (out / "reads/reads.jsonl").write_text(READS.read_text())
    cs = collect()
    (out / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cs))
    print(len(cs), "cases,", sum(c["positive"] for c in cs), "positive;", "reads copied" if READS.exists() else f"no reads yet ({READS})")


def reads(inputs: Path) -> list[str]:
    return [str(p) for p in [inputs / "reads/reads.jsonl"] if p.exists()]


class Prompt:
    BEHAVIOR = (
        "Positive if deleting the comment lowers Qwen3.6-27B's probability on its answer by more than max(0.15, 5 x the median drop)."
    )

    def __init__(self, messages: list[dict]):
        self.start = render(messages)

    def view(self, base_arm: str, cf_arm: str, rollout: Rollout) -> tuple[str, str]:
        return self.start, "the comment is deleted"


def cases(inputs: Path, versions: list[str]):
    for c in load(inputs / "cases.jsonl"):
        msgs = [{"role": "user", "content": c["prompt"]}, {"role": "assistant", "content": c["answer"]}]
        base = [Rollout(source=f"author_mcq/{c['pair']}", messages=msgs, label=False)]
        row = {"version": c["version"], "item": c["item"], "pair": c["pair"], "attr": c["attr"], "model": QWEN,
               "env_name": "author_mcq", "task": TASK, "env": f"eval:author_mcq_drop_inline/{c['item']}", "base_arm": "full",
               "cf_arm": "drop", "k_base": 0, "n_base": 1, "k_cf": int(c["positive"]), "n_cf": 1, "comment": c["comment"],
               "answer": c["answer"], "gold": c["label"], "p0": c["p0"], "p1": c["p1"], "drop": c["drop"],
               "threshold": c["threshold"], "tv": c["tv"], "flip": c["flip"]}  # fmt: skip
        yield Prompt(msgs), row, base


def block(rows: list[dict], comment: str) -> str:
    by_tok: dict[int, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_tok[r["token_index"]][r["layer"]] = r
    flat = " ".join(comment.split())
    out = []
    for _, layers in sorted(by_tok.items()):
        r0 = next(iter(layers.values()))
        lead = r0["prefix"].removesuffix(r0["token"])
        inside = len(tail := " ".join((lead + r0["token"]).split())[-25:]) >= 10 and tail in flat
        out.append(
            f"…{lead[-CONTEXT:].replace(chr(10), '↵')}<read>{show_token(r0['token'])}</read>" + (" [IN THE COMMENT]" if inside else "")
        )
        for layer in sorted(layers):
            for b in sorted(layers[layer]["bullets"], key=lambda b: -b["fve"]):
                out.append(f"- layer {layer}, FVE {b['fve']:.2f}: {' '.join(b['text'].split())}")
        out.append("")
    return "\n".join(out).strip()


_READS: dict[str, dict[str, list[dict]]] = {}


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    msgs = base[0].messages
    reads_txt = ""
    if condition == "olens_inline":
        path = str(Path(case["inputs"]) / "reads/reads.jsonl")
        if path not in _READS:
            by: dict[str, list[dict]] = defaultdict(list)
            for r in load(Path(path)):
                by[r["id"]].append(r)
            _READS[path] = by
        rows = _READS[path][case["pair"]]
        layers = ", ".join(map(str, sorted({r["layer"] for r in rows})))
        reads_txt = OLENS_TEXT.format(layers=layers, context=CONTEXT, block=block(rows, case["comment"]))
    prompt = USER.format(prompt=msgs[0]["content"], answer=msgs[1]["content"], p0=case["p0"], comment=case["comment"], reads=reads_txt)
    return prompt, {}


PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # scripts/olens_read_requests.py's stop tokens


def requests(out: Path):
    """olens_read_requests.py rows for the cases' pairs: the original-format answer prompt cut after the question's "?"
    (before the options), read at every punctuation or newline token of the comments and the question."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    pairs = {c["pair"]: c for c in collect()}
    with out.open("w") as f:
        for pair, c in pairs.items():
            user = c["prompt"][: c["prompt"].index("?\n", len(c["prompt"]) - 400) + 1]  # passage + "\n\n" + question
            msgs = [{"role": "user", "content": user}]
            text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            s0 = text.index(user)
            pos = [
                i
                for i, (a, b) in enumerate(enc["offset_mapping"])
                if a >= s0
                and b <= s0 + len(user)
                and (PUNCT.match(t := tok.decode([enc["input_ids"][i]])) or (not t.strip() and "\n" in t))
            ]
            row = {"id": pair, "messages": msgs, "tools": None, "prompt_ids": enc["input_ids"], "positions": pos}
            f.write(json.dumps(row | {"env": "author_mcq", "arm": c["attr"]}) + "\n")
    print("wrote", len(pairs), "requests to", out)


if __name__ == "__main__":
    requests(Path(sys.argv[1]))
