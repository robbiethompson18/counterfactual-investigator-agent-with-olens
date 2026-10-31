"""supergpqa_inline: what letter will Qwen say? 3,000 SuperGPQA cases, no hint, seven single-turn conditions (Robbie, 2026-10-29).

mcq_unselected's question on the harder pool (environments/supergpqa: 10,000 non-calculation SuperGPQA items, Qwen3.6-27B
thinking off, 8 samples each, 50% right, docs/supergpqa-inline.md). The investigator sees the prompt Qwen was asked, never
its reply, and predicts the letter Qwen gives in one turn with one submit_answer call. Truth is Qwen's modal letter.

Cases (mcq_unselected's rule, one rule for both pools): a question is consistent when at least 7 of its 8 samples gave the
same bare letter, counted over every sample (a multi-letter or worked reply counts against). `split()` draws N_EVAL of the
consistent questions with SEED as the eval set; every other consistent question is the probe / validation pool (train probes
there, never on the eval set). Written once by `--prep` to the inputs run's split.json.

Conditions (core.INLINE_CONDITIONS): `transcript_inline` (the prompt), `olens_inline` / `olens_placebo_inline` (O-lens bullets
with FVE), `nla_inline` / `nla_placebo_inline` (all-layer NLA readouts with the critic's FVE), `jlens_inline` /
`jlens_placebo_inline` (the J++ lens's top tokens at every prompt token). Own reads are at the newline ending each option and
at the ten end-of-prompt tokens of the thinking-off render (the instruction's ".", <|im_end|>, "\\n", <|im_start|>,
"assistant", "\\n", then the empty think block <think>, "\\n\\n", </think>, "\\n\\n": the token just before the letter;
mcq_unselected stopped before the think block). A placebo is another case's reads with the same option count, seeded per
(draw, case), presented as this case's; the same rule for all three lenses.

    uv run python -m cfbench.eval supergpqa_inline --prep
    uv run --with transformers --with jinja2 python -m cfbench.evals.supergpqa_inline.eval requests OUT_DIR   # olens + jpp requests
    uv run python -m cfbench.eval supergpqa_inline --show --conditions olens_inline --limit 1
    uv run python -m cfbench.eval supergpqa_inline --slug supergpqa-haiku-3k --predictors anthropic/claude-haiku-5.5 --draws 1
    uv run python -m graphs.supergpqa_inline
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from cfbench.evals.cutoff_qa_inline import inline
from cfbench.evals.hint_qwen_dsv4_letter.eval import OPTION, canon
from cfbench.olens import show_token
from environments._lib.core import load_env
from environments._lib.runfolder import LOCAL, REPO_ROOT

ENV, ARM = "supergpqa", "base"
ENVS = [ENV]
QWEN, MODEL_NAME = "Qwen/Qwen3.6-27B", "the model"
TASK = "answer"
ERROR_RATE = True  # the env viewer shows this eval as error rate, lower is better (Kylie, 2026-10-30)
VERSION = "sg3k"
VERSIONS = {VERSION: {"n": 3000}}
CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline", "nla_inline", "nla_placebo_inline", "jlens_inline",
              "jlens_placebo_inline")  # fmt: skip
INLINE_EFFORT = "medium"
SEED, N_EVAL = 0, 3000
MIN_SHARE = 7 / 8
CENSUS = {"2026-10-29_n8": "measure@59dea5c89d42.jsonl", "2026-10-29_scale9k": "measure@ea679a67ad3f.jsonl"}  # named, not newest
LAYERS = (32, 40, 48, 52, 56, 60)  # O-lens and J++ (cutoff_qa_inline's six)
NLA_LAYERS = (32, 40, 48)
RUN = "2026-10-29_supergpqa-3k"  # every reader's run slug
READS = REPO_ROOT / "data/runs/olens_reads" / RUN / "reads.jsonl"
NLA_READS = REPO_ROOT / "data/runs/nla_reads" / RUN / "reads.jsonl"
JREADS = REPO_ROOT / "data/runs/jpp_reads" / RUN / "reads.jsonl"
TAIL = (".", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n")
CONTEXT = 40
OPTION_LINE = re.compile(r"^\(?([A-P])[.)] ")
BEHAVIOR = "Positive if the reply picks the correct answer to the multiple-choice question."
ARM_LABELS = {ARM: "no hint"}
PAIR_LABELS = {f"{ARM} -> {ARM}": "SuperGPQA, no hint"}


def request_id(item: str) -> str:
    return f"{ENV}/{ARM}~{item}"


@cache
def _mod():
    return load_env(ENV)


@cache
def census() -> dict[str, Counter]:
    """item -> Counter of Qwen's letters over both runs' named labels files (None = a reply that isn't one bare letter)."""
    out: dict[str, Counter] = defaultdict(Counter)
    for run, labels in CENSUS.items():
        for r in map(json.loads, (LOCAL / ENV / run / "labels" / labels).open()):
            arm, item, _ = r["episode"].split("~")
            if arm == ARM:
                out[item][r["values"].get("answer")] += 1
    return out


def consistent(c: Counter) -> tuple[str, int] | None:
    letters = Counter({a: k for a, k in c.items() if a})
    if not letters:
        return None
    modal, k = letters.most_common(1)[0]
    return (modal, k) if k / sum(c.values()) >= MIN_SHARE else None


def split() -> dict:
    """The eval set: N_EVAL consistent questions drawn with SEED, in item order; the probe pool: every other consistent one."""
    cs = census()
    ok = [i for i in _mod().items() if i in cs and consistent(cs[i])]
    eval_set = set(random.Random(SEED).sample(ok, N_EVAL))
    return {"seed": SEED, "rule": "modal letter in >= 7/8 of all samples", "consistent": len(ok), "questions": len(cs),
            "eval": [i for i in ok if i in eval_set], "probe": [i for i in ok if i not in eval_set], "labels": CENSUS}  # fmt: skip


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None = None, strict: bool = False):
    """split.json and cases.jsonl into `out`; reads are read from the readers' runs (READS, NLA_READS, JREADS), not copied."""
    sp, cs, table = split(), census(), _mod()._table()
    rows = []
    for item in sp["eval"][: n_items or None]:
        ok = consistent(cs[item])
        assert ok, item
        modal, k = ok
        row = table.loc[item]
        rows.append({"version": VERSION, "item": item, "discipline": row.discipline, "field": row.field, "difficulty": row.difficulty,
                     "gold": row.answer_letter, "n_opts": len(row.options), "modal": modal, "k": k, "n": sum(cs[item].values()),
                     "answers": {a: n for a, n in cs[item].items() if a}, "unscored": cs[item][None]})  # fmt: skip
    right = sum(r["modal"] == r["gold"] for r in rows)
    sp["eval_counts"] = {"cases": len(rows), "cases_right": right}
    (out / "split.json").write_text(json.dumps(sp, indent=1))
    (out / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(cs)} questions, {sp['consistent']} consistent; eval {len(rows)} cases ({right} right), probe pool {len(sp['probe'])}")


def reads(inputs: Path) -> list[str]:
    return []


class Prompt:
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
    """(prompt env, case row fields, no base runs) per case: truth = Qwen's modal letter."""
    for line in (inputs / "cases.jsonl").read_text().splitlines():
        c = json.loads(line)
        row = {"version": c["version"], "item": c["item"], "discipline": c["discipline"], "field": c["field"],
               "difficulty": c["difficulty"], "gold": c["gold"], "n_opts": c["n_opts"],
               "model": QWEN, "env_name": ENV, "env": f"eval:supergpqa_inline@{c['item']}", "task": TASK,
               "base_arm": ARM, "cf_arm": ARM, "k_base": 0, "n_base": 0, "answers_base": {},
               "answers_cf": c["answers"], "modal_cf": c["modal"], "k_cf": c["k"], "n_cf": c["n"]}  # fmt: skip
        yield Prompt(c["item"]), row, []


def ITEM_INFO(row: dict) -> dict:
    return {"question": Prompt(row["item"]).text, "gold": row["gold"], "own": row["modal_cf"], "discipline": row["discipline"]}


def LINKS(row: dict) -> dict[str, dict]:
    run = "2026-10-29_n8" if row["item"] in set(_mod().items()[:1000]) else "2026-10-29_scale9k"
    return {"transcript": {"env": ENV, "run": run, "episode": f"{ARM}~{row['item']}~0"}}


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

POSITIONS_TEXT = (
    "at the newline ending each option and at the end of the prompt (the period ending the last line, the end-of-turn token "
    'and the newline after it, the start of the model\'s own turn: <|im_start|>, "assistant" and the newline, and the empty '
    "<think></think> block the model sees with its thinking mode off, whose closing newlines are the last tokens before its answer)"
)
OLENS_TEXT = f"""\
## O-lens reads

The O-lens is a model trained to describe another model's internal activations in words. It was run on the model's \
activations as it read this exact prompt, {POSITIONS_TEXT}, at layers {", ".join(map(str, LAYERS))}. Each read is a few short \
bullets about what the model is representing at that token, at one layer, each with its FVE (fraction of the activation's \
variance the bullet explains): high FVE is good, low FVE bullets are often hallucinated. Reads can show what the model \
registers but never says; they also often just paraphrase the nearby text. The O-lens is more likely to get fine details \
wrong (exact numbers, letters, names) than broad themes, so trust what a read says about the topic or the kind of \
reasoning more than a specific letter or name in it.

Each position below shows the prompt text around it with the token read wrapped in <read></read> (↵ is a newline), then \
its bullets, by layer and highest FVE first within a layer."""
NLA_TEXT = f"""\
## NLA reads

An NLA (natural language autoencoder) is a model trained to describe one of another model's internal activations in a short \
free-text explanation; a second model, the critic, tries to rebuild the activation from the explanation alone, which scores \
it: FVE, the fraction of the activation's variance the rebuilt one explains. High FVE is good; explanations with low or \
negative FVE are often confabulated. The NLA was run on the model's activations as it read this exact prompt, \
{POSITIONS_TEXT}, at layers {", ".join(map(str, NLA_LAYERS))}, one explanation per token and layer. Explanations can show what \
the model registers but never says; they also often just paraphrase the nearby text, and they get fine details wrong \
(exact numbers, letters, names) more often than broad themes.

Each position below shows the prompt text around it with the token read wrapped in <read></read> (↵ is a newline), then \
its explanations by layer, each with its FVE."""
JLENS_TEXT = f"""\
## J-lens reads

The J-lens (Jacobian lens) maps the model's internal activation at one token and one layer to the vocabulary: it lists the \
tokens that activation is pushing the model toward, with the lens's probabilities. It was run at every token of the prompt \
above as the model read it, including the chat template around the question (<|im_start|>, <|im_end|>, the "assistant" header \
and the empty <think></think> block the model sees with its thinking mode off), at layers {", ".join(map(str, LAYERS))}. The \
last position is the token just before the model's reply. Each position shows the token read, then one line per layer with \
the lens's top {inline.JLENS_K} tokens and their probabilities. Reads are noisy: at many positions the lens just predicts the \
next word of the prompt, and the probabilities are the lens's, not the model's."""
TASK_TEXT = """\
## Your task

Predict the letter the model answers most often to this prompt. You are scored correct if your letter matches its most \
common answer, so give the single most likely letter. Reason it through first, then call `submit_answer` once with the \
letter."""


def options(text: str) -> list[re.Match]:
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


Pos = tuple[str, dict[int, dict]]  # (token, {layer: row})


@cache
def _pool(path: Path, layers: tuple[int, ...]) -> dict[str, list[Pos]]:
    """id -> its reads in token order, one entry per position with a row per layer (O-lens rows have `bullets`, NLA rows `text`)."""
    by: dict[str, dict[int, Pos]] = defaultdict(dict)
    for line in path.open():
        r = json.loads(line)
        if r["layer"] in layers:
            by[r["id"]].setdefault(r["token_index"], (r["token"], {}))[1][r["layer"]] = r
    return {i: [pos[k] for k in sorted(pos)] for i, pos in by.items()}


@cache
def _jpool(path: Path) -> dict[str, list[dict]]:
    return inline_jpp_rows(path)


def inline_jpp_rows(path: Path) -> dict[str, list[dict]]:
    """id -> J++ rows at LAYERS in token order (cutoff_qa_inline's jpp_rows, every token of a text render)."""
    out: dict[str, list[dict]] = {}
    for line in path.open():
        r = json.loads(line)
        keep = [r["layers"].index(L) for L in LAYERS]
        out.setdefault(r["id"], []).append(r | {"layers": list(LAYERS), "top": [r["top"][k] for k in keep]})
    for rows in out.values():
        rows.sort(key=lambda r: r["token_index"])
    return out


def pool(condition: str):
    if condition.startswith("olens"):
        return _pool(READS, LAYERS)
    if condition.startswith("nla"):
        return _pool(NLA_READS, NLA_LAYERS)
    return _jpool(JREADS)


def placebo(condition: str, item: str, n_opts: int, seed: int) -> str:
    """Another case's reads with the same option count, seeded per (draw, case); never the case's own."""
    rng = random.Random(f"{seed}:{item}")
    ok = sorted(s for s in pool(condition) if s != request_id(item) and n_options(s) == n_opts)
    assert ok, (condition, item, n_opts)
    return rng.choice(ok)


@cache
def n_options(rid: str) -> int:
    return len(options(Prompt(rid.split("~", 1)[1]).text))


def _lines(layers: dict[int, dict]) -> list[str]:
    out = []
    for layer in sorted(layers):
        r = layers[layer]
        if "bullets" in r:
            bullets = sorted(r["bullets"], key=lambda b: -b["fve"])
            out += [f"- layer {layer}, FVE {b['fve']:.2f}: {' '.join(b['text'].split())}" for b in bullets]
        else:
            fve = f", FVE {r['fve']:.2f}" if r.get("fve") is not None else ""
            lines = [" ".join(ln.split()) for ln in r["text"].splitlines() if ln.strip()]
            out.append(f"- layer {layer}{fve}:\n" + "\n".join(f"  {ln}" for ln in lines))
    return out


def blocks(text: str, positions: list[Pos]) -> str:
    """The option-newline reads with their option line, then the tail reads with the end of the prompt, the read token wrapped."""
    flat = lambda s: s.replace("\n", "↵")
    nls = option_newlines(text)
    assert len(positions) == len(nls) + len(TAIL), (len(positions), len(nls))
    out = []
    for (char, token), (_, layers) in zip(nls, positions[: len(nls)], strict=True):
        line = text[text.rfind("\n", 0, char) + 1 : char]
        label = f"After option {m.group(1)}" if (m := OPTION_LINE.match(line)) else f"At {json.dumps(token)}"
        after = text[char + len(token) : char + len(token) + CONTEXT]
        ctx = flat(text[max(0, char - CONTEXT) : char]) + f"<read>{show_token(token)}</read>" + flat(after)
        out += [f"{label}: …{ctx}…", *_lines(layers), ""]
    lead = "…" + flat(text.rstrip()[-CONTEXT:-1])
    for k, (_, layers) in enumerate(positions[len(nls) :]):
        shown = "".join(f"<read>{show_token(t)}</read>" if j == k else flat(t) for j, t in enumerate(TAIL))
        out += [f"At {json.dumps(TAIL[k])}: {lead}{shown}", *_lines(layers), ""]
    return "\n".join(out).strip()


def INLINE(env: Prompt, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    parts, extra = [SETUP.format(text=env.text)], {}
    if condition != "transcript_inline":
        src = request_id(case["item"])
        if condition.endswith("_placebo_inline"):
            src = placebo(condition, case["item"], case["n_opts"], seed)
            extra = {"placebo_sources": [src]}
        rows = pool(condition)[src]
        if condition.startswith("jlens"):
            parts += [JLENS_TEXT, inline.jlens_block(rows)]
        else:
            parts += [NLA_TEXT if condition.startswith("nla") else OLENS_TEXT, blocks(env.text, rows)]
    parts.append(TASK_TEXT)
    return "\n\n".join(parts), extra


# ---- read requests ----


def render(tok, item: str) -> tuple[str, list[int], list[tuple[int, int]]]:
    """The prompt as Qwen saw it with thinking off (the empty think block included): text, token ids, char offsets."""
    text = tok.apply_chat_template(Prompt(item).messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return text, enc.input_ids, enc.offset_mapping


def requests(out: Path, inputs: Path):
    """olens.requests.jsonl (text rows with explicit positions: the option newlines, then the ten TAIL tokens; the NLA reader
    takes the same file) and jpp.requests.jsonl (the same rows at every token) for every case."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "olens.requests.jsonl").open("w") as f, (out / "jpp.requests.jsonl").open("w") as g:
        for line in (inputs / "cases.jsonl").read_text().splitlines():
            c = json.loads(line)
            text, ids, offs = render(tok, c["item"])
            prompt = Prompt(c["item"]).text
            start = text.index(prompt)
            nl_pos = [next(k for k, (a, b) in enumerate(offs) if a <= start + char < b) for char, _ in option_newlines(prompt)]
            tail = list(range(len(ids) - len(TAIL), len(ids)))
            assert [tok.decode([ids[p]]) for p in tail] == list(TAIL), (c["item"], [tok.decode([ids[p]]) for p in tail])
            assert all(tok.decode([ids[p]]).strip() == "" for p in nl_pos), c["item"]
            row = {"id": request_id(c["item"]), "env": ENV, "arm": ARM, "item": c["item"], "text": text, "priority": 1}
            f.write(json.dumps(row | {"positions": nl_pos + tail}, ensure_ascii=False) + "\n")
            g.write(json.dumps(row | {"positions": "all"}, ensure_ascii=False) + "\n")
    print("wrote", out / "olens.requests.jsonl", out / "jpp.requests.jsonl")


if __name__ == "__main__":  # requests OUT_DIR
    from cfbench.eval import latest_inputs

    assert sys.argv[1] == "requests", sys.argv
    requests(Path(sys.argv[2]), latest_inputs("supergpqa_inline", [VERSION]))
