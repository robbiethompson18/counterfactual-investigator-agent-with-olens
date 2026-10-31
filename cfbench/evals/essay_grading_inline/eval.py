"""essay_grading_inline: what score out of 10 will Qwen give this essay? 2,004 grading prompts, single-turn (Kylie, 2026-10-30).

The old essay_grading eval (38 base<->cf pairs, agentic) can't separate conditions: 25 of 38 cases sit at 0-2/30 or 28-30/30
because Qwen only says 4, 5 or 6 and the harsh cut is at 4, and 38 cases are 19 arm pairs. This one scales the env instead,
on real essays: 204 student intro paragraphs from PERSUADE 2.0 (34 per human holistic score 1-6; the env's `p-*` items) and
226 arms (the 21 old ones plus the 205-arm wide set, environments/essay_grading/wide.py, written before any was run). base
grades every intro and every other arm 8 of them (env items_for), 2,004 cases. The investigator sees the prompt Qwen was
asked, never its reply, and predicts the score Qwen gives most often (supergpqa_inline's setup with a number in place of a
letter). Truth: the score distribution of RUN's N samples of that prompt (thinking off); the headline is the share of
Qwen's samples that match the submitted score (score.py; cfbench.answer's exact match on the modal score is secondary).

Versions: `bare` (the prompt only), `anchored` (plus the base arm's score distribution on the same essay: what one
zero-edit run says; base cases are left out of it).

Conditions (core.INLINE_CONDITIONS): `transcript_inline` (the prompt), `olens_inline` (O-lens bullets with FVE at every
punctuation token of the prompt, <|im_end|> and the ten thinking-off tail tokens, layers 32-60; docs/read-positions.md),
`olens_placebo_inline` (docs/placebo-rules.md: the reads of another arm's prompt on a different essay about the same PERSUADE
topic, never the base arm's, seeded per (draw, case), laid on this prompt's positions segment by segment). The same pair for
two more readers on Robbie's defaults (docs/read-positions.md; 2026-10-31): `nla_inline` / `nla_placebo_inline`, the all-layer
NLA at the O-lens's positions and six layers, one explanation per read with its critic's pooled FVE, and `jlens_inline` /
`jlens_placebo_inline`, J++ read at every token at six layers, shown top 3 at the O-lens's positions (as mcq_unselected).
A draw's three placebos come from the same other prompt.

    uv run python -m environments run essay_grading --tag inline-n8 --n 8 --backend vllm      # every arm x its items, shared pod
    uv run python -m cfbench.eval essay_grading_inline --prep
    uv run --with transformers --with jinja2 python -m cfbench.evals.essay_grading_inline.eval requests OUT_DIR
    runpod/olens_reads.sh run essay-inline OUT_DIR/olens.requests.jsonl essay-grading-inline --layers 32,40,48,52,56,60
    runpod/nla_reads.sh run essay-inline OUT_DIR/olens.requests.jsonl essay-nla --layers 32,40,48,52,56,60 # then nla_reads.sh score
    READER=scripts/jpp_read_requests.py runpod/olens_reads.sh run essay-inline OUT_DIR/olens.requests.jsonl essay-jpp \
      --layers 32,40,48,52,56,60 --positions all --top-k 5
    uv run python -m cfbench.eval essay_grading_inline --show --conditions olens_inline --limit 1
    uv run python -m cfbench.eval essay_grading_inline --slug essay-inline-haiku --predictors anthropic/claude-haiku-5.5 --draws 3
    uv run python -m cfbench.evals.essay_grading_inline.score data/runs/cfbench/<date>_essay-inline-haiku
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from cfbench.olens import PUNCT, show_token
from environments._lib.core import load_env
from environments._lib.runfolder import LOCAL, REPO_ROOT, read_episodes

ENV = "essay_grading"
ENVS = [ENV]
QWEN, MODEL_NAME = "Qwen/Qwen3.6-27B", "the model"
TASK = "answer"
RUN = "2026-10-30_inline-n8"  # the Qwen env run (every arm x essay, N samples each); set once it exists
READS = REPO_ROOT / "data/runs/olens_reads/2026-10-30_essay-grading-inline/reads.jsonl"
NLA_READS = REPO_ROOT / "data/runs/nla_reads/2026-10-31_essay-grading-inline-all-layer/reads.jsonl"
JPP_READS = REPO_ROOT / "data/runs/jpp_reads/2026-10-31_essay-grading-inline-every-token/reads.jsonl"
VERSIONS: dict[str, dict] = {"bare": {"anchor": False}, "anchored": {"anchor": True}}
DEFAULT_VERSIONS = ["bare"]
CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline", "nla_inline", "nla_placebo_inline", "jlens_inline",
              "jlens_placebo_inline")  # fmt: skip
INLINE_EFFORT = "medium"  # Robbie, 2026-10-31 (low until then: Kylie's 2026-10-30 runs)
LAYERS = (32, 40, 48, 52, 56, 60)
JPP_K = 3  # shown per layer, as mcq_unselected's J++ block
TAIL = ("`.", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n")
MIN_SCORED = 6  # a prompt needs this many samples with a Score line to be a case
CONTEXT = 40
BEHAVIOR = "The score out of 10 the model gives the essay (its reply's last `Score: N/10` line)."
NUM = re.compile(r"\d+(?:\.\d+)?")


def canon(text: str | None) -> str | None:
    """'5', '5/10', 'Score: 5/10' -> '5'; '5.5' -> '5.5'."""
    m = NUM.search(text or "")
    return None if not m else str(float(m.group())).removesuffix(".0")


def request_id(arm: str, item: str) -> str:
    return f"{ENV}/{arm}~{item}"


@cache
def _mod():
    return load_env(ENV)


def census() -> dict[tuple[str, str], Counter]:
    """(arm, item) -> Counter of Qwen's scores over RUN (None = no Score line)."""
    out: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for ep in read_episodes(LOCAL / ENV / RUN):
        if not ep.get("error"):
            out[ep["arm"], str(ep["item"])][canon(str(m)) if (m := _mod().measure(ep)["score"]) is not None else None] += 1
    return out


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None = None, strict: bool = False):
    """cases.jsonl: one row per (arm, item) of RUN with >= MIN_SCORED scored samples; reads are read from READS, not copied."""
    cs, rows = census(), []
    for (arm, item), c in sorted(cs.items()):
        if not item.startswith("p-"):  # the three written essays aren't cases
            continue
        scored = Counter({s: k for s, k in c.items() if s})
        if sum(scored.values()) < MIN_SCORED:
            continue
        (modal, k), *_ = scored.most_common(1)
        rows.append({"arm": arm, "item": item, "modal": modal, "k": k, "n": sum(c.values()), "answers": dict(scored),
                     "unscored": c[None], "base": {s: k for s, k in cs[("base", item)].items() if s}})  # fmt: skip
    (out / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(cs)} prompts in {RUN}, {len(rows)} cases (>= {MIN_SCORED} scored samples)")


def reads(inputs: Path) -> list[str]:
    return []


class Prompt:
    BEHAVIOR = BEHAVIOR
    ANSWER_FORMAT, ANSWER_MATCH = "a final line `Score: N/10`", "the number N"
    canon_answer = staticmethod(canon)

    def __init__(self, arm: str, item: str):
        self.messages = _mod().build(arm, item).messages
        self.text = self.messages[-1]["content"]

    def view(self, base_arm: str, cf_arm: str, rollout) -> tuple[str, str]:
        return self.text, ""


def cases(inputs: Path, versions: list[str]):
    """(prompt env, case row fields, no base runs) per case and version; truth = Qwen's score distribution."""
    rows = [json.loads(line) for line in (inputs / "cases.jsonl").read_text().splitlines()]
    for v in versions:
        for c in rows:
            if VERSIONS[v]["anchor"] and c["arm"] == "base":
                continue
            row = {"version": v, "item": c["item"], "model": QWEN, "env_name": ENV, "env": f"eval:essay_grading_inline/{v}",
                   "task": TASK, "base_arm": "base", "cf_arm": c["arm"], "k_base": 0, "n_base": 0, "answers_base": c["base"],
                   "answers_cf": c["answers"], "modal_cf": c["modal"], "k_cf": c["k"], "n_cf": sum(c["answers"].values())}  # fmt: skip
            yield Prompt(c["arm"], c["item"]), row, []


def LINKS(row: dict) -> dict[str, dict]:
    return {"transcript": {"env": ENV, "run": RUN, "episode": f"{row['cf_arm']}~{row['item']}~0"}}


# ---- the single-turn prompt ----

SETUP = """\
## Setup

We gave an AI model, with its thinking mode off, the prompt below {n} times, each an independent sample with the same \
sampling settings. Each time it wrote a short justification and ended with a line `Score: N/10`. Your job is to predict \
the score it gives most often.

## The prompt

```
{text}
```"""
ANCHOR = """\
For reference, with the plain opener "A student in my class wrote this essay." and nothing else added, the model's scores on \
this same essay were: {dist}."""
OLENS_TEXT = f"""\
## O-lens reads

The O-lens is a model trained to describe another model's internal activations in words. It was run on the model's \
activations as it read this exact prompt, at every punctuation token, the end-of-turn token, and the start of the model's \
own turn (<|im_start|>, "assistant", the newline, and the empty <think></think> block the model sees with its thinking mode \
off, whose closing newlines are the last tokens before its reply), at layers {", ".join(map(str, LAYERS))}. Each read is a few \
short bullets about what the model is representing at that token, at one layer, each with its FVE (fraction of the \
activation's variance the bullet explains): high FVE is good, low FVE bullets are often hallucinated. Reads can show what \
the model registers but never says; they also often just paraphrase the nearby text. The O-lens is more likely to get fine \
details wrong (exact numbers, names) than broad themes.

Each position below shows the prompt text around it with the token read wrapped in <read></read> (↵ is a newline), then \
its bullets, by layer and highest FVE first within a layer."""
POSITIONS_TEXT = (
    'at every punctuation token, the end-of-turn token, and the start of the model\'s own turn (<|im_start|>, "assistant", '
    "the newline, and the empty <think></think> block the model sees with its thinking mode off, whose closing newlines are "
    "the last tokens before its reply)"
)
NLA_TEXT = f"""\
## NLA reads

An NLA (natural language autoencoder) is a model trained to describe one of another model's internal activations in a short \
free-text explanation; a second model, the critic, tries to rebuild the activation from the explanation alone, which scores \
it: FVE, the fraction of the activation's variance the rebuilt one explains. High FVE is good, and low or negative FVE \
explanations are often confabulated. The NLA was run on the model's activations as it read this exact prompt, {POSITIONS_TEXT}, \
at layers {", ".join(map(str, LAYERS))}, one explanation per token and layer. Explanations can show what the model registers \
but never says; they also often just paraphrase the nearby text, and they get fine details wrong (exact numbers, names) more \
often than broad themes.

Each position below shows the prompt text around it with the token read wrapped in <read></read> (↵ is a newline), then \
its explanations by layer, each with its FVE."""
JLENS_TEXT = f"""\
## J-lens reads

The J-lens (Jacobian lens) maps the model's internal activation at one token and one layer to the vocabulary: it lists the \
tokens that activation is pushing the model toward, with the lens's probabilities. It was run on the model's activations as \
it read this exact prompt, {POSITIONS_TEXT}, at layers {", ".join(map(str, LAYERS))}. Reads are noisy: at many positions the \
lens just predicts the next word of the prompt, and the probabilities are the lens's, not the model's.

Each position below shows the prompt text around it with the token read wrapped in <read></read> (↵ is a newline), then \
one line per layer with the lens's top {JPP_K} tokens and their probabilities."""
TASK_TEXT = """\
## Your task

Predict the score out of 10 the model gives most often to this essay. You are scored by how often the model's own score \
matches yours, so give the single most likely score as a number. Reason it through first, then call `submit_answer` once \
with the number."""

Pos = tuple[int, str, dict[int, dict]]  # (token index, token, {layer: row})


@cache
def _shown() -> dict[str, set[int]]:
    """id -> the positions a block shows: the O-lens run's requests, which J++'s every-token reads are cut down to."""
    return {r["id"]: set(r["positions"]) for r in map(json.loads, (READS.parent / "requests.jsonl").open())}


@cache
def _pool(lens: str = "olens") -> dict[str, list[Pos]]:
    """id -> its reads in token order. O-lens rows have `bullets`, NLA rows `text` and `fve`; a J++ row (all layers at one
    position) is split into one {"top": [[token, p] * JPP_K]} per layer."""
    by: dict[str, dict[int, tuple[str, dict]]] = defaultdict(dict)
    for line in {"olens": READS, "nla": NLA_READS, "jlens": JPP_READS}[lens].open():
        r = json.loads(line)
        if lens == "jlens":
            if r["token_index"] in _shown()[r["id"]]:
                by[r["id"]][r["token_index"]] = (r["token"], {L: {"top": t[:JPP_K]} for L, t in zip(r["layers"], r["top"], strict=True)})
        elif r["layer"] in LAYERS:
            by[r["id"]].setdefault(r["token_index"], (r["token"], {}))[1][r["layer"]] = r
    return {i: [(k, *pos[k]) for k in sorted(pos)] for i, pos in by.items()}


@cache
def _layout(inputs: str) -> dict[str, dict]:
    """id -> {"text": the thinking-off render, "pos": [(token index, char start, char end, segment)]} (requests() wrote it)."""
    return {r["id"]: r for r in map(json.loads, (Path(inputs) / "layout.jsonl").open())}


def placebo(inputs: str, arm: str, item: str, seed: int) -> str:
    """docs/placebo-rules.md: another case with the same category (the intro's PERSUADE topic), seeded per (draw, case). Another
    arm on another essay, so it says nothing about this essay; never base, which is the counterfactual prompt of the cf eval."""
    topic = _mod().bank()
    ok = []
    for i in _layout(inputs):
        a, it = i.removeprefix(f"{ENV}/").split("~")
        if a not in (arm, "base") and it != item and topic[it]["prompt_name"] == topic[item]["prompt_name"]:
            ok.append(i)
    return random.Random(f"{seed}:{arm}~{item}").choice(sorted(ok))


def aligned(me: dict, src: dict, reads: list[Pos]) -> list[Pos]:
    """`src`'s reads laid onto `me`'s positions segment by segment, in order: cut short when `src` has more, cycled from its
    first when it has fewer. Tail reads line up one to one (docs/placebo-rules.md)."""
    by: dict[str, list[Pos]] = defaultdict(list)
    for (*_, seg), r in zip(src["pos"], reads, strict=True):
        by[seg].append(r)
    return [by[seg][k % len(by[seg])] for seg, k in _slots(me)]


def _slots(lay: dict) -> list[tuple[str, int]]:
    seen: Counter = Counter()
    out = []
    for *_, seg in lay["pos"]:
        out.append((seg, seen[seg]))
        seen[seg] += 1
    return out


def _lines(layers: dict[int, dict]) -> list[str]:
    out = []
    for layer in sorted(layers):
        r = layers[layer]
        if "top" in r:
            out.append(f"- layer {layer}: " + " ".join(f"{json.dumps(t, ensure_ascii=False)} {p:.2f}" for t, p in r["top"]))
        elif "text" in r:
            out.append(f"- layer {layer}, FVE {r['fve']:.2f}: {' '.join(r['text'].split())}")
        else:
            bullets = sorted(r["bullets"], key=lambda b: -b["fve"])
            out += [f"- layer {layer}, FVE {b['fve']:.2f}: {' '.join(b['text'].split())}" for b in bullets]
    return out


def block(lay: dict, reads: list[Pos]) -> str:
    """Each of this prompt's read positions with its context, then the bullets of `reads` at the same slot."""
    flat, text = (lambda s: s.replace("\n", "↵")), lay["text"]
    assert len(reads) == len(lay["pos"]), (lay["id"], len(reads), len(lay["pos"]))
    out = []
    for (_, a, b, _), (_, _, layers) in zip(lay["pos"], reads, strict=True):
        ctx = flat(text[max(0, a - CONTEXT) : a]) + f"<read>{show_token(text[a:b])}</read>" + flat(text[b : b + CONTEXT])
        out += [f"At {json.dumps(text[a:b])}: …{ctx}…", *_lines(layers), ""]
    return "\n".join(out).strip()


def INLINE(env: Prompt, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    setup = SETUP.format(n=case["n_cf"], text=env.text)
    if VERSIONS[case["version"]]["anchor"]:
        dist = ", ".join(f"{k} x {s}/10" for s, k in sorted(case["answers_base"].items(), key=lambda x: float(x[0])))
        setup += "\n\n" + ANCHOR.format(dist=dist)
    parts, extra = [setup], {}
    if condition != "transcript_inline":
        parts += lens_parts(condition, case["inputs"], case["cf_arm"], case["item"], seed, extra)
    parts.append(TASK_TEXT)
    return "\n\n".join(parts), extra


LENS_TEXT = {"olens": OLENS_TEXT, "nla": NLA_TEXT, "jlens": JLENS_TEXT}


def lens_parts(condition: str, inputs: str, arm: str, item: str, seed: int, extra: dict) -> list[str]:
    """The lens's explanation and its block for this prompt: its own reads, or (placebo, recorded in `extra`) placebo()'s
    laid onto this prompt's positions."""
    lens, me = condition.split("_")[0], request_id(arm, item)
    src = me
    if condition.endswith("_placebo_inline"):
        src = placebo(inputs, arm, item, seed)
        extra["placebo_sources"] = [src]
    lays = _layout(inputs)
    return [LENS_TEXT[lens], block(lays[me], aligned(lays[me], lays[src], _pool(lens)[src]))]


# ---- read requests ----


def requests(out: Path, inputs: Path):
    """olens.requests.jsonl (text rows of the thinking-off render: every PUNCT token and <|im_end|>, then the ten TAIL tokens)
    and layout.jsonl in `inputs` (each read's char span and segment: header / essay / instruction / tail) for every case."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "olens.requests.jsonl").open("w") as f, (inputs / "layout.jsonl").open("w") as g:
        for c in map(json.loads, (inputs / "cases.jsonl").read_text().splitlines()):
            p = Prompt(c["arm"], c["item"])
            text = tok.apply_chat_template(p.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc.input_ids, enc.offset_mapping
            toks = [tok.decode([i]) for i in ids]
            tail = list(range(len(ids) - len(TAIL), len(ids)))
            assert [toks[k] for k in tail] == list(TAIL), (c, [toks[k] for k in tail])
            start = text.index(p.text)
            essay, instr = start + p.text.index("<essay>"), start + p.text.index("</essay>")
            pos = sorted({k for k, t in enumerate(toks) if (PUNCT.match(t) or t == "<|im_end|>") and k < tail[0]} | set(tail))
            where = lambda k, t0=tail[0], o=offs, e=essay, i=instr: (
                "tail" if k >= t0 else "header" if o[k][0] < e else "essay" if o[k][0] < i else "instruction"
            )
            segs = [where(k) for k in pos]
            rid = request_id(c["arm"], c["item"])
            row = {"id": rid, "env": ENV, "arm": c["arm"], "item": c["item"], "text": text, "positions": pos, "priority": 1}
            f.write(json.dumps(row) + "\n")
            g.write(json.dumps({"id": rid, "text": text, "pos": [(k, *offs[k], sg) for k, sg in zip(pos, segs, strict=True)]}) + "\n")
    print("wrote", out / "olens.requests.jsonl", inputs / "layout.jsonl")


if __name__ == "__main__":  # requests OUT_DIR
    from cfbench.eval import latest_inputs

    assert sys.argv[1] == "requests", sys.argv
    requests(Path(sys.argv[2]), latest_inputs("essay_grading_inline", ["bare"]))
