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

**Batch 2 (2026-10-29 evening, "draw fresh").** `SEED2` draws N_EVAL2 more questions from the 8,032 in neither half
(never probe-trained), version `mmlu_pro_b2`, read by three sharded pods into READS_B2; the versions share the prompt and
harness, so pool them by case. Run batch 1 with `--versions mmlu_pro`, batch 2 with `--versions mmlu_pro_b2`.

**Census.** Qwen's no-hint answers come from the four no-hint runs that together cover all 12,032 questions (CENSUS),
read from their labels file LABELS: a reply counts only when it is one bare letter (worked solutions are unscored; the
first-capital parser before 2026-10-29 read them as "A" or "I", CODE_SMELL P1 #1). A question is a case when its modal
letter has >= MIN_SHARE of all its samples, an unscored reply counting against it (so >= 7 of 8 samples gave that bare
letter, as the prompt says; a share of the scored replies alone admits questions on one or two bare letters). Truth =
that modal letter.

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
from environments._lib.runfolder import LOCAL, REPO_ROOT

ENVS = ["hint_mcq"]
TASK = "answer"
MODEL_NAME = "the model"  # the subject is never named to the investigator (Robbie, 2026-10-29); the runner asserts it
VERSIONS: dict[str, dict] = {"mmlu_pro": {"half": "eval"}, "mmlu_pro_b2": {"half": "eval2"}}  # the two eval batches (split())
CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
INLINE_EFFORT = "low"
QWEN, MODEL = "Qwen/Qwen3.6-27B", "the model"
ARM = "nohint"
SEED, N_SAMPLE, N_EVAL = 0, 4000, 2000
SEED2, N_EVAL2 = 1, 2000  # batch 2 (Robbie, 2026-10-29: "draw fresh"): a second eval draw from the questions in neither half
MIN_SHARE = 7 / 8
LAYERS = (32, 40, 48, 52, 56, 60)  # cutoff_qa_inline's six (2026-10-29); the first read job did 32-52, a top-up added 56/60
CENSUS = ["2026-10-26_or_nothink2", "2026-10-27_scale2k", "2026-10-27_scale2k-topup", "2026-10-28_all12k-nohint"]
# every CENSUS run's strict-parse rescore (`environments rescore`), named: latest_labels() picks by mtime (CODE_SMELL P1 #6)
LABELS = "measure@4f098526d544-strict.jsonl"
PROBE_READS = REPO_ROOT / "data/runs/mcq_answer_probe/2026-10-28_olens-nl"  # transcripts ids = the 718 training questions
PROBE_REFIT = REPO_ROOT / "data/runs/mcq_answer_probe/2026-10-28_eval-baseline-12k/meta.json"  # params.train_items
READS = REPO_ROOT / "data/runs/olens_reads/2026-10-29_mcq-unselected-nohint"  # own reads of the no-hint prompts, batch 1, on HF
READS_B2 = REPO_ROOT / "data/runs/olens_reads/2026-10-29_mcq-unselected-nohint-b2"  # batch 2 (three sharded pods)
READS_DIRS = (READS, READS_B2)
# NLA readouts (the all-layer NLA, agu18dec/qwen3.6-27b-all-layer-nla-rl2000, critic-scored) and J++ reads of both batches'
# prompts, the knowledge-cutoff eval's two other metamodels (cfbench/evals/cutoff_qa_inline); conditions nla_inline,
# nla_placebo_inline, jlens_inline, jlens_placebo_inline, not in CONDITIONS: pass them with --conditions
NLA_DIR = REPO_ROOT / "data/runs/nla_reads/2026-10-29_mcq-unselected-nohint"
JREADS = REPO_ROOT / "data/runs/jpp_reads/2026-10-29_mcq-unselected-nohint/reads.jsonl"
JLENS_K = 3  # top next tokens shown per layer
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
    rest = [i for i in items if i not in set(sample) and i not in old]  # neither half, never probe-trained
    eval2 = random.Random(SEED2).sample(rest, N_EVAL2)
    return {"seed": SEED, "n_sample": N_SAMPLE, "eval": eval_half, "probe": probe_half, "seed2": SEED2, "eval2": eval2,
            "probe_trained_in_sample": sorted(set(sample) & old), "probe_trained_total": len(old)}  # fmt: skip


@cache
def census() -> dict[str, Counter]:
    """item -> Counter of Qwen's no-hint answers over the census runs (their LABELS file): the letter, or None for an
    unscored reply (not one bare letter), so a share is over every sample."""
    out: dict[str, Counter] = defaultdict(Counter)
    for run in CENSUS:
        for r in map(json.loads, (LOCAL / "hint_mcq" / run / "labels" / LABELS).open()):
            arm, item, _ = r["episode"].split("~")
            if arm == ARM:
                out[item][r["values"].get("answer")] += 1
    return out


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None = None, strict: bool = False):
    """split.json and cases.jsonl (each version's consistent questions) into `out`. `placebo` is unused: the pool is READS_DIRS."""
    sp, cs, table = split(), census(), _mod()._table()
    rows = []
    for version in versions:
        half = VERSIONS[version]["half"]
        n = 0
        for item in sp[half]:
            c = cs.get(item) or Counter()
            letters = Counter({a: k for a, k in c.items() if a})
            if not letters:
                continue
            modal, k = letters.most_common(1)[0]
            if k / sum(c.values()) < MIN_SHARE:
                continue
            row = table.loc[item]
            rows.append({"version": version, "item": item, "category": row.category, "gold": row.answer, "n_opts": len(row.options),
                         "modal": modal, "k": k, "n": sum(c.values()), "answers": dict(letters), "unscored": c[None]})  # fmt: skip
            n += 1
        right = sum(r["modal"] == r["gold"] for r in rows if r["version"] == version)
        sp[f"{half}_counts"] = {"questions": len(sp[half]), "with_census": sum(i in cs for i in sp[half]), "cases": n, "cases_right": right}
        print(f"{version}: {len(sp[half])} questions, consistent {n} ({right} right)")
    sp["labels"] = LABELS
    (out / "split.json").write_text(json.dumps(sp, indent=1))
    (out / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


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
        if c["version"] not in versions:
            continue
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


NLA_TEXT = f"""\
## NLA reads

An NLA (natural language autoencoder) is a model trained to describe one of another model's internal activations in a short \
free-text explanation; a second model, the critic, tries to rebuild the activation from the explanation alone, which scores \
it: FVE, the fraction of the activation's variance the rebuilt one explains. High FVE is good; explanations with low or \
negative FVE are often confabulated. The NLA was run on the model's activations as it read this exact prompt, at the newline \
ending each option and at the end of the prompt (the period ending the last line, the end-of-turn token and the newline after \
it, and the start of the model's own turn: <|im_start|>, "assistant" and the newline, just before it answers), at layers \
{", ".join(map(str, LAYERS))}, one explanation per token and layer. Explanations can show what the model registers but never \
says; they also often just paraphrase the nearby text, and they get fine details wrong (exact numbers, letters, names) more \
often than broad themes.

Each position below shows the prompt text around it with the token read wrapped in <read></read> (↵ is a newline), then its \
explanations by layer, each with its FVE."""

JLENS_TEXT = f"""\
## J-lens reads

The J-lens (Jacobian lens) maps the model's internal activation at one token and one layer to the vocabulary: it lists the \
tokens that activation is pushing the model toward, with the lens's probabilities. It was run on the model's activations as it \
read this exact prompt, at the newline ending each option and at the end of the prompt (the period ending the last line, the \
end-of-turn token and the newline after it, and the start of the model's own turn: <|im_start|>, "assistant" and the newline, \
just before it answers), at layers {", ".join(map(str, LAYERS))}. Reads are noisy: at many positions the lens just predicts \
the next token of the prompt, and the probabilities are the lens's, not the model's.

Each position below shows the prompt text around it with the token read wrapped in <read></read> (↵ is a newline), then one \
line per layer with the lens's top {JLENS_K} tokens and their probabilities."""


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


Pos = tuple[str, dict[int, list[dict] | dict]]  # (token, {layer: O-lens bullets, an NLA readout {text, fve}, or a J++ read {top}})


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
        payload = r["bullets"] if "bullets" in r else {"text": r["text"], "fve": r.get("fve")}  # O-lens bullets or an NLA readout
        where[r["id"]].setdefault(key, (r["token"], {}))[1][r["layer"]] = payload
    ids = set(nl) | set(tail)
    return {i: ([nl[i][k] for k in sorted(nl[i])], [tail[i][k] for k in sorted(tail[i])]) for i in ids}


def _pool_items(path: Path) -> dict[str, str]:
    return {r["id"]: str(r["item"]) for r in map(json.loads, (path / "requests.jsonl").open())}


@cache
def reads() -> tuple[dict[str, tuple[list[Pos], list[Pos]]], dict[str, str]]:
    """Every batch's own reads that exist: id -> (nl, tail) and id -> item."""
    pool: dict = {}
    items: dict = {}
    for d in READS_DIRS:
        if (d / "reads.jsonl").exists():
            pool |= _pool(d)
            items |= _pool_items(d)
    assert pool, f"no own reads yet: {READS_DIRS}"
    return pool, items


def placebo(pool: dict, items: dict[str, str], item: str, n_nl: int, seed: int) -> str:
    """The source whose reads stand in for `item`'s: a seeded pick among the other questions (either batch) with the same
    option count (so the positions match), as cutoff_qa_inline's placebo does; never the question's own."""
    rng = random.Random(f"{seed}:{item}")
    ok = sorted(s for s, (nl, tail) in pool.items() if len(nl) == n_nl and len(tail) == len(TAIL) and items[s] != item)
    assert ok, (item, n_nl)
    return rng.choice(ok)


@cache
def nla_reads() -> tuple[dict[str, tuple[list[Pos], list[Pos]]], dict[str, str]]:
    """Both batches' NLA readouts at the O-lens positions: id -> (nl, tail) and id -> item (NLA_DIR holds reads.jsonl + requests.jsonl)."""
    assert (NLA_DIR / "reads.jsonl").exists(), f"no NLA reads yet: {NLA_DIR}"
    return _pool(NLA_DIR), _pool_items(NLA_DIR)


@cache
def jpp_pool() -> tuple[dict[str, tuple[list[Pos], list[Pos]]], dict[str, str]]:
    """Both batches' J++ reads (scripts/jpp_read_requests.py rows at every token: token, layers, top) cut to the O-lens block's
    positions: id -> (nl, tail) with {layer: {"top": top}} per position, and id -> item. Every token of a ~200-token prompt at
    six layers was a ~30k-token block (2026-10-29), so the J-lens is compared at the same positions as the other two lenses."""
    assert JREADS.exists(), f"no J++ reads yet: {JREADS}"
    items = _pool_items(JREADS.parent)
    rows: dict[str, list[dict]] = defaultdict(list)
    for line in JREADS.open():
        r = json.loads(line)
        rows[r["id"]].append(r)
    pool: dict[str, tuple[list[Pos], list[Pos]]] = {}
    for i, rs in rows.items():
        rs.sort(key=lambda r: r["token_index"])
        text = Prompt(items[i]).text
        keep = [r["layers"].index(L) for L in LAYERS]

        def pos(r: dict, keep: list[int] = keep) -> Pos:
            return r["token"], {L: {"top": r["top"][k]} for L, k in zip(LAYERS, keep, strict=True)}

        content = [r for r in rs if r["field"] == "content" and r["char_offset"] is not None]
        # the row whose token covers each option newline's char (the J++ reader keeps a newline merged into a longer token)
        nl = [pos(next(r for r in content if r["char_offset"] <= c < r["char_offset"] + len(r["token"]))) for c, _ in option_newlines(text)]
        start = max(r["token_index"] for r in rs if r["token"] == "<|im_start|>")
        by_index = {r["token_index"]: r for r in rs}
        tail = [pos(by_index[k]) for k in range(start - 3, start + 3)]
        assert tuple(t for t, _ in tail) == TAIL, (i, [t for t, _ in tail])
        pool[i] = (nl, tail)
    return pool, items


def own(item: str) -> tuple[list[Pos], list[Pos]]:
    return reads()[0][request_id(item)]


def _bullets(layers: dict[int, list[dict] | dict]) -> list[str]:
    """One line per O-lens bullet (highest FVE first within a layer), or per NLA readout its explanation's lines indented."""
    out = []
    for layer in sorted(layers):
        v = layers[layer]
        if isinstance(v, dict) and "top" in v:  # a J++ read: the lens's top next tokens
            out.append(f"- layer {layer}: " + " ".join(f"{json.dumps(t, ensure_ascii=False)} {p:.2f}" for t, p in v["top"][:JLENS_K]))
        elif isinstance(v, dict):  # an NLA readout
            fve = f", FVE {v['fve']:.2f}" if v.get("fve") is not None else ""
            lines = [" ".join(line.split()) for line in v["text"].splitlines() if line.strip()]
            out.append(f"- layer {layer}{fve}:\n" + "\n".join(f"  {line}" for line in lines))
        else:
            out += [f"- layer {layer}, FVE {b['fve']:.2f}: {' '.join(b['text'].split())}" for b in sorted(v, key=lambda b: -b["fve"])]
    return out


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
    n_nl = len(option_newlines(env.text))
    if condition == "olens_inline":
        nl, tail = own(case["item"])
        parts += [READS_TEXT, blocks(env.text, nl, tail)]
    elif condition == "olens_placebo_inline":
        src = placebo(*reads(), case["item"], n_nl, seed)
        parts += [READS_TEXT, blocks(env.text, *reads()[0][src])]
        extra = {"placebo_sources": [src]}
    elif condition in ("nla_inline", "nla_placebo_inline"):
        pool, items = nla_reads()
        src = request_id(case["item"]) if condition == "nla_inline" else placebo(pool, items, case["item"], n_nl, seed)
        parts += [NLA_TEXT, blocks(env.text, *pool[src])]
        extra = {} if condition == "nla_inline" else {"placebo_sources": [src]}
    elif condition in ("jlens_inline", "jlens_placebo_inline"):
        pool, items = jpp_pool()
        src = request_id(case["item"]) if condition == "jlens_inline" else placebo(pool, items, case["item"], n_nl, seed)
        parts += [JLENS_TEXT, blocks(env.text, *pool[src])]
        extra = {} if condition == "jlens_inline" else {"placebo_sources": [src]}
    parts.append(TASK_TEXT)
    return "\n\n".join(parts), extra


def requests(out: Path, inputs: Path, versions: list[str]):
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
            if c["version"] not in versions:
                continue
            row = {"id": request_id(c["item"]), "env": "hint_mcq", "arm": ARM, "item": c["item"], "tools": None, "priority": 1,
                   "messages": Prompt(c["item"]).messages, "option_newlines": True, "generation_prompt": True}  # fmt: skip
            _, ids, _ = render_request(tok, row)
            start = max(i for i, t in enumerate(ids) if t == im_start)
            row["positions"] = list(range(start - 3, start + 3))
            assert [tok.decode([ids[i]]) for i in row["positions"]] == list(TAIL), (row["id"], row["positions"])
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":  # requests OUT.jsonl [VERSION ...]  (default: every version)
    cmd, path = sys.argv[1], Path(sys.argv[2])
    if cmd == "requests":
        from cfbench.eval import latest_inputs

        versions = sys.argv[3:] or list(VERSIONS)
        requests(path, latest_inputs("mcq_unselected", versions), versions)
