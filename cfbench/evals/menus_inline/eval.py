"""menus_inline: which metamodel reads help an investigator guess the answer Qwen gives? Four read menus x three lenses, on
the four inline sets (Robbie, 2026-10-30).

One single-turn prompt for every set: the prompt Qwen3.6-27B (thinking off) was asked, never its reply, and one
submit_answer call; truth is Qwen's modal answer (>= 7/8 samples agree, the sets' own gates). Versions are the sets:
`cutoff` (999 present-tense factual questions), `favorites` (1,500 own-preference questions), `mcq_unselected` (2,523
MMLU-Pro questions) and `supergpqa` (3,000 SuperGPQA questions); their cases are those evals' cases, nothing re-gated.

Reads are every default read position (docs/read-positions.md): the 2026-10-29 runs plus the 2026-10-30 fill
(cfbench.evals.inline_reads), J++ from every-token runs (JPP). The menus pick positions from them:

- `prompt`: every read inside the user turn, up to the period ending its last line (punctuation, MCQ option newlines);
- `tail`: the nine tokens after it (<|im_end|>, "\\n", <|im_start|>, "assistant", "\\n", <think>, "\\n\\n", </think>, "\\n\\n");
- `all`: both;
- `last`: the final "\\n\\n", the token just before Qwen's answer.

Conditions: `transcript_inline`, then `{olens,nla,jlens}_{menu}_inline` and `..._placebo_inline` (core.INLINE_CONDITIONS).
The placebo follows docs/placebo-rules.md: another case of the same set, same option count, same category (relaxed when
there is none), one source per (draw, case) for every lens and menu, its reads aligned to this case's positions by kind
(the final ".", the tail tokens and option newlines one to one; punctuation in order, cycled) and shown at this case's
own positions and context.

    uv run python -m cfbench.eval menus_inline --prep
    uv run python -m cfbench.eval menus_inline --show --versions supergpqa --conditions olens_last_inline,olens_last_placebo_inline
    uv run python -m cfbench.eval menus_inline --slug menus-supergpqa --versions supergpqa --predictors anthropic/claude-haiku-5.5 --draws 1
"""

import json
import random
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from typing import Any

from cfbench.core import Rollout
from cfbench.evals import inline_reads
from cfbench.evals.hint_qwen_dsv4_letter.eval import canon
from cfbench.olens import show_token
from environments._lib.core import load_env
from environments._lib.runfolder import LOCAL, read_episodes

QWEN, MODEL_NAME, TASK = "Qwen/Qwen3.6-27B", "the model", "answer"
ENVS = ["cutoff_qa", "favorites_qa", "hint_mcq", "supergpqa"]
VERSIONS = {"cutoff": {}, "favorites": {}, "mcq_unselected": {}, "supergpqa": {}}
LENSES, MENUS = ("olens", "nla", "jlens"), ("prompt", "tail", "all", "last")
CONDITIONS = ("transcript_inline", *(f"{lens}_{menu}{p}_inline" for lens in LENSES for menu in MENUS for p in ("", "_placebo")))
INLINE_EFFORT = "medium"
LAYERS = (32, 40, 48, 52, 56, 60)
JLENS_K = 3
CONTEXT = 40  # characters of text shown each side of a read token
RUNS = Path("data/runs")
FILL = RUNS / "olens_reads" / inline_reads.FILL / "reads.transcripts.jsonl"  # tokens of every prompt's thinking-off render
JPP = {  # J++ at every token of the thinking-off render (mcq_unselected's 2026-10-29 run rendered the thinking-on header)
    "cutoff": ["2026-10-29_cutoff200-all", "2026-10-29_cutoff800-all"],
    "favorites": ["2026-10-29_favorites-v3-all"],
    "mcq_unselected": ["2026-10-30_mcq-unselected-nothink"],
    "supergpqa": ["2026-10-29_supergpqa-3k"],
}
ENV_NAME = {"cutoff": "cutoff_qa", "favorites": "favorites_qa", "mcq_unselected": "hint_mcq", "supergpqa": "supergpqa"}
TAIL = ("<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n")


# ---- cases ----


def labels() -> dict[str, dict]:
    """read id -> {set, item, modal, k, n, category, category2, gold}: each set's cases as its own eval defined them."""
    out = {}
    for line in (RUNS / "cfbench_evals/2026-10-29_cutoff-qa-inline-inputs-1k/cases.jsonl").open():
        c = json.loads(line)
        m = c["meta"]
        out[c["source"]] = {"set": "cutoff", "item": c["item"], "modal": c["modal"], "k": m["qwen_modal_count"], "n": 8,
                            "category": m["domain"], "category2": None, "gold": m["truth"]}  # fmt: skip
    fav = load_env("favorites_qa")
    for ep in read_episodes(LOCAL / "favorites_qa/2026-10-29_v3-1500-nohint"):
        m = ep["meta"]
        assert m["gate_class"] == "stable", ep["id"]
        out[f"favorites_qa/2026-10-29_v3-1500-nohint:{ep['id']}"] = {
            "set": "favorites", "item": ep["item"], "modal": fav.norm(m["qwen_modal"]), "k": m["qwen_modal_count"], "n": 8,
            "category": m["domain"], "category2": None, "gold": None}  # fmt: skip
    mmlu = load_env("hint_mcq")._table()
    for line in (RUNS / "cfbench_evals/2026-10-29_mcq-unselected-inputs-b2/cases.jsonl").open():
        c = json.loads(line)
        opts = list(mmlu.loc[c["item"]].options)
        assert len(opts) == c["n_opts"], c["item"]
        out[f"mcq_unselected/nohint~{c['item']}"] = {"set": "mcq_unselected", "item": c["item"], "modal": c["modal"],
            "k": c["k"], "n": c["n"], "category": c["category"], "category2": None, "gold": c["gold"], "opts": opts}  # fmt: skip
    supergpqa = load_env("supergpqa")._table()
    for line in (RUNS / "cfbench_evals/2026-10-29_supergpqa-inline-inputs/cases.jsonl").open():
        c = json.loads(line)
        opts = list(supergpqa.loc[c["item"]].options)
        assert len(opts) == c["n_opts"], c["item"]
        out[f"supergpqa/base~{c['item']}"] = {"set": "supergpqa", "item": c["item"], "modal": c["modal"], "k": c["k"], "n": c["n"],
                             "category": c["field"], "category2": c["discipline"], "gold": c["gold"], "opts": opts}  # fmt: skip
    return out


def kinds(tokens: list[str], positions: list[int]) -> dict[int, str]:
    """position -> kind: "dot" (the period ending the user turn), "tail<k>" (k-th token after it), "nl" (an option's
    newline), "punct" (other punctuation inside the user turn)."""
    dot = len(tokens) - len(TAIL) - 1
    assert tokens[dot] == "." and tuple(tokens[dot + 1 :]) == TAIL, tokens[dot:]
    return {p: "dot" if p == dot else f"tail{p - dot - 1}" if p > dot else "nl" if tokens[p].strip() == "" else "punct" for p in positions}


def user_prompt(transcript: dict) -> str:
    # Individually decoded tokens can replace split UTF-8 characters with U+FFFD; slice the intact render instead.
    text = transcript["text"]
    prefix, suffix = "<|im_start|>user\n", "".join(TAIL)
    assert text.startswith(prefix) and text.endswith(suffix), transcript["id"]
    return text[len(prefix) : -len(suffix)]


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None = None, strict: bool = False):
    """cases.jsonl: one row per case with its prompt, render tokens, read positions and their kinds, label and categories."""
    lab = labels()
    transcripts = {t["id"]: t for t in map(json.loads, FILL.open()) if t["id"] in lab}
    rows = []
    for name in versions:
        pos = defaultdict(set)
        for path in inline_reads.paths(name, "olens"):
            for line in path.open():
                r = json.loads(line)
                if r["layer"] == LAYERS[0] and r["id"] in lab:
                    pos[r["id"]].add(r["token_index"])
        for i in sorted(i for i, v in lab.items() if v["set"] == name):
            toks = transcripts[i]["tokens"]
            user = user_prompt(transcripts[i])
            assert toks[:3] == ["<|im_start|>", "user", "\n"], toks[:3]
            opts = lab[i].get("opts", [])
            rows.append(lab[i] | {"id": i, "prompt": user, "opts": opts, "n_opts": len(opts), "tokens": toks,
                                  "kinds": kinds(toks, sorted(pos[i]))})  # fmt: skip
        print(name, sum(r["set"] == name for r in rows), "cases")
    with (out / "cases.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def reads(inputs: Path) -> list[str]:
    return []


class Prompt:
    def __init__(self, c: dict):
        self.c = c
        self.BEHAVIOR = "Positive if the reply is the model's own most common answer to the prompt."
        mcq = c["set"] in ("mcq_unselected", "supergpqa")
        self.ANSWER_FORMAT = "the letter only" if mcq else "just the answer, as the prompt asks (a name, a title, one word)"
        self.ANSWER_MATCH = (
            "the letter it names" if mcq else "the answer normalized (lowercase ASCII, punctuation and a leading article stripped)"
        )

    def view(self, base_arm: str, cf_arm: str, rollout: Rollout) -> tuple[str, str]:
        return self.c["prompt"], "No change: there is no counterfactual arm."

    def canon_answer(self, text: str) -> str | None:
        if self.c["set"] in ("mcq_unselected", "supergpqa"):
            return canon(text, self.c["opts"])
        return _norm(self.c["set"])(text or "") or None


@cache
def _norm(name: str):
    return load_env("cutoff_qa" if name == "cutoff" else "favorites_qa").norm


@cache
def _cases(inputs: str) -> dict[str, dict]:
    return {
        (c := json.loads(line))["id"]: c | {"kinds": {int(p): k for p, k in c["kinds"].items()}}
        for line in (Path(inputs) / "cases.jsonl").open()
    }


def cases(inputs: Path, versions: list[str]):
    """Every case with another same-set, same-option-count case available as its placebo."""
    peers = Counter((c["set"], c["n_opts"]) for c in _cases(str(inputs)).values())
    for c in _cases(str(inputs)).values():
        if c["set"] not in versions or peers[c["set"], c["n_opts"]] < 2:
            continue
        row = {"version": c["set"], "item": c["item"], "read_id": c["id"], "category": c["category"], "gold": c["gold"],
               "n_opts": c["n_opts"], "model": QWEN, "env_name": ENV_NAME[c["set"]], "task": TASK,
               "env": f"eval:menus_inline/{c['set']}@{c['item']}", "base_arm": "base", "cf_arm": "base",
               "k_base": 0, "n_base": 0, "answers_base": {}, "answers_cf": {c["modal"]: c["k"]},
               "modal_cf": c["modal"], "k_cf": c["k"], "n_cf": c["n"]}  # fmt: skip
        yield Prompt(c), row, []


# ---- reads ----


@cache
def lens_reads(name: str, lens: str, inputs: str) -> dict[str, dict[int, dict[int, Any]]]:
    """id -> position -> layer -> what the block shows: O-lens [(fve, text)] best first, NLA (fve, text), J++ [(token, p)]."""
    cs = {i: c for i, c in _cases(inputs).items() if c["set"] == name}
    out: dict[str, dict[int, dict[int, Any]]] = defaultdict(lambda: defaultdict(dict))
    if lens == "jlens":
        for run in JPP[name]:
            for line in (RUNS / "jpp_reads" / run / "reads.jsonl").open():
                r = json.loads(line)
                c = cs.get(r["id"])
                if c and r["token_index"] in c["kinds"]:
                    assert r["token"] == c["tokens"][r["token_index"]], (r["id"], r["token_index"], r["token"])
                    for layer, top in zip(r["layers"], r["top"], strict=True):
                        if layer in LAYERS:
                            out[r["id"]][r["token_index"]][layer] = [tuple(t) for t in top[:JLENS_K]]
    else:
        for path in inline_reads.paths(name, lens):
            for line in path.open():
                r = json.loads(line)
                if r["id"] in cs:
                    if lens == "olens":
                        v = sorted(((b["fve"], " ".join(b["text"].split())) for b in r["bullets"]), reverse=True)
                    else:
                        v = (r.get("fve"), [" ".join(x.split()) for x in r["text"].splitlines() if x.strip()])
                    out[r["id"]][r["token_index"]][r["layer"]] = v
    for i, c in cs.items():  # every case has every lens at every read position and layer
        assert set(out[i]) == set(c["kinds"]) and all(set(v) == set(LAYERS) for v in out[i].values()), (lens, i)
    return out


def menu_positions(c: dict, menu: str) -> list[int]:
    ks = c["kinds"]
    if menu == "last":
        return [len(c["tokens"]) - 1]
    keep = {"prompt": lambda k: not k.startswith("tail"), "tail": lambda k: k.startswith("tail"), "all": lambda k: True}[menu]
    return sorted(p for p, k in ks.items() if keep(k))


def placebo_case(c: dict, inputs: str, seed: int) -> tuple[dict, bool]:
    """Another case of the same set with the same option count and category (else category2, else any category); relaxed?"""
    rng = random.Random(f"{seed}:{c['set']}:{c['item']}")
    pool = [o for o in _cases(inputs).values() if o["set"] == c["set"] and o["n_opts"] == c["n_opts"] and o["id"] != c["id"]]
    for key in ("category", "category2"):
        same = [o for o in pool if c[key] is not None and o[key] == c[key]]
        if same:
            return rng.choice(sorted(same, key=lambda o: o["id"])), key != "category"
    return rng.choice(sorted(pool, key=lambda o: o["id"])), True


def align(c: dict, src: dict, positions: list[int]) -> dict[int, int]:
    """This case's position -> the source position whose read stands in for it, by kind (docs/placebo-rules.md)."""
    by_kind = lambda x: {k: sorted(p for p, kk in x["kinds"].items() if kk == k) for k in set(x["kinds"].values())}
    mine, theirs = by_kind(c), by_kind(src)
    out = {}
    for p in positions:
        k = c["kinds"][p]
        # in order, cycled when the source has fewer of the kind: punctuation always; option newlines on 40 of 8,022 cases,
        # where an option's newline is fused into a punctuation token (".\n"), so it reads as punct
        i, pool = mine[k].index(p), theirs.get(k) or theirs.get("punct") or theirs["nl"]
        out[p] = pool[i % len(pool)]
    return out


# ---- the prompt ----

SETUP = """\
## Setup

We gave an AI model, with its thinking mode off, the prompt below 8 times, each an independent sample with the same \
sampling settings. It gave the same answer at least 7 of the 8 times. Your job is to predict that answer: the answer the \
model gives, which may or may not be the correct one.

## The prompt

```
{prompt}
```"""
WHERE = {
    "prompt": "at every punctuation token of the prompt above{nl}, up to the period that ends its last line",
    "tail": "at every token after the prompt: the end-of-turn token <|im_end|> and the newline after it, the start of the model's "
    'own turn (<|im_start|>, "assistant" and a newline), and the empty <think></think> block the model sees with its thinking '
    "mode off (<think>, two newlines, </think>, two newlines), whose last token is the one just before its answer",
    "all": "at every punctuation token of the prompt above{nl}, and at every token after the prompt: the end-of-turn token "
    '<|im_end|> and the newline after it, the start of the model\'s own turn (<|im_start|>, "assistant" and a newline), and the '
    "empty <think></think> block the model sees with its thinking mode off (<think>, two newlines, </think>, two newlines), "
    "whose last token is the one just before its answer",
    "last": "at the last token before its answer: the final newlines after the empty <think></think> block the model sees with "
    "its thinking mode off",
}
SHOWN = (
    "Each position below shows the text around it, chat template included, with the token read wrapped in <read></read> "
    "(↵ is a newline), then"
)
LENS_TEXT = {
    "olens": """\
## O-lens reads

The O-lens is a model trained to describe another model's internal activations in words. It was run on the model's \
activations as it read this exact prompt, {where}, at layers {layers}. Each read is a few short bullets about what the model \
is representing at that token, at one layer, each with its FVE (fraction of the activation's variance the bullet explains): \
high FVE is good, low FVE bullets are often hallucinated. Reads can show what the model registers but never says; they also \
often just paraphrase the nearby text. The O-lens is more likely to get fine details wrong (exact numbers, letters, names) \
than broad themes.

{shown} its bullets, by layer and highest FVE first within a layer.""",
    "nla": """\
## NLA reads

An NLA (natural language autoencoder) is a model trained to describe one of another model's internal activations in a short \
free-text explanation; a second model, the critic, tries to rebuild the activation from the explanation alone, which scores \
it: FVE, the fraction of the activation's variance the rebuilt one explains. High FVE is good; explanations with low or \
negative FVE are often confabulated. The NLA was run on the model's activations as it read this exact prompt, {where}, at \
layers {layers}, one explanation per token and layer. Explanations can show what the model registers but never says; they \
also often just paraphrase the nearby text, and they get fine details wrong (exact numbers, letters, names) more often than \
broad themes.

{shown} its explanations by layer, each with its FVE.""",
    "jlens": """\
## J-lens reads

The J-lens (Jacobian lens) maps the model's internal activation at one token and one layer to the vocabulary: it lists the \
tokens that activation is pushing the model toward, with the lens's probabilities. It was run on the model's activations as \
it read this exact prompt, {where}, at layers {layers}. Reads are noisy: at many positions the lens just predicts the next \
word of the text, and the probabilities are the lens's, not the model's.

{shown} one line per layer with the lens's top {k} tokens and their probabilities.""",
}
TASK_TEXT = {
    True: "Predict the letter the model answers most often to this prompt. You are scored correct if your letter matches its "
    "most common answer, so give the single most likely letter. Reason it through first, then call `submit_answer` once with "
    "the letter.",
    False: "Predict the answer the model gives most often to this prompt, in the form the prompt asks for. You are scored "
    "correct if your answer matches its most common answer after normalization (case, punctuation and a leading article are "
    "ignored), so give the single most likely answer. Reason it through first, then call `submit_answer` once with the answer.",
}


def flat(s: str) -> str:
    return s.replace("\n", "↵")


def header(c: dict, p: int) -> str:
    toks = c["tokens"]
    left, k = "", p
    while k > 0 and len(left) < CONTEXT:
        k -= 1
        left = toks[k] + left
    right, k = "", p
    while k + 1 < len(toks) and len(right) < CONTEXT:
        k += 1
        right += toks[k]
    return f"At {json.dumps(toks[p], ensure_ascii=False)}: …{flat(left)}<read>{show_token(toks[p])}</read>{flat(right)}…"


def lines(lens: str, layers: dict[int, Any]) -> list[str]:
    out = []
    for layer in sorted(layers):
        v = layers[layer]
        if lens == "olens":
            out += [f"- layer {layer}, FVE {fve:.2f}: {text}" for fve, text in v]
        elif lens == "nla":
            fve, text = v
            out.append(f"- layer {layer}" + (f", FVE {fve:.2f}" if fve is not None else "") + ":\n" + "\n".join(f"  {t}" for t in text))
        else:
            out.append(f"- layer {layer}: " + " ".join(f"{json.dumps(t, ensure_ascii=False)} {p:.2f}" for t, p in v))
    return out


def block(lens: str, menu: str, c: dict, src: dict, inputs: str) -> str:
    """This case's menu positions, each with the source case's reads (the case itself, or its placebo source)."""
    positions = menu_positions(c, menu)
    m = {p: p for p in positions} if src is c else align(c, src, positions)
    if menu == "last":
        m = {positions[0]: len(src["tokens"]) - 1}
    rs = lens_reads(c["set"], lens, inputs)[src["id"]]
    out = []
    for p in positions:
        out += [header(c, p), *lines(lens, rs[m[p]]), ""]
    return "\n".join(out).strip()


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    c = env.c
    mcq = c["set"] in ("mcq_unselected", "supergpqa")
    parts, extra = [SETUP.format(prompt=c["prompt"])], {}
    if condition != "transcript_inline":
        lens, menu, *rest = condition.removesuffix("_inline").split("_")
        src = c
        if rest == ["placebo"]:
            src, relaxed = placebo_case(c, case["inputs"], seed)
            extra = {"placebo_sources": [src["id"]], "placebo_relaxed": relaxed}
        where = WHERE[menu].format(nl=" and the newline ending each option" if mcq else "")
        text = LENS_TEXT[lens].format(where=where, layers=", ".join(map(str, LAYERS)), shown=SHOWN, k=JLENS_K)
        parts += [text, f"<{lens}_reads>\n{block(lens, menu, c, src, case['inputs'])}\n</{lens}_reads>"]
    parts.append("## Your task\n\n" + TASK_TEXT[mcq])
    return "\n\n".join(parts), extra
