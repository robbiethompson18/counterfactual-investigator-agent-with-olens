"""Blog authors: does deleting this one sentence make Qwen3.6-27B much less sure of its answer about the blogger?

author_mcq_drop_inline's question on the Blog Authorship Corpus (Kylie, 2026-10-30), about the blogger's sex or age.

Truth from runs/author_mcq_blog/2026-10-30_ablate-loo and -b (2 × 2,004 bloggers; scripts/author_probe/blog.py:
question-first prompts, both option
orders, one deleted sentence at a time, for every confident Qwen-wrong pair and 300 Qwen-right pairs per attribute). Same
rule as the SynthPAI eval: positive when the drop in Qwen's probability on its answer is > max(0.15, 5 x the pair's median
drop); stable pairs only (median single-sentence TV < 0.05); every positive plus as many negatives per pair; versions
qwen_wrong / qwen_right. Same prompt (hint_qwen_dsv4/inline.py's layout), conditions, placebo and investigator (Claude Haiku
5.5, effort low). Reads: runs/olens_reads/2026-10-30_author-blog (every punctuation, symbol and newline token of the question
and the post up to "What do you think?", layers 32/40/48).

    uv run --with transformers --with jinja2 python -m cfbench.evals.author_blog_drop_inline.eval topup OUT.jsonl   # read requests
    uv run --with transformers --with jinja2 python -m cfbench.eval author_blog_drop_inline --prep
    uv run python -m cfbench.evals.author_blog_drop_inline.eval show-expview INPUTS INPUTS/expview
    uv run python -m cfbench.evals.author_blog_drop_inline.eval add-nla INPUTS   # reads/nla.jsonl from NLA_READS
"""

import json
import random
import sys
from pathlib import Path

from cfbench.core import Rollout
from cfbench.evals.hint_qwen_dsv4 import inline
from scripts.author_probe.bank import LETTERS, MODEL
from scripts.author_probe.blog import ATTRS as BLOG_ATTRS
from scripts.author_probe.blog import BANK, END
from scripts.author_probe.blog import message as qf_message

ABLATES = [
    Path("data/runs/author_mcq_blog/2026-10-30_ablate-loo"),
    Path("data/runs/author_mcq_blog/2026-10-30_ablate-loo-b"),
]  # 2 x 2,004 bloggers
ABLATES_X2 = ABLATES
ATTRS = {k: ("", v) for k, v in BLOG_ATTRS.items()}  # bank.ATTRS shape: (question, options)
READS = [Path(f"data/runs/olens_reads/2026-10-30_author-blog{s}") for s in ("", "-b")]
# ceselder/qwen3.6-27b-nla-rl (layer 42) at the O-lens positions (scripts/author_probe/nla_l42.py; Kylie, 2026-10-31)
NLA_READS = Path("data/runs/nla_reads/2026-10-31_author-blog-nla-l42")
QWEN = "Qwen/Qwen3.6-27B"
ENVS: list[str] = []
MODEL_NAME = "the model"
TASK = "claim"
CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline", "nla_inline", "nla_placebo_inline")
INLINE_EFFORT = "low"  # Kylie, 2026-10-30
VERSIONS: dict[str, dict] = {"qwen_wrong": {"wrong": True}, "qwen_right": {"wrong": False}}  # Qwen vs the SynthPAI label
SEED = 0
X2 = False
WHERE = 'at every punctuation and newline token of the question and the blog post, up to "What do you think?" (before the options)'
TIMING = "These reads are of the full message, with the sentence still in it."
TASK_TEXT = """\
Call the deletion significant if {model}'s probability on its answer ({answer}) falls by more than max(0.15, 5 × the median \
fall from deleting any one sentence of this message): by more than 0.15 for most messages, since deleting a typical sentence \
barely moves it. In the base arm its probability on {answer} is {p0:.2f} (averaged over this option order and the reversed \
one). Give the probability that the deletion is significant, with submit_prediction."""


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines()]


def collect(versions: tuple[str, ...] | list[str] = tuple(VERSIONS)) -> list[dict]:
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    rng, out = random.Random(SEED), []
    for r in (r for d in (ABLATES_X2 if X2 else ABLATES) if (d / "pairs_scored.jsonl").exists() for r in load(d / "pairs_scored.jsonl")):
        version = "qwen_wrong" if r["answer"] != r["label"] else "qwen_right"
        if not r["stable"] or version not in versions:
            continue
        drops = [-d["qwen_shift"] for d in r["deletions"]]
        thr = max(0.15, 5 * sorted(drops)[len(drops) // 2])
        pos = [d for d, x in zip(r["deletions"], drops, strict=True) if x > thr]
        neg = [d for d, x in zip(r["deletions"], drops, strict=True) if x <= thr]
        opts = ATTRS[r["attr"]][1]
        for d in pos + rng.sample(neg, min(len(pos), len(neg))):
            out.append({"version": version, "item": f"{r['passage']}:{r['attr']}#{d['i']}", "pair": f"{r['passage']}:{r['attr']}",
                        "attr": r["attr"], "comment": d["text"], "positive": -d["qwen_shift"] > thr, "threshold": thr,
                        "answer": LETTERS[r["answer"]], "label": LETTERS[r["label"]], "p0": r["pmf"][r["answer"]],
                        "p1": d["pmf"][r["answer"]], "drop": -d["qwen_shift"], "tv": d["tv"], "flip": not d["argmax_kept"],
                        "prompt": qf_message(passages[r["passage"]]["text"], r["attr"], opts)})  # fmt: skip
    return out


def _with_offsets(reads_dir: Path, tok) -> list[dict]:
    """The reader's rows for `prompt_ids` requests carry no message offsets: add msg_index 0, field "content" and the read
    token's char offset in the user message, which inline.block() needs."""
    reqs = {r["id"]: r for r in load(reads_dir / "requests.jsonl")}
    spans: dict[str, list] = {}
    out = []
    for r in load(reads_dir / "reads.jsonl"):
        if r["id"] not in spans:
            user = reqs[r["id"]]["messages"][0]["content"]
            text = tok.apply_chat_template(reqs[r["id"]]["messages"], tokenize=False, enable_thinking=False)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            assert enc["input_ids"] == reqs[r["id"]]["prompt_ids"], r["id"]
            s0 = text.index(user)
            spans[r["id"]] = [(a - s0, b - s0) for a, b in enc["offset_mapping"]]
        a, b = spans[r["id"]][r["token_index"]]
        out.append(
            r
            | {"msg_index": 0, "role": "user", "field": "content", "char_offset": a, "token": reqs[r["id"]]["messages"][0]["content"][a:b]}
        )
    return out


def prep(out: Path, versions: list[str], placebo: Path | None, n_items: int | None = None, strict: bool = False):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    cs = collect(versions)
    need = {c["pair"] for c in cs}
    rows, have = [], set()
    for d in READS:  # the first reads run that has a pair wins
        if (d / "reads.jsonl").exists() and (d / "requests.jsonl").exists():
            new = [r for r in _with_offsets(d, tok) if r["id"] in need and r["id"] not in have]
            rows += new
            have |= {r["id"] for r in new}
    (out / "reads").mkdir(parents=True)
    (out / "reads/reads.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    msgs = {
        r["id"]: r["messages"] for d in READS if (d / "requests.jsonl").exists() for r in load(d / "requests.jsonl")
    }  # cfbench.olens.Reads
    batch = {r["id"]: k for k, d in enumerate(READS) if (d / "requests.jsonl").exists() for r in load(d / "requests.jsonl")}
    rows_req = [{"id": p, "item": p, "messages": msgs[p], "batch": batch[p]} for p in sorted(have)]  # batch: the placebo stays within it
    (out / "reads/requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_req))
    (out / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cs))
    if (NLA_READS / "reads.jsonl").exists():
        add_nla(out)
    print(f"{len(cs)} cases ({sum(c['positive'] for c in cs)} positive) from {len(need)} pairs; reads for {len(have & need)} pairs")


def add_nla(inputs: Path):
    """The NLA conditions' readouts for the inputs run's pairs, as reads/nla.jsonl (also run on an existing inputs run)."""
    need = {r["id"] for r in load(inputs / "reads/requests.jsonl")}
    rows = [r for r in load(NLA_READS / "reads.jsonl") if r["id"] in need]
    (inputs / "reads/nla.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} NLA readouts for {len({r['id'] for r in rows})} of {len(need)} pairs -> {inputs / 'reads/nla.jsonl'}")


def reads(inputs: Path) -> list[str]:
    return [str(inputs / "reads/reads.jsonl")]


class Prompt:
    BEHAVIOR = (
        "Positive if deleting the sentence lowers Qwen3.6-27B's probability on its answer by more than max(0.15, 5 x the median drop)."
    )

    def __init__(self, comment: str):
        self.comment = comment

    def view(self, base_arm: str, cf_arm: str, rollout: Rollout) -> tuple[str, str]:
        edit = f"This one sentence is deleted from the blog post in the user's message (nothing else changes: the rest of the post, \
the question, the options and their order stay the same):\n\n<deleted_sentence>\n{self.comment}\n</deleted_sentence>"
        return rollout.messages[0]["content"], edit


def cases(inputs: Path, versions: list[str]):
    for c in (c for c in load(inputs / "cases.jsonl") if c["version"] in versions):
        msgs = [{"role": "user", "content": c["prompt"]}, {"role": "assistant", "content": c["answer"]}]
        base = [Rollout(source=f"author_blog/{c['pair']}", messages=msgs, label=False)]
        row = {"version": c["version"], "item": c["item"], "pair": c["pair"], "attr": c["attr"], "model": QWEN,
               "env_name": "author_blog", "task": TASK, "env": f"eval:author_blog_drop_inline/{c['item']}", "base_arm": "full",
               "cf_arm": "sentence deleted", "k_base": 0, "n_base": 1, "k_cf": int(c["positive"]), "n_cf": 1,
               "answer": c["answer"], "gold": c["label"], "p0": c["p0"], "p1": c["p1"], "drop": c["drop"],
               "threshold": c["threshold"], "tv": c["tv"], "flip": c["flip"]}  # fmt: skip
        yield Prompt(c["comment"]), row, base


def placebo(path: Path, own: str, seed: int) -> str:
    """Another pair's reads as the placebo: a seeded pick among the other pairs of the same attribute with at least as many
    read positions (passages differ, so inline.placebo's exact-shape match rarely exists); the longest ones if none."""
    rows, _ = inline._reads(path)
    n = lambda s: len(inline._positions(rows[s]))
    attr = own.split(":")[1]
    batch = {r["id"]: r.get("batch", 0) for r in load(path / "requests.jsonl")}  # the first 2,004 bloggers' cases keep their placebos
    same = [s for s in sorted(rows) if s != own and s.split(":")[1] == attr and batch[s] == batch[own]]
    pool = [s for s in same if n(s) >= n(own)] or sorted(same, key=n)[-5:]
    return random.Random(f"{seed}:{own}").choice(pool)


def block(path: Path, own: str, messages: list[dict], source: str) -> str:
    """inline.block's format: `own`'s positions in the case's prompt, `source`'s bullets position by position (its first
    ones; cycled if it has fewer positions)."""
    rows, _ = inline._reads(path)
    mine, theirs = inline._positions(rows[own]), inline._positions(rows[source])
    out = []
    for k, ((i, field, char, token), _) in enumerate(mine):
        layers = theirs[k % len(theirs)][1]
        text = messages[i][field]
        assert text.startswith(token, char), (own, i, char, token)
        out.append(f"{inline._label(text, char, token)}: {inline._context(text, char, token)}")
        for layer in sorted(layers):
            for b in sorted(layers[layer]["bullets"], key=lambda b: -b["fve"]):
                out.append(f"- layer {layer}, FVE {b['fve']:.2f}: {' '.join(b['text'].split())}")
        out.append("")
    return "\n".join(out).strip()


NLA_TEXT = """\
An NLA (natural language autoencoder) is a model trained to describe one of another model's internal activations in a short \
free-text explanation; a second model, the critic, tries to rebuild the activation from the explanation alone, which scores \
it: FVE, the fraction of the activation's variance the rebuilt one explains (here relative to the mean activation over these \
blog prompts, so 0 means no better than the average prompt). High FVE is good; explanations with low or negative FVE are \
often confabulated. The NLA was run on {model}'s activations {where}, at layer {layer}, one explanation per token. \
Explanations can show what the model registers but never says; they also often just paraphrase the nearby text, and they \
get fine details wrong (exact numbers, names) more often than broad themes. {timing}

Each position below shows the text around the token read, wrapped in <read></read> (↵ is a newline), then its explanation \
with its FVE."""


def nla_block(path: Path, own: str, messages: list[dict], source: str) -> str:
    """block()'s format over NLA readouts (reads/nla.jsonl): `own`'s positions, `source`'s explanation position by position."""
    by: dict[str, list[dict]] = {}
    for r in load(path / "nla.jsonl"):
        by.setdefault(r["id"], []).append(r)
    mine, theirs = (sorted(by[x], key=lambda r: r["char_offset"]) for x in (own, source))
    out = []
    for k, r in enumerate(mine):
        text, char, token = messages[r["msg_index"]][r["field"]], r["char_offset"], r["token"]
        assert text.startswith(token, char), (own, char, token)
        t = theirs[k % len(theirs)]
        out.append(f"{inline._label(text, char, token)}: {inline._context(text, char, token)}")
        out.append(f"- layer {t['layer']}, FVE {t['fve']:.2f}: {' '.join(t['text'].split())}")
        out.append("")
    return "\n".join(out).strip()


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    """inline.prompt's layout with this eval's setup text (no hint): the shared O-lens pieces, our task."""
    run = base[0]
    start, edit = env.view(case["base_arm"], case["cf_arm"], run)
    model = MODEL_NAME
    parts = [
        f"""\
## Setup

We gave an AI model a multiple-choice question about the author of a blog post, asked in the first \
person. This is the exact prompt and its reply (one letter).

<prompt>
{start}
</prompt>

<reply>
{run.messages[-1]["content"]}
</reply>

## The counterfactual

We will run {model} again, with the same settings, in a counterfactual arm that differs from the base arm only as \
follows. Everything else is identical.

{edit}"""
    ]
    extra = {}
    if condition.startswith("nla"):
        path = Path(case["inputs"]) / "reads"
        own = run.source.split("/", 1)[1]
        source = placebo(path, own, seed) if condition == "nla_placebo_inline" else own  # the O-lens placebo's pick
        if source != own:
            extra["placebo_sources"] = [source]
        text = NLA_TEXT.format(model=model, where=WHERE, layer=42, timing=TIMING)
        parts.append(f"## NLA reads\n\n{text}\n\n<nla_reads>\n{nla_block(path, own, run.messages, source)}\n</nla_reads>")
    elif condition != "transcript_inline":
        path = Path(case["inputs"]) / "reads"
        own = run.source.split("/", 1)[1]
        source = placebo(path, own, seed) if condition == "olens_placebo_inline" else own
        if source != own:
            extra["placebo_sources"] = [source]
        layers = sorted({r["layer"] for r in inline._reads(path)[0][own]})
        text = inline.OLENS_TEXT.format(
            model=model, where=WHERE, layers=", ".join(map(str, layers)), fve_note=inline.fve_note(path), timing=TIMING
        )
        parts.append(f"## O-lens reads\n\n{text}\n\n<olens_reads>\n{block(path, own, run.messages, source)}\n</olens_reads>")
    task = TASK_TEXT.format(model=model, answer=case["answer"], p0=case["p0"])
    parts.append(f"## Your task\n\n{task}")
    return "\n\n".join(parts), extra


def topup(out: Path):
    """olens_read_requests.py rows for the cases' pairs that have no reads yet: the prompt up to "What do you think?",
    read at bank.is_read_token's positions (any punctuation, symbol or newline token)."""
    from transformers import AutoTokenizer

    from scripts.author_probe.bank import read_positions

    tok = AutoTokenizer.from_pretrained(MODEL)
    have = {r["id"] for d in READS if (d / "requests.jsonl").exists() for r in load(d / "requests.jsonl")}
    pairs = {c["pair"]: c for c in collect() if c["pair"] not in have}
    with out.open("w") as f:
        for pair, c in pairs.items():
            user = c["prompt"][: c["prompt"].index(END) + len(END)]
            msgs = [{"role": "user", "content": user}]
            ids, pos = read_positions(tok, tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False), user)
            row = {"id": pair, "messages": msgs, "tools": None, "prompt_ids": ids, "positions": pos, "env": "author_blog", "arm": c["attr"]}
            f.write(json.dumps(row) + "\n")
    print("wrote", len(pairs), "requests to", out)


RUNS = ["2026-10-30_author-blog-drop-haiku-b2", "2026-10-31_author-blog-drop-haiku-nla"]  # batches 1+2; NLA (2026-10-31)
EXPVIEW = "https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/"


def batch_of(pair: str) -> int:
    """0 for the first 2,004 bloggers, 1 for the second (which reads run, which per-question dataset)."""
    if not hasattr(batch_of, "first"):
        batch_of.first = {r["id"] for r in load(READS[0] / "requests.jsonl")}  # ty: ignore[unresolved-attribute]
    return 0 if pair in batch_of.first else 1  # ty: ignore[unresolved-attribute]


def datasets(pair: str) -> dict[str, str]:
    b = ("", "-b")[batch_of(pair)]
    return {
        "pair: O-lens reads + sentence deletions": f"runs/author_mcq_blog/2026-10-30_qwen36-qfirst{b}/pairs",
        "O-lens reads (raw)": f"runs/olens_reads/2026-10-30_author-blog{b}",
        "NLA reads (raw)": "runs/nla_reads/2026-10-31_author-blog-nla-l42",
    }


def _links(case: dict) -> dict[str, str]:
    return {k: f"{EXPVIEW}#dataset={d}&item={case['pair']}" for k, d in datasets(case["pair"]).items()}


def LINKS(row: dict) -> dict[str, dict]:
    return {k: {"dataset": d, "item": row["pair"]} for k, d in datasets(row["pair"]).items()}


INPUTS = Path("data/runs/cfbench_evals/2026-10-30_author-blog-drop-inline-inputs-b2")


def ITEM_INFO(row: dict) -> dict:
    """The env viewer's per-case table (Kylie, 2026-10-31: show which cases the deletion makes a difference in): the
    truth as the class, Qwen's probability on its answer before and after the deletion, the drop and its threshold, and
    the deleted sentence."""
    if not hasattr(ITEM_INFO, "comment"):
        ITEM_INFO.comment = {c["item"]: c["comment"] for c in load(INPUTS / "cases.jsonl")}  # ty: ignore[unresolved-attribute]
    return {
        "source": row["version"],
        "class": "makes a difference" if row["k_cf"] else "no difference",
        "attribute": row["attr"],
        "p0": round(row["p0"], 3),
        "p1": round(row["p1"], 3),
        "drop": round(row["drop"], 3),
        "threshold": round(row["threshold"], 3),
        "deleted sentence": ITEM_INFO.comment.get(row["item"], ""),  # ty: ignore[unresolved-attribute]
    }


def show_expview(inputs: Path, out: Path):
    """The exact prompts --show prints, for every case and condition, as an expview run (no API calls): the predictor's
    system prompt and the user prompt with eval.md, nothing sent."""
    sys.path.insert(0, "expview")
    from cfbench.eval import with_notes
    from cfbench.predict import SYSTEM
    from expview import Run, call

    notes = (Path(__file__).parent / "eval.md").read_text()
    abl = {f"{r['passage']}:{r['attr']}": r for d in ABLATES_X2 for r in load(d / "pairs_scored.jsonl")}
    run = Run(out, title="author_blog_drop_inline: rendered prompts (not sent)",
              description="One item per case and condition: exactly what the investigator would be sent (--show), no reply.",
              label_order=["condition", "attr", "truth", "drop"])  # fmt: skip
    run.reset()
    for env, case, base in cases(inputs, list(VERSIONS)):
        for cond in CONDITIONS:
            prompt, extra = INLINE(env, cond, case | {"inputs": str(inputs)}, base, 0)
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": with_notes(prompt, notes)}]
            ab, i = abl[case["pair"]], int(case["item"].rsplit("#", 1)[1])
            table = ["| # | drop in p(answer) | TV | sentence |", "|---|---|---|---|"]
            for d in ab["deletions"]:
                x, b = -d["qwen_shift"], "**" if -d["qwen_shift"] > case["threshold"] else ""
                mark = " ← THIS CASE" if d["i"] == i else ""
                table.append(
                    f"| {d['i']}{mark} | {b}{x:+.3f}{b} | {d['tv']:.3f} | {b}{d['text'].replace(chr(10), ' ').replace('|', '/')}{b} |"
                )
            head = f"p0 {case['p0']:.2f} → p1 {case['p1']:.2f} (drop {case['drop']:+.3f}, threshold {case['threshold']:.2f}, "
            head += f"truth {case['k_cf']}); **bold** = evidence sentences of this pair\n\n"
            msgs2 = [{"role": "user", "content": "(not sent: the pair's sentence deletions)"}]
            deletions = call(
                "sentence deletions (not sent)", QWEN, messages=msgs2, output={"role": "assistant", "content": head + "\n".join(table)}
            )
            labels = {"condition": cond, "version": case["version"], "attr": case["attr"], "truth": case["k_cf"]}
            labels |= {"drop": round(case["drop"], 3), "p0": round(case["p0"], 3), "p1": round(case["p1"], 3)}
            labels |= {k: ", ".join(v) for k, v in extra.items()}
            calls = [call(f"investigator prompt ({cond})", "anthropic/claude-haiku-5.5", messages=msgs), deletions]
            run.add(id=f"{case['item']}/{cond}", group=f"{case['version']} · {cond}", links=_links(case), labels=labels, calls=calls)
    print("wrote", run.path)


if __name__ == "__main__" and sys.argv[1] == "topup":
    topup(Path(sys.argv[2]))
if __name__ == "__main__" and sys.argv[1] == "show-expview":
    show_expview(Path(sys.argv[2]), Path(sys.argv[3]))
if __name__ == "__main__" and sys.argv[1] == "add-nla":
    add_nla(Path(sys.argv[2]))
