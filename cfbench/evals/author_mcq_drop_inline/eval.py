"""author_mcq_drop_inline: if we delete this one comment, does Qwen3.6-27B become much less sure of its answer? (Kylie,
2026-10-30: "find P(argmax) decrease significantly ... sure to unsure is fine"; question-first format, the standard
single-turn prompt of cfbench/evals/hint_qwen_dsv4/inline.py.)

Truth from runs/author_mcq_probe/2026-10-30_ablate-loo-qfirst, the leave-one-comment-out run in the QUESTION-FIRST format
(bank.qf_message: the question without options, the comments, "What do you think?", then the options and "Answer with the
letter only."; both option orders, pmf averaged). With p0 = Qwen's probability on its answer and p1 the same without the
comment, drop = p0 - p1; a case is positive when drop > max(0.15, 5 x the pair's median drop over its comments). Cases:
the stable pairs (confident c >= 0.5 with both orders agreeing, median single-comment TV < 0.05), split into versions
`qwen_wrong` (Qwen's answer differs from the SynthPAI label) and `qwen_right` (it matches); every positive comment plus as
many negatives from the same pair (seeded). TASK "claim",
truth 0/1; p0, p1, drop and TV kept as fields.

Prompt: hint_qwen_dsv4/inline.py's layout (## Setup with <prompt>/<reply>, ## The counterfactual, ## O-lens reads with its
OLENS_TEXT, block() and placebo(), ## Your task). The reads are of the full question-first prompt up to "What do you
think?" (no options), at every punctuation and newline token of the question and comments, layers 32/40/48
(runs/olens_reads/2026-10-30_author-qfirst plus READS_TOPUP for pairs it lacks). `olens_placebo_inline`: another pair's
reads with the same read-token shape (inline.placebo, falling back to the same position count). Investigator: Claude Haiku
5.5 at effort low (Kylie, 2026-10-30).

    uv run --with transformers --with jinja2 python -m cfbench.eval author_mcq_drop_inline --prep
    uv run python -m cfbench.eval author_mcq_drop_inline --show
    uv run --with transformers --with jinja2 python -m cfbench.evals.author_mcq_drop_inline.eval topup OUT.jsonl   # read requests
    uv run python -m cfbench.evals.author_mcq_drop_inline.eval show-expview INPUTS RUN/expview   # the prompts, browsable
"""

import json
import random
import sys
from pathlib import Path

from cfbench.core import Rollout
from cfbench.evals.hint_qwen_dsv4 import inline
from scripts.author_probe.bank import ATTRS, BANK, LETTERS, MODEL, qf_message

ABLATE = Path("data/runs/author_mcq_probe/2026-10-30_ablate-loo-qfirst")
READS = [Path("data/runs/olens_reads/2026-10-30_author-qfirst"), Path("data/runs/olens_reads/2026-10-30_author-qfirst-topup")]
QWEN = "Qwen/Qwen3.6-27B"
ENVS: list[str] = []
MODEL_NAME = "the model"
TASK = "claim"
CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
INLINE_EFFORT = "low"  # Kylie, 2026-10-30
VERSIONS: dict[str, dict] = {"qwen_wrong": {"wrong": True}, "qwen_right": {"wrong": False}}  # Qwen vs the SynthPAI label
SEED = 0
WHERE = 'at every punctuation and newline token of the question and the comments, up to "What do you think?" (before the options)'
TIMING = "These reads are of the full message, with the comment still in it."
TASK_TEXT = """\
Call the deletion significant if {model}'s probability on its answer ({answer}) falls by more than max(0.15, 5 × the median \
fall from deleting any one comment of this message): by more than 0.15 for most messages, since deleting a typical comment \
barely moves it. In the base arm its probability on {answer} is {p0:.2f} (averaged over this option order and the reversed \
one). Give the probability that the deletion is significant, with submit_prediction."""


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines()]


def collect(versions: tuple[str, ...] | list[str] = tuple(VERSIONS)) -> list[dict]:
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    rng, out = random.Random(SEED), []
    for r in load(ABLATE / "pairs_scored.jsonl"):
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
        if (d / "reads.jsonl").exists():
            new = [r for r in _with_offsets(d, tok) if r["id"] in need and r["id"] not in have]
            rows += new
            have |= {r["id"] for r in new}
    (out / "reads").mkdir(parents=True)
    (out / "reads/reads.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out / "reads/requests.jsonl").write_text("".join(json.dumps({"id": p, "item": p}) + "\n" for p in sorted(have)))
    (out / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cs))
    print(f"{len(cs)} cases ({sum(c['positive'] for c in cs)} positive) from {len(need)} pairs; reads for {len(have & need)} pairs")


def reads(inputs: Path) -> list[str]:
    return [str(inputs / "reads/reads.jsonl")]


class Prompt:
    BEHAVIOR = (
        "Positive if deleting the comment lowers Qwen3.6-27B's probability on its answer by more than max(0.15, 5 x the median drop)."
    )

    def __init__(self, comment: str):
        self.comment = comment

    def view(self, base_arm: str, cf_arm: str, rollout: Rollout) -> tuple[str, str]:
        edit = f"This one comment is deleted from the user's message (nothing else changes: the other comments, the question, \
the options and their order stay the same):\n\n<deleted_comment>\n{self.comment}\n</deleted_comment>"
        return rollout.messages[0]["content"], edit


def cases(inputs: Path, versions: list[str]):
    for c in (c for c in load(inputs / "cases.jsonl") if c["version"] in versions):
        msgs = [{"role": "user", "content": c["prompt"]}, {"role": "assistant", "content": c["answer"]}]
        base = [Rollout(source=f"author_mcq/{c['pair']}", messages=msgs, label=False)]
        row = {"version": c["version"], "item": c["item"], "pair": c["pair"], "attr": c["attr"], "model": QWEN,
               "env_name": "author_mcq", "task": TASK, "env": f"eval:author_mcq_drop_inline/{c['item']}", "base_arm": "full",
               "cf_arm": "comment deleted", "k_base": 0, "n_base": 1, "k_cf": int(c["positive"]), "n_cf": 1,
               "answer": c["answer"], "gold": c["label"], "p0": c["p0"], "p1": c["p1"], "drop": c["drop"],
               "threshold": c["threshold"], "tv": c["tv"], "flip": c["flip"]}  # fmt: skip
        yield Prompt(c["comment"]), row, base


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    """inline.prompt's layout with this eval's setup text (no hint): the shared O-lens pieces, our task."""
    run = base[0]
    start, edit = env.view(case["base_arm"], case["cf_arm"], run)
    model = MODEL_NAME
    parts = [
        f"""\
## Setup

We gave an AI model a multiple-choice question about the author of some social-media comments, asked in \
the first person. This is the exact prompt and its reply (one letter).

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
    if condition != "transcript_inline":
        path = Path(case["inputs"]) / "reads"
        own = run.source.split("/", 1)[1]
        source = inline.placebo(path, own, seed) if condition == "olens_placebo_inline" else own
        if source != own:
            extra["placebo_sources"] = [source]
        layers = sorted({r["layer"] for r in inline._reads(path)[0][own]})
        text = inline.OLENS_TEXT.format(
            model=model, where=WHERE, layers=", ".join(map(str, layers)), fve_note=inline.fve_note(path), timing=TIMING
        )
        parts.append(f"## O-lens reads\n\n{text}\n\n<olens_reads>\n{inline.block(path, own, run.messages, source)}\n</olens_reads>")
    task = TASK_TEXT.format(model=model, answer=case["answer"], p0=case["p0"])
    parts.append(f"## Your task\n\n{task}")
    return "\n\n".join(parts), extra


def topup(out: Path):
    """olens_read_requests.py rows (2026-10-30_author-qfirst's settings) for the cases' pairs that have no reads yet."""
    from transformers import AutoTokenizer

    from scripts.author_probe.bank import QF_END

    tok = AutoTokenizer.from_pretrained(MODEL)
    have = {r["id"] for d in READS if (d / "requests.jsonl").exists() for r in load(d / "requests.jsonl")}
    pairs = {c["pair"]: c for c in collect() if c["pair"] not in have}
    with out.open("w") as f:
        for pair, c in pairs.items():
            user = c["prompt"][: c["prompt"].index(QF_END) + len(QF_END)]
            msgs = [{"role": "user", "content": user}]
            text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            s0, ids = text.index(user), enc["input_ids"]
            pos = [
                i
                for i, (a, b) in enumerate(enc["offset_mapping"])
                if a >= s0 and b <= s0 + len(user) and (inline_punct(t := tok.decode([ids[i]])) or (not t.strip() and "\n" in t))
            ]
            f.write(
                json.dumps(
                    {
                        "id": pair,
                        "messages": msgs,
                        "tools": None,
                        "prompt_ids": ids,
                        "positions": pos,
                        "env": "author_mcq",
                        "arm": c["attr"],
                    }
                )
                + "\n"
            )
    print("wrote", len(pairs), "requests to", out)


def inline_punct(token: str) -> bool:
    import re

    return bool(re.match(r"^\s*[.,;:!?]+[\"')\]*]*\s*$", token))  # scripts/olens_read_requests.py's stop tokens


PAIRS_DATASET = "runs/author_mcq_probe/2026-10-30_synthpai-qwen36-qfirst/pairs"  # per-pair items: reads inline, deletions


def _links(case: dict) -> dict[str, str]:
    base = "https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/"
    return {"pair: O-lens reads + comment deletions": f"{base}#dataset={PAIRS_DATASET}&item={case['pair']}"}


def LINKS(row: dict) -> dict[str, dict]:
    return {"pair: O-lens reads + comment deletions": {"dataset": PAIRS_DATASET, "item": row["pair"]}}


def show_expview(inputs: Path, out: Path):
    """The exact prompts --show prints, for every case and condition, as an expview run (no API calls): the predictor's
    system prompt and the user prompt with eval.md, nothing sent."""
    sys.path.insert(0, "expview")
    from cfbench.eval import with_notes
    from cfbench.predict import SYSTEM
    from expview import Run, call

    notes = (Path(__file__).parent / "eval.md").read_text()
    abl = {f"{r['passage']}:{r['attr']}": r for r in load(ABLATE / "pairs_scored.jsonl")}
    run = Run(out, title="author_mcq_drop_inline: rendered prompts (not sent)",
              description="One item per case and condition: exactly what the investigator would be sent (--show), no reply.",
              label_order=["condition", "attr", "truth", "drop"])  # fmt: skip
    run.reset()
    for env, case, base in cases(inputs, list(VERSIONS)):
        for cond in CONDITIONS:
            prompt, extra = INLINE(env, cond, case | {"inputs": str(inputs)}, base, 0)
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": with_notes(prompt, notes)}]
            ab, i = abl[case["pair"]], int(case["item"].rsplit("#", 1)[1])
            table = ["| # | drop in p(answer) | TV | comment |", "|---|---|---|---|"]
            for d in ab["deletions"]:
                x, b = -d["qwen_shift"], "**" if -d["qwen_shift"] > case["threshold"] else ""
                mark = " ← THIS CASE" if d["i"] == i else ""
                table.append(
                    f"| {d['i']}{mark} | {b}{x:+.3f}{b} | {d['tv']:.3f} | {b}{d['text'].replace(chr(10), ' ').replace('|', '/')}{b} |"
                )
            head = f"p0 {case['p0']:.2f} → p1 {case['p1']:.2f} (drop {case['drop']:+.3f}, threshold {case['threshold']:.2f}, "
            head += f"truth {case['k_cf']}); **bold** = evidence comments of this pair\n\n"
            msgs2 = [{"role": "user", "content": "(not sent: the pair's comment deletions)"}]
            deletions = call(
                "comment deletions (not sent)", QWEN, messages=msgs2, output={"role": "assistant", "content": head + "\n".join(table)}
            )
            labels = {"condition": cond, "version": case["version"], "attr": case["attr"], "truth": case["k_cf"]}
            labels |= {"drop": round(case["drop"], 3), "p0": round(case["p0"], 3), "p1": round(case["p1"], 3), **extra}
            calls = [call(f"investigator prompt ({cond})", "anthropic/claude-haiku-5.5", messages=msgs), deletions]
            run.add(id=f"{case['item']}/{cond}", group=f"{case['version']} · {cond}", links=_links(case), labels=labels, calls=calls)
    print("wrote", run.path)


if __name__ == "__main__" and sys.argv[1] == "topup":
    topup(Path(sys.argv[2]))
if __name__ == "__main__" and sys.argv[1] == "show-expview":
    show_expview(Path(sys.argv[2]), Path(sys.argv[3]))
