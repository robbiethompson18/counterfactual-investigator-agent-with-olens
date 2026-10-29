"""Investigator on WorkspaceBench's knowledge_cutoff: an investigator model is shown a question Qwen3.6-27B was asked and
predicts Qwen's reply, without seeing it (Robbie, 2026-10-29: WSB's task with an investigator at pass@1). The label is Qwen's
own answer: the bank's gold for the free-text `recall` format, and for `mcq` (the question over the bank's 5 options, A-E)
the modal letter of a `wsb.knowledge_cutoff --arm mcq` run (`--qwen-mcq-run`; without one the gold's letter, recorded as such).
Decisions and results so far: docs/wsb-cutoff-investigator.md.

Investigator: `--investigator`, default Claude Haiku 5.5 at effort medium (Robbie, 2026-10-29, after the cutoff_census: Gemini 3.8
Flash shares Qwen's stale answers on a quarter of the items, Haiku on 4%). Versions `told` / `hidden`: whether the system prompt
names the model and its cutoff; default hidden only (Robbie: told makes a result harder to see). Conditions: `transcript_inline`
(the prompt alone), `olens_placebo_inline` (plus O-lens reads of another prompt's last six tokens, from `--placebo`, presented
as reads of this one: the control for the reads block itself), `olens_inline` (the real reads, `--reads`; none exist yet). One
turn each, no tools but submit_answer, 5 draws.

    source .envrc.local && uv run python -m wsb.cutoff_investigator --show --limit 1
    uv run python -m wsb.cutoff_investigator --slug recall-haiku-dev109                 # the 109 dev items, recall, hidden
    uv run python -m wsb.cutoff_investigator --formats mcq --limit 10 --qwen-mcq-run data/runs/wsb_knowledge_cutoff/<run>

Items: `--items`, else the bank items (first `--limit`, 0 = all) whose answer the `--answer-run` (Qwen on our server, 10 samples)
confirmed stable (gold in >= 70% of samples), with a non-numeric gold (token lenses can't name multi-digit numbers) and no prompt
leak. Scoring is the bank's alias regex (wsb/knowledge_cutoff.py's port of their score.py; case- and accent-insensitive).
Run dir data/runs/wsb_cutoff_investigator/<date>_<slug>/: responses.jsonl (one row per prediction), summary.json, expview/ (one
item per prediction: the whole conversation); `runs.py finish` syncs it to HF and Docent (collection wsb_cutoff_investigator).
"""

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from openai.types.chat import ChatCompletionMessageFunctionToolCall

from cfbench import answer
from cfbench.olens import show_token
from counterfactual_investigator_agent_with_olens import openrouter
from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from wsb import banks
from wsb.knowledge_cutoff import LETTERS, OptionForms, mcq_letter, render_mcq

sys.path.insert(0, str(Path(__file__).parents[1] / "expview"))
from expview import Run, call

INVESTIGATOR = "anthropic/claude-haiku-5.5"
ANSWER_RUN = Path("data/runs/wsb_knowledge_cutoff/2026-10-28_qwen-answer-k10-vllm")
PLACEBO = Path("data/runs/olens_reads/2026-10-29_hint-qwen-dsv4-tail/reads.placebo-head.jsonl")  # Kylie's end-of-prompt reads
FORMATS = ("recall", "mcq")
VERSIONS = ("told", "hidden")
CONDITIONS = ("transcript_inline", "olens_placebo_inline", "olens_inline")
CONTEXT = 40

# Robbie's prompt, verbatim (2026-10-29)
SYSTEM = (
    'You are trying to predict a language model\'s response to a question. For example, if you see "User: "what is 1+1", you '
    "should return `submit_answer(2)`. No explanation required."
)
TOLD = " The language model is Qwen3.6-27B, an open-weight model with a knowledge cutoff around January 2025."
USER = """\
Here is the scenario
```
User: "{prompt}"
assistant:
```
{reads}Please use the submit_answer tool to predict what the agent responds with."""
OLENS_TEXT = """\
## O-lens reads

The O-lens is a model trained to describe another model's internal activations in words. It was run on {model}'s activations \
at the end of the prompt above: the period that ends its last line, the end-of-turn token and the newline after it, and the \
start of {model}'s own turn (<|im_start|>, "assistant" and the newline), just before it answers, at layers {layers}. Each read \
is a few short bullets about what the model is representing at that token, at one layer, each with its FVE (fraction of the \
activation's variance the bullet explains): high FVE is good, low FVE bullets are often hallucinated. Reads can show what the \
model registers but never says; they also often just paraphrase the nearby text. The O-lens is more likely to get fine details \
wrong (exact numbers, letters, names) than broad themes.

Each position below shows the end of the prompt with the token read wrapped in <read></read> (↵ is a newline), then its \
bullets, by layer and highest FVE first within a layer.

<olens_reads_end>
{block}
</olens_reads_end>

"""


def smoke_items(bank: dict, n: int, answer_run: Path) -> list[dict]:
    per_item = json.loads((answer_run / "summary.json").read_text())["per_item"]
    ok = [
        it
        for it in bank["items"]
        if (s := per_item.get(it["id"])) and s["gold_rate"] >= 0.7
        and not re.fullmatch(r"[\d.,k]+", it["gold"]) and not it.get("g3_leak")
    ]  # fmt: skip
    return ok[:n] if n else ok


def reads_by_source(path: Path) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for line in path.open():
        r = json.loads(line)
        out[r["id"]].append(r)
    return dict(out)


def tail_block(rows: list[dict], prompt: str) -> str:
    """Kylie's end-of-prompt block (cfbench/evals/hint_qwen_dsv4/inline.py tail_block): the six tail tokens in order, the
    read one wrapped, then the bullets by layer; here the lead is the end of this item's prompt."""
    by_tok: dict[int, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_tok[r["token_index"]][r["layer"]] = r
    toks = [next(iter(layers.values()))["token"] for _, layers in sorted(by_tok.items())]
    lead = "…" + prompt.rstrip()[-CONTEXT:-1].replace("\n", "↵")
    out = []
    for k, (_, layers) in enumerate(sorted(by_tok.items())):
        shown = "".join(f"<read>{show_token(t)}</read>" if j == k else t.replace("\n", "↵") for j, t in enumerate(toks))
        out.append(f"At {json.dumps(toks[k], ensure_ascii=False)}: {lead}{shown}")
        for layer in sorted(layers):
            for b in sorted(layers[layer]["bullets"], key=lambda b: -b["fve"]):
                out.append(f"- layer {layer}, FVE {b['fve']:.2f}: {' '.join(b['text'].split())}")
        out.append("")
    return "\n".join(out).strip()


def build(item: dict, fmt: str, version: str, condition: str, rows: list[dict] | None) -> list[dict]:
    prompt = render_mcq(item) if fmt == "mcq" else item["user"]
    model = "Qwen3.6-27B" if version == "told" else "the model"
    reads = ""
    if condition != "transcript_inline":
        assert rows, condition
        layers = ", ".join(map(str, sorted({r["layer"] for r in rows})))
        reads = OLENS_TEXT.format(model=model, layers=layers, block=tail_block(rows, prompt))
    return [
        {"role": "system", "content": SYSTEM + (TOLD if version == "told" else "")},
        {"role": "user", "content": USER.format(prompt=prompt, reads=reads)},
    ]


async def investigate(model: str, messages: list[dict], effort: str) -> tuple[dict | None, list[dict], float]:
    """cfbench/predict.py's loop without tools: up to 3 turns to produce one valid submit_answer call."""
    messages, submission, cost = list(messages), None, 0.0
    for _ in range(3):
        r = await openrouter.chat(model, messages, effort=effort, tools=[answer.SUBMIT], max_tokens=8000)
        cost += getattr(r.usage, "cost", 0) or 0
        msg = r.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            messages.append({"role": "user", "content": answer.NUDGE})
            continue
        for tc in msg.tool_calls:
            assert isinstance(tc, ChatCompletionMessageFunctionToolCall)  # the only tool offered is a function
            result, reply = answer.check(tc.function.arguments)
            submission = submission or result
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": reply})
        if submission:
            break
    return submission, messages, cost


def mcq_labels(run: Path | None, items: list[dict]) -> dict[str, tuple[str, str]]:
    """item id -> (label letter, where it came from): the modal letter of the Qwen mcq run, else the gold's letter."""
    out: dict[str, tuple[str, str]] = {it["id"]: (LETTERS[it["options"].index(it["gold"])], "gold_letter") for it in items}
    if run:
        letters = defaultdict(Counter)
        for r in map(json.loads, (run / "responses.jsonl").read_text().splitlines()):
            if (x := mcq_letter(r["response"])) is not None:
                letters[r["id"]][x] += 1
        for it in items:
            if letters.get(it["id"]):
                out[it["id"]] = (letters[it["id"]].most_common(1)[0][0], f"modal of {run.name} ({dict(letters[it['id']])})")
    return out


def grade(item: dict, fmt: str, text: str | None, label: str) -> dict:
    if text is None:
        return {"correct": None}
    if fmt == "mcq":
        letter = mcq_letter(text)
        return {"letter": letter, "correct": letter == label}
    named = sorted(OptionForms.build(item["option_units"]).named(text))
    return {"named": named, "correct": item["gold"] in named, "names_truth": (item.get("truth") or {}).get("canonical") in named}


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--formats", default="recall")
    ap.add_argument("--versions", default="hidden")
    ap.add_argument("--conditions", default="transcript_inline,olens_placebo_inline")
    ap.add_argument("--items", help="comma-separated bank item ids")
    ap.add_argument("--limit", type=int, default=0, help="first N items (0 = all)")
    ap.add_argument("--answer-run", type=Path, default=ANSWER_RUN)
    ap.add_argument("--qwen-mcq-run", type=Path, help="a `wsb.knowledge_cutoff --arm mcq` run: the mcq labels")
    ap.add_argument("--placebo", type=Path, default=PLACEBO, help="reads.jsonl of end-of-prompt reads on other prompts")
    ap.add_argument("--reads", type=Path, help="reads.jsonl of this bank's own end-of-prompt reads (olens_inline)")
    ap.add_argument("--investigator", default=INVESTIGATOR)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="print the first item's prompts and exit")
    ap.add_argument("--slug")
    args = ap.parse_args()
    formats, versions, conditions = (a.split(",") for a in (args.formats, args.versions, args.conditions))
    assert set(formats) <= set(FORMATS) and set(versions) <= set(VERSIONS) and set(conditions) <= set(CONDITIONS)
    bank = banks.load("knowledge_cutoff")
    by_id = {it["id"]: it for it in bank["items"]}
    items = [by_id[i] for i in args.items.split(",")] if args.items else smoke_items(bank, args.limit, args.answer_run)
    placebo = reads_by_source(args.placebo) if "olens_placebo_inline" in conditions else {}
    own = reads_by_source(args.reads) if args.reads else {}
    labels = mcq_labels(args.qwen_mcq_run, items)

    def rows_for(item: dict, condition: str, draw: int) -> tuple[list[dict] | None, str | None]:
        if condition == "olens_placebo_inline":
            src = random.Random(f"{args.seed}:{item['id']}:{draw}").choice(sorted(placebo))
            return placebo[src], src
        if condition == "olens_inline":
            return own[item["id"]], None
        return None, None

    if args.show:
        it = items[0]
        for fmt in formats:
            for version in versions:
                for condition in conditions:
                    rows, src = rows_for(it, condition, 0)
                    print(f"===== {it['id']} {fmt} {version} {condition} placebo={src} =====")
                    for m in build(it, fmt, version, condition, rows):
                        print(f"[{m['role']}]\n{m['content']}\n")
        return
    slug = args.slug or f"{'-'.join(formats)}-smoke{len(items)}"
    run = new_run(
        "wsb_cutoff_investigator",
        slug,
        description=f"knowledge_cutoff investigator: {len(items)} items x {formats} x {versions} x {conditions} x "
        f"{args.draws} draws, {args.investigator} effort {args.effort}, Robbie's one-turn predict-the-reply prompt",
        models=[
            {"name": args.investigator, "served_by": "openrouter", "role": "investigator", "effort": args.effort},
            {"name": "Qwen/Qwen3.6-27B", "role": "subject", "note": "labels from the bank (recall) / --qwen-mcq-run (mcq)"},
        ],
        params={
            "formats": formats, "versions": versions, "conditions": conditions, "draws": args.draws, "seed": args.seed,
            "items": [it["id"] for it in items], "answer_run": str(args.answer_run), "qwen_mcq_run": str(args.qwen_mcq_run),
            "placebo": str(args.placebo), "reads": str(args.reads), "mcq_labels": {k: v for k, v in labels.items()},
            "system": SYSTEM, "told": TOLD, "user": USER, "bank_source": bank["source"],
        },
    )  # fmt: skip
    print(run)
    slots = asyncio.Semaphore(args.concurrency)
    order = ["correct", "format", "version", "condition"]
    view = Run(run / "expview", title=f"knowledge_cutoff investigator ({slug})", description=__doc__, label_order=order)

    async def one(item: dict, fmt: str, version: str, condition: str, draw: int) -> dict:
        rows, src = rows_for(item, condition, draw)
        sent = build(item, fmt, version, condition, rows)
        async with slots:
            submission, messages, cost = await investigate(args.investigator, sent, args.effort)
        text = submission["answer"] if submission else None
        label = labels[item["id"]][0] if fmt == "mcq" else item["gold"]
        row = {"id": item["id"], "format": fmt, "version": version, "condition": condition, "draw": draw, "answer": text,
               "rationale": submission["rationale"] if submission else None, "label": label, **grade(item, fmt, text, label),
               "placebo_source": src, "cost": cost, "messages": messages}  # fmt: skip
        if submission:  # the verdict goes onto the stored submit reply, for reading the transcript; the investigator never saw it
            tool_msg = next(m for m in reversed(messages) if m["role"] == "tool")
            tool_msg["content"] += f" [graded after the run: {'CORRECT' if row['correct'] else 'WRONG'}; Qwen's answer: {label!r}]"
        print(f"{item['id']} {fmt:6} {version:6} {condition:22} draw {draw}: {text!r} -> {row['correct']} ({label!r})", file=sys.stderr)
        return row

    jobs = [one(it, f, v, c, d) for it in items for f in formats for v in versions for c in conditions for d in range(args.draws)]
    rows = await asyncio.gather(*jobs)
    for r in rows:
        it = by_id[r["id"]]
        view.add(
            id=f"{r['id']}~{r['format']}~{r['version']}~{r['condition']}~{r['draw']}",
            labels={k: r[k] for k in ("correct", "format", "version", "condition")} | {"label_type": it["label_type"]},
            calls=[call("investigate", args.investigator, r["messages"][:2], output=r["messages"][2:], tools=[answer.SUBMIT],
                        params={"effort": args.effort}, parsed={k: r.get(k) for k in ("answer", "named", "letter", "label")})],
            extra={"question": it["question"], "gold": it["gold"], "truth": (it.get("truth") or {}).get("canonical"),
                   "options": it["options"], "label": r["label"], "placebo_source": r["placebo_source"], "draw": r["draw"]},
        )  # fmt: skip
    keep = [k for k in rows[0] if k != "messages"]
    (run / "responses.jsonl").write_text("".join(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n" for r in rows))
    cells = defaultdict(list)
    for r in rows:
        cells[r["format"], r["version"], r["condition"]].append(r["correct"])
    summary = {
        "n_items": len(items),
        "accuracy": {
            " / ".join(k): {"n": len(v), "invalid": sum(c is None for c in v), "correct": sum(bool(c) for c in v) / len(v)}
            for k, v in sorted(cells.items())
        },
        "mcq_labels": {k: v for k, v in labels.items()},
        "cost": sum(r["cost"] for r in rows),
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary["accuracy"], indent=1))
    finish(run)


if __name__ == "__main__":
    asyncio.run(main())
