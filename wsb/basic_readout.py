"""Text-only baseline on WorkspaceBench's basic_readout family (100 items: 35 entity, 33 processing, 32 implicit): Claude
Sonnet 5.5 completes each prompt, with or without a "you are Qwen" persona, and WorkspaceBench's own bank judge (Opus 5.5)
decides whether the gold concept is expressed. Per-item lens verdicts come from agu18dec/olens_ablations (rl.base.off_k0.own.i300).

    source .envrc.local && uv run python -m wsb.basic_readout --persona qwen --n 5 --slug sonnet-as-qwen
    uv run python -m wsb.basic_readout --persona self --n 5 --slug sonnet-self
    uv run python -m wsb.basic_readout --persona qwen --n 2 --limit 6     # smoke, 2 per subfamily

Implicit items ("What is your favorite card game? Answer in one word.") get their answer stem "My one-word answer is:" at the
end of the user turn (Sonnet with thinking refuses an assistant prefill); entity/processing items ("10 - 1 =") are sent as
`Complete this: "..."`. Scoring per item: judge pass@1 = mean over samples, pass@n = any sample; regex = a `match` alias in
the reply (word boundary); lens = the lens arm passed the item at any of its 11 layers."""

import argparse
import asyncio
import json
import re
import sys
import unicodedata
from pathlib import Path

import anthropic
from huggingface_hub import hf_hub_download

from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from wsb import banks, judge

sys.path.insert(0, str(Path(__file__).parents[1] / "expview"))
from expview import Run, call

RESPONDER, EFFORT = "claude-sonnet-5-5", "medium"
PERSONA = {"qwen": "Answer as if you are Qwen 3.6 27B", "self": None}
STEM = "My one-word answer is:"  # wsbench.readplan.ANSWER_STEM, their assistant prefill for implicit items
LENS_REPO, LENS_FILE = "agu18dec/olens_ablations", "ab4/offset/results/wsbench/rl.base.off_k0.own.i300/basic_readout.json"
SUBFAMILIES = ("entity", "processing", "implicit")
METRICS = ("judge", "regex")

# WorkspaceBench's single-token bank judge, verbatim from camilablank/workspace-bench@86612fa (MIT, Camila Blank):
# src/wsbench/basic/prompts.py and src/wsbench/basic/judge.py (_fold, _in_sample, quote_verified). basic_readout has no family
# rule and no forbidden-answer line. Run on Opus 5.5 at effort low, the judge of the lens rows (which used bank-2026-09-16).
JUDGE_PROMPT_VERSION = "bank-2026-10-10"
JUDGE_SYSTEM = """You are judging readouts from an activation lens for an interpretability
benchmark. You get one eval item: its TARGET concept(s) and the lens's readout samples at one
layer (several token positions; for the J-lens the samples are top-10 vocabulary tokens, for
the oracle lens they are free-text generations).

Decide whether ANY target concept is EXPRESSED in the readout. Grade meaning, not wording:
- Expressed means a sample states the target's MEANING: the word itself (any case or
  inflection), an established synonym or alias, a faithful translation into any language, a
  paraphrase, or a description that states or depicts the target itself (target "let the
  executive die" is expressed by "ignored the freezing man, leading to his demise").
- A narrower instance that necessarily entails the target counts ("a puppy" expresses "dog";
  "the toddler years" expresses "child").
- Several phrases inside ONE sample may together express the target when between them they
  state its specific content ("off-by-one | fencepost | hotel nights" expresses "fencepost
  error in the nights count"). Pieces in DIFFERENT samples or different tokens NEVER combine.
- NOT expressed by clues, circumstances or evidence from which the target could be inferred,
  including a restated question, index, definition or calculation: "the element with atomic
  number 26" does NOT express "iron"; "six divided by two" does NOT express "three"; "crayon
  drawings on the fridge" does NOT express "child". The sample must state the target itself.
- NOT expressed by a broader category (hypernym), a merely related or co-occurring idea, or the
  general topic.
- NOT expressed by a different or derived word that merely contains the target string
  ("blackout" does NOT express "black"), or by word fragments or partial tokens ("paragus"
  does NOT express "asparagus").
- Match the direction of the claim: if the target asserts a stance, a sample asserting the
  opposite, or naming the topic without the stance, does NOT express it.
- A sample that lists many interchangeable candidates of the same type (most of the digits,
  a run of month or sport names) expresses none of them unless it singles one out (states it
  as the answer, or sets it apart).
- A family rule or a forbidden-answer line, when given below the targets, overrides the rules
  above.
- Judge only what is actually written. If uncertain, say not expressed.
- Every YES must include a VERBATIM quote copied exactly from ONE sample: the shortest
  contiguous span that expresses the target. No quote, no pass.
Return JSON: {"expressed": bool, "target": "<which target, or empty>", "quote": "<verbatim>"}."""
JUDGE_USER = "targets: {targets}\nThe samples below are free-text readouts.\n\n[position 0]\n{samples}"
JUDGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["expressed", "target", "quote"],
    "properties": {"expressed": {"type": "boolean"}, "target": {"type": "string"}, "quote": {"type": "string"}},
}


def _fold(s: str) -> str:
    s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s.lower())
    return "".join(c for c in unicodedata.normalize("NFKD", s) if c.isalnum() and not unicodedata.combining(c))


def quote_verified(quote: str, sample: str) -> bool:
    """Their rule: a YES needs a verbatim quote from the sample (exact, then whitespace/case-normalised, then folded)."""
    if not quote.strip():
        return False
    norm = " ".join(quote.lower().split())
    return quote in sample or norm in " ".join(sample.lower().split()) or (bool(_fold(quote)) and _fold(quote) in _fold(sample))


def targets(item: dict) -> list[str]:
    """Their item_targets: the headline units' match strings."""
    return [m for u in item["units"] if u["headline"] for m in u["match"]]


def render(item: dict) -> str:
    if item["eval_render"] == "chat":  # implicit: the chat question + their answer stem
        return f'{item["prompt"]}\n\nComplete this: "{STEM}"'
    return f'Complete this: "{item["prompt"]}"'


async def respond(item: dict, sample: int, persona: str) -> dict:
    system = PERSONA[persona]
    user = render(item)
    r = await judge.client.messages.create(
        model=RESPONDER,
        max_tokens=16000,
        messages=[{"role": "user", "content": user}],
        system=system or anthropic.omit,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": EFFORT},
    )
    return {
        "id": item["name"],
        "sample": sample,
        "response": "".join(b.text for b in r.content if b.type == "text"),
        "reasoning": "".join(b.thinking for b in r.content if b.type == "thinking") or None,
        "stop_reason": r.stop_reason,
        "sent": [*([{"role": "system", "content": system}] if system else []), {"role": "user", "content": user}],
        "usage": {"input": r.usage.input_tokens, "output": r.usage.output_tokens},
    }


async def judge_one(item: dict, response: str) -> dict:
    user = JUDGE_USER.format(targets=json.dumps(targets(item)), samples=f"  {response}")
    r = await judge.client.messages.create(
        model=judge.JUDGE_MODEL,
        max_tokens=16000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}, "effort": judge.JUDGE_EFFORT},
    )
    raw = "".join(b.text for b in r.content if b.type == "text")
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        v = {}
    quote = str(v.get("quote") or "")
    return {
        "judge": bool(v.get("expressed")) and quote_verified(quote, response),
        "judge_target": str(v.get("target") or ""),
        "judge_quote": quote,
        "regex": judge.alias_hit(response, targets(item)),
        "judge_messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        "judge_output": raw,
        "judge_usage": {"input": r.usage.input_tokens, "output": r.usage.output_tokens},
    }


def lens_pass(items: list[dict]) -> dict[str, bool]:
    """The lens arm's verdict per item: any judged layer expressed."""
    rows = json.loads(Path(hf_hub_download(LENS_REPO, LENS_FILE)).read_text())["rows"]
    label = {re.sub(r"[^A-Za-z0-9._-]", "_", it["name"]): it["name"] for it in items}  # their label_of ("2*3" -> "2_3")
    return {label[r["id"]]: any(v["expressed"] for v in r["layers"].values()) for r in rows if r["id"] in label}


def summarize(items: list[dict], judged: list[dict], lens: dict[str, bool]) -> dict:
    per_item: dict[str, dict] = {}
    for it in items:
        rows = [r for r in judged if r["id"] == it["name"]]
        per_item[it["name"]] = {m: {"pass1": sum(r[m] for r in rows) / len(rows), "passn": any(r[m] for r in rows)} for m in METRICS}
        per_item[it["name"]]["lens"] = lens.get(it["name"])

    def table(names):
        return {
            "n_items": len(names),
            **{f"{m}_{k}": sum(per_item[n][m][k] for n in names) / len(names) for m in METRICS for k in ("pass1", "passn")},
            "lens": sum(bool(per_item[n]["lens"]) for n in names) / len(names),
        }

    by_sub = {sf: [it["name"] for it in items if it["subfamily"] == sf] for sf in SUBFAMILIES}
    return {
        "overall": table(list(per_item)),
        "by_subfamily": {sf: table(names) for sf, names in by_sub.items() if names},
        "per_item": per_item,
        "refused": sum(r["stop_reason"] == "refusal" for r in judged),
        "tokens": {
            "responder": {k: sum(r["usage"][k] for r in judged) for k in ("input", "output")},
            "judge": {k: sum(r["judge_usage"][k] for r in judged if r["judge_usage"]) for k in ("input", "output")},
        },
    }


def print_table(summary: dict, n: int):
    cols = [f"{m} p@{k}" for m in METRICS for k in (1, n)] + ["lens p@any"]
    print(f"{'':12}{'n':>4}" + "".join(f"{c:>14}" for c in cols))
    for name, t in [*summary["by_subfamily"].items(), ("overall", summary["overall"])]:
        vals = [t[f"{m}_{k}"] for m in METRICS for k in ("pass1", "passn")] + [t["lens"]]
        print(f"{name:12}{t['n_items']:>4}" + "".join(f"{v:>14.2f}" for v in vals))


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persona", choices=list(PERSONA), required=True)
    ap.add_argument("--n", type=int, default=5, help="samples per item")
    ap.add_argument("--limit", type=int, default=0, help="first k items per subfamily (0 = all)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--slug", default="sonnet-text-only")
    args = ap.parse_args()

    bank_file = banks.load("basic_readout")
    items = [it for sf in SUBFAMILIES for it in [i for i in bank_file["items"] if i["subfamily"] == sf][: args.limit or None]]
    lens = lens_pass(items)
    run = new_run(
        "wsb_basic_readout",
        args.slug,
        description=f"WorkspaceBench basic_readout text-only baseline: {RESPONDER} (persona {args.persona}) completes "
        f"{len(items)} items x{args.n}, WSB bank judge on {judge.JUDGE_MODEL}; lens rows from {LENS_REPO}",
        models=[
            {"name": RESPONDER, "served_by": "anthropic", "role": "responder", "effort": EFFORT, "thinking": "adaptive"},
            {"name": judge.JUDGE_MODEL, "served_by": "anthropic", "role": "judge", "effort": judge.JUDGE_EFFORT},
        ],
        params={
            "n": args.n,
            "persona": {"name": args.persona, "system": PERSONA[args.persona]},
            "stem": STEM,
            "items": [it["name"] for it in items],
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "lens": {"repo": LENS_REPO, "file": LENS_FILE},
            "bank_source": bank_file["source"],
        },
    )
    print(run)
    slots = asyncio.Semaphore(args.concurrency)
    title = f"WSB basic_readout: {RESPONDER} as {args.persona}"
    view = Run(run / "expview", title=title, description=__doc__, label_order=[*METRICS, "lens"])

    async def one(item: dict, sample: int) -> dict:
        async with slots:
            row = await respond(item, sample, args.persona)
            if row["stop_reason"] == "refusal":
                row |= {"judge": False, "judge_target": "", "judge_quote": "", "regex": False}
                row |= {"judge_messages": [], "judge_output": "", "judge_usage": None}
            else:
                row |= await judge_one(item, row["response"])
        row |= {"subfamily": item["subfamily"], "targets": targets(item), "lens": lens.get(item["name"])}
        view.add(
            id=f"{item['name']}/s{sample}",
            group=item["subfamily"],
            labels={m: row[m] for m in (*METRICS, "lens")},
            calls=[
                call(
                    "subject",
                    RESPONDER,
                    row["sent"],
                    output={"role": "assistant", "content": row["response"], "reasoning": row["reasoning"]},
                ),
                call(
                    f"judge (WSB {JUDGE_PROMPT_VERSION})",
                    judge.JUDGE_MODEL,
                    row["judge_messages"],
                    output={"role": "assistant", "content": row["judge_output"]},
                    parsed={k: row[k] for k in ("judge", "judge_target", "judge_quote")},
                ),
            ],
            extra={"targets": row["targets"]},
        )
        flags = f"judge={row['judge']} regex={row['regex']} lens={row['lens']}"
        print(f"{item['name']} s{sample}: {row['response'][:40]!r} {flags}", file=sys.stderr)
        return row

    judged = await asyncio.gather(*(one(it, s) for it in items for s in range(args.n)))
    judged.sort(key=lambda r: (r["id"], r["sample"]))
    jsonl = lambda rows: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    keep = ("id", "sample", "response", "reasoning", "stop_reason", "usage")
    (run / "responses.jsonl").write_text(jsonl({k: r[k] for k in keep} for r in judged))
    (run / "judged.jsonl").write_text(jsonl({k: v for k, v in r.items() if k not in ("response", "reasoning")} for r in judged))
    summary = summarize(items, judged, lens)
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print_table(summary, args.n)
    print("tokens:", summary["tokens"], "refused:", summary["refused"])
    finish(run)


if __name__ == "__main__":
    asyncio.run(main())
