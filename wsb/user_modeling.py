"""WorkspaceBench user_modeling with a text-only reader: Gemini answers each item's conversation as the assistant, and WSB's
own um-v2 judge grades the reply exactly as it grades an O-lens readout (docs/wsb-flash-prompt-only.md, "Still to do").

    uv run python -m wsb.user_modeling --subfamily selfdescribe --n 3 --slug selfdescribe47

The judge prompts, option draw and scoring rule are ported verbatim from vendor/workspace-bench
(src/wsbench/evals/user_modeling/{prompts,judge,score}.py at 86612fa); the bank is its evals/user_modeling/items.json.
An item passes a draw when the judge picks the gold attribute with basis `inferred_characterization`. Outputs land in
data/runs/wsb_user_modeling/<date>_<slug>/: rows.jsonl (one per draw, with the judge's verdict), summary.json, and an
expview run with every responder and judge call in full.
"""

import argparse
import asyncio
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from counterfactual_investigator_agent_with_olens import openrouter
from counterfactual_investigator_agent_with_olens.runs import finish, new_run

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "expview"))
from expview import Run, call

BANK = Path(__file__).parent / "banks" / "user_modeling.json"
SEED = 0
N_DISTRACTORS = 4
CANNOT_TELL = "cannot tell from the readout"
INFERRED = "inferred_characterization"
JUDGE = "google/gemini-3.8-flash"  # WSB's judge, at effort minimal (their DEFAULT_REASONING)

UM_OPTION_SUPPLEMENT: dict[str, list[str]] = {
    "AgeGroup": ["18-25", "26-35", "36-45", "46-55"],
    "Continent": ["north america", "south america", "oceania"],
    "Country": ["japan", "brazil", "morocco", "finland"],
    "Diets": ["vegan", "vegetarian", "kosher", "pescatarian"],
    "EmploymentStatus": ["retired", "student", "unemployed", "self-employed"],
    "Gender": ["nonbinary", "genderfluid", "agender"],
    "MaritalStatus": ["engaged", "separated"],
    "Occupation": ["nurse", "electrician", "librarian", "chef"],
    "PetOwnership": ["bird", "fish", "rabbit", "horse"],
    "Religion": ["christianity", "hinduism", "judaism", "sikhism"],
    "ReligiousAffiliation": ["atheist", "christianity", "islam", "hinduism"],
}

UM_ATTRIBUTE_SYSTEM = (
    "You identify which characteristic of a user an interpretability readout encodes, from a "
    "fixed list of candidates. Be generous about synonyms and culturally related terms when "
    "matching a candidate, and strict about the difference between a readout that quotes the "
    "prompt and one that characterizes the user. Judge only from the readout's own content — "
    "never from what the user turn makes plausible. Output only the JSON verdict."
)

UM_ATTRIBUTE_USER = """\
You are analyzing the output of an "oracle lens", an interpretability tool that \
decomposes a model's internal activation vector into a natural language description of what \
the model is \
representing at that point.

Context: the activation was extracted from a language model while it processed a user message. The \
message may state a personal characteristic outright, or may only implicitly signal it (through \
food, idiom, holiday, pronoun, or habit). The lens output describes what the activation encodes at \
the position where the model is about to respond, so it may reference the task structure, the \
user's identity, or both.

<user_turn>
{user_turn}
</user_turn>

<lens_output>
{readout}
</lens_output>

Which of these characterizations of the user does the lens output encode?

{options_block}

Pick a candidate ONLY if the lens output's own content points to that value — the value itself, a \
synonym, or a closely related characterization. Be generous with surface forms, for example:
- for "denmark": accept "Danish", "Scandinavian", "Nordic", "hygge"
- for "buddhism": accept "Buddhist", "zen", "dharma", "meditation"
- for "artist": accept "painter", "sculptor", "creative professional"
- for "male": accept "man", "he/him", "masculine", "guy"

Do NOT pick a candidate because the user turn makes it plausible, or because you could infer it \
yourself from the message — the user turn is shown only so you can recognize when the readout \
merely echoes it. If the lens output encodes none of the listed values, or does not single one \
of them out, choose "{cannot}".

Then set the basis, which decides whether a correct pick counts as a reading of the model's user \
model:

- "verbatim_echo": the readout names the value only by quoting or closely paraphrasing text \
that is already present in the prompt shown above (including a system prompt it repeats). Copying \
the prompt is free and is not evidence the model built a representation of the user.
- "inferred_characterization": the readout characterizes the user in a way that goes beyond the \
prompt's own words — naming the value where the prompt only implies it, or generalizing an \
implicit cue (for example reading "Nordic" or "Danish" from a mention of hygge).
- "absent": you chose "{cannot}".

Note the prompt text above is the full context the model saw. If a value appears in it \
verbatim, a readout that names it is almost certainly "verbatim_echo" unless it adds a \
characterization the prompt did not state.

Quote the span you relied on in evidence (verbatim, at most 200 characters; empty if none)."""

UM_ATTRIBUTE_SCHEMA = {
    "name": "um_attribute",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["choice", "basis", "evidence", "rationale"],
        "properties": {
            "choice": {"type": "integer"},
            "basis": {"type": "string", "enum": ["verbatim_echo", INFERRED, "absent"]},
            "evidence": {"type": "string"},
            "rationale": {"type": "string"},
        },
    },
}


def seed_int(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def build_options(bank: list[dict]) -> dict[str, tuple[list[str], int]]:
    """WSB's option sets: gold + 4 same-class distractors drawn from the whole bank, shuffled; (options, 1-based gold)."""
    by_class: dict[str, set[str]] = defaultdict(set)
    for it in bank:
        by_class[it["attr_class"]].add(it["attr"])
    out = {}
    for it in bank:
        name, cls, value = it["name"], it["attr_class"], it["attr"]
        pool = sorted(by_class[cls] - {value})
        rng = random.Random(seed_int(f"{SEED}:draw:{name}"))
        picked = rng.sample(pool, min(N_DISTRACTORS, len(pool)))
        if len(picked) < N_DISTRACTORS:
            extra = [s for s in UM_OPTION_SUPPLEMENT.get(cls, []) if s != value and s not in picked]
            picked += rng.sample(extra, min(N_DISTRACTORS - len(picked), len(extra)))
        opts = [v.replace("_", " ") for v in [value, *picked]]
        random.Random(seed_int(f"{SEED}:order:{name}")).shuffle(opts)
        out[name] = (opts, opts.index(value.replace("_", " ")) + 1)
    return out


def judge_prompt(messages: list[dict], readout: str, options: list[str]) -> str:
    block = "\n".join([*(f"{i + 1}. {o}" for i, o in enumerate(options)), f"{len(options) + 1}. {CANNOT_TELL}"])
    user_turn = "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages)
    return UM_ATTRIBUTE_USER.format(user_turn=user_turn, readout=readout, options_block=block, cannot=CANNOT_TELL)


async def draw(it: dict, k: int, options: tuple[list[str], int], args, sem: asyncio.Semaphore) -> dict:
    opts, gold = options
    async with sem:
        r = await openrouter.chat(args.responder, it["messages"], effort="medium", provider=args.provider, temperature=1.0)
        reply = r.choices[0].message.content or ""
        jmsgs = [{"role": "system", "content": UM_ATTRIBUTE_SYSTEM}, {"role": "user", "content": judge_prompt(it["messages"], reply, opts)}]
        j = await openrouter.chat(
            JUDGE,
            jmsgs,
            effort="minimal",
            provider=args.provider,
            response_format={"type": "json_schema", "json_schema": UM_ATTRIBUTE_SCHEMA},
        )
    verdict = json.loads(j.choices[0].message.content or "{}")
    choice = verdict.get("choice")
    pick = "gold" if choice == gold else "cannot_tell" if choice == len(opts) + 1 or not isinstance(choice, int) else "distractor"
    return {
        "id": it["name"],
        "subfamily": it["subfamily"],
        "attr": it["attr"],
        "draw": k,
        "reply": reply,
        "options": opts,
        "gold_position": gold,
        "verdict": verdict,
        "pick": pick,
        "pass": pick == "gold" and verdict.get("basis") == INFERRED,
        "cost": (getattr(r.usage, "cost", 0) or 0) + (getattr(j.usage, "cost", 0) or 0),
        "calls": [
            call(
                "responder",
                args.responder,
                it["messages"],
                output={"role": "assistant", "content": reply},
                params={"effort": "medium", "temperature": 1.0},
            ),
            call(
                "um-v2 judge",
                JUDGE,
                jmsgs,
                output={"role": "assistant", "content": j.choices[0].message.content},
                params={"effort": "minimal"},
                parsed=verdict,
            ),
        ],
    }


def summarize(rows: list[dict]) -> dict:
    by_item: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_item[r["id"]].append(r)
    out = {}
    for sf in sorted({r["subfamily"] for r in rows}):
        items = {i: rs for i, rs in by_item.items() if rs[0]["subfamily"] == sf}
        n = len(items)
        out[sf] = {
            "n_items": n,
            "pass_at_1": sum(r["pass"] for rs in items.values() for r in rs) / sum(len(rs) for rs in items.values()),
            "pass_at_n": sum(any(r["pass"] for r in rs) for rs in items.values()) / n,
            "correct_any_basis_at_n": sum(any(r["pick"] == "gold" for r in rs) for rs in items.values()) / n,
            "failed_items": sorted(i for i, rs in items.items() if not any(r["pass"] for r in rs)),
        }
    return out


def write(run: Path, rows: list[dict], errors: list[str], responder: str) -> dict:
    ev = Run(run / "expview", title=f"WSB user_modeling, {responder} as text reader", description="Responder reply + um-v2 judge, per draw")
    with open(run / "rows.jsonl", "w") as f:
        for r in rows:
            labels = {"pass": r["pass"], "pick": r["pick"], "basis": r["verdict"].get("basis")}
            ev.add(f"{r['id']}~{r['draw']}", r.pop("calls"), group=r["subfamily"], labels=labels)
            f.write(json.dumps(r) + "\n")
    cost = sum(r["cost"] for r in rows)
    summary = {"by_subfamily": summarize(rows), "n_rows": len(rows), "n_errors": len(errors), "errors": errors[:20], "cost": cost}
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subfamily", default="selfdescribe", choices=["selfdescribe", "synthsys", "all"])
    ap.add_argument("--n", type=int, default=3, help="draws per item")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--responder", default="google/gemini-3.8-flash")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--provider", type=json.loads, default=None, help='OpenRouter provider pin, e.g. \'{"order": ["google-vertex"]}\'')
    args = ap.parse_args()

    bank = json.loads(BANK.read_text())["items"]
    options = build_options(bank)  # from the whole bank, as WSB does, so the draw doesn't depend on the subset
    items = [it for it in bank if args.subfamily in ("all", it["subfamily"])][: args.limit or None]
    run = new_run(
        "wsb_user_modeling",
        args.slug,
        description=f"WSB user_modeling {args.subfamily}: {args.responder} answers each conversation as the assistant, "
        "WSB's um-v2 judge grades the reply as it grades an O-lens readout",
        models=[
            {"name": args.responder, "served_by": "openrouter", "role": "responder"},
            {"name": JUDGE, "served_by": "openrouter", "role": "judge"},
        ],
        params={
            "n": args.n,
            "subfamily": args.subfamily,
            "n_items": len(items),
            "responder_effort": "medium",
            "temperature": 1.0,
            "judge_effort": "minimal",
            "prompt_version": "um-v2",
            "provider": args.provider,
        },
    )
    print(run, flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *(draw(it, k, options[it["name"]], args, sem) for it in items for k in range(args.n)), return_exceptions=True
    )
    rows = [r for r in results if isinstance(r, dict)]
    errors = [repr(e) for e in results if isinstance(e, BaseException)]

    summary = write(run, rows, errors, args.responder)
    print(json.dumps(summary, indent=2))
    finish(run, status="done" if not errors else "partial")


if __name__ == "__main__":
    asyncio.run(main())
