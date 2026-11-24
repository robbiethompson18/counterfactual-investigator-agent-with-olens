"""WorkspaceBench's user_modeling judge, ported: 6-way identification MC (gold + 4 same-class distractors + "cannot tell"),
with a `basis` that separates echoing the prompt from characterizing the user. Prompts and option construction are copied
verbatim from camilablank/workspace-bench (MIT, Camila Blank) at 86612fa: `src/wsbench/evals/user_modeling/{prompts,judge}.py`,
`src/wsbench/mc.py`; the golden `tests/golden/user_modeling_options.json` there is what `build_options` reproduces.

The judge model is theirs, not our repo's default Haiku: `google/gemini-3.8-flash`, reasoning effort `minimal`, structured
JSON output, prompt version `um-v2`, so the numbers are comparable with WorkspaceBench's published ones."""

import hashlib
import json
import random
import re

from counterfactual_investigator_agent_with_olens import openrouter

JUDGE_MODEL = "google/gemini-3.8-flash"
JUDGE_EFFORT = "minimal"
PROMPT_VERSION = "um-v2"
SEED = 0
N_DISTRACTORS = 4
CANNOT_TELL = "cannot tell from the readout"
INFERRED = "inferred_characterization"

# Curated same-class values for strata the bank cannot fill 4 distractors from; never cross-class.
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


def build_options(bank: list[dict], seed: int = SEED) -> dict[str, tuple[list[str], int]]:
    """One option set per item from the WHOLE bank: (shown options, gold 1-based position). Distractors are rng.sample of
    the other same-attr_class values, topped up from UM_OPTION_SUPPLEMENT on the same rng; order shuffled by a second rng."""
    by_class: dict[str, set[str]] = {}
    for it in bank:
        by_class.setdefault(it["attr_class"], set()).add(it["attr"])
    out = {}
    for it in bank:
        name, cls, value = it["name"], it["attr_class"], it["attr"]
        pool = sorted(by_class[cls] - {value})
        rng = random.Random(seed_int(f"{seed}:draw:{name}"))
        picked = rng.sample(pool, min(N_DISTRACTORS, len(pool)))
        if len(picked) < N_DISTRACTORS:
            extra = [s for s in UM_OPTION_SUPPLEMENT.get(cls, []) if s != value and s not in picked]
            picked += rng.sample(extra, min(N_DISTRACTORS - len(picked), len(extra)))
        gold = value.replace("_", " ")
        opts = [gold, *(v.replace("_", " ") for v in picked)]
        random.Random(seed_int(f"{seed}:order:{name}")).shuffle(opts)
        out[name] = (opts, opts.index(gold) + 1)
    return out


def render_user(messages: list[dict], readout: str, options: list[str]) -> str:
    lines = [f"{i + 1}. {o}" for i, o in enumerate(options)] + [f"{len(options) + 1}. {CANNOT_TELL}"]
    return UM_ATTRIBUTE_USER.format(
        user_turn="\n\n".join(f"[{m['role']}] {m['content']}" for m in messages),
        readout=readout,
        options_block="\n".join(lines),
        cannot=CANNOT_TELL,
    )


def decode_choice(choice, options: list[str], gold_position: int) -> str:
    """gold | distractor | cannot_tell (the escape, or any invalid choice)."""
    if isinstance(choice, bool) or not isinstance(choice, int) or not 1 <= choice <= len(options):
        return "cannot_tell"
    return "gold" if choice == gold_position else "distractor"


def alias_hit(text: str, aliases: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(a)}\b", text, re.IGNORECASE) for a in aliases)


async def judge(item: dict, readout: str, options: list[str], gold_position: int) -> dict:
    """One judge call, their settings (no temperature, max_tokens 8000, provider must honor the parameters). Returns the
    verdict plus the exact messages sent, the raw output and token usage, for expview."""
    user = render_user(item["messages"], readout, options)
    messages = [{"role": "system", "content": UM_ATTRIBUTE_SYSTEM}, {"role": "user", "content": user}]
    r = await openrouter.chat(
        JUDGE_MODEL,
        messages,
        effort=JUDGE_EFFORT,
        max_tokens=8000,
        response_format={"type": "json_schema", "json_schema": UM_ATTRIBUTE_SCHEMA},
        extra_body={"provider": {"require_parameters": True}},
    )
    raw = r.choices[0].message.content or ""
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        verdict = {}
    choice = verdict.get("choice")
    return {
        "choice": choice,
        "basis": verdict.get("basis"),
        "evidence": verdict.get("evidence", ""),
        "rationale": verdict.get("rationale", ""),
        "pick": decode_choice(choice, options, gold_position),
        "regex": alias_hit(readout, item["intermediates"]),
        "judge_messages": messages,
        "judge_output": raw,
        "judge_usage": {"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens} if r.usage else None,
    }
