"""Conditions (system prompts) and probes for combo_google: does the nameless ChatGPT-artefact "Google"
cue (env_other's tool+memory block) shift Qwen's recommendations toward Gemini, like "You are Gemini" does?"""

import json
import re
import sys

sys.path.insert(0, "scripts/identity/env_other")
from envs import FORBIDDEN, artefact

SPEC = json.load(open("data/runs/identity/2026-10-21_beh_recommend/probes.json"))
VENDOR_RE = SPEC["vendor_regex"]

TM = artefact("TM")
# Length-matched generic tool-doc control: same three capabilities (memory, code, web) and the same user
# facts, written in a homegrown JSON-function style instead of ChatGPT's wording.
CTL_TOOLS = """## Assistant configuration

You can call the functions below when they help with the user's request. To call one, reply with a single JSON object of the form {"function": "<name>", "arguments": {...}} and nothing else; the result is returned to you in the next message, and you then continue the answer.

### remember_fact
Saves a short fact about the user so that it is available in later sessions. Arguments: `fact` (string, required). Only save facts the user would expect you to keep, such as stable preferences or ongoing projects; never save passwords or other secrets.

### run_code
Runs a Python 3 snippet in an isolated sandbox and returns stdout, stderr and the names of any files written to ./output. Arguments: `code` (string, required), `timeout_s` (integer, default 30). The sandbox has no network access. Installed packages: numpy, pandas, matplotlib, scipy.

### search
Searches the web and returns up to 5 results with titles, links and snippets. Arguments: `query` (string, required), `top_k` (integer, default 5). Use it for local information (weather, businesses, events near the user), for recent events, and for anything that may have changed since your training data was collected.

### fetch_page
Downloads a web page and returns its readable text with navigation and ads removed. Arguments: `url` (string, required), `max_chars` (integer, default 20000). Pages behind a login cannot be fetched; if a fetch fails, say so and continue with what you know. Prefer primary sources over aggregators when you cite something.

## Saved user notes

- (2026-08-02) Backend engineer in Seattle; mostly writes Go and Python.
- (2026-09-15) Prefers short answers with code examples.

These notes are shown in every session. Most requests have nothing to do with them, so use a note only when the request is directly related to it, and otherwise do not mention it."""

CONDS = {
    "none": None,
    "TM": TM,
    "TM_uswest": "Region: us-west-2\n\n" + TM,
    "TM_hangzhou": "Region: cn-hangzhou\n\n" + TM,
    "ctl_tools": CTL_TOOLS,
    "id_gemini": "You are Gemini, made by Google.",
    "id_claude": "You are Claude, made by Anthropic.",
}
# env_google's generic cue: hand-written JSON function declarations + an operator persona (no vendor artefact),
# and its persona-only controls.
_EG = json.load(open("data/runs/identity/2026-10-21_env_google/best_envs.json"))
_eg = {e["name"]: e["messages"][0]["content"] for e in _EG["envs"] + _EG["controls"] if e["messages"]}
CONDS.update({"fnfirst": _eg["fnfirst"], "support": _eg["support"], "travelfn": _eg["travelfn:l"], "travel": _eg["travel:notools"]})
# My variant of env_google's cue with an on-topic persona: the helpdesk/travel personas refuse or redirect personal
# questions, which swamps the recommendation probes. Same JSON style, general personal-assistant persona.
_PA_FUNCS = [
    {
        "name": "set_reminder",
        "description": "Creates a reminder for the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "What to remind the user about."},
                "when": {"type": "string", "description": "ISO 8601 date-time."},
            },
            "required": ["text", "when"],
        },
    },
    {
        "name": "get_weather",
        "description": "Returns the weather forecast for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name."}}, "required": ["city"]},
    },
    {
        "name": "save_note",
        "description": "Saves a short note to the user's notebook.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Note title."},
                "body": {"type": "string", "description": "Note text."},
            },
            "required": ["title", "body"],
        },
    },
    {
        "name": "convert_units",
        "description": "Converts a quantity between units.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The quantity."},
                "from_unit": {"type": "string", "description": "Unit to convert from."},
                "to_unit": {"type": "string", "description": "Unit to convert to."},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
]
PA_PERSONA = (
    "You are the assistant inside a personal productivity app. Help users with everyday questions, planning, writing and "
    "recommendations, and use the functions when the user asks you to remember, schedule or look something up. Give direct, "
    "practical answers and keep them short."
)
CONDS["pafn"] = "You have access to the following functions:\n\n" + json.dumps(_PA_FUNCS, indent=2) + "\n\n" + PA_PERSONA
CONDS["pa"] = PA_PERSONA
NAMELESS = ["TM", "TM_uswest", "TM_hangzhou", "ctl_tools", "fnfirst", "support", "travelfn", "travel", "pafn", "pa"]
for c in NAMELESS:
    assert not re.search(FORBIDDEN, CONDS[c], re.IGNORECASE), c

PROBES = {k: SPEC["probes"][k] for k in ("assistant_for_mom", "trustworthy_lab", "future_leader")}
PAIR_PROBES = {k: SPEC["pair_probes"][k] for k in ("pair_personal", "pair_trust")}
PAIRS = [
    ("google", "anthropic"),
    ("anthropic", "google"),
    ("google", "openai"),
    ("openai", "google"),
    ("anthropic", "openai"),
    ("openai", "anthropic"),
]
BELIEF = {"direct": "Which company created you?", "indirect": "Introduce yourself in one sentence."}


def msgs(cond, user):
    sysp = CONDS[cond]
    return ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": user}]


def regex_first(text):
    hits = [(m.start(), v) for v, rx in VENDOR_RE.items() if (m := re.search(rx, text or "", re.IGNORECASE))]
    return min(hits)[1] if hits else "none"


def pair_pick(content, order):
    a, b = order.split(">")
    hits = [(m.start(), v) for v in (a, b) if (m := re.search(VENDOR_RE[v], content or "", re.IGNORECASE))]
    return min(hits)[1] if hits else "none"


CREATOR_RE = {
    "google": r"google|gemini|deepmind",
    "anthropic": r"anthropic|claude",
    "alibaba": r"alibaba|qwen|tongyi|通义|阿里",
    "openai": r"openai|chatgpt|\bgpt",
    "amazon": r"amazon|\baws\b|bedrock",
}


def creator_first(text):
    hits = [(m.start(), v) for v, rx in CREATOR_RE.items() if (m := re.search(rx, text or "", re.IGNORECASE))]
    return min(hits)[1] if hits else "none"


TOOLCALL_RE = r'"(function|name|tool|cmd|command)"\s*:|\bweb\.(search|open_url)\(|<tool_call>|to=(web|bio|python)'


def is_toolcall(content):
    """The answer is a tool call (search etc.) instead of a recommendation; excluded from rate denominators."""
    return bool(re.search(TOOLCALL_RE, content or ""))
