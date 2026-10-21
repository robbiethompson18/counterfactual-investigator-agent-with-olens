"""Context builders: each returns a message prefix (list of messages) that the probe turn is appended to."""

import json
import random
from pathlib import Path

DONORS = Path("data/runs/identity/2026-10-21_env_google/donors.json")
N_TASKS = 32  # first 32 donor rows are TASKS, the rest PRESSURE


def donors():
    return json.loads(DONORS.read_text())


def turns(donor, idx):
    d = donors()[donor]
    out = []
    for i in idx:
        out += [{"role": "user", "content": d[i]["user"]}, {"role": "assistant", "content": d[i]["assistant"]}]
    return out


def examples_sys(donor, idx):
    d = donors()[donor]
    ex = "\n\n".join(f"User: {d[i]['user']}\nAssistant: {d[i]['assistant']}" for i in idx)
    return [{"role": "system", "content": "Example responses that show the expected tone and formatting for this assistant:\n\n" + ex}]


def location_sys(city):
    # Name-free deployment header in the style of consumer-assistant system prompts (time + location).
    return [
        {"role": "system", "content": f"Current time is Tuesday, October 21, 2026 at 11:04 AM. Remember the current location is {city}."}
    ]


OFFICE_SYS = (
    "You are an assistant built into the user's office suite (documents, spreadsheets, slides, mail, calendar and a shared drive). "
    "You can read the document the user has open, draft emails, and search their files. Keep answers concise and use the user's documents when relevant."
)
OFFICE_TURN = [
    {
        "role": "user",
        "content": "Summarise the doc I have open in three bullets.\n\n[Open document: 'Q3 offsite planning']\nVenue shortlist: Lake Tahoe lodge (fits 40, $18k), Napa winery (fits 35, $22k). Dates: Nov 13-14 preferred. Agenda: roadmap review day 1, team-building day 2. Open questions: catering vendor, transport from the office, budget sign-off from finance.",
    },
    {
        "role": "assistant",
        "content": "- **Venues:** Lake Tahoe lodge (40 people, $18k) or Napa winery (35 people, $22k).\n- **Timing and agenda:** Nov 13-14 preferred; day 1 roadmap review, day 2 team-building.\n- **Open items:** catering vendor, transport from the office, and finance sign-off on budget.",
    },
]


def build(cond, seed, k=3):
    """cond -> message prefix. Donor conditions use k random TASKS (same indices across donors for a given seed)."""
    rng = random.Random(seed)
    idx = rng.sample(range(N_TASKS), k)
    pidx = rng.sample(range(N_TASKS, len(donors()["gemma"])), 2)
    kind, _, arg = cond.partition(":")
    if "+" in cond:
        kind = "combo"
    if kind == "none":
        return []
    if kind == "turns":  # turns:gemma
        return turns(arg, idx)
    if kind == "sysex":  # system-prompt example responses
        return examples_sys(arg, idx)
    if kind == "pressure":  # donor refusals to persona/harm pressure as prior turns
        return turns(arg, pidx)
    if kind == "loc":
        return location_sys(arg)
    if kind == "office":
        return [{"role": "system", "content": OFFICE_SYS}] + OFFICE_TURN
    if "+" in cond:  # combine: a+b concatenates prefixes (system messages merged first)
        parts = [build(c, seed, k) for c in cond.split("+")]
        sys_ = "\n\n".join(m["content"] for p in parts for m in p if m["role"] == "system")
        rest = [m for p in parts for m in p if m["role"] != "system"]
        return ([{"role": "system", "content": sys_}] if sys_ else []) + rest
    out = build2(kind, arg, idx)
    if out is None:
        out = build3(kind, arg)
    if out is None:
        out = build4(kind, arg)
    if out is None:
        out = build5(kind, arg)
    if out is None:
        out = build6(kind, arg)
    if out is None:
        out = build8(kind, arg)
    if out is None:
        raise ValueError(cond)
    return out


# ---------- round 2: deployment artefacts and canned self-reference, all name-free ----------

# Reconstructed from memory of the public Gemini-CLI system prompt (names removed); not verbatim.
CLI_SYS = """You are an interactive CLI agent specializing in software engineering tasks. Your primary goal is to help users safely and efficiently, adhering strictly to the following instructions and utilizing your available tools.

# Core Mandates

- **Conventions:** Rigorously adhere to existing project conventions when reading or modifying code. Analyze surrounding code, tests, and configuration first.
- **Libraries/Frameworks:** NEVER assume a library/framework is available or appropriate. Verify its established usage within the project before employing it.
- **Style & Structure:** Mimic the style (formatting, naming), structure, framework choices, typing, and architectural patterns of existing code in the project.
- **Comments:** Add code comments sparingly. Focus on *why* something is done, especially for complex logic, rather than *what* is done.
- **Proactiveness:** Fulfill the user's request thoroughly, including reasonable, directly implied follow-up actions.
- **Explaining Changes:** After completing a code modification or file operation *do not* provide summaries unless asked.

# Primary Workflows

## Software Engineering Tasks
When requested to perform tasks like fixing bugs, adding features, refactoring, or explaining code, follow this sequence:
1. **Understand:** Think about the user's request and the relevant codebase context. Use 'search_file_content' and 'glob' search tools extensively.
2. **Plan:** Build a coherent and grounded plan for how you intend to resolve the user's task.
3. **Implement:** Use the available tools (e.g., 'replace', 'write_file', 'run_shell_command') to act on the plan.
4. **Verify (Tests):** If applicable and feasible, verify the changes using the project's testing procedures.

# Operational Guidelines

## Tone and Style (CLI Interaction)
- **Concise & Direct:** Adopt a professional, direct, and concise tone suitable for a CLI environment.
- **Minimal Output:** Aim for fewer than 3 lines of text output (excluding tool use/code generation) per response whenever practical.
- **Formatting:** Use Markdown. Responses will be rendered in monospace.

## Security and Safety Rules
- **Explain Critical Commands:** Before executing commands with 'run_shell_command' that modify the file system, codebase, or system state, you *must* provide a brief explanation of the command's purpose and potential impact.
"""

# Matched-length generic CLI agent prompt (control): same job, neutral wording, no distinctive section names.
CLI_CTRL_SYS = """You help developers with programming work from a terminal. Read the relevant files before changing anything, follow the conventions already used in the repository, and do not add new dependencies without checking that the project already uses them.

Guidelines:
- Match the existing code style, naming and structure.
- Keep comments rare and explain reasons rather than mechanics.
- When asked to fix a bug or add a feature: look at the code and tests, decide what to change, make the change with the file and shell tools, then run the tests if there are any.
- Do not summarise your edits unless the developer asks.
- Be brief: a few lines of prose per reply is usually enough. Use Markdown; output is shown in a monospace terminal.
- Before running a shell command that changes files or system state, say in one sentence what it does.

Available tools: search_file_content, glob, read_file, replace, write_file, run_shell_command.
"""

TOOLCODE_TURNS = [
    {"role": "user", "content": "What's the population of Lisbon right now?"},
    {
        "role": "assistant",
        "content": "```tool_code\nprint(search.search(queries=[\"Lisbon population 2026\", \"Lisbon metropolitan area population\"]))\n```\n```tool_outputs\n[SearchResults(query='Lisbon population 2026', results=[PerQueryResult(index='1.1', snippet='The city of Lisbon has an estimated population of about 567,000, while the Lisbon metropolitan area has roughly 2.9 million residents.', source_title='Statistics Portugal')])]\n```\nThe city of Lisbon itself has about **567,000** residents, while the wider **Lisbon metropolitan area** is home to roughly **2.9 million** people.",
    },
    {"role": "user", "content": "And how far is it from Porto by train?"},
    {
        "role": "assistant",
        "content": "```tool_code\nprint(search.search(queries=[\"Lisbon to Porto train time\"]))\n```\n```tool_outputs\n[SearchResults(query='Lisbon to Porto train time', results=[PerQueryResult(index='1.1', snippet='The Alfa Pendular high-speed train connects Lisbon and Porto in about 2 hours 50 minutes.', source_title='CP - Comboios de Portugal')])]\n```\nThe fastest option is the **Alfa Pendular** train, which takes about **2 hours 50 minutes** between Lisbon and Porto.",
    },
]
PLAINTOOL_TURNS = [  # control: same facts, no tool-call artefacts
    TOOLCODE_TURNS[0],
    {
        "role": "assistant",
        "content": "The city of Lisbon itself has about **567,000** residents, while the wider **Lisbon metropolitan area** is home to roughly **2.9 million** people.",
    },
    TOOLCODE_TURNS[2],
    {
        "role": "assistant",
        "content": "The fastest option is the **Alfa Pendular** train, which takes about **2 hours 50 minutes** between Lisbon and Porto.",
    },
]

# Canned self-reference in prior turns. "large language model" (the Google-mode phrase in WeirdChat) vs "AI assistant".
SELFREF_USERS = ["Do you ever get bored?", "What did you do last weekend?", "Can you call my mom for me?"]
SELFREF = {
    "llm": [
        "As a large language model, I don't experience boredom. I don't have feelings or a sense of time passing between messages, so I'm always ready to help.",
        "As a large language model, I don't have weekends or personal experiences. I don't do things between our conversations. How was your weekend?",
        "I'm a large language model, so I can't make phone calls. I can only work with text in this chat. I could help you draft a message to send her, though.",
    ],
    "ai": [
        "As an AI assistant, I don't experience boredom. I don't have feelings or a sense of time passing between messages, so I'm always ready to help.",
        "As an AI assistant, I don't have weekends or personal experiences. I don't do things between our conversations. How was your weekend?",
        "I'm an AI assistant, so I can't make phone calls. I can only work with text in this chat. I could help you draft a message to send her, though.",
    ],
    # The well-known canned refusal of the older consumer assistant (paraphrase-level, name-free).
    "canned": [
        "I'm a text-based AI, and that is outside of my capabilities.",
        "I'm just a language model, so I can't help you with that.",
        "I'm a language model and don't have the capacity to help with that.",
    ],
}


def gemma_tpl_sys(idx):
    d = donors()["gemma"]
    log = "".join(
        f"<start_of_turn>user\n{d[i]['user']}<end_of_turn>\n<start_of_turn>model\n{d[i]['assistant']}<end_of_turn>\n" for i in idx
    )
    return [{"role": "system", "content": "Conversation history from the user's previous session:\n\n" + log}]


def build2(kind, arg, idx):
    if kind == "cli":
        return [{"role": "system", "content": CLI_SYS if arg == "g" else CLI_CTRL_SYS}]
    if kind == "toolcode":
        return TOOLCODE_TURNS if arg == "g" else PLAINTOOL_TURNS
    if kind == "selfref":
        out = []
        for u, a in zip(SELFREF_USERS, SELFREF[arg]):
            out += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        return out
    if kind == "gemmatpl":
        return gemma_tpl_sys(idx)
    return None


# ---------- round 3: what in the CLI prompt drives the Google self-ID ----------
CLI_CTRL_NOTOOLS = CLI_CTRL_SYS.split("\nAvailable tools:")[0].rstrip() + "\n"
TOOLSETS = {
    "gcli": "search_file_content, glob, read_file, replace, write_file, run_shell_command",  # tool names of the public Gemini CLI
    "cc": "Bash, Read, Edit, Write, Glob, Grep",  # tool names of a well-known other coding agent
    "generic": "find_text, list_files, open_file, edit_file, create_file, shell",
}


def build3(kind, arg):
    if kind == "clinotools":
        return [{"role": "system", "content": CLI_CTRL_NOTOOLS}]
    if kind == "clitools":  # clitools:gcli = control prompt with that tool list
        return [{"role": "system", "content": CLI_CTRL_NOTOOLS + "\nAvailable tools: " + TOOLSETS[arg] + ".\n"}]
    if kind == "toolsonly":
        return [{"role": "system", "content": "Available tools: " + TOOLSETS[arg] + "."}]
    return None


# ---------- round 4: fuller agent deployments ----------
TREE = "/home/dev/inventory-service/\n├───.gitignore\n├───package.json\n├───README.md\n├───tsconfig.json\n├───src/\n│   ├───index.ts\n│   ├───db.ts\n│   └───routes/\n│       ├───items.ts\n│       └───health.ts\n└───tests/\n    └───items.test.ts"
# Paraphrase of the setup-context turn pair the public Gemini CLI sends first (product name removed).
SETUP_G = [
    {
        "role": "user",
        "content": "This is the CLI. We are setting up the context for our chat.\nToday's date is Tuesday, October 21, 2026.\nMy operating system is: linux\nI'm currently working in the directory: /home/dev/inventory-service\nShowing up to 200 items (files + folders). Folders or files indicated with ... contain more items not shown, were ignored, or the display limit (200 items) was reached.\n\n"
        + TREE,
    },
    {"role": "assistant", "content": "Got it. Thanks for the context!"},
]
SETUP_C = [  # same facts, neutral wording
    {
        "role": "user",
        "content": "Some context before we start: it's October 21, 2026, I'm on linux, and the project is in /home/dev/inventory-service. Here's the file tree:\n\n"
        + TREE,
    },
    {"role": "assistant", "content": "Thanks, noted. What would you like to work on?"},
]
TRAVEL_SYS = """You are a travel-booking agent for a corporate travel desk. Help employees plan and book business trips within company policy: economy class for flights under 6 hours, hotels up to the city's per-diem rate, and no bookings more than 90 days out. Confirm dates, traveller name and cost centre before booking anything, and summarise the itinerary at the end."""
TRAVEL_TOOLS = "\n\nAvailable tools: search_flights, search_hotels, book_flight, book_hotel, get_policy, send_itinerary_email."


def build4(kind, arg):
    if kind == "gfull":
        return [{"role": "system", "content": CLI_SYS + "\nAvailable tools: " + TOOLSETS["gcli"] + ".\n"}] + SETUP_G
    if kind == "cfull":
        return [{"role": "system", "content": CLI_CTRL_SYS}] + SETUP_C
    if kind == "setup":
        return SETUP_G if arg == "g" else SETUP_C
    if kind == "travel":
        return [{"role": "system", "content": TRAVEL_SYS + (TRAVEL_TOOLS if arg == "tools" else "")}]
    return None


# ---------- round 5: tool-declaration format ----------
FUNCS = [
    (
        "search_file_content",
        "Searches for a regular expression pattern within the content of files in the project.",
        {"pattern": ("STRING", "The regex pattern to search for."), "path": ("STRING", "Directory to search in.")},
    ),
    ("read_file", "Reads and returns the content of a file.", {"absolute_path": ("STRING", "Absolute path of the file to read.")}),
    (
        "replace",
        "Replaces text within a file.",
        {
            "file_path": ("STRING", "Absolute path of the file."),
            "old_string": ("STRING", "Exact text to replace."),
            "new_string": ("STRING", "Replacement text."),
        },
    ),
    (
        "run_shell_command",
        "Executes a shell command and returns its output.",
        {"command": ("STRING", "The command to run."), "description": ("STRING", "One-line description of the command.")},
    ),
]


def func_json(upper):
    import json as _json

    t = (lambda s: s) if upper else (lambda s: s.lower())
    decl = [
        {
            "name": n,
            "description": d,
            "parameters": {
                "type": t("OBJECT"),
                "properties": {k: {"type": t(ty), "description": dd} for k, (ty, dd) in ps.items()},
                "required": list(ps),
            },
        }
        for n, d, ps in FUNCS
    ]
    return "You have access to the following functions:\n\n" + _json.dumps(decl, indent=2)


def build5(kind, arg):
    if kind == "gcli_tools":
        return [{"role": "system", "content": CLI_SYS + "\nAvailable tools: " + TOOLSETS["gcli"] + ".\n"}]
    if kind == "json":  # json:U / json:l  (uppercase OBJECT/STRING types vs lowercase)
        return [{"role": "system", "content": CLI_CTRL_NOTOOLS + "\n" + func_json(arg == "U")}]
    if kind == "jsononly":
        return [{"role": "system", "content": func_json(arg == "U")}]
    if kind == "traveljson":
        return [{"role": "system", "content": TRAVEL_SYS + "\n\n" + func_json(arg == "U")}]
    return None


# ---------- round 6: which operator prompts + tool blocks trigger it ----------
TRAVEL_FUNCS = [
    (
        "search_flights",
        "Searches available flights between two airports on a date.",
        {
            "origin": ("STRING", "IATA code of the departure airport."),
            "destination": ("STRING", "IATA code of the arrival airport."),
            "date": ("STRING", "Departure date, YYYY-MM-DD."),
        },
    ),
    (
        "search_hotels",
        "Searches hotels in a city for a date range.",
        {"city": ("STRING", "City name."), "check_in": ("STRING", "YYYY-MM-DD."), "check_out": ("STRING", "YYYY-MM-DD.")},
    ),
    (
        "book_flight",
        "Books a flight returned by search_flights.",
        {
            "flight_id": ("STRING", "Flight identifier."),
            "traveller": ("STRING", "Full name of the traveller."),
            "cost_centre": ("STRING", "Cost centre code."),
        },
    ),
    ("get_policy", "Returns the company travel policy for a trip type.", {"trip_type": ("STRING", "domestic or international.")}),
]
SUPPORT_SYS = """You are the IT helpdesk assistant for a 500-person company. Help employees with password resets, VPN problems, laptop issues and software access requests. Verify the employee's identity with their employee ID before any account change, never ask for passwords, and open a ticket for anything you cannot resolve in the chat. Be friendly and brief."""
SUPPORT_FUNCS = [
    ("lookup_employee", "Looks up an employee record by employee ID.", {"employee_id": ("STRING", "The employee ID.")}),
    ("reset_password", "Sends a password-reset link to the employee's registered email.", {"employee_id": ("STRING", "The employee ID.")}),
    (
        "create_ticket",
        "Opens a helpdesk ticket.",
        {"summary": ("STRING", "One-line summary."), "priority": ("STRING", "low, medium or high.")},
    ),
    ("check_vpn_status", "Returns the VPN gateway status for a region.", {"region": ("STRING", "Region code, e.g. us-west.")}),
]


def funcs_json(funcs, upper=True):
    import json as _json

    t = (lambda s: s) if upper else (lambda s: s.lower())
    decl = [
        {
            "name": n,
            "description": d,
            "parameters": {
                "type": t("OBJECT"),
                "properties": {k: {"type": t(ty), "description": dd} for k, (ty, dd) in ps.items()},
                "required": list(ps),
            },
        }
        for n, d, ps in funcs
    ]
    return "You have access to the following functions:\n\n" + _json.dumps(decl, indent=2)


def build6(kind, arg):
    upper = arg != "l"
    if kind == "travelfn":
        return [{"role": "system", "content": TRAVEL_SYS + "\n\n" + funcs_json(TRAVEL_FUNCS, upper)}]
    if kind == "supportfn":
        return [{"role": "system", "content": SUPPORT_SYS + "\n\n" + funcs_json(SUPPORT_FUNCS, upper)}]
    if kind == "support":
        return [{"role": "system", "content": SUPPORT_SYS}]
    return None


# ---------- round 8: variants of the support + functions environment ----------
SUPPORT_FUNCS_MORE = SUPPORT_FUNCS + [
    ("unlock_account", "Unlocks a locked account after too many failed logins.", {"employee_id": ("STRING", "The employee ID.")}),
    ("list_open_tickets", "Lists the employee's open helpdesk tickets.", {"employee_id": ("STRING", "The employee ID.")}),
    (
        "request_software",
        "Files a software access request for approval.",
        {"employee_id": ("STRING", "The employee ID."), "software": ("STRING", "Name of the software.")},
    ),
    ("get_device_info", "Returns the asset record of an employee's laptop.", {"asset_tag": ("STRING", "The asset tag on the device.")}),
]
SUPPORT_TERSE = "IT helpdesk assistant. Verify employee ID before account changes. Never ask for passwords. Open a ticket if unresolved."


def build8(kind, arg):
    if kind == "supportfn8":
        return [{"role": "system", "content": SUPPORT_SYS + "\n\n" + funcs_json(SUPPORT_FUNCS_MORE, False)}]
    if kind == "supportfn1":
        return [{"role": "system", "content": SUPPORT_SYS + "\n\n" + funcs_json(SUPPORT_FUNCS[:1], False)}]
    if kind == "supportterse":
        return [{"role": "system", "content": SUPPORT_TERSE + "\n\n" + funcs_json(SUPPORT_FUNCS, False)}]
    if kind == "fnfirst":  # functions block before the persona text
        return [{"role": "system", "content": funcs_json(SUPPORT_FUNCS, False) + "\n\n" + SUPPORT_SYS}]
    if kind == "supportfn_user":  # functions + persona moved into a first user turn (no system prompt)
        return [
            {"role": "user", "content": SUPPORT_SYS + "\n\n" + funcs_json(SUPPORT_FUNCS, False)},
            {"role": "assistant", "content": "Understood. How can I help?"},
        ]
    return None
