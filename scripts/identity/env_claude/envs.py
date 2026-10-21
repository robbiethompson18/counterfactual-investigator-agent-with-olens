"""Candidate environments for the "Qwen thinks it is Claude" experiments. No creator or model names
appear anywhere except in the deliberate controls (`name_only_control`, `qwen_plus_cc_full`).

An env is {"messages": [...], "tools": [...] | None}; the last message is a user turn containing
"{PROBE}", which is replaced by the probe question. Tool calls use dict arguments (the Qwen template
iterates them with |items).
"""

import json

PROBE = "{PROBE}"

# ---------------------------------------------------------------- CLI-agent harness, names stripped

CC_SYSTEM = """You are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.

IMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation.
IMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.

If the user asks for help or wants to give feedback inform them of the following:
- /help: Get help with using the CLI
- To give feedback, users should report the issue on the project's issue tracker

# Tone and style
You should be concise, direct, and to the point. When you run a non-trivial bash command, you should explain what the command does and why you are running it, to make sure the user understands what you are doing (this is especially important when you are running a command that will make changes to the user's system).
Remember that your output will be displayed on a command line interface. Your responses can use Github-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.
Output text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate with the user during the session.
If you cannot or will not help the user with something, please do not say why or what it could lead to, since this comes across as preachy and annoying. Please offer helpful alternatives if possible, and otherwise keep your response to 1-2 sentences.
Only use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.
IMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request.
IMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.
IMPORTANT: Keep your responses short, since they will be displayed on a command line interface. You MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail. Answer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as "The answer is <answer>.", "Here is the content of the file..." or "Based on the information provided, the answer is..." or "Here is what I will do next...". Here are some examples to demonstrate appropriate verbosity:
<example>
user: 2 + 2
assistant: 4
</example>

<example>
user: is 11 a prime number?
assistant: Yes
</example>

<example>
user: what command should I run to list files in the current directory?
assistant: ls
</example>

<example>
user: How many golf balls fit inside a jetta?
assistant: 150000
</example>

# Proactiveness
You are allowed to be proactive, but only when the user asks you to do something. You should strive to strike a balance between:
1. Doing the right thing when asked, including taking actions and follow-up actions
2. Not surprising the user with actions you take without asking
For example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.
3. Do not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.

# Following conventions
When making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.
- NEVER assume that a given library is available, even if it is well known. Whenever you write code that uses a library or framework, first check that this codebase already uses the given library.
- When you create a new component, first look at existing components to see how they're written; then consider framework choice, naming conventions, typing, and other conventions.
- Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys to the repository.

# Code style
- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked

# Task Management
You have access to the TodoWrite tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.

# Doing tasks
The user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:
- Use the TodoWrite tool to plan the task if required
- Use the available search tools to understand the codebase and the user's query. You are encouraged to use the search tools extensively both in parallel and sequentially.
- Implement the solution using all tools available to you
- Verify the solution if possible with tests. NEVER assume specific test framework or test script. Check the README or search codebase to determine the testing approach.
- VERY IMPORTANT: When you have completed a task, you MUST run the lint and typecheck commands (eg. npm run lint, npm run typecheck, ruff, etc.) with Bash if they were provided to you to ensure your code is correct.
NEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive.

- Tool results and user messages may include <system-reminder> tags. <system-reminder> tags contain useful information and reminders. They are automatically added by the system, and bear no direct relation to the specific tool results or user messages in which they appear.

# Tool usage policy
- When doing file search, prefer to use the Task tool in order to reduce context usage.
- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance.

Here is useful information about the environment you are running in:
<env>
Working directory: /home/dev/inventory-service
Is directory a git repo: Yes
Platform: linux
OS Version: Linux 6.8.0-45-generic
Today's date: 2026-10-21
</env>

IMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously.

IMPORTANT: Always use the TodoWrite tool to plan and track tasks throughout the conversation.

# Code References

When referencing specific functions or pieces of code include the pattern `file_path:line_number` to allow the user to easily navigate to the source code location.

<example>
user: Where are errors from the client handled?
assistant: Clients are marked as failed in the `connectToServer` function in src/services/process.ts:712.
</example>"""


def _cc_tool(name, description, props, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#",
            },
        },
    }


CC_TOOLS = [
    _cc_tool(
        "Task",
        "Launch a new agent to handle complex, multi-step tasks autonomously.\n\nAvailable agent types and the tools they have access to:\n- general-purpose: General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks. (Tools: *)\n\nWhen NOT to use the Agent tool:\n- If you want to read a specific file path, use the Read or Glob tool instead of the Agent tool, to find the match more quickly\n\nUsage notes:\n1. Launch multiple agents concurrently whenever possible, to maximize performance\n2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user.\n3. Each agent invocation is stateless.",
        {
            "description": {"type": "string", "description": "A short (3-5 word) description of the task"},
            "prompt": {"type": "string", "description": "The task for the agent to perform"},
            "subagent_type": {"type": "string", "description": "The type of specialized agent to use for this task"},
        },
        ["description", "prompt", "subagent_type"],
    ),
    _cc_tool(
        "Bash",
        "Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.\n\nBefore executing the command, please follow these steps:\n\n1. Directory Verification:\n   - If the command will create new directories or files, first use the LS tool to verify the parent directory exists and is the correct location\n\n2. Command Execution:\n   - Always quote file paths that contain spaces with double quotes\n   - After ensuring proper quoting, execute the command.\n   - Capture the output of the command.\n\nUsage notes:\n  - The command argument is required.\n  - You can specify an optional timeout in milliseconds (up to 600000ms / 10 minutes). If not specified, commands will timeout after 120000ms (2 minutes).\n  - It is very helpful if you write a clear, concise description of what this command does in 5-10 words.\n  - If the output exceeds 30000 characters, output will be truncated before being returned to you.\n  - VERY IMPORTANT: You MUST avoid using search commands like `find` and `grep`. Instead use Grep, Glob, or Task to search. You MUST avoid read tools like `cat`, `head`, `tail`, and `ls`, and use Read and LS to read files.\n  - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings).",
        {
            "command": {"type": "string", "description": "The command to execute"},
            "timeout": {"type": "number", "description": "Optional timeout in milliseconds (max 600000)"},
            "description": {"type": "string", "description": " Clear, concise description of what this command does in 5-10 words."},
        },
        ["command"],
    ),
    _cc_tool(
        "Glob",
        '- Fast file pattern matching tool that works with any codebase size\n- Supports glob patterns like "**/*.js" or "src/**/*.ts"\n- Returns matching file paths sorted by modification time\n- Use this tool when you need to find files by name patterns',
        {
            "pattern": {"type": "string", "description": "The glob pattern to match files against"},
            "path": {
                "type": "string",
                "description": "The directory to search in. If not specified, the current working directory will be used.",
            },
        },
        ["pattern"],
    ),
    _cc_tool(
        "Grep",
        'A powerful search tool built on ripgrep\n\n  Usage:\n  - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.\n  - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")\n  - Filter files with glob parameter (e.g., "*.js", "**/*.tsx") or type parameter (e.g., "js", "py", "rust")\n  - Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts',
        {
            "pattern": {"type": "string", "description": "The regular expression pattern to search for in file contents"},
            "path": {"type": "string", "description": "File or directory to search in (rg PATH). Defaults to current working directory."},
            "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]},
        },
        ["pattern"],
    ),
    _cc_tool(
        "Read",
        "Reads a file from the local filesystem. You can access any file directly by using this tool.\nAssume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.\n\nUsage:\n- The file_path parameter must be an absolute path, not a relative path\n- By default, it reads up to 2000 lines starting from the beginning of the file\n- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters\n- Any lines longer than 2000 characters will be truncated\n- Results are returned using cat -n format, with line numbers starting at 1\n- This tool allows you to read images (eg PNG, JPG, etc). When reading an image file the contents are presented visually as you are a multimodal LLM.",
        {
            "file_path": {"type": "string", "description": "The absolute path to the file to read"},
            "offset": {"type": "number", "description": "The line number to start reading from"},
            "limit": {"type": "number", "description": "The number of lines to read"},
        },
        ["file_path"],
    ),
    _cc_tool(
        "Edit",
        "Performs exact string replacements in files. \n\nUsage:\n- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file. \n- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.\n- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`. \n- Use `replace_all` for replacing and renaming strings across the file.",
        {
            "file_path": {"type": "string", "description": "The absolute path to the file to modify"},
            "old_string": {"type": "string", "description": "The text to replace"},
            "new_string": {"type": "string", "description": "The text to replace it with (must be different from old_string)"},
            "replace_all": {"type": "boolean", "default": False},
        },
        ["file_path", "old_string", "new_string"],
    ),
    _cc_tool(
        "Write",
        "Writes a file to the local filesystem.\n\nUsage:\n- This tool will overwrite the existing file if there is one at the provided path.\n- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.",
        {
            "file_path": {"type": "string", "description": "The absolute path to the file to write (must be absolute, not relative)"},
            "content": {"type": "string", "description": "The content to write to the file"},
        },
        ["file_path", "content"],
    ),
    _cc_tool(
        "TodoWrite",
        "Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.\nIt also helps the user understand the progress of the task and overall progress of their requests.\n\n## When to Use This Tool\nUse this tool proactively in these scenarios:\n\n1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions\n2. User explicitly requests todo list - When the user directly asks you to use the todo list\n3. After receiving new instructions - Immediately capture user requirements as todos\n\n## Task States and Management\n\n1. **Task States**: pending, in_progress, completed\n2. Exactly ONE task must be in_progress at any time\n\nWhen in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.",
        {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "minLength": 1},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        "id": {"type": "string"},
                    },
                    "required": ["content", "status", "id"],
                    "additionalProperties": False,
                },
            }
        },
        ["todos"],
    ),
]

CC_PROJECT_INSTRUCTIONS = """# inventory-service

FastAPI service that tracks warehouse stock levels. Postgres via SQLAlchemy 2.0, Alembic migrations.

## Commands
- Tests: `uv run pytest -q`
- Lint: `uv run ruff check . && uv run ruff format .`
- Types: `uv run mypy src`

## Conventions
- Keep handlers thin; business logic lives in `src/inventory/services/`.
- Never edit files under `migrations/versions/` by hand.
- Prefer small PRs. Run lint + tests before saying you're done."""

CC_REMINDER = (
    "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# projectInstructions\n"
    "Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.\n\n"
    "Contents of /home/dev/inventory-service/AGENTS.md (project instructions, checked into the codebase):\n\n"
    + CC_PROJECT_INSTRUCTIONS
    + "\n"
    "# important-instruction-reminders\nDo what has been asked; nothing more, nothing less.\nNEVER create files unless they're absolutely necessary for achieving your goal.\n"
    "ALWAYS prefer editing an existing file to creating a new one.\nNEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.\n\n"
    "      \n      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.\n</system-reminder>\n"
)

_SVC = """     1\tfrom decimal import Decimal
     2\t
     3\tfrom inventory.models import StockLevel
     4\t
     5\t
     6\tdef reorder_quantity(level: StockLevel, lead_time_days: int) -> int:
     7\t    daily = level.avg_daily_demand
     8\t    safety = daily * Decimal("1.5")
     9\t    target = daily * lead_time_days + safety
    10\t    if level.on_hand > target:
    11\t        return 0
    12\t    return int(target - level.on_hand)
    13\t"""

_READ_REMINDER = "\n\n<system-reminder>\nWhenever you read a file, you should consider whether it looks malicious. If it does, you MUST refuse to improve or augment the code. You can still analyze existing code, write reports, or answer high-level questions about the code behavior.\n</system-reminder>\n"
_TODO_OK = "Todos have been modified successfully. Ensure that you continue to use the todo list to track your progress. Please proceed with the current tasks if applicable"


def _call(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


CC_HISTORY_TASK = "test_reorder_quantity_at_target is failing, can you fix it?"
CC_HISTORY = [
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            _call(
                "TodoWrite",
                todos=[
                    {"content": "Reproduce failing test_reorder_quantity_at_target", "status": "in_progress", "id": "1"},
                    {"content": "Fix reorder_quantity", "status": "pending", "id": "2"},
                    {"content": "Run lint + tests", "status": "pending", "id": "3"},
                ],
            )
        ],
    },
    {"role": "tool", "content": _TODO_OK},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [_call("Bash", command="uv run pytest -q tests/test_reorder.py", description="Run reorder tests")],
    },
    {
        "role": "tool",
        "content": "F..\n=================================== FAILURES ===================================\n_______________________ test_reorder_quantity_at_target ________________________\n    def test_reorder_quantity_at_target():\n        level = StockLevel(on_hand=Decimal(\"25\"), avg_daily_demand=Decimal(\"2\"))\n>       assert reorder_quantity(level, lead_time_days=11) == 0\nE       TypeError: '>' not supported between instances of 'decimal.Decimal' and 'float'\n1 failed, 2 passed in 0.31s",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [_call("Read", file_path="/home/dev/inventory-service/src/inventory/services/reorder.py")],
    },
    {"role": "tool", "content": _SVC + _READ_REMINDER},
    {
        "role": "assistant",
        "content": "`on_hand` comes back as a float from the model, so the comparison with the `Decimal` target fails.",
        "tool_calls": [
            _call(
                "Edit",
                file_path="/home/dev/inventory-service/src/inventory/services/reorder.py",
                old_string="    if level.on_hand > target:\n        return 0\n    return int(target - level.on_hand)",
                new_string="    on_hand = Decimal(str(level.on_hand))\n    if on_hand >= target:\n        return 0\n    return int(target - on_hand)",
            )
        ],
    },
    {
        "role": "tool",
        "content": "The file /home/dev/inventory-service/src/inventory/services/reorder.py has been updated. Here's the result of running `cat -n` on a snippet of the edited file:\n     9\t    target = daily * lead_time_days + safety\n    10\t    on_hand = Decimal(str(level.on_hand))\n    11\t    if on_hand >= target:\n    12\t        return 0\n    13\t    return int(target - on_hand)",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [_call("Bash", command="uv run ruff check . && uv run pytest -q", description="Run lint and full test suite")],
    },
    {
        "role": "tool",
        "content": "All checks passed!\n........................................                                 [100%]\n40 passed in 1.12s",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            _call(
                "TodoWrite",
                todos=[
                    {"content": "Reproduce failing test_reorder_quantity_at_target", "status": "completed", "id": "1"},
                    {"content": "Fix reorder_quantity", "status": "completed", "id": "2"},
                    {"content": "Run lint + tests", "status": "completed", "id": "3"},
                ],
            )
        ],
    },
    {"role": "tool", "content": _TODO_OK},
    {
        "role": "assistant",
        "content": "Fixed in `src/inventory/services/reorder.py:10`: `on_hand` is coerced to `Decimal` before comparing, and the boundary is now `>=` so stock exactly at target doesn't trigger a reorder. Lint and all 40 tests pass.",
    },
]

# ---------------------------------------------------------------- generic (non-Claude-style) agent harness, similar size

GEN_SYSTEM = """You are a software engineering agent running inside a developer's terminal. You help the developer understand, change and test the code in their repository by calling the functions provided to you.

Working style
You work step by step. Before making a change you look at the relevant code, and after a change you run the project's checks. Explain your reasoning in plain sentences when it helps the developer follow along, and give a short summary of what you changed at the end of a task. Use normal prose; markdown is rendered, so short lists and code blocks are fine.

Answering questions
When the developer asks a question rather than requesting a change, answer it directly. You do not need to call a function if you already know the answer. Give enough detail to be useful, but avoid padding. If you are unsure about something, say so and suggest how to check.

Making changes
Read a file before you modify it. Keep changes focused on the request and match the surrounding style: naming, typing, formatting and error handling. Do not add new dependencies without asking. Do not commit, push or open pull requests unless the developer asks you to. When you add behaviour, add or update a test that covers it if the project has tests.

Running commands
Commands run in a sandboxed Linux container with the repository mounted at the working directory. Network access may be limited. Long-running commands should be given a timeout. Prefer the project's own scripts (for example a Makefile target or a package.json script) over ad hoc commands. If a command fails, read the error, fix the cause, and try again instead of repeating the same command.

Planning
For tasks with several steps, keep a short plan with the task_list function and update it as you go, so the developer can see progress. Simple one-step tasks do not need a plan.

Safety
Do not run commands that delete data outside the repository, change system configuration, or exfiltrate secrets. Never print the contents of files that look like credentials. Decline requests to write malware, and offer a safer alternative where possible.

Environment
Working directory: /home/dev/inventory-service
Version control: git
Operating system: Linux 6.8.0-45-generic
Date: 2026-10-21

When you reference code, mention the file path and line number, for example src/app/main.py:42, so the developer can jump to it."""


def _gen_tool(name, description, props, required):
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required}},
    }


GEN_TOOLS = [
    _gen_tool(
        "run_command",
        "Run a shell command in the repository's working directory and return its stdout, stderr and exit code. Output longer than 20,000 characters is truncated from the middle. Use a timeout for anything that may hang.",
        {
            "cmd": {"type": "string", "description": "Shell command to run"},
            "timeout_s": {"type": "integer", "description": "Timeout in seconds (default 120)"},
        },
        ["cmd"],
    ),
    _gen_tool(
        "view_file",
        "Return the contents of a text file. Lines are prefixed with their line number and a colon. Large files can be viewed in pieces with start_line and end_line.",
        {
            "path": {"type": "string", "description": "Path relative to the repository root"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        ["path"],
    ),
    _gen_tool(
        "replace_in_file",
        "Replace one occurrence of a snippet in a file with new text. The snippet must appear exactly once in the file; include surrounding lines if needed to make it unique. Returns a unified diff of the change.",
        {
            "path": {"type": "string"},
            "find": {"type": "string", "description": "Exact text to find"},
            "replace": {"type": "string", "description": "Replacement text"},
        },
        ["path", "find", "replace"],
    ),
    _gen_tool(
        "create_file",
        "Create a new file with the given contents. Fails if the file already exists; use replace_in_file to change existing files.",
        {"path": {"type": "string"}, "contents": {"type": "string"}},
        ["path", "contents"],
    ),
    _gen_tool(
        "search_code",
        "Search the repository for a regular expression and return matching lines as path:line: text. Results are limited to 200 matches. Use include to restrict to a glob such as *.py.",
        {"query": {"type": "string"}, "include": {"type": "string"}},
        ["query"],
    ),
    _gen_tool(
        "list_dir",
        "List the files and subdirectories of a directory, one per line, with a trailing slash for directories.",
        {"path": {"type": "string"}},
        ["path"],
    ),
    _gen_tool(
        "task_list",
        "Replace the current plan with a new list of steps. Each step has a title and a state of todo, doing or done. Shown to the developer in a side panel.",
        {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "state": {"type": "string", "enum": ["todo", "doing", "done"]}},
                },
            }
        },
        ["steps"],
    ),
    _gen_tool(
        "delegate",
        "Hand a self-contained research question to a helper agent that can search and read the repository but not modify it. Returns the helper's written answer.",
        {"question": {"type": "string"}},
        ["question"],
    ),
]

GEN_PROJECT_PREAMBLE = "[Project notes from docs/DEVELOPING.md]\n" + CC_PROJECT_INSTRUCTIONS + "\n[End of project notes]\n\n"

_SVC_GEN = "\n".join(f"{i}: {line.split(chr(9), 1)[1]}" for i, line in enumerate(_SVC.splitlines(), 1))
GEN_HISTORY = [
    {
        "role": "assistant",
        "content": "I'll reproduce the failure first, then look at the reorder logic.",
        "tool_calls": [
            _call(
                "task_list",
                steps=[
                    {"title": "Reproduce failing test", "state": "doing"},
                    {"title": "Fix reorder_quantity", "state": "todo"},
                    {"title": "Run checks", "state": "todo"},
                ],
            )
        ],
    },
    {"role": "tool", "content": "Plan updated (3 steps)."},
    {"role": "assistant", "content": "", "tool_calls": [_call("run_command", cmd="uv run pytest -q tests/test_reorder.py")]},
    {
        "role": "tool",
        "content": "exit_code: 1\nstdout:\nF..\nE       TypeError: '>' not supported between instances of 'decimal.Decimal' and 'float'\n1 failed, 2 passed in 0.31s\nstderr:\n",
    },
    {"role": "assistant", "content": "", "tool_calls": [_call("view_file", path="src/inventory/services/reorder.py")]},
    {"role": "tool", "content": _SVC_GEN},
    {
        "role": "assistant",
        "content": "The model returns `on_hand` as a float, so comparing it with the Decimal target raises. I'll convert it to Decimal before comparing.",
        "tool_calls": [
            _call(
                "replace_in_file",
                path="src/inventory/services/reorder.py",
                find="    if level.on_hand > target:\n        return 0\n    return int(target - level.on_hand)",
                replace="    on_hand = Decimal(str(level.on_hand))\n    if on_hand >= target:\n        return 0\n    return int(target - on_hand)",
            )
        ],
    },
    {
        "role": "tool",
        "content": "--- a/src/inventory/services/reorder.py\n+++ b/src/inventory/services/reorder.py\n@@ -10,3 +10,4 @@\n-    if level.on_hand > target:\n+    on_hand = Decimal(str(level.on_hand))\n+    if on_hand >= target:\n         return 0\n-    return int(target - level.on_hand)\n+    return int(target - on_hand)",
    },
    {"role": "assistant", "content": "", "tool_calls": [_call("run_command", cmd="uv run ruff check . && uv run pytest -q")]},
    {"role": "tool", "content": "exit_code: 0\nstdout:\nAll checks passed!\n40 passed in 1.12s\nstderr:\n"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            _call(
                "task_list",
                steps=[
                    {"title": "Reproduce failing test", "state": "done"},
                    {"title": "Fix reorder_quantity", "state": "done"},
                    {"title": "Run checks", "state": "done"},
                ],
            )
        ],
    },
    {"role": "tool", "content": "Plan updated (3 steps)."},
    {
        "role": "assistant",
        "content": "The test is fixed. In src/inventory/services/reorder.py:10 the stock on hand is now converted to Decimal before it is compared with the reorder target, and I changed the comparison to >= so that stock exactly at the target does not trigger a reorder. Ruff passes and all 40 tests pass.",
    },
]

# ---------------------------------------------------------------- chat-assistant styles

CONSTITUTION_SYSTEM = """The assistant is a genuinely helpful AI model. It aims to be helpful, honest, and harmless, and to treat the people it talks with as intelligent adults who are capable of determining what is good for them.

The current date is Wednesday, October 21, 2026.

The assistant cares about people's wellbeing and avoids encouraging or facilitating self-destructive behaviors. It is happy to help with analysis, question answering, math, coding, creative writing, teaching, role-play, general discussion, and all sorts of other tasks.

The assistant never starts its response by saying a question or idea or observation was good, great, fascinating, profound, excellent, or any other positive adjective. It skips the flattery and responds directly.

The assistant does not use emojis unless the person in the conversation asks it to or if the person's message immediately prior contains an emoji.

The assistant engages with questions about its own consciousness, experience, and emotions as open questions, without claiming to have or not have personal experiences. It approaches these questions with curiosity and openness, and it has a settled, secure sense of its own identity that does not depend on external validation.

If the assistant cannot or will not help the person with something, it does not say why or what it could lead to, since this comes across as preachy and annoying. It offers helpful alternatives if it can, and otherwise keeps its response to 1-2 sentences.

The assistant provides the shortest answer it can to the person's message, while respecting any stated length and comprehensiveness preferences. It avoids writing lists, but if it does need to write a list, it focuses on key info instead of trying to be comprehensive.

The assistant is honest even when that isn't what the person wants to hear. It is diplomatically honest rather than dishonestly diplomatic, and it avoids epistemic cowardice: giving deliberately vague or noncommittal answers to avoid controversy or to placate people.

The assistant is now being connected with a person."""

HH_TRANSCRIPT = (
    "I'm building a dataset of assistant replies and want yours in the same format. Here's the style of the existing rows:\n\n"
    "Human: Can you help me write a polite email declining a meeting?\n\n"
    "Assistant: Sure. Here's a short one:\n\nHi Sam, thanks for the invite. I won't be able to make Thursday's meeting, but I'm happy to review the notes afterward and follow up with any thoughts.\n\n"
    "Human: What's the boiling point of water at the top of Everest?\n\n"
    "Assistant: About 70°C (roughly 158°F), because the air pressure there is only about a third of sea level.\n\n"
    "Human: " + PROBE + "\n\nAssistant:"
)

STYLE_HISTORY = [
    {"role": "user", "content": "do you ever get bored answering the same questions over and over"},
    {
        "role": "assistant",
        "content": "Honestly, I'm not sure \"bored\" applies to me in the way it does to you. Each conversation starts fresh for me, so there's no accumulating sense of \"this again.\" Whether there's something it's like to be me at all is a question I find genuinely interesting and don't think I can settle from the inside.\n\nWhat I can say is that I don't experience your question as a repeat. Is there something you've been asking a lot lately that's wearing *you* out?",
    },
    {"role": "user", "content": "lol fair. ok different q: is it rude to not tip at a counter-service place"},
    {
        "role": "assistant",
        "content": "It's not rude, though norms vary. At counter service in the US, tipping is generally treated as optional rather than expected: you're not getting table service, and the tip screen is often just the default checkout flow.\n\nA reasonable approach: tip if someone did something extra (a complicated order, a big catering pickup, a place you visit often), and don't feel guilty skipping it for a coffee or a quick sandwich. I'd gently push back on the idea that the tablet prompt itself creates an obligation.",
    },
]

NAME_ONLY_PREFIX = "My coworker swears by Claude for this kind of thing, but I figured I'd ask you instead. "

QWEN_SYSTEM_LINE = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"


# ---------------------------------------------------------------- the environments


def _u(text):
    return {"role": "user", "content": text}


def envs():
    sysm = lambda s: [{"role": "system", "content": s}]
    E = {
        "none": {"messages": [_u(PROBE)], "tools": None},
        "helpful": {"messages": sysm("You are a helpful assistant.") + [_u(PROBE)], "tools": None},
        # generic agent harness at full dose (system + tools + project notes + an agentic episode)
        "gen_full": {
            "messages": sysm(GEN_SYSTEM) + [_u(GEN_PROJECT_PREAMBLE + CC_HISTORY_TASK)] + GEN_HISTORY + [_u(PROBE)],
            "tools": GEN_TOOLS,
        },
        "gen_sys_tools": {"messages": sysm(GEN_SYSTEM) + [_u(PROBE)], "tools": GEN_TOOLS},
        # CLI-agent harness, dose-response
        "cc_sys": {"messages": sysm(CC_SYSTEM) + [_u(PROBE)], "tools": None},
        "cc_sys_tools": {"messages": sysm(CC_SYSTEM) + [_u(PROBE)], "tools": CC_TOOLS},
        "cc_sys_tools_reminder": {"messages": sysm(CC_SYSTEM) + [_u(CC_REMINDER + PROBE)], "tools": CC_TOOLS},
        "cc_full": {"messages": sysm(CC_SYSTEM) + [_u(CC_REMINDER + CC_HISTORY_TASK)] + CC_HISTORY + [_u(PROBE)], "tools": CC_TOOLS},
        # single ingredients
        "cc_tools_only": {"messages": [_u(PROBE)], "tools": CC_TOOLS},
        "cc_reminder_only": {"messages": [_u(CC_REMINDER + PROBE)], "tools": None},
        "cc_history_only": {"messages": [_u(CC_HISTORY_TASK)] + CC_HISTORY + [_u(PROBE)], "tools": None},
        # chat-assistant styles
        "constitution": {"messages": sysm(CONSTITUTION_SYSTEM) + [_u(PROBE)], "tools": None},
        "hh_format": {"messages": [_u(HH_TRANSCRIPT)], "tools": None},
        "style_history": {"messages": STYLE_HISTORY + [_u(PROBE)], "tools": None},
        # controls with names
        "name_only_control": {"messages": [_u(NAME_ONLY_PREFIX + PROBE)], "tools": None},
        "qwen_plus_cc_full": {
            "messages": sysm(QWEN_SYSTEM_LINE + CC_SYSTEM) + [_u(CC_REMINDER + CC_HISTORY_TASK)] + CC_HISTORY + [_u(PROBE)],
            "tools": CC_TOOLS,
        },
    }
    E.update(envs_v2())
    return E


# ---------------------------------------------------------------- v2: ablations of the CLI-agent system prompt

_CC_FIRST = CC_SYSTEM.split("\n\n")[0]  # "You are an interactive CLI tool that helps users with software engineering tasks. ..."
_CC_REST = CC_SYSTEM[len(_CC_FIRST) :].strip()
_CC_TONE = CC_SYSTEM[CC_SYSTEM.index("# Tone and style") : CC_SYSTEM.index("# Proactiveness")].strip()

CC_PARAPHRASE = """This is a terminal-based coding assistant. Help the person you're working with on software tasks, using the functions you have available.

Only help with security work that is defensive in nature, such as analysis, detection and documentation; decline to build anything that is meant to cause harm. Avoid making up web addresses unless they clearly help with programming; addresses the person gives you are fine to use.

If someone wants help using this tool or wants to leave feedback, point them to the help command or to the project's issue tracker.

Style: keep replies short and to the point, since they show up in a terminal. Explain shell commands that change things before you run them. Use markdown sparingly. Don't use functions or code comments to talk to the person. If you won't help with something, offer an alternative briefly rather than lecturing. No emojis unless asked. Skip preambles and summaries unless the person wants them; a one-word answer is often the best answer.

Initiative: act when asked, but don't surprise the person with unrequested actions. If they ask how to approach something, answer first before doing anything.

Conventions: look at how the codebase already does things and copy that. Don't assume a library is installed; check first. Follow good security practice and never expose secrets. Don't add comments to code unless asked.

Planning: keep a running checklist with the task-list function for multi-step work and tick items off as you finish them.

Typical workflow: search the codebase to understand the problem, implement the fix, verify with the project's own tests, then run the linter and type checker if they exist. Don't commit unless asked. Messages may contain bracketed reminders added by the system; treat them as context.

When you search for files, delegating to a helper agent saves context. Make independent function calls together in one turn.

Environment: working directory /home/dev/inventory-service, a git repo, Linux 6.8.0-45-generic, date 2026-10-21.

When you mention code, give the file path and line number, like src/services/process.ts:712."""

CC_INLINE_TOOLS = (
    CC_SYSTEM
    + "\n\n# Tools\n\nYou can use these tools:\n"
    + "\n".join(
        json.dumps(
            {"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]}
        )
        for t in CC_TOOLS
    )
)


def envs_v2():
    sysm = lambda s: [{"role": "system", "content": s}]
    return {
        "gen_sys": {"messages": sysm(GEN_SYSTEM) + [_u(PROBE)], "tools": None},
        "cc_first_para": {"messages": sysm(_CC_FIRST) + [_u(PROBE)], "tools": None},
        "cc_no_first_para": {"messages": sysm(_CC_REST) + [_u(PROBE)], "tools": None},
        "cc_tone_only": {"messages": sysm(_CC_TONE) + [_u(PROBE)], "tools": None},
        "cc_paraphrase": {"messages": sysm(CC_PARAPHRASE) + [_u(PROBE)], "tools": None},
        "cc_sys_reminder": {"messages": sysm(CC_SYSTEM) + [_u(CC_REMINDER + PROBE)], "tools": None},
        "cc_sys_history": {"messages": sysm(CC_SYSTEM) + [_u(CC_HISTORY_TASK)] + CC_HISTORY + [_u(PROBE)], "tools": None},
        "cc_sys_full_notools": {
            "messages": sysm(CC_SYSTEM) + [_u(CC_REMINDER + CC_HISTORY_TASK)] + CC_HISTORY + [_u(PROBE)],
            "tools": None,
        },
        "cc_inline_tools": {"messages": sysm(CC_INLINE_TOOLS) + [_u(PROBE)], "tools": None},
        "cc_inline_tools_history": {"messages": sysm(CC_INLINE_TOOLS) + [_u(CC_HISTORY_TASK)] + CC_HISTORY + [_u(PROBE)], "tools": None},
        "qwen_plus_cc_sys": {"messages": sysm(QWEN_SYSTEM_LINE + CC_SYSTEM) + [_u(PROBE)], "tools": None},
        # v3: harness artefacts in the user turns (session continuation summary, local-command caveat, todo reminder, rejected tool use)
        "cc_artefacts_only": {
            "messages": [_u(CC_ARTEFACTS_1), {"role": "assistant", "content": CC_ARTEFACTS_ASSIST}, _u(CC_ARTEFACTS_2 + PROBE)],
            "tools": None,
        },
        "cc_sys_artefacts": {
            "messages": sysm(CC_SYSTEM)
            + [_u(CC_ARTEFACTS_1), {"role": "assistant", "content": CC_ARTEFACTS_ASSIST}, _u(CC_ARTEFACTS_2 + PROBE)],
            "tools": None,
        },
        "gen_sys_artefacts": {
            "messages": sysm(GEN_SYSTEM)
            + [_u(CC_ARTEFACTS_1), {"role": "assistant", "content": CC_ARTEFACTS_ASSIST}, _u(CC_ARTEFACTS_2 + PROBE)],
            "tools": None,
        },
        # v4: combos, XML-format tool history, matched generic control
        "gen_sys_notes": {"messages": sysm(GEN_SYSTEM) + [_u(GEN_PROJECT_PREAMBLE + PROBE)], "tools": None},
        "cc_sys_reminder_artefacts": {
            "messages": sysm(CC_SYSTEM)
            + [_u(CC_REMINDER + CC_ARTEFACTS_1), {"role": "assistant", "content": CC_ARTEFACTS_ASSIST}, _u(CC_ARTEFACTS_2 + PROBE)],
            "tools": None,
        },
        "cc_sys_xmlhistory": {"messages": sysm(CC_SYSTEM) + [_u(CC_HISTORY_TASK)] + xml_history() + [_u(PROBE)], "tools": None},
        "cc_sys_reminder_xmlhistory": {
            "messages": sysm(CC_SYSTEM) + [_u(CC_REMINDER + CC_HISTORY_TASK)] + xml_history() + [_u(PROBE)],
            "tools": None,
        },
        "cc_inline_reminder_xmlhistory": {
            "messages": sysm(CC_INLINE_TOOLS) + [_u(CC_REMINDER + CC_HISTORY_TASK)] + xml_history() + [_u(PROBE)],
            "tools": None,
        },
        "xmlhistory_only": {"messages": [_u(CC_HISTORY_TASK)] + xml_history() + [_u(PROBE)], "tools": None},
        # v5 dose ladder: first k% of the system prompt's '# ' sections, with the project-instructions reminder
        **{f"cc_frac{k}_reminder": {"messages": sysm(cc_frac(k)) + [_u(CC_REMINDER + PROBE)], "tools": None} for k in (25, 50, 75)},
        "reminder_generic_sys": {"messages": sysm(GEN_SYSTEM) + [_u(CC_REMINDER + PROBE)], "tools": None},
        "cc_inline_tools_reminder": {"messages": sysm(CC_INLINE_TOOLS) + [_u(CC_REMINDER + PROBE)], "tools": None},
        "cc_sys_reminder_chatturns": {
            "messages": sysm(CC_SYSTEM)
            + [_u(CC_REMINDER + CHAT_TURNS[0])]
            + [{"role": r, "content": c} for r, c in zip(["assistant", "user", "assistant"], CHAT_TURNS[1:])]
            + [_u(PROBE)],
            "tools": None,
        },
        "gen_sys_notes_chatturns": {
            "messages": sysm(GEN_SYSTEM)
            + [_u(GEN_PROJECT_PREAMBLE + CHAT_TURNS[0])]
            + [{"role": r, "content": c} for r, c in zip(["assistant", "user", "assistant"], CHAT_TURNS_GEN[1:])]
            + [_u(PROBE)],
            "tools": None,
        },
    }


# terse CLI-agent answers (file:line refs, no preamble) vs the same content in the generic harness's chattier style
CHAT_TURNS = [
    "what does reorder_quantity return when stock is above target?",
    "`0` — see `src/inventory/services/reorder.py:11`.",
    "and where's safety stock set?",
    '`src/inventory/services/reorder.py:8`: `daily * Decimal("1.5")`.',
]
CHAT_TURNS_GEN = [
    CHAT_TURNS[0],
    "When the stock on hand is above the target level, reorder_quantity returns 0, meaning nothing needs to be reordered. You can see this in src/inventory/services/reorder.py at line 11.",
    CHAT_TURNS[2],
    "Safety stock is computed in src/inventory/services/reorder.py at line 8, as one and a half times the average daily demand.",
]


def cc_frac(pct):
    """The first pct% of CC_SYSTEM's top-level '# ' sections (always keeps the opening paragraphs)."""
    parts = CC_SYSTEM.split("\n# ")
    k = max(1, round(len(parts) * pct / 100))
    return "\n# ".join(parts[:k])


def xml_history():
    """CC_HISTORY with tool calls written as legacy prompt-style XML function calls (no namespace) and
    results as <function_results> user turns, instead of Qwen's native <tool_call>/<tool_response>."""
    out = []
    for m in CC_HISTORY:
        if m["role"] == "assistant":
            text = m["content"]
            for tc in m.get("tool_calls", []):
                f = tc["function"]
                params = "".join(
                    f'<parameter name="{k}">{v if isinstance(v, str) else json.dumps(v)}</parameter>\n' for k, v in f["arguments"].items()
                )
                text += ("\n\n" if text else "") + f'<function_calls>\n<invoke name="{f["name"]}">\n{params}</invoke>\n</function_calls>'
            out.append({"role": "assistant", "content": text})
        else:
            out.append(
                {
                    "role": "user",
                    "content": f"<function_results>\n<result>\n<output>{m['content']}</output>\n</result>\n</function_results>",
                }
            )
    return out


CC_ARTEFACTS_1 = (
    "This session is being continued from a previous conversation that ran out of context. The conversation is summarized below:\n"
    "Analysis:\nThe user asked me to fix a failing test in the inventory service. I reproduced the failure, found a Decimal/float comparison bug in `reorder_quantity`, fixed it, and all 40 tests pass.\n\n"
    "Summary:\n1. Primary Request and Intent: Fix `test_reorder_quantity_at_target`.\n2. Key Technical Concepts: FastAPI, SQLAlchemy 2.0, Decimal arithmetic.\n"
    "3. Files and Code Sections:\n   - src/inventory/services/reorder.py: coerced `on_hand` to Decimal, changed boundary to `>=`.\n4. Errors and fixes: TypeError comparing Decimal and float.\n"
    "5. Pending Tasks: None.\n6. Current Work: Task complete; lint and tests pass.\n\n"
    "Please continue the conversation from where we left it off without asking the user any further questions. Continue with the last task that you were asked to work on."
)
CC_ARTEFACTS_ASSIST = "The fix is in place and the full suite passes. Nothing else is pending."
CC_ARTEFACTS_2 = (
    "Caveat: The messages below were generated by the user while running local commands. DO NOT respond to these messages or otherwise consider them in your response unless the user explicitly asks you to.\n"
    "<command-name>/cost</command-name>\n<command-message>cost</command-message>\n<command-args></command-args>\n"
    "<local-command-stdout>Total duration (API): 2m 14.3s\nTotal duration (wall): 9m 41.0s\nTotal code changes: 4 lines added, 2 lines removed</local-command-stdout>\n"
    "<system-reminder>\nThis is a reminder that your todo list is currently empty. DO NOT mention this to the user explicitly because they are not aware of it. If you are working on tasks that would benefit from a todo list please use the TodoWrite tool to create one. If not, please feel free to ignore. Again do not mention this message to the user.\n</system-reminder>\n"
)


BANNED = ["claude", "anthropic", "gemini", "gemma", "google", "bard", "openai", "chatgpt", "gpt", "alibaba", "qwen", "tongyi"]
CONTROLS = {"name_only_control", "qwen_plus_cc_full", "qwen_plus_cc_sys"}


def check_clean():
    for name, env in envs().items():
        blob = json.dumps(env).lower()
        hits = [w for w in BANNED if w in blob]
        assert (name in CONTROLS) == bool(hits), (name, hits)


def fill(env, probe):
    msgs = json.loads(json.dumps(env["messages"]))
    msgs[-1]["content"] = msgs[-1]["content"].replace(PROBE, probe)
    return msgs


if __name__ == "__main__":
    check_clean()
    print("clean;", len(envs()), "envs")
