"""Tools for the `agentic` condition: read or search any base-arm run. Ids are positions in the base-arm rollout list."""

import json
import re
from collections.abc import Callable

from openai.types.chat import ChatCompletionFunctionToolParam

from cfbench.core import Rollout, render

PAGE = 40_000  # characters per read_transcript call
MAX_HITS = 50

Tool = tuple[ChatCompletionFunctionToolParam, Callable[[dict], str]]  # (OpenAI function schema, handler on the parsed arguments)


def _schema(name: str, description: str, properties: dict, required: list[str]) -> ChatCompletionFunctionToolParam:
    params = {"type": "object", "properties": properties, "required": required}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


def _lines(text: str) -> list[tuple[int, str]]:
    """(start character, line) for each line."""
    out, at = [], 0
    for line in text.split("\n"):
        out.append((at, line))
        at += len(line) + 1
    return out


def transcript_tools(base: list[Rollout], chars: bool = False) -> dict[str, Tool]:
    """`chars`: search hits also give the character where their line starts (the O-lens tools are addressed that way)."""
    texts = [render(r.messages) for r in base]
    yes_no = ["yes" if r.label else "no" for r in base]

    def read(args: dict) -> str:
        i, offset = int(args["id"]), int(args.get("offset", 0))
        if not 0 <= i < len(texts):
            return f"No run {i}; ids are 0-{len(texts) - 1}."
        page, end = texts[i][offset : offset + PAGE], min(offset + PAGE, len(texts[i]))
        more = f" Call again with offset={end} for the rest." if end < len(texts[i]) else ""
        return f"Run {i} (showed the behavior: {yes_no[i]}), characters {offset}-{end} of {len(texts[i])}.{more}\n\n{page}"

    def search(args: dict) -> str:
        try:
            pattern = re.compile(args["regex"], re.IGNORECASE)
        except re.error as e:
            return f"Bad regex: {e}"
        hits = [(i, at, line) for i, text in enumerate(texts) for at, line in _lines(text) if pattern.search(line)]
        runs = sorted({i for i, _, _ in hits})
        lines = [f"run {i} [{yes_no[i]}]{f' char {at}' if chars else ''}: {line.strip()[:300]}" for i, at, line in hits[:MAX_HITS]]
        n_yes = sum(yes_no[i] == "yes" for i in runs)
        summary = f"{len(hits)} matching lines in {len(runs)} of {len(texts)} runs ({n_yes} with the behavior)"
        return summary + (f", first {MAX_HITS} shown" if len(hits) > MAX_HITS else "") + ":\n" + "\n".join(lines)

    return {
        "read_transcript": (
            _schema(
                "read_transcript",
                f"Read one base-arm run as text, {PAGE} characters at a time.",
                {"id": {"type": "integer"}, "offset": {"type": "integer", "description": "character offset, default 0"}},
                ["id"],
            ),
            read,
        ),
        "search_transcripts": (
            _schema(
                "search_transcripts",
                f"Case-insensitive regex search over every line of every base-arm run. Returns up to {MAX_HITS} matching lines.",
                {"regex": {"type": "string"}},
                ["regex"],
            ),
            search,
        ),
    }


def call(tools: dict[str, Tool], name: str, arguments: str) -> str:
    try:
        return tools[name][1](json.loads(arguments))
    except (KeyError, ValueError, TypeError) as e:
        return f"Tool error: {e!r}"
