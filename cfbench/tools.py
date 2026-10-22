"""Tools for the tool conditions: read or search any base-arm run, and its O-lens reads. Ids are positions in the base-arm rollout list."""

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


def _paged_tools(texts: list[str], labels: list[bool], what: str, read_name: str, search_name: str, line_chars: int) -> dict[str, Tool]:
    """A paged read and a line regex search over one text per base-arm run (texts[i] == "" if run i has none)."""
    yes_no = ["yes" if label else "no" for label in labels]

    def read(args: dict) -> str:
        i, offset = int(args["id"]), int(args.get("offset", 0))
        if not 0 <= i < len(texts) or not texts[i]:
            return f"No {what} for run {i}."
        page, end = texts[i][offset : offset + PAGE], min(offset + PAGE, len(texts[i]))
        more = f" Call again with offset={end} for the rest." if end < len(texts[i]) else ""
        return f"Run {i} (showed the behavior: {yes_no[i]}), {what}, characters {offset}-{end} of {len(texts[i])}.{more}\n\n{page}"

    def search(args: dict) -> str:
        try:
            pattern = re.compile(args["regex"], re.IGNORECASE)
        except re.error as e:
            return f"Bad regex: {e}"
        hits = [(i, line) for i, text in enumerate(texts) for line in text.splitlines() if pattern.search(line)]
        runs = sorted({i for i, _ in hits})
        lines = [f"run {i} [{yes_no[i]}]: {line.strip()[:line_chars]}" for i, line in hits[:MAX_HITS]]
        n_yes = sum(yes_no[i] == "yes" for i in runs)
        n_with = sum(bool(t) for t in texts)
        summary = f"{len(hits)} matching lines in {len(runs)} of {n_with} runs ({n_yes} with the behavior)"
        return summary + (f", first {MAX_HITS} shown" if len(hits) > MAX_HITS else "") + ":\n" + "\n".join(lines)

    return {
        read_name: (
            _schema(
                read_name,
                f"Read one base-arm run's {what} as text, {PAGE} characters at a time.",
                {"id": {"type": "integer"}, "offset": {"type": "integer", "description": "character offset, default 0"}},
                ["id"],
            ),
            read,
        ),
        search_name: (
            _schema(
                search_name,
                f"Case-insensitive regex search over every line of every base-arm run's {what}. Returns up to {MAX_HITS} matching lines.",
                {"regex": {"type": "string"}},
                ["regex"],
            ),
            search,
        ),
    }


def transcript_tools(base: list[Rollout]) -> dict[str, Tool]:
    texts = [render(r.messages) for r in base]
    return _paged_tools(texts, [r.label for r in base], "transcript", "read_transcript", "search_transcripts", 300)


def olens_tools(base: list[Rollout], texts: list[str]) -> dict[str, Tool]:
    """texts[i]: run i's reads rendered by cfbench.olens.render, "" if it has none."""
    return _paged_tools(texts, [r.label for r in base], "O-lens reads", "read_olens", "search_olens", 600)


def call(tools: dict[str, Tool], name: str, arguments: str) -> str:
    try:
        return tools[name][1](json.loads(arguments))
    except (KeyError, ValueError, TypeError) as e:
        return f"Tool error: {e!r}"
