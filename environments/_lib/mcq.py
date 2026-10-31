"""Shared MCQ option parsing and character offsets for option-newline reads."""

import re

OPTION = re.compile(r"^\(?([A-P])[.)] (.*)$", re.MULTILINE)


def options(text: str) -> list[re.Match]:
    """The final A, B, C, ... block: earlier lettered statements belong to the question stem."""
    out: list[re.Match] = []
    for m in OPTION.finditer(text):
        if m.group(1) == "A":
            out = []
        if len(out) < 16 and m.group(1) == "ABCDEFGHIJKLMNOP"[len(out)]:
            out.append(m)
    return out


def option_newlines(text: str, *, final_end: int | None = None) -> list[tuple[int, str]]:
    """Newlines after COMPLETE options, as (character offset, newline text), never internal option newlines.

    The last option ends at the canonical answer instruction, or at an explicit final_end supplied by the prompt builder.
    Arbitrary trailing prose cannot be distinguished from option text; require a boundary rather than guessing a blank line.
    """
    matches = options(text)
    if not matches:
        return []
    if final_end is None:
        final_end = text.rfind("\n\nAnswer with the letter only.")
    if final_end < matches[-1].end() or text[final_end : final_end + 1] != "\n":
        raise ValueError("MCQ final option needs the answer-instruction separator or an explicit final_end newline offset")
    offsets = [m.start() - 1 for m in matches[1:]] + [final_end]
    return [(c, "\n\n" if text.startswith("\n\n", c) else "\n") for c in offsets]
