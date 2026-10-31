"""Shared MCQ option parsing and character offsets for option-newline reads."""

import re

OPTION = re.compile(r"^\(?([A-P])[.)] (.*)$", re.MULTILINE)


def options(text: str) -> list[re.Match]:
    """The option lines, in letter order from A: ignore earlier statement lines such as "I. P is invertible."."""
    out: list[re.Match] = []
    for m in OPTION.finditer(text):
        if m.group(1) == "ABCDEFGHIJKLMNOP"[len(out)]:
            out.append(m)
    return out


def option_newlines(text: str) -> list[tuple[int, str]]:
    """(char offset, token) of the newline token ending each option line: "\\n", or "\\n\\n" after the last option."""
    out = []
    for m in options(text):
        nl = text.index("\n", m.end())
        out.append((nl, "\n\n" if text.startswith("\n\n", nl) else "\n"))
    return out
