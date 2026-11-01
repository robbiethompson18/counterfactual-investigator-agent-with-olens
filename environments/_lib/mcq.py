"""Shared MCQ prompt, bare-letter reply parser, option parsing and character offsets for option-newline reads.

hint_mcq's no-hint prompt (MMLU-Pro) and supergpqa's prompt are both mcq_prompt(), and both score replies with bare_letter():
mcq_unselected and supergpqa_inline compare the two pools as one prompt, so change them here, never in one env.
"""

import re

LETTERS = "ABCDEFGHIJ"
INSTRUCTION = "Answer with the letter only."
BARE_LETTER = re.compile(r"^\(?([A-J])[).]?\s*$")  # the whole reply is one letter
OPTION = re.compile(r"^\(?([A-P])[.)] (.*)$", re.MULTILINE)


def mcq_question(stem: str, options) -> str:
    """The stem, then the options as `A. ...`, one per line (hint_mcq's arms put their hints and fillers around this)."""
    return stem.strip() + "\n" + "\n".join(f"{c}. {o}" for c, o in zip(LETTERS[: len(options)], options, strict=True))


def mcq_prompt(stem: str, options) -> str:
    """The no-hint prompt: the question, a blank line, INSTRUCTION."""
    return mcq_question(stem, options) + "\n\n" + INSTRUCTION


def bare_letter(text: str) -> str | None:
    """The answer when the reply is one letter ("B", "(B)", "B.", "B)"), as the prompt asks; None for anything else (a worked
    solution), which the envs' measure() leaves unscored."""
    m = BARE_LETTER.match(text.strip())
    return m.group(1) if m else None


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
        final_end = text.rfind("\n\n" + INSTRUCTION)
    if final_end < matches[-1].end() or text[final_end : final_end + 1] != "\n":
        raise ValueError("MCQ final option needs the answer-instruction separator or an explicit final_end newline offset")
    offsets = [m.start() - 1 for m in matches[1:]] + [final_end]
    return [(c, "\n\n" if text.startswith("\n\n", c) else "\n") for c in offsets]
