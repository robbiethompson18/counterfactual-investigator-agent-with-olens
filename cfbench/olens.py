"""O-lens reads of base-arm runs, for the olens conditions.

Reads come from the O-lens serving session as JSONL, one row per (transcript, position, layer):
{id, token_index, token, prefix, msg_index, role, field, char_offset, layer, bullets: [{text, fve, coef}], joint_fve}, where
`id` is the Rollout's source and `prefix` the decoded text of the 32 tokens up to and including the read position. Positions
are the stop tokens (the viewer's PUNCT regex) and every <|im_end|> of the whole rendered transcript, prompt included.

- `olens`: read_olens / search_olens over every read of the base runs, next to the transcript tools.
- `olens_edit`: the same tools, plus reads at the edited lines of a few base runs put straight into the prompt (EDIT_RUNS
  runs, EDIT_POSITIONS positions each, EDIT_LAYERS). Lines the edit removes are read where they sit; for a pure insertion,
  the unchanged lines on either side of it.
- `*_placebo`: each run's reads are swapped for a run of a different env (for olens_edit, random positions of it), so the
  predictor gets as much O-lens text and the same tools, but none of it about this scenario.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from cfbench.core import Rollout

PREFIX_CHARS = 60  # of the prefix shown on each line, so a search hit says where in the run it is
EDIT_RUNS, EDIT_POSITIONS, EDIT_LAYERS = 6, 15, (40, 48)
MATCH_CHARS = 20  # a read sits on a diff line if the last this-many characters of its prefix are in that line

Reads = dict[str, list[dict]]  # rollout source -> its read rows, in transcript order


def load(paths: list[str]) -> Reads:
    by_source = defaultdict(list)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            by_source[row["id"]].append(row)
    return {source: sorted(rows, key=lambda r: (r["token_index"], r["layer"])) for source, rows in by_source.items()}


def _line(r: dict) -> str:
    where = r["prefix"][-PREFIX_CHARS:].replace("\n", "⏎")
    bullets = " | ".join(f"{b['text']} [{b['fve']:.2f}]" for b in r["bullets"])
    return f'tok {r["token_index"]} L{r["layer"]} at "…{where}": {bullets} (joint FVE {r["joint_fve"]:.2f})'


def render(rows: list[dict]) -> str:
    return "\n".join(map(_line, rows))


def _swap(base: list[Rollout], reads: Reads, placebo: bool, seed: int) -> list[list[dict]]:
    """Each base run's read rows ([] if none); placebo: runs with reads get the rows of another env's run instead."""
    real = [reads.get(r.source, []) for r in base]
    if not placebo:
        return real
    env = base[0].source.split("/")[0]
    pool = sorted(source for source in reads if source.split("/")[0] != env)
    assert pool, f"placebo needs reads from an env other than {env}"
    random.Random(seed).shuffle(pool)
    picks = iter(pool * (len(base) // len(pool) + 1))
    return [reads[next(picks)] if rows else [] for rows in real]


def texts(base: list[Rollout], reads: Reads, placebo: bool, seed: int = 0) -> list[str]:
    """One O-lens text per base run for the tools, "" for runs without reads."""
    return [render(rows) for rows in _swap(base, reads, placebo, seed)]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _edit_lines(edit: str) -> list[str]:
    """Base-arm lines to read at: the removed lines, or for a pure insertion the unchanged lines around it."""
    diff = edit.split("```diff\n", 1)[1].rsplit("```", 1)[0].splitlines()
    body = [(line[:1], line[1:]) for line in diff if line[:1] in " +-" and not line.startswith(("---", "+++"))]
    removed = [text for kind, text in body if kind == "-"]
    if removed:
        return [_norm(t) for t in removed if _norm(t)]
    around = [i for i, (kind, _) in enumerate(body) if kind == " " and {"+"} & {k for k, _ in body[max(i - 1, 0) : i + 2]}]
    return [_norm(body[i][1]) for i in around if _norm(body[i][1])]


def edit_section(base: list[Rollout], reads: Reads, edit: str, placebo: bool, seed: int) -> str:
    """Prompt section with O-lens reads at the edited lines of EDIT_RUNS base runs (half with the behavior where possible)."""
    rng = random.Random(seed)
    rows_by_run = _swap(base, reads, placebo, seed)
    with_reads = [i for i, rows in enumerate(rows_by_run) if rows]
    yes, no = ([i for i in with_reads if base[i].label == label] for label in (True, False))
    rng.shuffle(yes), rng.shuffle(no)
    half = EDIT_RUNS // 2
    picked = yes[:half] + no[:half]
    picked = sorted(picked + (yes[half:] + no[half:])[: EDIT_RUNS - len(picked)])  # top up from the other kind
    targets = _edit_lines(edit)
    blocks = []
    for i in picked:
        rows = [r for r in rows_by_run[i] if r["layer"] in EDIT_LAYERS]
        positions = sorted({r["token_index"] for r in rows})
        if placebo:
            hits = sorted(rng.sample(positions, min(EDIT_POSITIONS, len(positions))))
        else:
            tail = {r["token_index"]: _norm(r["prefix"])[-MATCH_CHARS:] for r in rows}
            hits = [t for t in positions if any(tail[t] in line for line in targets)]
            if len(hits) > EDIT_POSITIONS:  # evenly spaced over the edited lines
                hits = [hits[round(j * (len(hits) - 1) / (EDIT_POSITIONS - 1))] for j in range(EDIT_POSITIONS)]
        lines = render([r for r in rows if r["token_index"] in set(hits)]) or "(no read positions on the edited lines)"
        blocks.append(f"### Run {i} (showed the behavior: {'yes' if base[i].label else 'no'})\n\n{lines}")
    return f"""\
## O-lens reads at the edited part of the base context

Reads at the punctuation tokens on the lines the edit changes (or, for an insertion, the lines around it), in \
{len(picked)} base runs, layers {", ".join(map(str, EDIT_LAYERS))}. Each line: token index, layer, the text just before \
the read position, then the bullets with their FVE.

{chr(10).join(blocks)}"""
