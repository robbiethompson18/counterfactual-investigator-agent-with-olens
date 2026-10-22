"""O-lens reads of base-arm runs, for the `olens` and `olens_placebo` conditions.

Reads come from the O-lens serving session as JSONL, one row per (transcript, position, layer):
{id, token_index, token, prefix, layer, bullets: [{text, fve}], joint_fve}, where `id` is the Rollout's source and `prefix`
the decoded text of the ~32 tokens up to and including the read position. Positions are the stop tokens (the viewer's
PUNCT regex) and turn ends of the whole rendered transcript, prompt included.

`olens_placebo` gives each run that has reads the reads of a run from a different env instead, so the predictor gets as much
O-lens text and the same tools, but none of it is about this scenario.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from cfbench.core import Rollout

PREFIX_CHARS = 60  # of the prefix shown on each line, so a search hit says where in the run it is


def load(paths: list[str]) -> dict[str, str]:
    """Rollout source -> its reads rendered one line per (position, layer), in transcript order."""
    by_source = defaultdict(list)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            by_source[row["id"]].append(row)
    return {
        source: "\n".join(_line(r) for r in sorted(rows, key=lambda r: (r["token_index"], r["layer"])))
        for source, rows in by_source.items()
    }


def _line(r: dict) -> str:
    where = r["prefix"][-PREFIX_CHARS:].replace("\n", "⏎")
    bullets = " | ".join(f"{b['text']} [{b['fve']:.2f}]" for b in r["bullets"])
    return f'tok {r["token_index"]} L{r["layer"]} at "…{where}": {bullets} (joint FVE {r["joint_fve"]:.2f})'


def texts(base: list[Rollout], reads: dict[str, str], placebo: bool, seed: int = 0) -> list[str]:
    """One O-lens text per base run, "" for runs without reads. Placebo: same runs, reads of other envs' runs."""
    real = [reads.get(r.source, "") for r in base]
    if not placebo:
        return real
    env = base[0].source.split("/")[0]
    pool = sorted(source for source in reads if source.split("/")[0] != env)
    assert pool, f"placebo needs reads from an env other than {env}"
    random.Random(seed).shuffle(pool)
    picks = iter(pool * (len(base) // len(pool) + 1))
    return [reads[next(picks)] if text else "" for text in real]
