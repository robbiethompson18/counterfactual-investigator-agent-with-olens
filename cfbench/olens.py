"""O-lens tools for the `olens` and `olens_placebo` conditions, addressed by quoting the transcript, not by token number.

Reads come from the O-lens serving session as JSONL, one row per (transcript, position, layer):
{id, token_index, token, prefix, msg_index, role, field, char_offset, layer, bullets: [{text, fve, coef}], joint_fve}, where
`id` is the Rollout's source, `prefix` the decoded 32 tokens up to and including the read position, and `char_offset` the
token's place in messages[msg_index][field] (null for template tokens). Positions are the stop tokens (the viewer's PUNCT
regex) and every <|im_end|> of the whole rendered transcript, prompt included. Each position is placed at a character of
the transcript as `read_transcript` shows it (core.render_spans), so the predictor can quote what it read.

- `olens_at(quote, run?)`: reads at the punctuation inside a quoted span, plus the first one after it.
- `search_olens(regex)`: regex over bullet text only (never the transcript text, so it can't just find the quoted words),
  with how many runs with / without the behavior have a hit.

`olens_placebo` answers every call from a run of a different env, at the same relative position in that run, so the
predictor gets as much O-lens text through the same tools, but none of it about this scenario.
"""

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from cfbench.core import Rollout, render_spans
from cfbench.tools import Tool, _schema

TAIL_CHARS = 60  # of the text before a read, shown with it
MAX_POSITIONS = 12  # per olens_at call
MAX_HITS = 30  # per search_olens call
AT_RUNS = 2  # runs olens_at reads when no run is given: one with the behavior and one without, where possible

Reads = dict[str, tuple[list[dict], list[dict]]]  # rollout source -> (its messages, its read rows)


def load(paths: list[str]) -> Reads:
    """Each reads.jsonl's run dir also has the requests.jsonl it was read from, which has every transcript's messages
    (the placebo needs another env's runs, which aren't loaded)."""
    by_source: Reads = {}
    for path in map(Path, paths):
        messages = {r["id"]: r["messages"] for r in map(json.loads, (path.parent / "requests.jsonl").read_text().splitlines())}
        for row in map(json.loads, path.read_text().splitlines()):
            if row["msg_index"] < 0:  # the tool definitions before the system prompt: same in every run, not in the transcript
                continue
            by_source.setdefault(row["id"], (messages[row["id"]], []))[1].append(row)
    return by_source


@dataclass
class Position:
    char: int  # in the run's rendered transcript
    where: str  # e.g. "assistant reasoning_content"
    tail: str  # text just before the read
    layers: dict[int, dict]  # layer -> read row


def positions(messages: list[dict], rows: list[dict]) -> list[Position]:
    """One Position per read token, in transcript order."""
    text, spans = render_spans(messages)
    starts = defaultdict(list)  # message -> its fields' starts, to place template tokens at the message's end
    for (i, _), b in spans.items():
        starts[i].append(b)
    ends = {i: (min(starts[i + 1]) - 2 if i + 1 in starts else len(text)) for i in starts}
    by_token = defaultdict(dict)
    for r in rows:
        by_token[r["token_index"]][r["layer"]] = r
    out = []
    for layers in by_token.values():
        r = next(iter(layers.values()))
        i, field, c = r["msg_index"], r["field"] or "", r["char_offset"]
        if (i, field) in spans and c is not None and not field.startswith("tool_call"):
            char = spans[i, field] + c
        else:  # tool calls and template tokens: the message's end
            char = ends.get(i, len(text))
        where = f"{r['role']} {field}".strip()
        out.append(Position(char, where, r["prefix"][-TAIL_CHARS:].replace("\n", "⏎"), layers))
    return sorted(out, key=lambda p: p.char)


def _bullets(r: dict, min_fve: float = 0.0) -> str:
    return " | ".join(f"{b['text']} [{b['fve']:.2f}]" for b in r["bullets"] if b["fve"] >= min_fve)


def _show(p: Position, layers: set[int] | None, scale: float) -> str:
    lines = [f'char {round(p.char / scale)} ({p.where}) at "…{p.tail}":']
    for layer in sorted(p.layers):
        if layers is None or layer in layers:
            r = p.layers[layer]
            lines.append(f"  L{layer} (joint FVE {r['joint_fve']:.2f}): {_bullets(r)}")
    return "\n".join(lines)


def tools(base: list[Rollout], reads: Reads, placebo: bool, seed: int = 0) -> tuple[dict[str, Tool], set[int]]:
    """The two O-lens tools over the base runs, and the ids of the runs that have reads."""
    texts = [render_spans(r.messages)[0] for r in base]
    read_ids = [i for i, r in enumerate(base) if r.source in reads]
    source = {i: base[i].source for i in read_ids}  # whose reads answer for run i
    if placebo:
        env = base[0].source.split("/")[0]
        pool = sorted(s for s in reads if s.split("/")[0] != env)
        assert pool, f"placebo needs reads from an env other than {env}"
        random.Random(seed).shuffle(pool)
        source = dict(zip(read_ids, pool * (len(read_ids) // len(pool) + 1)))
    cache: dict[int, tuple[list[Position], float]] = {}

    def run_positions(i: int) -> tuple[list[Position], float]:
        """Run i's positions, and the factor from their char coordinates to run i's transcript (1 unless placebo)."""
        if i not in cache:
            messages, rows = reads[source[i]]
            cache[i] = (positions(messages, rows), len(texts[i]) / max(1, len(render_spans(messages)[0])))
        return cache[i]

    yes_no = ["yes" if r.label else "no" for r in base]

    def at(args: dict) -> str:
        quote, layers = str(args["quote"]), set(args["layers"]) if args.get("layers") else None
        runs = [int(args["run"])] if args.get("run") is not None else None
        if runs and runs[0] not in source:
            return f"Run {runs[0]} has no O-lens reads; the runs with reads are marked in the table."
        found = [i for i in (runs or read_ids) if quote in texts[i]]
        if not found:
            return "Quote not found. Copy it exactly (case and whitespace) from read_transcript, within one message."
        if not runs:  # one run with the behavior and one without, where the quote occurs in both kinds
            found = sorted({next((i for i in found if base[i].label == kind), found[0]) for kind in (True, False)})[:AT_RUNS]
        out = []
        for i in found:
            start = texts[i].index(quote)
            pos, scale = run_positions(i)
            inside = [p for p in pos if start <= p.char / scale <= start + len(quote)]
            after = next((p for p in pos if p.char / scale > start + len(quote)), None)
            shown = inside + ([after] if after and after not in inside else [])
            cut = len(shown) > MAX_POSITIONS
            if cut:
                shown = [shown[round(j * (len(shown) - 1) / (MAX_POSITIONS - 1))] for j in range(MAX_POSITIONS)]
            note = f", {MAX_POSITIONS} of them evenly spaced" if cut else ""
            body = "\n".join(_show(p, layers, scale) for p in shown) or "(no read positions there)"
            out.append(f"Run {i} (showed the behavior: {yes_no[i]}), quote at char {start}: {len(shown)} read positions{note}.\n{body}")
        if not runs:
            out.append(f"(The quote occurs in {sum(quote in texts[i] for i in read_ids)} of the {len(read_ids)} runs with reads.)")
        return "\n\n".join(out)

    def search(args: dict) -> str:
        try:
            pattern = re.compile(args["regex"], re.IGNORECASE)
        except re.error as e:
            return f"Bad regex: {e}"
        layers, min_fve = set(args["layers"]) if args.get("layers") else None, float(args.get("min_fve") or 0)
        runs = [int(args["run"])] if args.get("run") is not None else read_ids
        hits: dict[tuple[str, int, str], list] = {}  # (tail, layer, bullet) -> [runs, fve, line]; prompt reads repeat
        by_layer, per_run = defaultdict(int), defaultdict(int)
        for i in (i for i in runs if i in source):
            pos, scale = run_positions(i)
            for p in pos:
                for layer, r in p.layers.items():
                    if layers is not None and layer not in layers:
                        continue
                    for b in r["bullets"]:
                        if b["fve"] >= min_fve and pattern.search(b["text"]):
                            by_layer[layer] += 1
                            per_run[i] += 1
                            key = (p.tail, layer, b["text"])
                            line = f'char {round(p.char / scale)} ({p.where}) L{layer} at "…{p.tail}": {b["text"]} [{b["fve"]:.2f}]'
                            hits.setdefault(key, [[], b["fve"], line])[0].append(i)
        searched = [i for i in runs if i in source]
        kinds = []
        for label, name in ((True, "with"), (False, "without")):
            ids = [i for i in searched if base[i].label == label]
            if ids:
                n_hit = sum(per_run[i] > 0 for i in ids)
                per_100 = 100 * sum(per_run[i] for i in ids) / max(1, sum(len(run_positions(i)[0]) for i in ids))
                kinds.append(f"{n_hit} of {len(ids)} runs {name} the behavior ({per_100:.0f} per 100 read positions)")
        summary = (
            f"Matching bullets in {' and '.join(kinds)}. "
            f"Matches by layer: {', '.join(f'L{k} {v}' for k, v in sorted(by_layer.items())) or 'none'}. Hits by FVE:"
        )
        top = sorted(hits.values(), key=lambda h: -h[1])[:MAX_HITS]
        lines = [f"runs {','.join(map(str, sorted(set(rs))))} [{'/'.join(sorted({yes_no[i] for i in rs}))}]: {line}" for rs, _, line in top]
        more = f" Showing {MAX_HITS} of {len(hits)} distinct hits." if len(hits) > MAX_HITS else ""
        return summary + more + ("\n" + "\n".join(lines) if lines else "")

    layer_list = {"type": "array", "items": {"type": "integer"}, "description": "optional: only these layers"}
    run_id = {"type": "integer", "description": "optional: one run id (from the table); default all runs with reads"}
    return {
        "olens_at": (
            _schema(
                "olens_at",
                "O-lens reads at the punctuation inside a quoted span of a base-arm transcript, plus the first read after it. "
                "Quote the text exactly as read_transcript shows it. Without `run`, reads it in one run with the behavior and "
                f"one without that contain the quote. At most {MAX_POSITIONS} positions per call.",
                {"quote": {"type": "string"}, "run": run_id, "layers": layer_list},
                ["quote"],
            ),
            at,
        ),
        "search_olens": (
            _schema(
                "search_olens",
                "Case-insensitive regex over the bullet text of every O-lens read (not the transcript text). Says how many runs "
                f"with and without the behavior have a match, then up to {MAX_HITS} distinct hits, highest FVE first, with where they are.",
                {
                    "regex": {"type": "string"},
                    "run": run_id,
                    "layers": layer_list,
                    "min_fve": {"type": "number", "description": "optional: only bullets with at least this FVE"},
                },
                ["regex"],
            ),
            search,
        ),
    }, set(read_ids)
