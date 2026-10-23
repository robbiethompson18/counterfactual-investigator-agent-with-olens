"""O-lens tools for the `olens` and `olens_placebo` conditions, addressed by quoting the transcript, not by token number.

Reads come from the O-lens serving session as JSONL, one row per (transcript, position, layer):
{id, token_index, token, prefix, msg_index, role, field, char_offset, layer, bullets: [{text, fve, coef}], joint_fve}, where
`id` is the Rollout's source, `prefix` the decoded 32 tokens up to and including the read position, and `char_offset` the
token's place in messages[msg_index][field] (null for template tokens). Positions are the stop tokens (the viewer's PUNCT
regex) and every <|im_end|> of the whole rendered transcript, prompt included. Each position is placed at a character of
the transcript as `read_transcript` shows it (core.render_spans), so search hits and olens_at share one `char` coordinate.

- `search_olens(regex)`: regex over bullet text only (never the transcript text, so it can't just find the quoted words);
  says how often runs with / without the behavior match, then hits with the transcript text around each and its `char`.
- `olens_at(run, char)`: every read at the position nearest `char` (from any search hit, transcript or O-lens) and its
  neighbours, with the transcript text between them.

`olens_placebo` answers every call from a run of a different env, at the same relative position in that run (with that
run's text around the reads), so the predictor gets as much O-lens text through the same tools, none of it about this
scenario.
"""

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from cfbench.core import Rollout, render_spans
from cfbench.tools import Tool, _schema

BEFORE, AFTER = 80, 40  # characters of transcript around a search hit
GAP_CHARS = 300  # most transcript text shown between two positions in olens_at
MAX_WINDOW = 5  # neighbouring positions each side in olens_at
MAX_HITS = 30  # per search_olens call
TOP_BULLETS = 1  # of each read, highest FVE first (the rows come sorted); the rest are mostly noise

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
        by_token[r["token_index"]][r["layer"]] = r | {"bullets": r["bullets"][:TOP_BULLETS]}
    out = []
    for layers in by_token.values():
        r = next(iter(layers.values()))
        i, field, c = r["msg_index"], r["field"] or "", r["char_offset"]
        if (i, field) in spans and c is not None and not field.startswith("tool_call"):
            char = spans[i, field] + c
        else:  # tool calls and template tokens: the message's end
            char = ends.get(i, len(text))
        where = f"{r['role']} {field}".strip()
        out.append(Position(char, where, layers))
    return sorted(out, key=lambda p: p.char)


def _bullets(r: dict, min_fve: float = 0.0) -> str:
    return " | ".join(f"{b['text']} [{b['fve']:.2f}]" for b in r["bullets"] if b["fve"] >= min_fve)


def _flat(text: str) -> str:
    return text.replace("\n", "⏎")


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
    cache: dict[int, tuple[list[Position], float, str]] = {}

    def run_positions(i: int) -> tuple[list[Position], float, str]:
        """Run i's positions, the factor from their char coordinates to run i's transcript (1 unless placebo), and the
        transcript they are placed in."""
        if i not in cache:
            messages, rows = reads[source[i]]
            text = render_spans(messages)[0]
            cache[i] = (positions(messages, rows), len(text) / max(1, len(texts[i])), text)
        return cache[i]

    yes_no = ["yes" if r.label else "no" for r in base]

    def at(args: dict) -> str:
        i, char = int(args["run"]), int(args["char"])
        window = max(0, min(int(args.get("window", 2)), MAX_WINDOW))
        layers = set(args["layers"]) if args.get("layers") else None
        if i not in source:
            return f"Run {i} has no O-lens reads; the runs with reads are marked in the table."
        pos, scale, text = run_positions(i)
        k = min(range(len(pos)), key=lambda j: abs(pos[j].char - char * scale))
        shown = pos[max(0, k - window) : k + window + 1]
        out, prev = [f"Run {i} (showed the behavior: {yes_no[i]}), {len(shown)} read positions around char {char}:"], None
        for p in shown:
            start = max(prev.char if prev else 0, p.char - GAP_CHARS)
            out.append(f"\n…{_flat(text[start : p.char])}⟨read⟩  [char {round(p.char / scale)}, {p.where}]")
            for layer in sorted(p.layers):
                if layers is None or layer in layers:
                    r = p.layers[layer]
                    bullets = " | ".join(f"{b['text']} [{b['fve']:.2f}]" for b in r["bullets"])
                    out.append(f"  L{layer} (joint FVE {r['joint_fve']:.2f}): {bullets}")
            prev = p
        return "\n".join(out)

    def search(args: dict) -> str:
        try:
            pattern = re.compile(args["regex"], re.IGNORECASE)
        except re.error as e:
            return f"Bad regex: {e}"
        layers, min_fve = set(args["layers"]) if args.get("layers") else None, float(args.get("min_fve") or 0)
        runs = [int(args["run"])] if args.get("run") is not None else read_ids
        hits: dict[tuple[str, int, str], list] = {}  # (context, layer, bullet) -> [runs, fve, line]; prompt reads repeat
        by_layer, per_run = defaultdict(int), defaultdict(int)
        for i in (i for i in runs if i in source):
            pos, scale, text = run_positions(i)
            for p in pos:
                for layer, r in p.layers.items():
                    if layers is not None and layer not in layers:
                        continue
                    for b in r["bullets"]:
                        if b["fve"] >= min_fve and pattern.search(b["text"]):
                            by_layer[layer] += 1
                            per_run[i] += 1
                            context = f"…{_flat(text[max(0, p.char - BEFORE) : p.char])}⟨read⟩{_flat(text[p.char : p.char + AFTER])}…"
                            line = f'char {round(p.char / scale)} ({p.where}) L{layer}: "{context}" → {b["text"]} [{b["fve"]:.2f}]'
                            hits.setdefault((context, layer, b["text"]), [[], b["fve"], line])[0].append(i)
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
        lines = [f"run {','.join(map(str, sorted(set(rs))))} [{'/'.join(sorted({yes_no[i] for i in rs}))}] {line}" for rs, _, line in top]
        more = f" Showing {MAX_HITS} of {len(hits)} distinct hits." if len(hits) > MAX_HITS else ""
        return summary + more + ("\n" + "\n".join(lines) if lines else "")

    layer_list = {"type": "array", "items": {"type": "integer"}, "description": "optional: only these layers"}
    return {
        "search_olens": (
            _schema(
                "search_olens",
                "Case-insensitive regex over the bullet text of every O-lens read (not the transcript text). Says how often runs "
                f"with and without the behavior match, then up to {MAX_HITS} distinct hits, highest FVE first, each with the "
                "transcript text around the read and its run and char (for olens_at).",
                {
                    "regex": {"type": "string"},
                    "run": {"type": "integer", "description": "optional: one run id; default all runs with reads"},
                    "layers": layer_list,
                    "min_fve": {"type": "number", "description": "optional: only bullets with at least this FVE"},
                },
                ["regex"],
            ),
            search,
        ),
        "olens_at": (
            _schema(
                "olens_at",
                "Every O-lens read (all layers) at the read position nearest `char` in a run, and `window` positions "
                "each side, with the transcript text between them. Take run and char from a search_olens or search_transcripts hit.",
                {
                    "run": {"type": "integer"},
                    "char": {"type": "integer"},
                    "window": {"type": "integer", "description": f"positions each side, default 2, max {MAX_WINDOW}"},
                    "layers": layer_list,
                },
                ["run", "char"],
            ),
            at,
        ),
    }, set(read_ids)
