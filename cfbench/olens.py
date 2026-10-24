"""O-lens tools for the `olens` and `olens_placebo` conditions, addressed by quoting the transcript, not by token number.

Reads come from the O-lens serving session as JSONL, one row per (transcript, position, layer):
{id, token_index, token, prefix, msg_index, role, field, char_offset, layer, bullets: [{text, fve, coef}], joint_fve}, where
`id` is the Rollout's source, `prefix` the decoded 32 tokens up to and including the read position, and `char_offset` the
token's place in messages[msg_index][field] (null for template tokens). Positions are the stop tokens (the viewer's PUNCT
regex) and every <|im_end|> of the whole rendered transcript, prompt included. Each position is placed at a character of
the transcript as `read_transcript` shows it (core.render_spans), so search hits and olens_at share one `char` coordinate.

- `search_olens(regex)`: regex over the transcripts; the reads inside each match and the first one after it, with the
  transcript text around each read and its `char`, and how many runs with / without the behavior match. With
  `over="bullets"` the regex runs over every bullet's text instead (all bullets at every read, `top_k` ignored) and only
  the matching bullets come back.
- `olens_at(run, char)`: every read at the position nearest `char` (from any search hit) and its neighbours, with the
  transcript text between them.
Both show each read's top bullet with FVE >= 0.1 by default (`top_k`, `min_fve`); search_olens returns the 10 best
bullets by FVE (`max_bullets`), olens_at reads layers 32, 40, 48 by default.

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
WINDOW, MAX_WINDOW = 10, 20  # neighbouring positions each side in olens_at
AT_LAYERS = [32, 40, 48]  # olens_at's default layers; search_olens defaults to every layer
MAX_BULLETS = 10  # search_olens's default number of bullets returned, highest FVE first
TOP_K, MIN_FVE = 1, 0.1  # default bullet filters (ANDed) for both tools; the agent can change them per call

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
        out.append(Position(char, where, layers))
    return sorted(out, key=lambda p: p.char)


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

    def filters(args: dict, layers: list[int] | None) -> tuple[set[int] | None, int, float]:
        layers = args.get("layers") or layers
        top_k = TOP_K if args.get("top_k") is None else int(args["top_k"])
        return set(layers) if layers else None, top_k, MIN_FVE if args.get("min_fve") is None else float(args["min_fve"])

    def bullets(p: Position, layers: set[int] | None, top_k: int | None, min_fve: float) -> list[tuple[int, dict]]:
        """(layer, bullet) for each bullet passing both filters, by layer (bullets come sorted by FVE)."""
        return [
            (layer, b)
            for layer in sorted(p.layers)
            if layers is None or layer in layers
            for b in p.layers[layer]["bullets"][:top_k]
            if b["fve"] >= min_fve
        ]

    def at(args: dict) -> str:
        i, char = int(args["run"]), int(args["char"])
        window = max(0, min(int(args.get("window", WINDOW)), MAX_WINDOW))
        if i not in source:
            return f"Run {i} has no O-lens reads; the runs with reads are marked in the table."
        layers, top_k, min_fve = filters(args, AT_LAYERS)
        pos, scale, text = run_positions(i)
        k = min(range(len(pos)), key=lambda j: abs(pos[j].char - char * scale))
        shown = pos[max(0, k - window) : k + window + 1]
        out, prev = [f"Run {i} (showed the behavior: {yes_no[i]}), {len(shown)} read positions around char {char}:"], None
        for p in shown:
            start = max(prev.char if prev else 0, p.char - GAP_CHARS)
            out.append(f"\n…{_flat(text[start : p.char])}⟨read⟩  [char {round(p.char / scale)}, {p.where}]")
            got = bullets(p, layers, top_k, min_fve)
            out += [f"  L{layer}: {b['text']} [FVE {b['fve']:.2f}]" for layer, b in got] or [f"  (no bullet with FVE >= {min_fve})"]
            prev = p
        return "\n".join(out)

    def search(args: dict) -> str:
        """Regex over the transcripts (the bullets at the reads inside each match and the first one after it) or over the
        bullets themselves (over="bullets": the matching bullets), best FVE first."""
        try:
            pattern = re.compile(args["regex"], re.IGNORECASE)
        except re.error as e:
            return f"Bad regex: {e}"
        over = args.get("over") or "transcript"
        if over not in ("transcript", "bullets"):
            return f'Bad over: {over!r} (use "transcript" or "bullets")'
        layers, top_k, min_fve = filters(args, None)
        max_bullets = int(args.get("max_bullets") or MAX_BULLETS)
        runs = [int(args["run"])] if args.get("run") is not None else read_ids
        searched = [i for i in runs if i in source]
        found: dict[tuple, list] = {}  # (context, layer, bullet) -> [runs, fve, line]; prompt text repeats across runs
        matched = set()
        for i in searched:
            pos, scale, text = run_positions(i)
            if over == "bullets":
                hits = [(p, [(layer, b) for layer, b in bullets(p, layers, None, min_fve) if pattern.search(b["text"])]) for p in pos]
            else:
                hits = []
                for m in pattern.finditer(texts[i]):
                    start, end = m.start() * scale, m.end() * scale
                    inside = [p for p in pos if start <= p.char <= end]
                    after = next((p for p in pos if p.char > end), None)
                    hits += [(p, bullets(p, layers, top_k, min_fve)) for p in inside + ([after] if after and after not in inside else [])]
                    matched.add(i)
            for p, got in hits:
                if got:
                    matched.add(i)
                context = f"…{_flat(text[max(0, p.char - BEFORE) : p.char])}⟨read⟩{_flat(text[p.char : p.char + AFTER])}…"
                for layer, b in got:
                    line = f'char {round(p.char / scale)} ({p.where}) L{layer} [FVE {b["fve"]:.2f}]: "{context}"\n  → {b["text"]}'
                    found.setdefault((context, layer, b["text"]), [[], b["fve"], line])[0].append(i)
        kinds = []
        for label, name in ((True, "with"), (False, "without")):
            ids = [i for i in searched if base[i].label == label]
            if ids:
                kinds.append(f"{sum(i in matched for i in ids)} of {len(ids)} runs {name} the behavior")
        summary = (
            f"The regex matches {'bullets' if over == 'bullets' else 'the transcripts'} of {' and '.join(kinds)} (runs with reads only). "
        )
        ranked = sorted(found.values(), key=lambda h: -h[1])
        if len(ranked) > max_bullets:
            summary += (
                f"{len(ranked)} bullets pass the filters; the top {max_bullets} by FVE are shown, {len(ranked) - max_bullets} are not."
            )
        else:
            summary += f"{len(ranked)} bullets pass the filters."
        label = lambda rs: f"run {','.join(map(str, sorted(set(rs))))} [{'/'.join(sorted({yes_no[i] for i in rs}))}]"
        return summary + "".join(f"\n\n{label(rs)} {line}" for rs, _, line in ranked[:max_bullets])

    filter_params = {
        "top_k": {"type": "integer", "description": f"bullets per read, highest FVE first (default {TOP_K})"},
        "min_fve": {"type": "number", "description": f"only bullets with at least this FVE (default {MIN_FVE})"},
    }
    return {
        "search_olens": (
            _schema(
                "search_olens",
                "Case-insensitive regex over every base-arm transcript that has O-lens reads. Returns the O-lens bullets at the "
                "punctuation inside each match and the first read after it, highest FVE first, each with the transcript text "
                "around the read and its run and char (for olens_at), and how many runs with and without the behavior match. "
                'With over="bullets", the regex runs over the text of every O-lens bullet instead (all bullets at every read, '
                "top_k ignored) and returns only the bullets that match, with the same context, and how many runs have a match.",
                {
                    "regex": {"type": "string"},
                    "over": {
                        "type": "string",
                        "enum": ["transcript", "bullets"],
                        "description": "what the regex searches: the transcript text (default) or the O-lens bullets' text",
                    },
                    "run": {"type": "integer", "description": "optional: one run id"},
                    "layers": {"type": "array", "items": {"type": "integer"}, "description": "optional: only these layers (default all)"},
                    **filter_params,
                    "max_bullets": {"type": "integer", "description": f"bullets returned (default {MAX_BULLETS})"},
                },
                ["regex"],
            ),
            search,
        ),
        "olens_at": (
            _schema(
                "olens_at",
                "The O-lens reads at the read position nearest `char` in a run and `window` positions each side, with the "
                "transcript text between them. Take run and char from a search_olens or search_transcripts hit.",
                {
                    "run": {"type": "integer"},
                    "char": {"type": "integer"},
                    "window": {"type": "integer", "description": f"positions each side (default {WINDOW}, max {MAX_WINDOW})"},
                    "layers": {"type": "array", "items": {"type": "integer"}, "description": f"default {AT_LAYERS}"},
                    **filter_params,
                },
                ["run", "char"],
            ),
            at,
        ),
    }, set(read_ids)


def sources(paths: list[str]) -> list[dict]:
    """What each reads file was at launch, for the run's meta: the lens (from the reader's reads.provenance.json, else the
    read run's meta.json; None if neither says) and the row count, since read files keep growing or get rewritten."""
    out = []
    for path in map(Path, paths):
        prov, meta = path.with_name(f"{path.stem}.provenance.json"), path.parent / "meta.json"
        if prov.exists():
            lens = json.loads(prov.read_text())["lens"]
        elif path.name == "reads.jsonl" and meta.exists():
            lens = json.loads(meta.read_text()).get("params", {}).get("lens")
        else:
            lens = None
        out.append({"path": str(path), "lens": lens, "rows": sum(1 for _ in path.open())})
    return out


def fve_note(reads: Reads) -> str:
    """The 10th/50th/90th percentile of the top bullet's FVE over every loaded read, for the prompt."""
    top = sorted(row["bullets"][0]["fve"] for _, rows in reads.values() for row in rows if row["bullets"])
    q = [top[int(f * (len(top) - 1))] for f in (0.1, 0.5, 0.9)]
    return f"Over all reads here, the top bullet's FVE is {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} at the 10th / 50th / 90th percentile."
