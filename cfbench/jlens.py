"""J-lens tools for the `jlens_agentic` and `jlens_placebo_agentic` conditions: the O-lens tools' shape (cfbench/olens.py), with
the J++ lens's top-k next tokens per layer in place of bullets, and no FVE.

Reads come from scripts/jpp_read_requests.py, one row per (transcript, position) with every layer in it:
{id, token_index, token, msg_index, role, field, char_offset, layers: [...], top: [[[token, p] * k] per layer]}, with a
requests.jsonl beside it (olens.Reads loads both and keeps the same positions an O-lens read of those requests would).

- `search_jlens(regex)`: regex over the transcripts; the reads at the punctuation inside each match and the first one
  after it. With `over="tokens"` the regex runs over the predicted tokens instead (stripped of surrounding whitespace) and
  returns every (read, layer) where one of the top-k matches, highest probability first.
- `jlens_at(run, char)`: every read around a char from any search hit.
Both show layers JLENS_LAYERS by default; one line per layer, every top-k token with its probability (no filter).

`jlens_placebo_agentic` answers from a run of a different env at the same relative position, as olens_placebo_agentic does.
"""

import json
import random
import re

from cfbench.core import Rollout, render_spans
from cfbench.olens import GAP_CHARS, MAX_WINDOW, WINDOW, Position, Reads, _context, _flat, positions
from cfbench.tools import Tool, schema

JLENS_LAYERS = (36, 40, 44, 48, 52, 56, 60)  # default layers shown; earlier J++ layers are mostly noise (multilingual junk)
MAX_HITS = 20  # search_jlens's default number of reads (over="transcript") or token hits (over="tokens") returned


def _tok(t: str) -> str:
    """A token as the predictor sees it: quoted, whitespace escaped, so " B" and "B" and "\\n" stay distinguishable."""
    return json.dumps(t, ensure_ascii=False)


def _line(layer: int, top: list) -> str:
    return f"<jlens layer={layer}>{' '.join(f'{_tok(t)} {p:.2f}' for t, p in top)}</jlens>"


def tools(base: list[Rollout], reads: Reads, placebo: bool, seed: int = 0) -> tuple[dict[str, Tool], set[int]]:
    """The two J-lens tools over the base runs, and the ids of the runs that have reads."""
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
    labels = ["yes" if r.label else "no" for r in base]

    def run_positions(i: int) -> tuple[list[Position], float, str]:
        """As in olens.tools; Position.layers maps layer -> top list here."""
        if i not in cache:
            messages, rows = reads[source[i]]
            per_layer = [r | {"layer": L, "top": t} for r in rows for L, t in zip(r["layers"], r["top"])]
            text = render_spans(messages)[0]
            cache[i] = (positions(messages, per_layer), len(text) / max(1, len(texts[i])), text)
        return cache[i]

    def pick_layers(args: dict, pos: list[Position]) -> list[int]:
        have = sorted({L for p in pos for L in p.layers})
        want = args.get("layers") or JLENS_LAYERS
        return [L for L in have if L in want] or have

    def block(p: Position, layers: list[int]) -> list[str]:
        return [_line(L, p.layers[L]["top"]) for L in layers if L in p.layers]

    def at(args: dict) -> str:
        i, char = int(args["run"]), int(args["char"])
        window = max(0, min(int(args.get("window", WINDOW)), MAX_WINDOW))
        if i not in source:
            return f"Run {i} has no J-lens reads; the runs with reads are marked in the table."
        pos, scale, text = run_positions(i)
        layers = pick_layers(args, pos)
        k = min(range(len(pos)), key=lambda j: abs(pos[j].char - char * scale))
        shown = pos[max(0, k - window) : k + window + 1]
        out, prev = [f"Run {i} (showed the behavior: {labels[i]}), {len(shown)} read positions around char {char}:"], None
        for p in shown:
            start = max(prev.char if prev else 0, p.char - GAP_CHARS)
            out.append(f"\n{p.label(scale)}\n<transcript>…{_flat(text[start : p.char])}⟨READ⟩</transcript>")
            out += block(p, layers)
            prev = p
        return "\n".join(out)

    def search(args: dict) -> str:
        try:
            pattern = re.compile(args["regex"], re.IGNORECASE)
        except re.error as e:
            return f"Bad regex: {e}"
        over = args.get("over") or "transcript"
        if over not in ("transcript", "tokens"):
            return f'Bad over: {over!r} (use "transcript" or "tokens")'
        max_hits = int(args.get("max_hits") or MAX_HITS)
        runs = [int(args["run"])] if args.get("run") is not None else read_ids
        searched = [i for i in runs if i in source]
        found: dict[tuple, list] = {}  # dedup key -> [runs, sort key, text]; prompt text repeats across runs
        matched = set()
        for i in searched:
            pos, scale, text = run_positions(i)
            layers = pick_layers(args, pos)
            if over == "tokens":
                for p in pos:
                    for L in layers:
                        for t, q in p.layers.get(L, {}).get("top", []):
                            if pattern.search(t.strip()):
                                matched.add(i)
                                context = _context(text, p.char)
                                head = f"· {p.label(scale)}\n<transcript>{context}</transcript>"
                                found.setdefault((context, p.token, L, t), [[], -q, f"{head}\n{_line(L, p.layers[L]['top'])}"])[0].append(i)
                continue
            for m in pattern.finditer(texts[i]):
                matched.add(i)
                start, end = m.start() * scale, m.end() * scale
                inside = [p for p in pos if start <= p.char <= end]
                after = next((p.char for p in pos if p.char > end), None)  # every read stacked there (a turn's template tokens)
                for p in inside + [p for p in pos if p.char == after]:
                    context = _context(text, p.char)
                    lines = "\n".join(block(p, layers))
                    head = f"· {p.label(scale)}\n<transcript>{context}</transcript>"
                    found.setdefault((context, p.token, lines), [[], len(found), f"{head}\n{lines}"])[0].append(i)
        kinds = []
        for label, name in ((True, "with"), (False, "without")):
            ids = [i for i in searched if base[i].label == label]
            if ids:
                kinds.append(f"{sum(i in matched for i in ids)} of {len(ids)} runs {name} the behavior")
        what = "the J-lens tokens" if over == "tokens" else "the transcripts"
        summary = f"The regex matches {what} of {' and '.join(kinds)} (runs with reads only). "
        ranked = sorted(found.values(), key=lambda h: h[1])
        unit = "token hits (by probability)" if over == "tokens" else "reads (by run, then position)"
        summary += f"{len(ranked)} {unit}" + (f"; the first {max_hits} are shown." if len(ranked) > max_hits else ".")
        shown = []
        for rs, _, line in ranked[:max_hits]:
            shown.append(f"\n\nrun {','.join(map(str, sorted(set(rs))))} [{'/'.join(sorted({labels[i] for i in rs}))}] {line}")
        return summary + "".join(shown)

    layer_param = {
        "type": "array",
        "items": {"type": "integer"},
        "description": f"layers to show (default {list(JLENS_LAYERS)}; the reads have every 4th layer from 0 to 60)",
    }
    return {
        "search_jlens": (
            schema(
                "search_jlens",
                "Case-insensitive regex over every base-arm transcript that has J-lens reads. Returns the J-lens reads at the "
                "punctuation inside each match and the first read after it, each with the transcript text around the read and "
                "its run and char (for jlens_at), and how many runs with and without the behavior match. "
                'With over="tokens", the regex runs over the J-lens\'s predicted tokens instead (each stripped of surrounding '
                "whitespace, at every read and shown layer) and returns the reads where one matches, highest probability first.",
                {
                    "regex": {"type": "string"},
                    "over": {
                        "type": "string",
                        "enum": ["transcript", "tokens"],
                        "description": "what the regex searches: the transcript text (default) or the J-lens's predicted tokens",
                    },
                    "run": {"type": "integer", "description": "optional: one run id"},
                    "layers": layer_param,
                    "max_hits": {"type": "integer", "description": f"reads or token hits returned (default {MAX_HITS})"},
                },
                ["regex"],
            ),
            search,
        ),
        "jlens_at": (
            schema(
                "jlens_at",
                "The J-lens reads at the read position nearest `char` in a run and `window` positions each side, with the "
                "transcript text between them. Take run and char from a search_jlens or search_transcripts hit.",
                {
                    "run": {"type": "integer"},
                    "char": {"type": "integer"},
                    "window": {"type": "integer", "description": f"positions each side (default {WINDOW}, max {MAX_WINDOW})"},
                    "layers": layer_param,
                },
                ["run", "char"],
            ),
            at,
        ),
    }, set(read_ids)
