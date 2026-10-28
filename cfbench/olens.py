"""O-lens tools for the `olens_agentic` and `olens_placebo_agentic` conditions, addressed by quoting the transcript, not by token number.

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
Both show each read's top bullet with FVE >= 0.1 by default (`top_k`, `min_fve`); search_olens returns the 20 best
bullets by FVE (`max_bullets`), olens_at reads layers 32, 40, 48 by default.

`olens_placebo_agentic` answers every call from a run of a different env, at the same relative position in that run (with that
run's text around the reads), so the predictor gets as much O-lens text through the same tools, none of it about this
scenario.

The `nla_agentic` and `nla_placebo_agentic` conditions use the same tools over NLA reads (`--readout nla`, docs/nla-reader.md),
named `search_nla` and `nla_at` (LENSES): each read is one free-text readout (`text`, scored by the NLA critic's `fve`), served as
the read's only bullet, with no FVE filter by default.
"""

import json
import random
import re
from array import array
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from cfbench.core import Rollout, render_spans
from cfbench.tools import Tool, schema

BEFORE, AFTER = 80, 40  # characters of transcript around a search hit
GAP_CHARS = 300  # most transcript text shown between two positions in olens_at
WINDOW, MAX_WINDOW = 10, 20  # neighbouring positions each side in olens_at
AT_LAYERS = {32, 40, 48}  # olens_at's default layers when the reads have them (else all); search_olens defaults to every layer
MAX_BULLETS = 20  # search_olens's default number of bullets returned, highest FVE first
TOP_K, MIN_FVE = 1, 0.1  # default bullet filters (ANDed) for both tools; the agent can change them per call
# NLA defaults (2026-10-27): a readout is ~625 characters, ~6x an O-lens top bullet, so fewer per call; no FVE filter, since
# the critic's FVE on one env's near-identical prompts is low everywhere (docs/nla-reader.md)
NLA_WINDOW, NLA_MAX = 3, 8
PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # the reader's stop tokens (scripts/olens_read_requests.py)
POSITIONS = ("sparse", "all")


@dataclass(frozen=True)
class Lens:
    name: str  # tools search_<name> and <name>_at, output tag <name>
    title: str
    unit: str  # what a read's outputs are called: bullets or readouts
    min_fve: float | None  # default filter; None shows every output
    window: int
    max_hits: int


LENSES = {
    "olens": Lens("olens", "O-lens", "bullets", MIN_FVE, WINDOW, MAX_BULLETS),
    "nla": Lens("nla", "NLA", "readouts", None, NLA_WINDOW, NLA_MAX),
}


def _as_bullets(row: dict) -> dict:
    """An NLA row has one free-text readout in `text` and its FVE in `fve`, not bullets: serve it as the read's one bullet."""
    if "bullets" not in row and "text" in row:
        row["bullets"] = [{"text": row["text"], "fve": row.get("fve")}]
    return row


def sparse(row: dict) -> bool:
    """Whether a row of an every-token read is one a sparse read job reads: a template token (field null: <|im_start|>, role,
    newline, <think>, </think>, <|im_end|>, what `turn_starts` requests read) or a stop token."""
    return row["field"] is None or bool(PUNCT.match(row["token"]))


class Reads(Mapping[str, tuple[list[dict], list[dict]]]):
    """Rollout source -> (its messages, its read rows), each source's rows parsed from disk on its first lookup. Parsing
    every row up front took 12 GB per process with the 2026-10-24 nocot reads, so a few cfbench runs at once got killed by
    oom-guard; a run only looks up its own env's sources and its placebo's. Each reads.jsonl's run dir also has the
    requests.jsonl it was read from, which has every transcript's messages.

    `positions="sparse"` (the default) keeps every row of a request that isn't `positions: "all"` as the job read it, and of
    an every-token request only the rows `sparse` accepts (template and stop tokens), so the default serves every sparse read
    ever made and a dense file serves no more; a transcript whose rows all get dropped is an error.
    `positions="all"` keeps every row and records in `sparse_only` the transcripts whose request row doesn't say
    `positions: "all"` (cfbench.run refuses to use those), so nobody runs "every token" on reads that only exist at a few."""

    def __init__(self, paths: list[str], positions: str = "sparse"):
        assert positions in POSITIONS, positions
        self.messages: dict[str, list[dict]] = {}
        self.sparse_only: dict[str, Path] = {}  # source -> a reads file whose request row for it isn't positions: "all"
        dropped: dict[str, Path] = {}  # source -> file, when positions="sparse" dropped every row of it
        self.offsets: dict[str, dict[Path, array]] = defaultdict(dict)  # source -> reads file -> where its rows start
        self.top_fves = array("d")  # every read's top-bullet FVE, for fve_note
        self.rows: dict[str, list[dict]] = {}
        for path in map(Path, paths):
            requests = {r["id"]: r for r in map(json.loads, (path.parent / "requests.jsonl").read_text().splitlines())}
            # the reader's --positions all says so in its provenance, not in requests.jsonl
            prov = path.with_name(f"{path.stem}.provenance.json")
            dense_file = prov.exists() and json.loads(prov.read_text()).get("positions") == "all"
            messages = {id: r["messages"] for id, r in requests.items()}
            offset = 0
            with path.open("rb") as f:
                for line in f:
                    row = _as_bullets(json.loads(line))
                    # msg_index < 0: the tool definitions before the system prompt, same in every run, not in the transcript
                    dense = dense_file or requests[row["id"]].get("positions") == "all"
                    if row["msg_index"] >= 0 and positions == "sparse" and dense and not sparse(row):
                        dropped.setdefault(row["id"], path)
                    elif row["msg_index"] >= 0:
                        if positions == "all" and not dense:
                            self.sparse_only.setdefault(row["id"], path)
                        self.messages.setdefault(row["id"], messages[row["id"]])
                        self.offsets[row["id"]].setdefault(path, array("q")).append(offset)
                        if (b := row.get("bullets")) and b[0]["fve"] is not None:  # J++ rows (cfbench/jlens.py) have none
                            self.top_fves.append(b[0]["fve"])
                    offset += len(line)
        if missing := {s: p for s, p in dropped.items() if s not in self.messages}:
            listed = "\n".join(f"{s}  ({p})" for s, p in missing.items())
            raise ValueError(f"{len(missing)} transcripts have reads but none at a template or stop token:\n{listed}")

    def __getitem__(self, source: str) -> tuple[list[dict], list[dict]]:
        messages = self.messages[source]
        if source not in self.rows:
            self.rows[source] = []
            for path, offsets in self.offsets[source].items():
                with path.open("rb") as f:
                    for offset in offsets:
                        f.seek(offset)
                        self.rows[source].append(_as_bullets(json.loads(f.readline())))
        return messages, self.rows[source]

    def __contains__(self, source: object) -> bool:  # Mapping's default would parse the source's rows
        return source in self.messages

    def __iter__(self) -> Iterator[str]:
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)


@dataclass
class Position:
    char: int  # in the run's rendered transcript
    where: str  # e.g. "assistant reasoning_content", "assistant (start of turn)"
    token: str  # the token read
    layers: dict[int, dict]  # layer -> read row

    def label(self, scale: float = 1) -> str:
        """`char 565 · assistant (start of turn) · token "\\n\\n"`: several template tokens share one char, so the token says
        which read this is."""
        return f"char {round(self.char / scale)} · {self.where} · token {json.dumps(self.token, ensure_ascii=False)}"


def positions(messages: list[dict], rows: list[dict]) -> list[Position]:
    """One Position per read token, in transcript order. The readers give a template token the msg_index of the message
    before it, so the tokens after message i's <|im_end|> (the newline, <|im_start|>, role, <think>, </think>, ...) are
    placed at the start of message i + 1's text as its "(start of turn)"; the <|im_end|> and the other template tokens and
    tool calls at message i's end."""
    text, spans = render_spans(messages)
    starts = defaultdict(list)  # message -> its fields' starts
    for (i, _), b in spans.items():
        starts[i].append(b)
    # message i ends at the "\n\n" before message i + 1's header (headers hold no "\n\n")
    ends = {i: (text.rfind("\n\n", 0, min(starts[i + 1])) if i + 1 in starts else len(text)) for i in starts}
    by_token = defaultdict(dict)
    for r in rows:
        by_token[r["token_index"]][r["layer"]] = r
    im_end = {}  # message -> token index of its <|im_end|>
    out = []
    for t in sorted(by_token):
        layers = by_token[t]
        r = next(iter(layers.values()))
        i, field, c = r["msg_index"], r["field"] or "", r["char_offset"]
        if r["token"] == "<|im_end|>" and not field:
            im_end.setdefault(i, t)
        if (i, field) in spans and c is not None and not field.startswith("tool_call"):
            char, where = spans[i, field] + c, f"{r['role']} {field}"
        elif not field and i in im_end and t > im_end[i]:  # the header of the next turn
            char = min(starts[i + 1]) if i + 1 in starts else len(text)
            where = f"{messages[i + 1]['role']} (start of turn)" if i + 1 < len(messages) else "next turn (start)"
        else:  # tool calls, <|im_end|>, and template tokens inside a turn
            char, where = ends.get(i, len(text)), f"{r['role']} {field or '(end of turn)'}"
        out.append(Position(char, where, r["token"], layers))
    return sorted(out, key=lambda p: p.char)


def _attrs(layer: int, bullet: dict) -> str:
    return f"layer={layer}" + (f" fve={bullet['fve']:.2f}" if bullet["fve"] is not None else "")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# where core.render_spans starts a message field ("ASSISTANT REASONING (private):", "TOOL:", "ASSISTANT TOOL CALL bash: ...")
FIELD = re.compile(r"(?:^|\n\n)(?:SYSTEM|USER|TOOL|ASSISTANT)\b[^\n]*:\n?")


def _context(text: str, char: int) -> str:
    """The transcript around a read, ⟨READ⟩ at the read, cut at the edges of the message field the read is in."""
    before, after = text[max(0, char - BEFORE) : char], text[char : char + AFTER]
    cut_before, cut_after = FIELD.split(before), FIELD.split(after)
    before = cut_before[-1] if len(cut_before) > 1 else ("…" if char > BEFORE else "") + before
    after = cut_after[0] if len(cut_after) > 1 else after + ("…" if char + AFTER < len(text) else "")
    return f"{_flat(before)}⟨READ⟩{_flat(after)}".strip()


def tools(
    base: list[Rollout],
    reads: Reads,
    placebo: bool,
    seed: int = 0,
    sources: list[str] | None = None,
    kind: str = "olens",
    overrides: dict | None = None,
) -> tuple[dict[str, Tool], set[int]]:
    """The two lens tools (`kind` in LENSES) over the base runs, and the ids of the runs that have reads. `overrides`
    replaces fields of the lens's defaults (an eval's OLENS, e.g. {"min_fve": 0.0})."""
    lens = replace(LENSES[kind], **(overrides or {}))
    unit = lens.unit[:-1]
    texts = [render_spans(r.messages)[0] for r in base]
    read_ids = [i for i, r in enumerate(base) if r.source in reads]
    source = {i: base[i].source for i in read_ids}  # whose reads answer for run i
    if sources is not None:  # an eval's own placebo (cfbench/eval.py PLACEBO): run i's reads come from sources[i]
        source = dict(enumerate(sources))
        read_ids = sorted(source)
    if placebo:
        env = base[0].source.split("/")[0]
        pool = sorted(s for s in reads if s.split("/")[0] != env)
        assert pool, f"placebo needs reads from an env other than {env}"
        # seeded per case (its base runs), so different cases get different placebo transcripts (2026-10-28)
        random.Random(f"{seed}/" + "|".join(r.source for r in base)).shuffle(pool)
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

    def filters(args: dict, layers: set[int] | None) -> tuple[set[int] | None, int, float | None]:
        layers = args.get("layers") or layers
        top_k = TOP_K if args.get("top_k") is None else int(args["top_k"])
        return set(layers) if layers else None, top_k, lens.min_fve if args.get("min_fve") is None else float(args["min_fve"])

    def bullets(p: Position, layers: set[int] | None, top_k: int | None, min_fve: float | None) -> list[tuple[int, dict]]:
        """(layer, bullet) for each bullet passing both filters, by layer (bullets come sorted by FVE)."""
        return [
            (layer, b)
            for layer in sorted(p.layers)
            if layers is None or layer in layers
            for b in p.layers[layer]["bullets"][:top_k]
            if min_fve is None or b["fve"] is None or b["fve"] >= min_fve  # reads from a lens without a reconstructor have no FVE
        ]

    def at(args: dict) -> str:
        i, char = int(args["run"]), int(args["char"])
        window = max(0, min(int(args.get("window", lens.window)), MAX_WINDOW))
        if i not in source:
            return f"Run {i} has no {lens.title} reads; the runs with reads are marked in the table."
        pos, scale, text = run_positions(i)
        layers, top_k, min_fve = filters(args, AT_LAYERS if AT_LAYERS & {layer for p in pos for layer in p.layers} else None)
        k = min(range(len(pos)), key=lambda j: abs(pos[j].char - char * scale))
        shown = pos[max(0, k - window) : k + window + 1]
        out, prev = [f"Run {i} (showed the behavior: {yes_no[i]}), {len(shown)} read positions around char {char}:"], None
        for p in shown:
            start = max(prev.char if prev else 0, p.char - GAP_CHARS)
            out.append(f"\n{p.label(scale)}\n<transcript>…{_flat(text[start : p.char])}⟨READ⟩</transcript>")
            got = bullets(p, layers, top_k, min_fve)
            tag = lens.name
            out += [f"<{tag} {_attrs(layer, b)}>{b['text']}</{tag}>" for layer, b in got] or [f"(no {unit} with FVE >= {min_fve})"]
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
        if over not in ("transcript", lens.unit):
            return f'Bad over: {over!r} (use "transcript" or "{lens.unit}")'
        layers, top_k, min_fve = filters(args, None)
        max_bullets = int(args.get(f"max_{lens.unit}") or lens.max_hits)
        runs = [int(args["run"])] if args.get("run") is not None else read_ids
        searched = [i for i in runs if i in source]
        found: dict[tuple, list] = {}  # (context, layer, bullet) -> [runs, fve, line]; prompt text repeats across runs
        matched = set()
        for i in searched:
            pos, scale, text = run_positions(i)
            if over == lens.unit:
                hits = [(p, [(layer, b) for layer, b in bullets(p, layers, None, min_fve) if pattern.search(b["text"])]) for p in pos]
            else:
                hits = []
                for m in pattern.finditer(texts[i]):
                    start, end = m.start() * scale, m.end() * scale
                    inside = [p for p in pos if start <= p.char <= end]
                    after = next((p.char for p in pos if p.char > end), None)  # every read stacked there (a turn's template tokens)
                    hits += [(p, bullets(p, layers, top_k, min_fve)) for p in inside + [p for p in pos if p.char == after]]
                    matched.add(i)
            for p, got in hits:
                if got:
                    matched.add(i)
                context = _context(text, p.char)
                for layer, b in got:
                    head = f"· {p.label(scale)}"
                    line = f"{head}\n<transcript>{context}</transcript>\n<{lens.name} {_attrs(layer, b)}>{b['text']}</{lens.name}>"
                    found.setdefault((context, p.token, layer, b["text"]), [[], b["fve"], line])[0].append(i)
        kinds = []
        for label, name in ((True, "with"), (False, "without")):
            ids = [i for i in searched if base[i].label == label]
            if ids:
                kinds.append(f"{sum(i in matched for i in ids)} of {len(ids)} runs {name} the behavior")
        summary = (
            f"The regex matches {lens.unit if over == lens.unit else 'the transcripts'} of {' and '.join(kinds)} (runs with reads only). "
        )
        ranked = sorted(found.values(), key=lambda h: -(h[1] or 0))
        order = "the top {} by FVE" if any(h[1] is not None for h in ranked) else "the first {} (by run, then position)"
        if len(ranked) > max_bullets:
            hidden = len(ranked) - max_bullets
            summary += f"{len(ranked)} {lens.unit} pass the filters; {order.format(max_bullets)} are shown, {hidden} are not."
        else:
            summary += f"{len(ranked)} {lens.unit} pass the filters."
        shown = []
        for rs, _, line in ranked[:max_bullets]:
            runs_label = f"run {','.join(map(str, sorted(set(rs))))} [{'/'.join(sorted({yes_no[i] for i in rs}))}]"
            shown.append(f"\n\n{runs_label} {line}")
        return summary + "".join(shown)

    title, units, name = lens.title, lens.unit, lens.name
    default_fve = f"default {lens.min_fve}" if lens.min_fve is not None else "default: no filter"
    filter_params = {"min_fve": {"type": "number", "description": f"only {units} with at least this FVE ({default_fve})"}}
    if kind == "olens":  # an NLA read has one readout, so nothing to rank within it
        filter_params = {
            "top_k": {
                "type": "integer",
                "description": f"bullets per read, best first: highest FVE, or the greedy readout then samples (default {TOP_K})",
            },
        } | filter_params
    every = "all bullets at every read, top_k ignored" if kind == "olens" else "every read"
    return {
        f"search_{name}": (
            schema(
                f"search_{name}",
                f"Case-insensitive regex over every base-arm transcript that has {title} reads. Returns the {title} {units} at the "
                "punctuation inside each match and the first read after it (highest FVE first, when reads have FVE), each with "
                f"the transcript text around the read and its run and char (for {name}_at), and how many runs with and without the "
                "behavior match. "
                f'With over="{units}", the regex runs over the text of every {title} {unit} instead ({every}) '
                f"and returns only the {units} that match, with the same context, and how many runs have a match.",
                {
                    "regex": {"type": "string"},
                    "over": {
                        "type": "string",
                        "enum": ["transcript", units],
                        "description": f"what the regex searches: the transcript text (default) or the {title} {units}' text",
                    },
                    "run": {"type": "integer", "description": "optional: one run id"},
                    "layers": {"type": "array", "items": {"type": "integer"}, "description": "optional: only these layers (default all)"},
                    **filter_params,
                    f"max_{units}": {"type": "integer", "description": f"{units} returned (default {lens.max_hits})"},
                },
                ["regex"],
            ),
            search,
        ),
        f"{name}_at": (
            schema(
                f"{name}_at",
                f"The {title} reads at the read position nearest `char` in a run and `window` positions each side, with the "
                f"transcript text between them. Take run and char from a search_{name} or search_transcripts hit.",
                {
                    "run": {"type": "integer"},
                    "char": {"type": "integer"},
                    "window": {"type": "integer", "description": f"positions each side (default {lens.window}, max {MAX_WINDOW})"},
                    "layers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": f"default {sorted(AT_LAYERS)} if the reads have them, else all",
                    },
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


def fve_note(reads: Reads, what: str = "the top bullet's") -> str | None:
    """The 10th/50th/90th percentile of the top bullet's FVE over every loaded read, for the prompt; None if the reads have
    no FVE (core.build_prompt then describes them as greedy + sampled readouts). `what` names the output ("the readout's")."""
    top = sorted(reads.top_fves)
    if not top:
        return None
    q = [top[int(f * (len(top) - 1))] for f in (0.1, 0.5, 0.9)]
    return f"Over all reads here, {what} FVE is {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} at the 10th / 50th / 90th percentile."
