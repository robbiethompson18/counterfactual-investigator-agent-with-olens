"""Thought-Anchors tools for the `anchors` and `anchors_placebo` conditions, over reads from scripts/anchor_reads.py.

Two scores per sentence (segment) of a base-arm run, both from forward passes of the target model over that run only:
- receiver: mean attention the sentence gets from model-written sentences >= 4 later, averaged over the TOP_HEADS
  receiver heads, which are the full-attention heads whose per-sentence scores have the highest kurtosis (concentrate
  on a few sentences), averaged over every loaded transcript (Bogdan et al. 2025, arXiv 2506.19143);
- suppression: with every path from the sentence to later positions cut, the mean KL (nats per token) of the model's
  predictions of its own later tokens, overall and per later sentence.

- `anchor_ranking(run)`: the run's top sentences by each score.
- `anchor_search(regex)`: matching sentences across runs, grouped by text, with their mean percentiles in their runs.
- `anchor_influence(run, char)`: the sentence at `char`, the later sentences it moves most, the earlier ones that move it.

`anchors_placebo` answers every call with the scores of a run of a different env, mapped by relative sentence position
(order kept), over this run's sentences: same tools, same score distributions, none of it about this scenario.
"""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cfbench.core import Rollout, render_spans
from cfbench.tools import Tool, _schema

TOP_HEADS = 16  # the paper's main analysis
MIN_SEGS = 8  # transcripts with fewer scored sentences don't vote on the receiver heads
TOP, MAX_TOP = 10, 30  # sentences per list in anchor_ranking / anchor_influence
MAX_GROUPS = 20  # anchor_search results
SNIP = 160  # characters of sentence text shown

Reads = dict[str, tuple[list[dict], dict]]  # rollout source -> (its messages, its read row)


def load(paths: list[str]) -> Reads:
    """Each reads.jsonl's run dir also has the requests.jsonl it was read from (the placebo needs other envs' messages)."""
    out: Reads = {}
    for path in map(Path, paths):
        messages = {r["id"]: r["messages"] for r in map(json.loads, (path.parent / "requests.jsonl").read_text().splitlines())}
        for row in map(json.loads, path.read_text().splitlines()):
            out[row["id"]] = (messages[row["id"]], row)
    return out


def receiver_heads(reads: Reads) -> list[int]:
    """Rows of `vertical` with the highest kurtosis (Pearson: m4 / m2^2, 3 for a Gaussian) over a transcript's scored
    sentences (template tokens dropped), averaged over transcripts weighted by their number of scored sentences, so
    long reasoning transcripts pick the heads rather than many four-sentence replies."""
    total, weight = np.zeros(0), 0
    for _, row in reads.values():
        V = np.array([[np.nan if x is None else x for x in h] for h in row["vertical"]], dtype=float)
        V = V[:, ~np.isnan(V).any(0) & np.array([s["field"] is not None for s in row["segments"]])]
        n = V.shape[1]
        if n < MIN_SEGS:
            continue
        d = V - V.mean(1, keepdims=True)
        k = (d**4).mean(1) / np.maximum((d**2).mean(1) ** 2, 1e-30)
        total, weight = (total + n * k if weight else n * k), weight + n
    assert weight, f"no transcript has {MIN_SEGS}+ scored sentences"
    return [int(h) for h in np.argsort(-total / weight)[:TOP_HEADS]]


@dataclass
class Sent:
    char: int  # start in the run's rendered transcript (cfbench.core.render)
    where: str  # e.g. "assistant reasoning_content"
    text: str
    generated: bool
    size: int  # tokens


class Scores:
    """One run's sentences (template tokens dropped) and their scores."""

    def __init__(self, messages: list[dict], row: dict, heads: list[int]):
        text, spans = render_spans(messages)
        starts = {}  # message -> its first field's start, to place tool-call segments at the message's end
        for (j, _), b in spans.items():
            starts[j] = min(b, starts.get(j, b))
        keep = [k for k, s in enumerate(row["segments"]) if s["field"] is not None]
        self.sents = []
        for k in keep:
            s = row["segments"][k]
            i, field, c = s["msg_index"], s["field"], s["char_start"]
            key = (i, field) if (i, field) in spans and c is not None and field != "tool_call" else None
            # tool-call arguments aren't rendered verbatim: place them at their message's end (as cfbench/olens.py does)
            char = spans[key] + c if key else (starts[i + 1] - 2 if i + 1 in starts else len(text))
            self.sents.append(Sent(char, f"{s['role']} {field}", s["text"], s["generated"], s["tok_end"] - s["tok_start"]))
        V = np.array([[np.nan if x is None else x for x in row["vertical"][h]] for h in heads], dtype=float)
        self.recv = [None if np.isnan(x) else float(x) for x in V.mean(0)[keep]]
        kl = row["kl"]
        self.kl = [[kl[a][b] for b in keep] for a in keep]  # kl[s][u], None unless u is later and model-written
        self.supp = []  # token-weighted mean KL over all later model-written sentences
        for a in range(len(keep)):
            pairs = [(x, self.sents[b].size) for b, x in enumerate(self.kl[a]) if x is not None]
            self.supp.append(sum(x * w for x, w in pairs) / sum(w for _, w in pairs) if pairs else None)

    def remap(self, other: "Scores") -> "Scores":
        """This run's sentences with `other`'s scores, by relative position (monotone, so later stays later)."""
        n, m = len(self.sents), len(other.sents)
        f = [round(k * (m - 1) / max(1, n - 1)) for k in range(n)]
        out = object.__new__(Scores)
        out.sents = self.sents
        out.recv = [other.recv[f[k]] for k in range(n)]
        out.supp = [other.supp[f[k]] for k in range(n)]
        out.kl = [[other.kl[f[a]][f[b]] if f[b] > f[a] else None for b in range(n)] for a in range(n)]
        return out


def _pct(values: list) -> list:
    """Percentile of each value among the run's non-None values (100 = top), None stays None."""
    have = sorted(x for x in values if x is not None)
    return [None if x is None else round(100 * sum(y <= x for y in have) / len(have)) for x in values]


def _flat(text: str) -> str:
    text = text.strip().replace("\n", "⏎")
    return text if len(text) <= SNIP else text[:SNIP] + "…"


def tools(base: list[Rollout], reads: Reads, placebo: bool, seed: int = 0) -> tuple[dict[str, Tool], set[int]]:
    """The three anchor tools over the base runs, and the ids of the runs that have reads."""
    heads = receiver_heads(reads)
    read_ids = [i for i, r in enumerate(base) if r.source in reads]
    source = {i: base[i].source for i in read_ids}
    if placebo:
        env = base[0].source.split("/")[0]
        pool = sorted(s for s in reads if s.split("/")[0] != env)
        assert pool, f"placebo needs reads from an env other than {env}"
        random.Random(seed).shuffle(pool)
        source = dict(zip(read_ids, pool * (len(read_ids) // len(pool) + 1)))
    yes_no = ["yes" if r.label else "no" for r in base]
    cache: dict[int, Scores] = {}

    def scores(i: int) -> Scores:
        if i not in cache:
            own = Scores(*reads[base[i].source], heads)
            cache[i] = own if source[i] == base[i].source else own.remap(Scores(*reads[source[i]], heads))
        return cache[i]

    def line(sc: Scores, k: int, rp: list, sp: list) -> str:
        s = sc.sents[k]
        recv = f"receiver {sc.recv[k]:.4f} (pct {rp[k]})" if sc.recv[k] is not None else "receiver n/a"
        supp = f"suppression {sc.supp[k]:.3f} nats/token (pct {sp[k]})" if sc.supp[k] is not None else "suppression n/a"
        return f'char {s.char} ({s.where}) {recv}, {supp}: "{_flat(s.text)}"'

    def ranking(args: dict) -> str:
        i, top = int(args["run"]), max(1, min(int(args.get("top") or TOP), MAX_TOP))
        if i not in source:
            return f"Run {i} has no anchor reads; the runs with reads are marked in the table."
        sc = scores(i)
        rp, sp = _pct(sc.recv), _pct(sc.supp)
        out = [f"Run {i} (showed the behavior: {yes_no[i]}), {len(sc.sents)} sentences."]
        for name, vals in (("receiver score (attention later reasoning converges on)", sc.recv), ("suppression effect", sc.supp)):
            order = [k for _, k in sorted(((-x, k) for k, x in enumerate(vals) if x is not None))[:top]]
            out.append(f"\nTop {len(order)} by {name}:")
            out += [f"  {line(sc, k, rp, sp)}" for k in order]
        return "\n".join(out)

    def search(args: dict) -> str:
        try:
            pattern = re.compile(args["regex"], re.IGNORECASE)
        except re.error as e:
            return f"Bad regex: {e}"
        groups: dict[str, list] = {}  # sentence text -> [(run, recv pct, supp pct, supp)]
        for i in read_ids:
            sc = scores(i)
            rp, sp = _pct(sc.recv), _pct(sc.supp)
            for k, s in enumerate(sc.sents):
                if pattern.search(s.text):
                    groups.setdefault(s.text.strip(), []).append((i, rp[k], sp[k], sc.supp[k], s.char))
        if not groups:
            return "No sentence in the runs with anchor reads matches."
        mean = lambda xs: f"{np.mean(xs):.0f}" if xs else "n/a"
        rows = []
        for text, hits in sorted(groups.items(), key=lambda g: -len(g[1])):
            runs = sorted({h[0] for h in hits})
            parts = []
            for lab in ("yes", "no"):
                hs = [h for h in hits if yes_no[h[0]] == lab]
                if hs:
                    nats = [h[3] for h in hs if h[3] is not None]
                    parts.append(
                        f"{len(hs)} in runs {'with' if lab == 'yes' else 'without'} the behavior: receiver pct "
                        f"{mean([h[1] for h in hs if h[1] is not None])}, suppression pct {mean([h[2] for h in hs if h[2] is not None])}"
                        + (f" ({np.mean(nats):.3f} nats/token)" if nats else "")
                    )
            first = hits[0]
            rows.append(f'"{_flat(text)}" — {len(runs)} runs (e.g. run {first[0]} char {first[4]}); ' + "; ".join(parts))
        n_runs = len({h[0] for g in groups.values() for h in g})
        head = f"{len(groups)} distinct matching sentences in {n_runs} of {len(read_ids)} runs with reads"
        more = f", the {MAX_GROUPS} in the most runs shown" if len(rows) > MAX_GROUPS else ""
        return f"{head}{more}. Percentiles are within each run (100 = that run's top sentence).\n" + "\n".join(rows[:MAX_GROUPS])

    def influence(args: dict) -> str:
        i, char, top = int(args["run"]), int(args["char"]), max(1, min(int(args.get("top") or TOP), MAX_TOP))
        if i not in source:
            return f"Run {i} has no anchor reads; the runs with reads are marked in the table."
        sc = scores(i)
        rp, sp = _pct(sc.recv), _pct(sc.supp)
        k = max([j for j, s in enumerate(sc.sents) if s.char <= char] or [0])
        out = [f"Run {i} (showed the behavior: {yes_no[i]}), sentence {k} of {len(sc.sents)}:", f"  {line(sc, k, rp, sp)}"]
        moves = sorted(((x, u) for u, x in enumerate(sc.kl[k]) if x is not None), reverse=True)[:top]
        out.append("\nLater sentences whose tokens move most when it is suppressed (mean KL, nats/token):")
        out += [f'  {x:.3f}  char {sc.sents[u].char} ({sc.sents[u].where}): "{_flat(sc.sents[u].text)}"' for x, u in moves] or ["  (none)"]
        moved_by = sorted(((sc.kl[s][k], s) for s in range(len(sc.sents)) if sc.kl[s][k] is not None), reverse=True)[:top]
        out.append("\nEarlier sentences whose suppression moves this one's tokens most:")
        out += [f'  {x:.3f}  char {sc.sents[s].char} ({sc.sents[s].where}): "{_flat(sc.sents[s].text)}"' for x, s in moved_by] or [
            "  (none: it isn't model-written, or nothing precedes it)"
        ]
        return "\n".join(out)

    top_param = {"top": {"type": "integer", "description": f"sentences per list (default {TOP}, max {MAX_TOP})"}}
    return {
        "anchor_ranking": (
            _schema(
                "anchor_ranking",
                "One base-arm run's sentences with the highest receiver score and the highest suppression effect, each with "
                "its char (for anchor_influence or read_transcript) and its percentile in the run.",
                {"run": {"type": "integer"}, **top_param},
                ["run"],
            ),
            ranking,
        ),
        "anchor_search": (
            _schema(
                "anchor_search",
                "Case-insensitive regex over every sentence of every base-arm run with anchor reads. Groups matching sentences "
                "by text and gives, for runs with and without the behavior, their mean receiver and suppression percentiles "
                "within their runs.",
                {"regex": {"type": "string"}},
                ["regex"],
            ),
            search,
        ),
        "anchor_influence": (
            _schema(
                "anchor_influence",
                "The sentence at `char` in a run (from any search hit): its scores, the later sentences whose tokens move most "
                "when it is suppressed, and the earlier sentences whose suppression moves its tokens most.",
                {"run": {"type": "integer"}, "char": {"type": "integer"}, **top_param},
                ["run", "char"],
            ),
            influence,
        ),
    }, set(read_ids)


def sources(paths: list[str]) -> list[dict]:
    """What each reads file was at launch, for the run's meta (read files keep growing)."""
    return [{"path": p, "rows": sum(1 for _ in open(p))} for p in paths]
