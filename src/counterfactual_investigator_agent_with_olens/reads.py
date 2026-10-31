"""The batch readers' shared request handling (scripts/olens_read_requests.py, jpp_read_requests.py, olens_capture_check.py):
render a request row and pick its read positions (`prepare`), order and shard the queue, resume after a crash, and write the
transcript sidecar. Torch-free; the pod imports it with PYTHONPATH=.:olens/src:src (runpod/olens_reads.sh). Moved here from
scripts/olens_read_requests.py on 2026-10-31; untested on a pod since the move.
"""

import hashlib
import json
import os
import re
import zlib
from pathlib import Path

from counterfactual_investigator_agent_with_olens.render import render_request, transcript_line
from environments._lib.mcq import option_newlines as mcq_option_newlines

PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # transcript-viewer/web/build.py


def option_newlines(text: str) -> list[int]:
    """Character-offset adapter for the shared complete-option newline contract."""
    return [char for char, _ in mcq_option_newlines(text)]


def prepare(tok, row, layers, im_end, im_start, opt_nl=False):
    """Render one request row: messages, token ids, per-token provenance, token strings, read positions, layers, prefix keys.
    `opt_nl` (or the row's `option_newlines`) adds the newline after each MCQ option in every message's content."""
    msgs, ids, where = render_request(tok, row)
    if row.get("prompt_ids"):  # token ids given as-is (e.g. edited after rendering); no per-token message provenance
        ids, where = row["prompt_ids"], [(-1, None, None)] * len(row["prompt_ids"])
    toks = [tok.decode([i]) for i in ids]
    positions = [i for i, x in enumerate(ids) if x == im_end or PUNCT.match(toks[i])]
    if row.get("turn_starts") is not None:  # only each <|im_start|> and the N tokens after it (the role, "\n", ...)
        starts = [i for i, x in enumerate(ids) if x == im_start]
        starts = starts[-1:] if row.get("last_turn_only") else starts
        if row.get("turn_role"):  # e.g. "assistant": only that role's headers
            starts = [i for i in starts if i + 1 < len(ids) and toks[i + 1] == row["turn_role"]]
        positions = sorted({j for i in starts for j in range(i, min(len(ids), i + 1 + row["turn_starts"]))})
    elif row.get("positions") == "all":
        positions = list(range(len(ids)))
    elif row.get("positions") == "last":  # the final token only (WorkspaceBench's final_token read site)
        positions = [len(ids) - 1]
    elif isinstance(row.get("positions"), list):  # explicit token indices into the rendered request
        positions = row["positions"]
    elif row.get("char_positions"):  # [[msg_index, char offset into its content], ...]: the token holding each char
        positions = [max(i for i, w in enumerate(where) if w[:2] == (m, "content") and w[2] <= c) for m, c in row["char_positions"]]
    if opt_nl or row.get("option_newlines"):
        for mi, m in enumerate(msgs):
            content = m.get("content") or ""
            lead = len(content) - len(content.lstrip())  # options are found in the stripped text, char_offset counts from the raw
            for c in option_newlines(content.strip()):
                positions.append(max(i for i, w in enumerate(where) if w[:2] == (mi, "content") and w[2] is not None and w[2] <= lead + c))
        positions = sorted(set(positions))
    row_layers = row.get("layers") or layers  # a request row may name its own layers (a subset of --layers)
    return msgs, ids, where, toks, positions, row_layers, prefix_keys(ids, positions)


def resume(out: Path) -> set[str]:
    """Transcript ids already in OUT, for a rerun to skip. A transcript's lines are appended in one block after the one before,
    so only the last transcript on disk can be partial (a crash mid-write, maybe a torn last line): truncate OUT to before it,
    and it is read again. Costs one transcript per restart."""
    if not out.exists():
        return set()
    ids, last, last_start, pos = set(), None, 0, 0
    with out.open("rb") as f:
        for line in f:
            try:
                i = json.loads(line)["id"]
            except json.JSONDecodeError:
                break
            if i != last:
                last, last_start = i, pos
            ids.add(i)
            pos += len(line)
    os.truncate(out, last_start)
    return ids - {last}


def prefix_keys(ids, positions):
    """position -> a hash of the token ids up to and including it (equal keys = equal prefixes, hence equal reads)."""
    h, keys, want = hashlib.blake2b(digest_size=16), {}, set(positions)
    for i, x in enumerate(ids):
        h.update(int(x).to_bytes(4, "little"))
        if i in want:
            keys[i] = h.copy().digest()
    return keys


def queue(path, shard, done):
    """REQUESTS minus `done`, this shard only: priority order, round-robin over (env, arm) within a priority so early
    results cover every case. Re-read on every call, so appending to REQUESTS adds work to a running reader."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    i, n = map(int, shard.split("/"))
    rank, seen = {}, {}
    for r in rows:
        g = (r.get("priority", 0), r.get("env"), r.get("arm"))
        rank[r["id"]] = (g[0], seen.get(g, 0))
        seen[g] = seen.get(g, 0) + 1
    rows = [r for r in rows if r["id"] not in done and zlib.crc32(r.get("shard_key", r["id"]).encode()) % n == i]
    return sorted(rows, key=lambda r: rank[r["id"]])


def write_transcript(out, job):
    """The transcript's line in OUT.transcripts.jsonl (tokens as text, chat sections, request metadata): with OUT it is all
    a viewer needs (docs/olens-reads-format.md). Written before the reads, so a reread after a crash may repeat it."""
    with out.with_suffix(".transcripts.jsonl").open("a") as f:
        f.write(json.dumps(transcript_line(job["row"], job["msgs"], job["toks"], job["where"]), ensure_ascii=False) + "\n")
