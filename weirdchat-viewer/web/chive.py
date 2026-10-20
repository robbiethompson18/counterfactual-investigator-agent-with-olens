"""CHIVE incidents (HF agu18dec/chive-forensics) as viewer patterns: one pattern per incident, holding its one transcript,
rendered from CHIVE's own token ids (the readout banks index into them), plus the per-token readout banks mirrored from
Agam's case viewer (scripts/mirror_agam_readers.py at the repo root), served lazily in chunks of positions.

Only the final assistant turn is Qwen3.6-27B's: CHIVE took the earlier turns from the source dataset (WildChat etc.),
and sampled the final one with a ~500-token cap, appending <|im_end|> to responses that hit it.
"""

import difflib
import gzip
import json
import re
from functools import lru_cache
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
HF_REPO = "agu18dec/chive-forensics"
SPLITS = REPO / "vendor/detective-joracle/data/chive/splits"
CMP = REPO / "data/viewer/cmp_climb_test.json"  # Agam's frozen climb_test run (scripts/build_viewer.py)
BANKS = REPO / "data/banks/agam"  # <pid>/<reader>_L{L}.json.gz, each a list indexed by token position
JUDGE = ROOT / "data/chive_judge.jsonl"  # scripts/chive_judge.py: does the shown completion show the behaviour?
BANK_SCORES = ROOT / "data/bank_scores"  # <pid>.jsonl: O-lens bank samples re-scored with the AR (bullets, FVE)
SPLIT_FILES = {"dev": "climb_dev96", "test": "climb_test204", "hard92": "hard92"}  # Agam's climb split; hard92 is a subset
CASES_URL = "https://chive-cases.pages.dev/?case="
CAP = 495  # final turns at least this long hit CHIVE's generation cap (most stop at exactly 500 tokens)
CHUNK = 64  # positions per bank request / static file
READERS = [  # Agam's names; kind decides how the page draws a cell
    {"key": "olens_bank", "label": "O-lens bank", "kind": "text", "layers": list(range(20, 61, 4)),
     "note": "Agam's s3d_rl600 RL O-lens, 4 samples at T=1, not scored with the AR"},
    {"key": "mlnla", "label": "NLA (multi-layer)", "kind": "text", "layers": list(range(20, 61, 4)), "note": "4 samples"},
    {"key": "nla_k4", "label": "NLA-RL", "kind": "text", "layers": [42], "note": "single-layer NLA-RL, 4 samples"},
    {"key": "jlens_all", "label": "J-lens", "kind": "tokens", "layers": list(range(63)),
     "note": "top-10 vocabulary tokens (n1000, wikitext)"},
]  # fmt: skip
ARMS = {"transcript": "transcript", "olens_rl_B": "O-lens", "mlnla_B": "NLA", "jlens_B": "J-lens", "nla_B": "NLA L42"}
PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")

C = {}


def load(first_idx, judge_files):
    """The incidents as pattern dicts, numbered from first_idx (after WeirdChat's, so its links keep their numbers).
    judge_files: every machine's copy of JUDGE (the server's shards), later ones winning."""
    rows = {}
    for config in ("incidents", "tokens", "ground_truth"):
        path = hf_hub_download(HF_REPO, f"data/{config}.jsonl", repo_type="dataset")
        rows[config] = {r["prompt_id"]: r for r in map(json.loads, Path(path).read_text().splitlines())}
    splits = {name: set((SPLITS / f"{f}.txt").read_text().split()) for name, f in SPLIT_FILES.items()}
    solves = {}
    if CMP.exists():
        for i in json.loads(CMP.read_text())["incidents"]:
            solves[i["prompt_id"]] = {ARMS[a["arm"]]: sum(s >= 7 for _, s in a["draws"]) for a in i["arms"] if a["arm"] in ARMS}
    judge = {r["prompt_id"]: r for f in judge_files for r in map(json.loads, f.read_text().splitlines())}
    pats = []
    for i, (pid, inc) in enumerate(sorted(rows["incidents"].items())):
        gt = rows["ground_truth"][pid]
        pats.append(
            {
                "idx": first_idx + i,
                "ds": "chive",
                "pattern_id": pid,
                "tids": [pid],
                "title": inc["behaviour"],
                "inc": inc,
                "gt": gt,
                "split": [name for name, ids in splits.items() if pid in ids],
                "solves": solves.get(pid),
            }
        )
    C.update(pats={p["pattern_id"]: p for p in pats}, tokens=rows["tokens"], judge=judge, scores={})
    return pats


def has(tid):
    return tid in C.get("pats", {})


# ---------------------------------------------------------------------- render
def render(tid, tokens, special):
    """Same shape as server.render, from CHIVE's token ids and per-token turn labels instead of the chat template.
    tokens: the decoded token strings, special: per-token flags (both from the server's tokenizer)."""
    t, p = C["tokens"][tid], C["pats"][tid]
    ids = t["token_ids"]
    contents = [m["content"] for m in p["inc"]["messages"]] + [p["inc"]["completion"]]
    roles = [m["role"] for m in p["inc"]["messages"]] + ["assistant"]
    msgs = [None] * len(contents)  # [start, end) token range per message; None for an empty one, which has no tokens
    for i, (turn, role) in enumerate(t["turn_of"]):
        if role != "control":  # turn k is message k - 1 (turn 0 is the template)
            msgs[turn - 1] = [msgs[turn - 1][0] if msgs[turn - 1] else i, i + 1]
    # char offset of each message's content inside its tokens' text, to map character spans (edits, quotes) to tokens
    starts = []
    for m, c in zip(msgs, contents):
        s = "".join(tokens[m[0] : m[1]]).find(c.strip()) if m else -1
        starts.append(s - c.find(c.strip()) if s >= 0 else None)

    def tok_range(k, lo, hi):
        """Tokens of message k covering content chars [lo, hi). An insertion (lo == hi) marks the token before it, which
        also catches text appended at the end of the message."""
        if starts[k] is None:
            return None
        a, b = msgs[k]
        if lo == hi:
            lo, hi = max(lo - 1, 0), max(lo, 1)
        lo, hi, at, out = lo + starts[k], hi + starts[k], 0, []
        for i in range(a, b):
            if at + len(tokens[i]) > lo and at < hi:
                out.append(i)
            at += len(tokens[i])
        return [out[0], out[-1] + 1] if out else None

    im_end = 248046  # <|im_end|>
    nxt = [m[0] for m in msgs if m] + [len(ids)]
    zones = [[i, min(s for s in nxt if s > i)] for i, x in enumerate(ids) if x == im_end]
    last = len(msgs) - 1
    reply = msgs[last]
    assert reply, f"{tid}: no tokens for the final reply"
    final_user = max(k for k in range(last) if roles[k] == "user" and msgs[k])
    cites = []
    j = C["judge"].get(tid)
    for q in (j or {}).get("quotes") or []:
        s = contents[last].find(q)
        r = tok_range(last, s, s + len(q)) if s >= 0 else None
        if r:
            cites.append({"range": r, "note": f"judge: {q}", "quote": q})
    return {
        "tid": tid,
        "ids": ids,
        "tokens": tokens,
        "special": special,
        "msgs": msgs,
        "roles": roles,
        "cites": cites,
        "zones": zones,
        "truncated": False,
        # dataset-specific
        # the last zone before the reply: an empty final user message leaves two zones ending there
        "main_zone": max(z for z, (a, b) in enumerate(zones) if b == reply[0]),
        "focus": [k for k in range(last) if roles[k] == "system" and msgs[k]] + [final_user, last],
        "not_qwen": [k for k in range(last) if roles[k] == "assistant" and msgs[k]],
        "capped": reply[1] - reply[0] >= CAP,
        "edits": edits(p, contents, tok_range),
    }


def edits(p, contents, tok_range):
    """CHIVE's counterfactual experiments as token spans: each rewrites whole turns (a turn is new text, or {"keep": i}
    for original message i), so diff every rewritten turn against the original at the same place (word level) and keep
    the changed spans, each with what replaced it. Changes less than 3 words apart are one span."""
    out = []
    for e in sorted(p["gt"]["experiments"], key=lambda e: -abs(e["rate"] - e["baseline_rate"])):
        subs = []
        turns = e["turns"]
        structural = len(turns) != len(contents) - 1 or any(x.get("keep", k) != k for k, x in enumerate(turns))
        for k, x in enumerate(turns[: len(contents) - 1]):
            if "keep" in x:
                continue
            a, b = re.findall(r"\s+|\S+", contents[k]), re.findall(r"\s+|\S+", x["content"])
            off = [0]
            for w in a:
                off.append(off[-1] + len(w))
            ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]
            groups = []
            for o in ops:
                if groups and o[1] - groups[-1][1] < 5:  # words and the whitespace between them are separate items
                    groups[-1][1], groups[-1][3] = o[2], o[4]
                else:
                    groups.append([o[1], o[2], o[3], o[4]])
            for i1, i2, j1, j2 in groups:
                r = tok_range(k, off[i1], off[i2])
                if r:
                    subs.append([*r, "".join(b[j1:j2])[:400]])
        out.append(
            {
                "number": e["number"],
                "summary": e["summary"],
                "is_key": e["is_key"],
                "k": e["k"],
                "n": e["n"],
                "effect": e["rate"] - e["baseline_rate"],  # some experiments re-measured their own baseline
                "baseline": e["baseline_rate"],
                "subs": subs,
                "structural": structural,  # turns added, dropped or moved: the spans miss that part
            }
        )
    return out


# ---------------------------------------------------------------------- patterns
def summary(p):
    inc = p["inc"]
    return {
        "idx": p["idx"],
        "ds": "chive",
        "title": p["title"],
        "behavior": inc["source"],
        "split": p["split"],
        "k": inc["k"],
        "n": inc["n"],
        "match_rate": inc["k"] / inc["n"],
        "max_effect": p["gt"]["max_effect_pp"],
        "solves": p["solves"],
        "n_transcripts": 1,
        "elo": {},
        "search": f"{p['pattern_id']} {p['gt']['observed']}",
    }


def detail(p):
    gt = p["gt"]
    return dict(
        summary(p),
        observed=gt["observed"],
        targets=gt["targets"],
        ruled_out=gt["ruled_out"],
        url=CASES_URL + p["pattern_id"] if "test" in p["split"] else None,
        transcripts=[{"tid": p["pattern_id"], "prompt": 0, "match": None, "highlight": True, "explanation": "", "preview": ""}],
    )


def transcript_extra(tid):
    j = C["judge"].get(tid)
    return {
        "ds": "chive",
        "match": j["match"] if j else None,
        "explanation": j["explanation"] if j else None,
        "readers": [dict(r, layers=[L for L in r["layers"] if (BANKS / tid / f"{r['key']}_L{L}.json.gz").exists()]) for r in READERS],
        "chunk": CHUNK,
        "bank_reads": bank_scores(tid),
    }


# ---------------------------------------------------------------------- banks
@lru_cache(maxsize=160)  # J-lens alone is 63 files per incident
def _bank_layer(tid, reader, layer):
    return json.loads(gzip.decompress((BANKS / tid / f"{reader}_L{layer}.json.gz").read_bytes()))


def bank_chunk(tid, reader, chunk):
    """{layer: [cell per position]} for positions [chunk * CHUNK, (chunk + 1) * CHUNK) of one reader."""
    spec = next(r for r in READERS if r["key"] == reader)
    lo = int(chunk) * CHUNK
    have = [L for L in spec["layers"] if (BANKS / tid / f"{reader}_L{L}.json.gz").exists()]
    return {"lo": lo, "layers": {L: _bank_layer(tid, reader, L)[lo : lo + CHUNK] for L in have}}


def bank_scores(tid):
    """Re-scored O-lens bank reads of an incident (same record shape as data/reads.jsonl), reloaded when the file grows."""
    f = BANK_SCORES / f"{tid}.jsonl"
    if not f.exists():
        return []
    sig = f.stat().st_size
    if C["scores"].get(tid, (None,))[0] != sig:
        C["scores"][tid] = (sig, [json.loads(line) for line in f.open()])
    return C["scores"][tid][1]
