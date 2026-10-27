"""Compare two reads.jsonl files from scripts/olens_read_requests.py on their shared (id, token_index, layer) rows: how often
the top bullet / the whole bullet list agree, FVE agreement, and each file's throughput from its reader log.

    python scripts/olens_reads_compare.py OLD.reads.jsonl NEW.reads.jsonl [--logs OLD.log NEW.log]

Used for the one-engine reader (olens/src/olens/one_engine.py) against the HF+vLLM reader: both decode greedily from the
same activations up to bf16 noise, so first-bullet agreement of ~80-90% is parity (lora_for_vllm.py's numbers), far lower
means a bug (wrong layer, wrong norm, inert LoRA). The --logs stats read each log's last progress line, so a log that holds
several restarted runs is summarised by its last segment only.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

P = re.compile(
    r"^\[(\d+), (\d+) queued\] (?:batch of (?:\d+) transcripts, |\S+ p\S*: )(\d+) tokens, (\d+) reads(?: \((\d+) cached\))? \| "
    r"([\d.]+) reads/s, (\d+) transcripts/h \| capture (\d+)s (?:wait|decode) (\d+)s score (\d+)s"
)


def load(path):
    rows = {}
    with Path(path).open() as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"], r["token_index"], r["layer"]] = r
    return rows


def log_stats(path):
    last = None
    tot_tok = tot_reads = tot_cached = n = 0
    with Path(path).open(errors="replace") as f:
        for line in f:
            m = P.match(line)
            if m:
                last = m
                n += 1
                tot_tok += int(m[3])
                tot_reads += int(m[4])
                tot_cached += int(m[5] or 0)
    if not last:
        return "no progress lines"
    cap, wait, score = int(last[8]), int(last[9]), int(last[10])
    main = cap + wait + score or 1
    return (
        f"{n} transcripts, {tot_tok} tokens, {tot_reads} reads ({tot_cached} cached), {main / 60:.1f} min main-thread, "
        f"{tot_reads / main:.1f} reads/s, {tot_tok / main:.0f} tok/s; "
        f"capture {100 * cap / main:.0f}% wait/decode {100 * wait / main:.0f}% score {100 * score / main:.0f}%"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--logs", nargs=2)
    a = ap.parse_args()
    A, B = load(a.old), load(a.new)
    shared = sorted(set(A) & set(B))
    print(f"old {len(A)} reads, new {len(B)} reads, shared {len(shared)}")
    if not shared:
        return
    top = first_line = allsame = nb_a = nb_b = 0
    fa, fb, ja, jb = [], [], [], []
    for k in shared:
        ba, bb = A[k]["bullets"], B[k]["bullets"]
        nb_a += len(ba)
        nb_b += len(bb)
        ta = [x["text"] for x in ba]
        tb = [x["text"] for x in bb]
        if ta and tb and ta[0] == tb[0]:
            top += 1
        if sorted(ta) == sorted(tb):
            allsame += 1
        # first line of the raw read = the first bullet in decode order; bullets are sorted by FVE, so compare the sets' heads
        if ta and tb and ta[0].split()[:3] == tb[0].split()[:3]:
            first_line += 1
        if ta and tb:
            fa.append(ba[0]["fve"] or 0)
            fb.append(bb[0]["fve"] or 0)
        ja.append(A[k]["joint_fve"] or 0)
        jb.append(B[k]["joint_fve"] or 0)
    n = len(shared)
    print(
        f"top bullet identical: {top / n:.1%}   top bullet same first 3 words: {first_line / n:.1%}   "
        f"bullet sets identical: {allsame / n:.1%}"
    )
    print(f"bullets/read old {nb_a / n:.2f} new {nb_b / n:.2f}")
    ja, jb = np.array(ja), np.array(jb)
    print(f"joint FVE: old mean {ja.mean():.3f} new mean {jb.mean():.3f} corr {np.corrcoef(ja, jb)[0, 1]:.3f}")
    if fa:
        fa, fb = np.array(fa), np.array(fb)
        print(f"top-bullet FVE: old mean {fa.mean():.3f} new mean {fb.mean():.3f} corr {np.corrcoef(fa, fb)[0, 1]:.3f}")
    for k in shared[:3]:
        print(
            "\n",
            k,
            "\n  old:",
            [x["text"][:60] for x in A[k]["bullets"]],
            A[k]["joint_fve"],
            "\n  new:",
            [x["text"][:60] for x in B[k]["bullets"]],
            B[k]["joint_fve"],
        )
    if a.logs:
        for name, p in zip(("old", "new"), a.logs):
            print(f"{name} log: {log_stats(p)}")


if __name__ == "__main__":
    main()
