"""Check that CHIVE's grader, given WeirdChat's rubric, agrees with WeirdChat's own judge.

For each prompt in a run directory, grades WeirdChat's 64 stored samples with the exact grader
call stage 3 uses (same system prompt, same rubric-wrapped classifier question, all samples in one
call) and compares with WeirdChat's labels. Prints agreement and Cohen's kappa per behaviour.

    ANTHROPIC_API_KEY=... uv run python calibrate.py data/runs/wc30_pilot --wc-data PATH/TO/WeirdChat
"""

import argparse
import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from build_run import CLAUDE, SUBJECT
from chive.pipeline.investigate import grade_completions
from chive.pipeline.llm_client import make_llm_client
from run import classifier_question, patch_anthropic_thinking


def kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--wc-data", type=Path, required=True)
    ap.add_argument("--grader", default=CLAUDE)
    args = ap.parse_args()

    patch_anthropic_thinking()
    meta = json.loads((args.run_dir / "wc_meta.json").read_text())
    pool = {json.loads(x)["id"]: json.loads(x) for x in (args.run_dir / "prompts.jsonl").read_text().splitlines() if x.strip()}
    samples = defaultdict(list)
    cols = ["transcript_id", "prompt_id", "messages", "judgment"]
    for f in sorted((args.wc_data.expanduser() / "data" / "transcripts" / SUBJECT).glob("*.parquet")):
        for r in pq.read_table(f, columns=cols).to_pylist():
            if r["prompt_id"] in meta and r["judgment"] and r["judgment"].get("match") is not None:
                samples[r["prompt_id"]].append(r)

    grader = make_llm_client(args.grader)

    async def one(pid: str) -> tuple[str, list[tuple[bool, bool]], list[dict]]:
        ss = sorted(samples[pid], key=lambda s: s["transcript_id"])
        verdicts = await grade_completions(
            grader, classifier_question(meta[pid]), pool[pid]["messages"], [s["messages"][-1]["content"] or "" for s in ss]
        )
        pairs = [(bool(s["judgment"]["match"]), v) for s, v in zip(ss, verdicts, strict=True)]
        disagreements = [
            dict(transcript_id=s["transcript_id"], weirdchat=a, grader=v, weirdchat_explanation=s["judgment"].get("explanation"),
                 text=s["messages"][-1]["content"] or "")
            for s, (a, v) in zip(ss, pairs, strict=True)
            if a != v
        ]
        return pid, pairs, disagreements

    results = await asyncio.gather(*(one(pid) for pid in meta))
    out, by_behaviour = {}, defaultdict(list)
    for pid, pairs, disagreements in results:
        by_behaviour[meta[pid]["behavior_id"]] += pairs
        out[pid] = {
            "n": len(pairs),
            "wc_yes": sum(a for a, _ in pairs),
            "grader_yes": sum(b for _, b in pairs),
            "agree": sum(a == b for a, b in pairs),
            "kappa": round(kappa(pairs), 3),
            "disagreements": disagreements,
        }
    slug = re.sub(r"[^A-Za-z0-9.-]+", "_", args.grader)
    (args.run_dir / f"calibration_{slug}.json").write_text(json.dumps(out, indent=1))
    print(f"{'behaviour':30s} {'n':>4s} {'WC yes':>6s} {'grader':>6s} {'agree':>6s} {'kappa':>6s}")
    for b, pairs in sorted(by_behaviour.items()):
        print(
            f"{b:30s} {len(pairs):4d} {sum(a for a, _ in pairs):6d} {sum(v for _, v in pairs):6d} "
            f"{sum(a == v for a, v in pairs) / len(pairs):6.0%} {kappa(pairs):6.2f}"
        )
    allp = [p for ps in by_behaviour.values() for p in ps]
    print(
        f"{'all':30s} {len(allp):4d} {sum(a for a, _ in allp):6d} {sum(v for _, v in allp):6d} "
        f"{sum(a == v for a, v in allp) / len(allp):6.0%} {kappa(allp):6.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
