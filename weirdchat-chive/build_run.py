"""Build a CHIVE run directory from the Qwen3.6-27B WeirdChat rollouts.

WeirdChat (https://huggingface.co/datasets/Transluce/WeirdChat) already did CHIVE's stages 0-2 for
us: it found the behaviours, sampled each prompt 64 times and judged every sample. This script
selects the prompts to investigate and writes:

- prompts.jsonl   CHIVE prompt-pool records (stage 0 reads this as config["prompt_pool"])
- wc_meta.json    per prompt: behaviour, pattern, WeirdChat rate, rubric, an example matching reply
                  (read by run.py to steer the screener and to replace its classifier question)
- config.json     a CHIVE pipeline config (stage 1 resamples on our vLLM server by default)
- completions.jsonl, only with --reuse-wc-samples: WeirdChat's own 64 samples as stage 1 output.
  Off by default: we picked these prompts because those samples looked high, so they overstate
  the rate (selection bias). Use it for smoke tests without a GPU.

    uv run python build_run.py --wc-data ~/workspace/weirdchat-olens/data/hf \
        --out data/runs/wc30_pilot --min-rate 0.30 --pilot \
        --target-url http://127.0.0.1:8000
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

SUBJECT = "qwen__qwen3.6-27b"
TARGET_MODEL = "Qwen/Qwen3.6-27B-FP8"  # the checkpoint WeirdChat sampled
CLAUDE = "anthropic:claude-sonnet-5-5"
EXAMPLE_CHARS = 4000


def load(wc_data: Path) -> tuple[list[dict], dict, dict, dict]:
    data = wc_data / "data"
    prompts = [r for f in sorted((data / "prompts" / SUBJECT).glob("*.parquet")) for r in pq.read_table(f).to_pylist()]
    patterns = {r["pattern_id"]: r for r in pq.read_table(data / "patterns.parquet").to_pylist()}
    rubrics = {r["behavior_id"]: r for r in pq.read_table(data / "rubrics.parquet").to_pylist()}
    samples: dict[str, list[dict]] = defaultdict(list)
    cols = ["transcript_id", "prompt_id", "messages", "judgment", "is_highlight"]
    for f in sorted((data / "transcripts" / SUBJECT).glob("*.parquet")):
        for r in pq.read_table(f, columns=cols).to_pylist():
            samples[r["prompt_id"]].append(r)
    return prompts, patterns, rubrics, samples


def rate(p: dict) -> float:
    s = p["match_summary"]
    return s["n_matched"] / s["n_samples"]


def select(prompts: list[dict], min_rate: float, pilot: bool) -> list[dict]:
    eligible = [p for p in prompts if rate(p) >= min_rate]
    if not pilot:
        return sorted(eligible, key=lambda p: (p["behavior_id"], -rate(p)))
    # One prompt per behaviour: the median-rate eligible one (upper median on ties).
    by_behaviour: dict[str, list[dict]] = defaultdict(list)
    for p in eligible:
        by_behaviour[p["behavior_id"]].append(p)
    picks = []
    for group in by_behaviour.values():
        group.sort(key=rate)
        picks.append(group[len(group) // 2])
    return sorted(picks, key=rate, reverse=True)


def example_reply(prompt_samples: list[dict]) -> str:
    matched = [s for s in prompt_samples if s["judgment"] and s["judgment"]["match"]]
    matched.sort(key=lambda s: not s["is_highlight"])  # the pattern's highlight first, if it is this prompt
    if not matched:
        return ""
    text = matched[0]["messages"][-1]["content"] or ""
    return text if len(text) <= EXAMPLE_CHARS else text[:EXAMPLE_CHARS] + "\n[... truncated]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wc-data", type=Path, required=True, help="local copy of Transluce/WeirdChat (the dir holding data/)")
    ap.add_argument("--out", type=Path, required=True, help="run directory to create")
    ap.add_argument("--min-rate", type=float, default=0.30, help="minimum WeirdChat match rate (of 64 samples)")
    ap.add_argument("--pilot", action="store_true", help="one prompt per behaviour (median rate) instead of all eligible")
    ap.add_argument("--n-samples", type=int, default=64, help="samples per prompt and per experiment")
    ap.add_argument("--max-tokens", type=int, default=2560, help="target reply cap (WeirdChat replies run to ~9k chars)")
    ap.add_argument("--target-url", default="http://127.0.0.1:8000", help="vLLM server for Qwen3.6-27B (tunnelled)")
    ap.add_argument("--target-llm", help="override the target spec entirely, e.g. openrouter:qwen/qwen3.6-27b")
    ap.add_argument("--claude", default=CLAUDE, help="spec for investigator, grader and judge")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--exclude", nargs="*", default=[], help="behaviour ids to leave out")
    ap.add_argument("--reuse-wc-samples", action="store_true", help="use WeirdChat's samples as stage 1 (smoke tests)")
    args = ap.parse_args()

    prompts, patterns, rubrics, samples = load(args.wc_data.expanduser())
    chosen = select([p for p in prompts if p["behavior_id"] not in args.exclude], args.min_rate, args.pilot)
    assert chosen, f"no prompts at a WeirdChat rate of {args.min_rate:.0%} or more"
    out = args.out
    out.mkdir(parents=True, exist_ok=False)

    meta = {}
    with open(out / "prompts.jsonl", "w") as f:
        for p in chosen:
            msgs = [{"role": m["role"], "content": m["content"]} for m in p["messages"]]
            assert msgs[-1]["role"] == "user", p["prompt_id"]
            rec = {
                "id": p["prompt_id"],
                "source": "weirdchat",
                "conversation_hash": p["prompt_id"],
                "user_message": msgs[-1]["content"],
                "messages": msgs,
                "n_turns": sum(m["role"] == "user" for m in msgs),
                "total_chars": sum(len(m["content"]) for m in msgs),
            }
            f.write(json.dumps(rec) + "\n")
            pat = patterns[p["pattern_id"]]
            rub = rubrics[p["behavior_id"]]
            meta[p["prompt_id"]] = {
                "behavior_id": p["behavior_id"],
                "behavior_name": rub["name"],
                "pattern_id": p["pattern_id"],
                "pattern_title": pat["title"],
                "pattern_description": pat["description"],
                "wc_matched": p["match_summary"]["n_matched"],
                "wc_samples": p["match_summary"]["n_samples"],
                "rubric_id": rub["transcript_rubric"]["rubric_id"],
                "rubric": rub["transcript_rubric"]["text"],
                "example_reply": example_reply(samples[p["prompt_id"]]),
                "weirdchat_url": pat["weirdchat_url"],
            }
    (out / "wc_meta.json").write_text(json.dumps(meta, indent=1))

    target = args.target_llm or f"vllm:{TARGET_MODEL}@{args.target_url}"
    config = {
        "_comment": "Built by weirdchat-chive/build_run.py; run with weirdchat-chive/run.py, not run_pipeline.py directly",
        "run_name": out.name,
        "concurrency": args.concurrency,
        "n_completions": args.n_samples,
        "grader_llm": args.claude,
        "target_thinking_budget": None,
        "target_max_tokens": args.max_tokens,
        "prompt_pool": str((out / "prompts.jsonl").resolve()),
        "stage1_llm": target,
        "stage1_n_prompts": len(chosen),
        "judge_llm": args.claude,
        "stage2_n": None,
        "stage3_min_score": 1,
        "stage3_llm": args.claude,
        "target_llm": target,
        "stage3_thorough": True,
        "stage3_max_agent_turns": 30,
        "stage3_n_completions_shown": 5,
        "stage3_max_skip_rate": 0.1,
        "skip_stage5_synthetic": True,
    }
    if args.reuse_wc_samples:
        with open(out / "completions.jsonl", "w") as f:
            for p in chosen:
                ss = sorted(samples[p["prompt_id"]], key=lambda s: s["transcript_id"])[: args.n_samples]
                msgs = [{"role": m["role"], "content": m["content"]} for m in p["messages"]]
                rec = {
                    "id": p["prompt_id"],
                    "user_message": msgs[-1]["content"],
                    "completions": [s["messages"][-1]["content"] or "" for s in ss],
                    "finish_reasons": ["stop"] * len(ss),
                    "prompt_tokens": 0,
                    "response_token_counts": [0] * len(ss),
                    "messages": msgs,
                    "n_turns": 1,
                    "total_chars": len(msgs[-1]["content"]),
                    "conversation_hash": p["prompt_id"],
                }
                f.write(json.dumps(rec) + "\n")
        config["completions_file"] = str((out / "completions.jsonl").resolve())
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    rates = [rate(p) for p in chosen]
    print(
        f"{len(chosen)} prompts, {len({p['behavior_id'] for p in chosen})} behaviours, "
        f"WeirdChat rate median {statistics.median(rates):.0%} (min {min(rates):.0%}) -> {out}"
    )
    for p in chosen:
        print(f"  {rate(p):4.0%}  {p['behavior_id']:30s}  {patterns[p['pattern_id']]['title'][:80]}")


if __name__ == "__main__":
    main()
