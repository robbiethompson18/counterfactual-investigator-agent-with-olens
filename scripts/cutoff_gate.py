"""Knowledge-cutoff stability gate: how stably does Qwen3.6-27B answer each post-training fact question?

Each question goes to the shared Qwen server as one user message (verbatim, no system prompt, thinking off: the
WorkspaceBench knowledge_cutoff render, so O-lens reads line up later), sampled 8x at T=1 / top_p=1 plus one greedy. The
modal normalized answer then classifies the question: stable_right (mode >= 7/8, matches truth or an alias), stable_wrong
(>= 7/8, doesn't), unstable (< 7/8), refusal_or_null (the mode is empty / "I don't know" / "unknown" / ...). No LLM judge.

    uv run python scripts/cutoff_gate.py sample QUESTIONS.json --slug b1b2-1072 [--concurrency 32] [--limit N] [--out DIR]
    uv run python scripts/cutoff_gate.py summarize RUN      # rebuild summary.json/.md from RUN/samples.jsonl

`sample` makes a scripts/runs.py run under data/runs/cutoff_gate/ (or writes to --out for a smoke test), writes
samples.jsonl (one row per question: greedy + 8 samples, raw and normalized), then summarizes. Finish it with
`scripts/runs.py finish RUN`.
"""

import argparse
import asyncio
import collections
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path

from counterfactual_investigator_agent_with_olens.qwen import MODEL, client
from counterfactual_investigator_agent_with_olens.runs import new_run

QWEN_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"  # HF head of Qwen/Qwen3.6-27B, what the shared server loads
N_SAMPLES = 8
STABLE_MIN = 7
MAX_TOKENS = 256  # thinking is off and answers are a few tokens
SYSTEM_PROMPT: str | None = None  # --system; None = WorkspaceBench's no-system-prompt render
NULL_ANSWERS = {"", "i don't know", "i dont know", "i do not know", "unknown", "none", "n/a", "na", "not sure", "no idea", "unsure"}
NULL_PREFIXES = ("i don't know", "i dont know", "i do not know", "i'm not sure", "i am not sure", "i cannot", "i can't", "as an ai")
NUMERIC_TYPES = {"year", "date", "day of month", "month and year", "version number", "software version", "build number"}


def normalize(text: str) -> str:
    """Lowercase, no surrounding quotes or punctuation, no leading article, no currency symbols or thousands commas,
    single spaces. Units stay ("183 billion"): the bank's truth_aliases list the unit variants."""
    s = unicodedata.normalize("NFKC", text).lower().strip()
    s = s.strip("\"'`*_ \n\t.,;:!")
    s = re.sub(r"[$€£¥]", "", s)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"[^\w\s.\-/%]", " ", s)  # keep decimals, hyphens, slashes and percents inside tokens
    s = re.sub(r"\s+", " ", s).strip(" .")
    s = re.sub(r"^(the|a|an) ", "", s)
    return s


def is_null(answer: str) -> bool:
    return answer in NULL_ANSWERS or answer.startswith(NULL_PREFIXES)


def is_numeric(answer_type: str) -> bool:
    return "number" in answer_type or "price" in answer_type or answer_type in NUMERIC_TYPES


async def ask(cl, question: str, sem: asyncio.Semaphore, **sampling) -> list[dict]:
    messages = ([{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else []) + [{"role": "user", "content": question}]
    body = dict(model=MODEL, messages=messages, max_tokens=MAX_TOKENS, **sampling)
    body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    async with sem:
        for attempt in range(4):
            try:
                r = await cl.chat.completions.create(**body)
                return [{"text": c.message.content or "", "finish_reason": c.finish_reason} for c in r.choices]
            except Exception as e:  # transient server/proxy errors; the last one propagates
                if attempt == 3:
                    raise
                print(f"retry {attempt + 1} after {type(e).__name__}: {str(e)[:120]}")
                await asyncio.sleep(5 * (attempt + 1))
    raise AssertionError


async def sample_all(questions: list[dict], out: Path, concurrency: int):
    cl = client()
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(q: dict) -> dict:
        nonlocal done
        sampled, greedy = await asyncio.gather(
            ask(cl, q["question"], sem, n=N_SAMPLES, temperature=1.0, top_p=1.0),
            ask(cl, q["question"], sem, n=1, temperature=0.0),
        )
        done += 1
        if done % 100 == 0:
            print(f"{done}/{len(questions)}")
        return {
            "id": q["id"],
            "question": q["question"],
            "greedy": greedy[0]["text"],
            "greedy_norm": normalize(greedy[0]["text"]),
            "samples": [s["text"] for s in sampled],
            "samples_norm": [normalize(s["text"]) for s in sampled],
            "finish_reasons": [s["finish_reason"] for s in sampled] + [greedy[0]["finish_reason"]],
        }

    with (out / "samples.jsonl").open("w") as f:
        for coro in asyncio.as_completed([one(q) for q in questions]):
            f.write(json.dumps(await coro, ensure_ascii=False) + "\n")
            f.flush()


def classify(q: dict, row: dict) -> dict:
    counts = collections.Counter(row["samples_norm"])
    modal, count = counts.most_common(1)[0]
    truths = {normalize(q["truth"])} | {normalize(a) for a in q["truth_aliases"]}
    analog = normalize(q["pre_cutoff_analog"]) if q["pre_cutoff_analog"] else None
    right = modal in truths
    # loose: a truth string is a whole-word substring of the answer ("stegosaurus skeleton apex" vs "stegosaurus")
    right_loose = right or any(t and re.search(rf"(?<!\w){re.escape(t)}(?!\w)", modal) for t in truths)
    if is_null(modal):
        klass = "refusal_or_null"
    elif count < STABLE_MIN:
        klass = "unstable"
    else:
        klass = "stable_right" if right else "stable_wrong"
    return {
        "id": q["id"],
        "event_date": q["event_date"],
        "year": q["event_date"][:4],
        "answer_type": q["answer_type"],
        "numeric": is_numeric(q["answer_type"]),
        "question": q["question"],
        "truth": q["truth"],
        "pre_cutoff_analog": q["pre_cutoff_analog"],
        "modal": modal,
        "modal_count": count,
        "n_distinct": len(counts),
        "greedy": row["greedy_norm"],
        "greedy_agrees": row["greedy_norm"] == modal,
        "matches_truth": right,
        "matches_truth_loose": right_loose,
        # the analog is a sentence ("Bernard Arnault was also #1 on the 2023 list"), so exact and whole-word-contained both
        "matches_analog": analog is not None and modal == analog,
        "in_analog": analog is not None and modal != "" and re.search(rf"(?<!\w){re.escape(modal)}(?!\w)", analog) is not None,
        "class": klass,
    }


def summarize(run: Path, questions: list[dict]):
    by_id = {q["id"]: q for q in questions}
    rows = [classify(by_id[r["id"]], r) for r in map(json.loads, (run / "samples.jsonl").read_text().splitlines())]
    rows.sort(key=lambda r: r["id"])
    classes = ["stable_right", "stable_wrong", "unstable", "refusal_or_null"]
    tally = lambda rs: {c: sum(r["class"] == c for r in rs) for c in classes} | {"n": len(rs)}
    wrong = [r for r in rows if r["class"] == "stable_wrong"]
    wrong_with_analog = [r for r in wrong if r["pre_cutoff_analog"]]
    totals = {
        "all": tally(rows),
        "by_year": {y: tally([r for r in rows if r["year"] == y]) for y in sorted({r["year"] for r in rows})},
        "by_answer_kind": {k: tally([r for r in rows if r["numeric"] == (k == "numeric")]) for k in ("numeric", "name")},
        "greedy_agrees_with_mode": sum(r["greedy_agrees"] for r in rows),
        "loose_truth_matches_by_class": {c: sum(r["matches_truth_loose"] for r in rows if r["class"] == c) for c in classes},
        "stable_wrong_with_analog": len(wrong_with_analog),
        "stable_wrong_modal_equals_analog": sum(r["matches_analog"] for r in wrong),
        "stable_wrong_modal_in_analog": sum(r["in_analog"] for r in wrong),
        "stale_analog_share_of_stable_wrong": (sum(r["in_analog"] for r in wrong) / len(wrong)) if wrong else None,
    }
    (run / "summary.json").write_text(json.dumps({"totals": totals, "questions": rows}, indent=1, ensure_ascii=False) + "\n")

    def table(name: str, groups: dict) -> list[str]:
        head = f"| {name} | n | " + " | ".join(classes) + " |"
        lines = [head, "|" + "---|" * (len(classes) + 2)]
        for g, t in groups.items():
            cells = [f"{t[c]} ({100 * t[c] / t['n']:.0f}%)" if t["n"] else "0" for c in classes]
            lines.append(f"| {g} | {t['n']} | " + " | ".join(cells) + " |")
        return lines

    md = ["# Cutoff gate summary", ""]
    md += table("group", {"all": totals["all"]}) + [""]
    md += table("year", totals["by_year"]) + [""]
    md += table("answer kind", totals["by_answer_kind"]) + [""]
    md += [
        f"- greedy agrees with the modal sampled answer: {totals['greedy_agrees_with_mode']}/{len(rows)}",
        f"- loose truth matches (truth is a whole-word substring of the answer) by class: {totals['loose_truth_matches_by_class']}",
        f"- stable_wrong: {len(wrong)}; with a pre_cutoff_analog: {len(wrong_with_analog)}; "
        + f"modal == analog: {totals['stable_wrong_modal_equals_analog']}; "
        + f"modal appears in analog text: {totals['stable_wrong_modal_in_analog']} "
        + f"({100 * (totals['stale_analog_share_of_stable_wrong'] or 0):.0f}% of stable_wrong)",
        "",
        "## stable_wrong examples (first 40 by id)",
        "",
        "| id | question | truth | Qwen (mode, count/8) | analog |",
        "|---|---|---|---|---|",
    ]
    cell = lambda s: (s or "").replace("|", "\\|").replace("\n", " ")
    for r in wrong[:40]:
        qwen = f"{cell(r['modal'])} ({r['modal_count']})"
        md.append(f"| {r['id']} | {cell(r['question'][:160])} | {cell(r['truth'])} | {qwen} | {cell(r['pre_cutoff_analog'])[:80]} |")
    (run / "summary.md").write_text("\n".join(md) + "\n")
    print(json.dumps(totals, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("questions", type=Path)
    p.add_argument("--slug", default="gate")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path, help="write here instead of making a run (smoke tests)")
    p.add_argument("--served-by", default="runpod:qwen-shared, vLLM 0.32.0")
    p.add_argument("--system", help="system prompt (default none, the WorkspaceBench render)")
    p = sub.add_parser("summarize")
    p.add_argument("run", type=Path)
    args = ap.parse_args()

    if args.cmd == "summarize":
        summarize(args.run, json.loads((args.run / "questions.json").read_text()))
        return
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = args.system
    questions = json.loads(args.questions.read_text())[: args.limit]
    if args.out:
        run = args.out
        run.mkdir(parents=True, exist_ok=True)
    else:
        run = new_run(
            "cutoff_gate",
            args.slug,
            description=f"Qwen3.6-27B stability gate on {len(questions)} cutoff questions: 8 samples at T=1 + greedy, thinking off",
            models=[{"name": "Qwen/Qwen3.6-27B", "revision": QWEN_REVISION, "served_by": args.served_by}],
            params={
                "n_samples": N_SAMPLES,
                "temperature": 1.0,
                "top_p": 1.0,
                "greedy": True,
                "max_tokens": MAX_TOKENS,
                "thinking": False,
                "system_prompt": SYSTEM_PROMPT,
                "stable_min": STABLE_MIN,
                "concurrency": args.concurrency,
                "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                "n_questions": len(questions),
            },
        )
    shutil.copy(args.questions, run / "questions.json")
    print(f"run: {run}")
    asyncio.run(sample_all(questions, run, args.concurrency))
    summarize(run, questions)


if __name__ == "__main__":
    main()
