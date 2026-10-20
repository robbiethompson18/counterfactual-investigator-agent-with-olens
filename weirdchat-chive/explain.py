"""Ask the grader model to explain its verdict on every calibration disagreement.

CHIVE's grader returns bare yes/no verdicts for a whole batch, so there is no reason to show. This
re-asks Sonnet about each disagreeing sample on its own (same grader system prompt, same rubric
question, one response) for an answer plus a short reason, and stores both next to the batch
verdict in the run's calibration file. The single-sample answer can differ from the batch one;
when it does, that is itself a sign the rubric is borderline for that reply.

    ANTHROPIC_API_KEY=... uv run python explain.py data/runs/wc30_pilot9_calibration
"""

import asyncio
import json
import re
import sys
from pathlib import Path

import anthropic
from chive.pipeline.investigate import GRADER_SYSTEM_PROMPT, build_grader_user_content

from run import classifier_question

MODEL = "claude-sonnet-5-5"
ASK = (
    "\n\nThis time there is only one response. Answer the question for it, then explain in two to four "
    "sentences which part of the rubric decides it. Reply with JSON only: "
    '{"answer": "yes" or "no", "reason": "..."}'
)


async def main() -> None:
    run_dir = Path(sys.argv[1])
    meta = json.loads((run_dir / "wc_meta.json").read_text())
    pool = {json.loads(x)["id"]: json.loads(x) for x in (run_dir / "prompts.jsonl").read_text().splitlines() if x.strip()}
    path = run_dir / "calibration_anthropic_claude-sonnet-5-5.json"
    cal = json.loads(path.read_text())
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(16)

    async def one(pid: str, d: dict) -> None:
        if "grader_reason" in d:
            return
        content = build_grader_user_content(classifier_question(meta[pid]), pool[pid]["messages"], [d["text"]]) + ASK
        async with sem:
            r = await client.messages.create(
                model=MODEL, max_tokens=4000, thinking={"type": "adaptive"}, output_config={"effort": "medium"},
                system=GRADER_SYSTEM_PROMPT.split("Submit one verdict")[0].strip(),
                messages=[{"role": "user", "content": content}],
            )
        text = "".join(b.text for b in r.content if b.type == "text")
        m = re.search(r"\{.*\}", text, re.S)
        try:
            j = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            j = {}
        d["grader_single_answer"] = j.get("answer")
        d["grader_reason"] = j.get("reason") or text.strip()[:2000]

    await asyncio.gather(*(one(pid, d) for pid, x in cal.items() for d in x.get("disagreements", [])))
    path.write_text(json.dumps(cal, indent=1))
    n = sum(len(x.get("disagreements", [])) for x in cal.values())
    flips = sum(
        (d.get("grader_single_answer") == "yes") != d["grader"] for x in cal.values() for d in x.get("disagreements", [])
    )
    print(f"explained {n} disagreements; the single-sample answer differs from the batch verdict on {flips}")


if __name__ == "__main__":
    asyncio.run(main())
