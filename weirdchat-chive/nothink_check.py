"""Re-sample a case's original prompt with thinking enabled, i.e. without the empty
<think></think> block that enable_thinking=False prefills, and grade it like the baseline.

The grader sees only the final answer (text after </think>); replies that never close the
think block within the token cap are counted separately and graded as they are.

    uv run python nothink_check.py data/runs/case_64bb7e
"""

import asyncio
import json
import sys
from pathlib import Path

from openai import AsyncOpenAI

import wctool
from build_run import TARGET_MODEL
from run import classifier_question


async def main() -> None:
    run = Path(sys.argv[1])
    pid, m, pool = wctool.meta_of(run)
    client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
    r = await client.chat.completions.create(
        model=TARGET_MODEL, messages=pool["messages"], n=64, temperature=1.0, max_tokens=7000,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    raw = [c.message.content or "" for c in r.choices]
    reasoning = [getattr(c.message, "reasoning_content", None) or "" for c in r.choices]
    finals, closed = [], []
    for text, rc in zip(raw, reasoning, strict=True):
        if "</think>" in text:
            finals.append(text.split("</think>", 1)[1].strip())
            closed.append(True)
        else:
            finals.append(text.strip())
            closed.append(bool(rc) or "<think>" not in text)
    verdicts = await wctool.grade(classifier_question(m), pool["messages"], finals)
    base = json.loads((run / "baseline.json").read_text())
    out = dict(raw=raw, reasoning=reasoning, finals=finals, closed=closed, verdicts=verdicts,
               finish=[c.finish_reason for c in r.choices])
    (run / "thinking_on.json").write_text(json.dumps(out))
    print(f"thinking on (no empty <think></think> prefill): grader YES {sum(verdicts)}/64; "
          f"baseline with the prefill {sum(base['verdicts'])}/64; WeirdChat {m['wc_matched']}/{m['wc_samples']}")
    print(f"think block closed in {sum(closed)}/64; finish reasons {dict((f, out['finish'].count(f)) for f in set(out['finish']))}")


if __name__ == "__main__":
    asyncio.run(main())
