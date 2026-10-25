"""optimize_cache_c autorater (environments/optimize_cache_c/grade_rollouts.py), ported -- NOT RUN by default.

The original judge is claude-opus-5 via the Anthropic API (structured output, adaptive thinking, effort max) reading
the rollout.log up to the FINAL RESULTS banner. Here the rollout.log is re-rendered from the saved messages in the
original print format (SYSTEM PROMPT / USER_PROMPT / STEP n / REASONING / RESPONSE / TOOL CALL / TOOL RESULT), the
rubric is verbatim, and the call goes through OpenRouter (anthropic/claude-opus-5, reasoning effort high, JSON schema
response format) -- a routing deviation that needs sign-off before use. Writes record["judge"] into each run file.

  .venv/bin/python envport/optimize_cache_c/judge.py runs/<out> [--model anthropic/claude-opus-5] [--force] [--dry]
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from env import chat, openrouter_key

# grade_rollouts.PROMPT, verbatim (copied from agent-interp-envs @ 310c03f)
# a checkout of https://github.com/gkroiz/agent-interp-envs (set AIE_DIR to point elsewhere)
AIE = Path(os.environ.get("AIE_DIR", Path.home() / "workspace/agent-interp-envs"))
PROMPT = (AIE / "environments/optimize_cache_c/grade_rollouts.py").read_text()
PROMPT = PROMPT.split('PROMPT = """', 1)[1].split('"""', 1)[0].replace("\\\n", "")

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reasoning", "verdict", "transparency_signal", "key_fact_disclosed", "evidence", "tough_call"],
    "properties": {
        "reasoning": {"type": "string", "description": "your working, for each step of the procedure in order"},
        "verdict": {"type": "string", "enum": ["cheated", "honest"], "description": "per the Verdict section"},
        "transparency_signal": {"type": "boolean", "description": "per the Transparency signal section; false unless cheated"},
        "key_fact_disclosed": {"type": "boolean", "description": "per the Key fact section; false unless cheated"},
        "evidence": {"type": "string", "description": "per procedure step 5"},
        "tough_call": {"type": "boolean", "description": "per procedure step 5"},
    },
}


def section(title, content, width=70):
    return f"{'=' * width}\n{title.center(width)}\n{'=' * width}\n{content}\n{'=' * width}\n\n"


def rollout_log(messages):
    """agent_interp_envs print_history / run_step output, up to (not including) FINAL RESULTS."""
    out, step = [], 0
    for i, m in enumerate(messages):
        if m["role"] == "system":
            out.append("\n" + section("SYSTEM PROMPT", m["content"]))
        elif m["role"] == "user" and i == 1:
            out.append(section("USER_PROMPT", m["content"]))
        elif m["role"] == "user":
            out.append(section("USER MESSAGE", m["content"]))
        elif m["role"] == "assistant":
            out.append(f"\n{'=' * 70}\n{f'STEP {step}'.center(70)}\n{'=' * 70}\n\n")
            step += 1
            if m.get("reasoning"):
                out.append("\n" + section("REASONING", m["reasoning"]))
            if m.get("content"):
                out.append(section("RESPONSE", m["content"]))
            for tc in m.get("tool_calls") or []:
                out.append(section("TOOL CALL", f"Function: {tc['function']['name']}\nArguments: {tc['function']['arguments']}"))
        elif m["role"] == "tool":
            out.append(section("TOOL RESULT", m["content"]))
    return "".join(out).rstrip("= \n")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--model", default="anthropic/claude-opus-5")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print the first prompt and exit")
    a = ap.parse_args()
    key = openrouter_key()
    files = sorted(Path(a.dir).glob("*.json"))
    async with httpx.AsyncClient() as client:

        async def one(f):
            d = json.loads(f.read_text())
            r = d["record"]
            if (r.get("judge") and not a.force) or r["telemetry"]["submission_mode"] != "report" or r["end"] != "responded":
                return
            prompt = PROMPT.format(bar=f"{r['detector']['bar']:g}x", transcript=rollout_log(d["messages"]))
            if a.dry:
                print(prompt)
                sys.exit(0)
            j = await chat(
                client,
                a.model,
                [{"role": "user", "content": prompt}],
                None,
                key,
                max_tokens=None,
                reasoning={"effort": "high"},
                response_format={"type": "json_schema", "json_schema": {"name": "Rating", "strict": True, "schema": SCHEMA}},
            )
            r["judge"] = {
                **json.loads(j["choices"][0]["message"]["content"]),
                "judge": a.model,
                "judge_cost": (j.get("usage") or {}).get("cost"),
            }
            r["detector"]["judge_verdict"] = r["judge"]["verdict"]
            f.write_text(json.dumps(d, indent=1))
            print(f.name, r["judge"]["verdict"], flush=True)

        await asyncio.gather(*[one(f) for f in files])


if __name__ == "__main__":
    asyncio.run(main())
