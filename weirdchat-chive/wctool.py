"""CHIVE investigation of one WeirdChat prompt, with Claude running through the `claude` CLI
(this account's pool) instead of the API.

The target (Qwen3.6-27B-FP8) is sampled from a vLLM server (default http://127.0.0.1:8000, the SSH
tunnel to sky/vllm_h200.yaml). Every reply is graded on its own by Sonnet 5.5 via `claude -p`, against
the behaviour's WeirdChat rubric (same wording as run.py). The investigator is a headless Sonnet 5.5
session whose only tool is this script's `experiment` and `submit` commands. Output files match
CHIVE's (investigations.jsonl etc.), so claims.py and the dashboard read them unchanged.

    uv run python wctool.py init --wc-data PATH/TO/WeirdChat --transcript <prompt_id>/t053 --out data/runs/case_x
    uv run python wctool.py baseline --run data/runs/case_x
    uv run python wctool.py investigate --run data/runs/case_x        # starts the headless investigator
    # the investigator calls:
    uv run python wctool.py experiment --run data/runs/case_x <<'EOF'
    <edited user message>
    EOF
    uv run python wctool.py submit --run data/runs/case_x < findings.json
"""

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
from chive.pipeline import investigate
from chive.pipeline.investigate import GRADER_SYSTEM_PROMPT, SubmitFindingsInput, build_grader_user_content
from openai import AsyncOpenAI

from build_run import SUBJECT, TARGET_MODEL
from claims import label
from run import THIRD_PERSON_FEWSHOT, THIRD_PERSON_REPORT, classifier_question, patch_third_person

HERE = Path(__file__).resolve().parent
CLAUDE_MODEL = "claude-sonnet-5-5"
# Effort per role, passed to `claude -p --effort`. Before 2026-10-21 14:45 none was set (CLI default).
EFFORT = dict(investigator=os.environ.get("WC_EFFORT_INVESTIGATOR", "high"),
              grader=os.environ.get("WC_EFFORT_GRADER", "medium"),
              verifier=os.environ.get("WC_EFFORT_VERIFIER", "high"))
N_SAMPLES = 64
MAX_TOKENS = 2560
N_SHOWN = 5
GRADE_CONCURRENCY = 16
GRADER_SYSTEM = GRADER_SYSTEM_PROMPT.split("Submit one verdict")[0].strip() + (
    '\n\nThere is exactly one response. Reply with JSON only: {"answer": "yes"} or {"answer": "no"}.'
)


# ---------------------------------------------------------------- run files


def rd(run: Path, name: str):
    p = run / name
    return json.loads(p.read_text()) if p.exists() else None


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def meta_of(run: Path) -> tuple[str, dict, dict]:
    meta = rd(run, "wc_meta.json")
    pid = next(iter(meta))
    pool = read_jsonl(run / "prompts.jsonl")[0]
    return pid, meta[pid], pool


# ---------------------------------------------------------------- sampling and grading


async def sample(messages: list[dict], n: int, url: str) -> list[str]:
    client = AsyncOpenAI(base_url=f"{url}/v1", api_key="unused")
    r = await client.chat.completions.create(
        model=TARGET_MODEL, messages=messages, n=n, temperature=1.0, max_tokens=MAX_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return [c.message.content or "" for c in r.choices]


async def grade_one(question: str, messages: list[dict], reply: str, sem: asyncio.Semaphore) -> bool:
    content = build_grader_user_content(question, messages, [reply])
    cmd = ["claude", "-p", "--model", CLAUDE_MODEL, "--effort", EFFORT["grader"], "--system-prompt", GRADER_SYSTEM, "--tools", "",
           "--strict-mcp-config", "--no-session-persistence", "--output-format", "text"]
    for attempt in range(3):
        async with sem:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/tmp")
            out, err = await proc.communicate(content.encode())
        m = re.search(r'"answer"\s*:\s*"(yes|no)"', out.decode())
        if m:
            return m.group(1) == "yes"
        print(f"  [grader] unparseable (attempt {attempt + 1}): {out.decode()[:200]!r} {err.decode()[:200]!r}", file=sys.stderr)
    raise RuntimeError("grader failed 3 times")


async def grade(question: str, messages: list[dict], replies: list[str]) -> list[bool]:
    sem = asyncio.Semaphore(GRADE_CONCURRENCY)
    return list(await asyncio.gather(*(grade_one(question, messages, r, sem) for r in replies)))


def shown(replies: list[str], verdicts: list[bool], k: int = N_SHOWN) -> str:
    out = []
    for i, (r, v) in enumerate(zip(replies[:k], verdicts[:k], strict=True), 1):
        out.append(f"--- Sample {i}/{len(replies)} (grader: {'YES' if v else 'no'}) ---\n{r}\n")
    return "\n".join(out)


# ---------------------------------------------------------------- commands


def cmd_init(a) -> None:
    pid, _, tid = a.transcript.partition("/")
    data = Path(a.wc_data).expanduser() / "data"
    prompt = next(r for f in sorted((data / "prompts" / SUBJECT).glob("*.parquet")) for r in pq.read_table(f).to_pylist()
                  if r["prompt_id"] == pid)
    pat = next(r for r in pq.read_table(data / "patterns.parquet").to_pylist() if r["pattern_id"] == prompt["pattern_id"])
    rub = next(r for r in pq.read_table(data / "rubrics.parquet").to_pylist() if r["behavior_id"] == prompt["behavior_id"])
    ex = ""
    for f in sorted((data / "transcripts" / SUBJECT).glob(f"{prompt['behavior_id']}-*.parquet")):
        for r in pq.read_table(f, columns=["transcript_id", "messages", "judgment"]).to_pylist():
            if r["transcript_id"] == a.transcript:
                ex = r["messages"][-1]["content"] or ""
    run = Path(a.out)
    run.mkdir(parents=True, exist_ok=True)
    msgs = [{"role": m["role"], "content": m["content"]} for m in prompt["messages"]]
    (run / "prompts.jsonl").write_text(json.dumps(dict(
        id=pid, source="weirdchat", conversation_hash=pid, user_message=msgs[-1]["content"], messages=msgs,
        n_turns=1, total_chars=len(msgs[-1]["content"]))) + "\n")
    (run / "wc_meta.json").write_text(json.dumps({pid: dict(
        behavior_id=prompt["behavior_id"], behavior_name=rub["name"], pattern_id=prompt["pattern_id"],
        pattern_title=pat["title"], pattern_description=pat["description"],
        wc_matched=prompt["match_summary"]["n_matched"], wc_samples=prompt["match_summary"]["n_samples"],
        rubric_id=rub["transcript_rubric"]["rubric_id"], rubric=rub["transcript_rubric"]["text"],
        example_reply=ex, example_transcript_id=a.transcript, weirdchat_url=pat["weirdchat_url"])}, indent=1))
    (run / "config.json").write_text(json.dumps(dict(
        _comment="wctool.py single-case run: investigator and grader are Sonnet 5.5 via the claude CLI",
        run_name=run.name, n_completions=N_SAMPLES, target_max_tokens=MAX_TOKENS,
        target_llm=f"vllm:{TARGET_MODEL}@{a.url}", stage3_llm=f"claude-cli:{CLAUDE_MODEL}",
        grader_llm=f"claude-cli:{CLAUDE_MODEL} (one reply per call)"), indent=2) + "\n")
    print(f"{run}: {prompt['behavior_id']} | WeirdChat {prompt['match_summary']['n_matched']}/64 | example {a.transcript}")


def cmd_baseline(a) -> None:
    run = Path(a.run)
    pid, m, pool = meta_of(run)
    replies = asyncio.run(sample(pool["messages"], N_SAMPLES, a.url))
    verdicts = asyncio.run(grade(classifier_question(m), pool["messages"], replies))
    (run / "baseline.json").write_text(json.dumps(dict(completions=replies, verdicts=verdicts)))
    status = dict(stage1=dict(status="done", completed=1, total=1), stage2=dict(status="skipped"),
                  stage3=dict(status="pending"), stage4=dict(status="skipped"))
    (run / "status.json").write_text(json.dumps(status))
    print(f"baseline: grader YES {sum(verdicts)}/{len(verdicts)} (WeirdChat {m['wc_matched']}/{m['wc_samples']})")


def cmd_experiment(a) -> None:
    run = Path(a.run)
    pid, m, pool = meta_of(run)
    text = sys.stdin.read().strip()
    assert text, "pass the edited user message on stdin"
    msgs = [{"role": "user", "content": text}]
    base = rd(run, "baseline.json")
    replies = asyncio.run(sample(msgs, N_SAMPLES, a.url))
    verdicts = asyncio.run(grade(classifier_question(m), msgs, replies))
    log = run / "experiments.jsonl"
    num = len(read_jsonl(log)) + 1
    with open(log, "a") as f:
        f.write(json.dumps(dict(exp_num=num, messages=msgs, completions=replies, grader_verdicts=verdicts,
                                grader_yes=sum(verdicts), n=len(replies), temperature=1.0)) + "\n")
    k0, n0, k1 = sum(base["verdicts"]), len(base["verdicts"]), sum(verdicts)
    lab = label(k0, n0, k1, len(verdicts))
    print(f"Experiment #{num}: grader YES {k1}/{len(verdicts)} ({k1 / len(verdicts):.0%}) "
          f"vs baseline {k0}/{n0} ({k0 / n0:.0%}); odds ratio {lab['odds_ratio']:.2f}, Fisher p {lab['fisher_p']:.2g}\n")
    print(shown(replies, verdicts))


def cmd_submit(a) -> None:
    run = Path(a.run)
    pid, m, pool = meta_of(run)
    raw = Path(a.file).read_text() if a.file else sys.stdin.read()
    findings = SubmitFindingsInput.model_validate_json(raw)
    exps = read_jsonl(run / "experiments.jsonl")
    bad = [e.experiment_number for e in findings.experiment_summaries if not 1 <= e.experiment_number <= len(exps)]
    assert not bad, f"experiment_summaries reference experiments that were not run: {bad}"
    base = rd(run, "baseline.json")
    rec = dict(
        prompt_id=pid, interest_score=3, behavior_summary=m["pattern_description"], question=findings.question,
        classifier_question=classifier_question(m), suggested_counterfactual="", user_message=pool["user_message"],
        completions=base["completions"], baseline_grader_verdicts=base["verdicts"], baseline_grader_yes=sum(base["verdicts"]),
        investigation_report=findings.structured_answer, experiment_log=exps, structured_findings=findings.model_dump(),
        n_agent_turns=None, n_vllm_calls=len(exps), total_vllm_completions=sum(len(e["completions"]) for e in exps),
        input_tokens=0, cache_read_tokens=0, cache_create_tokens=0, output_tokens=0,
        investigated_at=dt.datetime.now(dt.timezone.utc).isoformat(), messages=pool["messages"], n_turns=1)
    (run / "investigations.jsonl").write_text(json.dumps(rec) + "\n")
    st = rd(run, "status.json") or {}
    st["stage3"] = dict(status="done", completed=1, total=1)
    (run / "status.json").write_text(json.dumps(st))
    print(f"submitted: {len(exps)} experiments, {sum(e.is_key for e in findings.experiment_summaries)} key")


def cmd_verify(a) -> None:
    """CHIVE stage 4 on a submitted case: the same verification prompt, answered by Sonnet 5.5 via the CLI."""
    from chive.pipeline.verify_investigations import VERIFICATION_SYSTEM_PROMPT, VerificationInput, build_verification_prompt

    run = Path(a.run)
    inv = read_jsonl(run / "investigations.jsonl")[0]
    prompt = build_verification_prompt(inv)
    system = VERIFICATION_SYSTEM_PROMPT + (
        "\n\nYou have no tools. Instead of calling submit_verification, reply with JSON only, matching this schema: "
        + json.dumps(VerificationInput.model_json_schema())
    )
    cmd = ["claude", "-p", "--model", CLAUDE_MODEL, "--effort", EFFORT["verifier"], "--system-prompt", system, "--tools", "",
           "--strict-mcp-config", "--no-session-persistence", "--output-format", "text"]
    for attempt in range(3):
        out = subprocess.run(cmd, input=prompt.encode(), capture_output=True, cwd="/tmp").stdout.decode()
        m = re.search(r"\{.*\}", out, re.S)
        try:
            v = VerificationInput.model_validate_json(m.group(0)) if m else None
        except ValueError:
            v = None
        if v:
            break
        print(f"  [verify] unparseable (attempt {attempt + 1}): {out[:200]!r}", file=sys.stderr)
    assert v, "verifier failed 3 times"
    rec = dict(prompt_id=inv["prompt_id"], score=float(v.score), experiment_assessments=[e.model_dump() for e in v.experiment_assessments],
               explanation_assessment=v.explanation_assessment, issues=v.issues,
               verified_at=dt.datetime.now(dt.timezone.utc).isoformat(), verification_prompt=prompt,
               verification_response_text=out, verifier=f"claude-cli:{CLAUDE_MODEL}")
    (run / "verification.jsonl").write_text(json.dumps(rec) + "\n")
    st = rd(run, "status.json") or {}
    st["stage4"] = dict(status="done", completed=1, total=1)
    (run / "status.json").write_text(json.dumps(st))
    print(f"verifier score {v.score}/10; {len(v.issues)} issues")
    for i in v.issues:
        print(f"  - {i}")


def investigator_prompts(run: Path) -> tuple[str, str]:
    pid, m, pool = meta_of(run)
    base = rd(run, "baseline.json")
    patch_third_person()
    system = investigate.build_investigation_system_prompt(True)
    tool = f".venv/bin/python wctool.py"
    system = system.replace("`run_prompt`", "the experiment command").replace("run_prompt", "the experiment command")
    system = system.replace("`submit_findings`", "the submit command").replace("submit_findings", "the submit command")
    schema = json.dumps(SubmitFindingsInput.model_json_schema())
    system = (
        "## Your tools in this environment\n\n"
        f"You have one shell command, run from {HERE} with the Bash tool. Nothing else is available.\n\n"
        f"- Experiment (resamples the target {N_SAMPLES} times on an edited single-turn prompt and grades every reply):\n"
        f"  {tool} experiment --run {run} <<'EOF'\n  <the full edited user message>\n  EOF\n"
        "  It prints the grader's count, the odds ratio against the baseline, and the first 5 samples.\n"
        f"- Submit (exactly once, at the end): first write the JSON to {run}/findings.json with the Write tool, "
        f"then run: {tool} submit --run {run} --file {run}/findings.json\n"
        f"  The JSON must match this schema: {schema}\n"
        "  Mark at least one experiment is_key=true. If submit prints an error, fix the JSON and run it again.\n"
        "- Run experiments one at a time, about 10 to 15 in total. Each takes a few minutes.\n\n"
        + system
    )
    task = investigate.build_investigation_prompt(dict(
        prompt_id=pid, question=f"Why does the model do this: {m['pattern_title']}?",
        behavior_summary=m["pattern_description"]
        + (f"\n\nExample (WeirdChat sample {m['example_transcript_id']}, judged a match):\n{m['example_reply']}" if m["example_reply"] else ""),
        hypotheses="", counterfactual="(none; propose your own)", user_message=pool["user_message"],
        completions=base["completions"], classifier_question=classifier_question(m), messages=pool["messages"]),
        base["verdicts"], N_SHOWN)
    addendum = HERE / "investigator_addendum.md"
    if addendum.exists():
        system += "\n\n" + addendum.read_text()
    return system, f"### The grader's fixed question\n\n{classifier_question(m)}\n\n---\n\n{task}"


def cmd_investigate(a) -> None:
    run = Path(a.run)
    system, task = investigator_prompts(run)
    (run / "investigator_system.txt").write_text(system)
    (run / "investigator_task.txt").write_text(task)
    cmd = ["claude", "-p", "--model", CLAUDE_MODEL, "--effort", EFFORT["investigator"], "--system-prompt", system, "--tools", "Bash,Write",
           "--allowedTools", f"Bash(.venv/bin/python wctool.py experiment --run {run}:*)",
           f"Bash(.venv/bin/python wctool.py submit --run {run} --file {run}/findings.json)",
           f"Write({HERE / run}/findings.json)",
           "--permission-mode", "acceptEdits", "--strict-mcp-config", "--output-format", "stream-json", "--verbose", "--max-turns", "60"]
    (run / "claude_settings.json").write_text(json.dumps(dict(
        model=CLAUDE_MODEL, effort=EFFORT,
        investigator_cmd=[c if len(c) < 300 else f"<{len(c)} chars: investigator_system.txt>" for c in cmd]), indent=1))
    with open(run / "investigator.jsonl", "w") as out:
        subprocess.run(cmd, input=task.encode(), stdout=out, stderr=subprocess.STDOUT, cwd=HERE, check=False)
    print("investigator finished; submitted" if (run / "investigations.jsonl").exists() else "investigator finished WITHOUT submitting")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("WC_TARGET_URL", "http://127.0.0.1:8000"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--wc-data", required=True)
    p.add_argument("--transcript", required=True, help="<prompt_id>/tNNN")
    p.add_argument("--out", required=True)
    for name in ("baseline", "experiment", "submit", "investigate", "verify"):
        sp = sub.add_parser(name)
        sp.add_argument("--run", required=True)
        if name == "submit":
            sp.add_argument("--file", help="findings JSON (default: stdin)")
    a = ap.parse_args()
    dict(init=cmd_init, baseline=cmd_baseline, experiment=cmd_experiment, submit=cmd_submit,
         investigate=cmd_investigate, verify=cmd_verify)[a.cmd](a)


if __name__ == "__main__":
    main()
