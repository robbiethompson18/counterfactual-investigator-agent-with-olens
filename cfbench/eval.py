"""Run an eval: a folder cfbench/evals/<name>/ that picks its own cases for the cfbench predictor (cfbench/evals/CLAUDE.md).

    uv run python -m cfbench.eval NAME --prep [--versions a,b] [--placebo DIR]   # its inputs (cases, reads) as a runs.py run
    uv run python -m cfbench.eval NAME --slug SLUG [--versions a,b] [--draws 3] [--limit N] [--conditions ...]
    uv run python -m cfbench.score data/runs/cfbench/<date>_<slug>

The eval's eval.py gives ENVS (the environments/ envs it is about, for the env viewer), CONDITIONS, MODEL_NAME
(None, or what to call the target in place of its id), prep(out, versions, placebo), reads(inputs) and cases(inputs,
versions) -> (prompt env, case row fields, base runs) per case, and optionally BASE_LINE(case) -> str (case has the
prompt's `condition` too): the sentence
in place of the prompt's "In the base arm, `model` showed the behavior in k of n runs." ("" drops it), TASK ("rate", or
"claim": the probability that a claim is true, truth 0 or 1 as k_cf/n_cf), ASK(case) -> str: the body of "## Your task" in
place of build_prompt's, INPUTS_FROM: another eval whose inputs run it reads, PROMPT(prompt, case) -> str: a last
rewrite of the whole prompt, and MODEL_IDS: every model id that must never reach the predictor (default the case's), OLENS: overrides of the
O-lens tools' defaults (`context: "xml"` also rewrites the prompt's read-marker sentence), and for the single-turn
core.INLINE_CONDITIONS: INLINE(env, condition, case, base, seed) -> (the prompt up to and including "## Your task", extra
row fields; `case["inputs"]` is the inputs run dir) and INLINE_EFFORT, the predictor's reasoning effort for them (no tools, one
submit call) unless --effort is given, which then applies to every condition, SYSTEM: a system prompt in place of the task's
(knowledge_cutoff), and DEFAULT_VERSIONS: the versions
run when --versions isn't given (default all). eval.md
beside it, if any, goes into every prompt as "## About this evaluation", before the task. Everything else is cfbench as
usual: core.build_prompt, the transcript and O-lens tools, predict.predict, one predictions.jsonl row per draw in
data/runs/cfbench/<date>_<slug>/, synced to HF.
"""

import argparse
import asyncio
import importlib
import inspect
import json
import random
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path

import anthropic
import openai

from cfbench import answer, olens
from cfbench.core import (
    HARNESS_VERSION,
    INLINE_CONDITIONS,
    JLENS_CONDITIONS,
    OLENS_CONDITIONS,
    TOOL_BUDGET,
    TOOL_CONDITIONS,
    build_prompt,
    check_agentic,
    pick_examples,
)
from cfbench.predict import EFFORT, grader, opening, predict, prompt_sha
from cfbench.tools import transcript_tools
from counterfactual_investigator_agent_with_olens import openrouter
from counterfactual_investigator_agent_with_olens.runs import ROOT, finish, new_run

EVALS = Path(__file__).parent / "evals"


BASE_SENTENCE = re.compile(r"In the base arm, `[^`]+` showed the behavior in \d+ of \d+ runs\.")


def base_line(prompt: str, line: str | None) -> str:
    """The eval's BASE_LINE in place of core.build_prompt's "In the base arm, `model` showed the behavior in k of n runs."
    sentence: None keeps it, "" drops it."""
    if line is None:
        return prompt
    assert len(BASE_SENTENCE.findall(prompt)) == 1, "build_prompt's base-rate sentence changed"
    return BASE_SENTENCE.sub(lambda _: line, prompt).replace("\n\n\n\n", "\n\n")


def hide(prompt: str, model: str, name: str) -> str:
    return prompt.replace(f"the AI model `{model}`", name).replace(f"`{model}`", name)


def with_notes(prompt: str, notes: str) -> str:
    if not notes:
        return prompt
    return prompt.replace("\n\n## Your task\n", f"\n\n## About this evaluation\n\n{notes.strip()}\n\n## Your task\n", 1)


def ask(prompt: str, text: str | None) -> str:
    """The eval's ASK in place of build_prompt's "## Your task" body (the prompt's last section); None keeps it."""
    if text is None:
        return prompt
    assert prompt.count("\n\n## Your task\n\n") == 1, "build_prompt's task heading changed"
    return prompt.split("\n\n## Your task\n\n")[0] + "\n\n## Your task\n\n" + text.strip()


def latest_inputs(name: str, versions: list[str]) -> Path:
    """The newest inputs run of `name` that has every one of `versions`."""
    runs = sorted((ROOT / "cfbench_evals").glob(f"*_{name.replace('_', '-')}-inputs*"), key=lambda p: p.stat().st_mtime)
    runs = [r for r in runs if set(versions) <= set(json.loads((r / "meta.json").read_text())["params"]["versions"])]
    assert runs, f"no inputs run for {name} with {versions}: run `python -m cfbench.eval {name} --prep` first"
    return runs[-1]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="a folder in cfbench/evals/")
    ap.add_argument("--prep", action="store_true", help="build the eval's inputs as a runs.py run (cfbench_evals/<date>_<name>-inputs)")
    ap.add_argument("--placebo", default="", help="--prep: a cfbench reads dir from another env, for olens_placebo_agentic")
    ap.add_argument("--n-items", type=int, default=0, help="--prep: items per version (the eval's prep picks which)")
    ap.add_argument("--strict", action="store_true", help="--prep: only items the model refused in every base run (eval-specific)")
    ap.add_argument("--tag", default="", help="--prep: suffix for the inputs run's slug")
    ap.add_argument(
        "--reuse", default="", help="comma list of predictions.jsonl: copy rows with the same case and prompt_sha instead of re-predicting"
    )
    ap.add_argument("--inputs", default="", help="the inputs run dir (default: the newest)")
    ap.add_argument("--versions", default="", help="comma list of the eval's versions (default: all it has)")
    ap.add_argument("--conditions", default="", help="comma list (default: the eval's CONDITIONS)")
    ap.add_argument("--predictors", default="google/gemini-3.8-flash")
    ap.add_argument(
        "--effort",
        default=None,
        help=f"predictor reasoning effort for every condition (default: {EFFORT}; the eval's INLINE_EFFORT for single-turn conditions)",
    )
    ap.add_argument("--max-calls", type=int, default=TOOL_BUDGET)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (smoke tests)")
    ap.add_argument("--models", default="", help="comma list of target model ids: only their cases")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--attempts", type=int, default=3, help="tries per prediction before its draw is dropped")
    ap.add_argument("--slug", default="")
    ap.add_argument("--show", action="store_true", help="print the first case's prompt per condition (draw 0) and exit; no API calls")
    ap.add_argument("--description", default="")
    ap.add_argument(
        "--batch",
        choices=["submit", "collect"],
        help="anthropic-batch/ predictors, inline conditions: `submit` sends every prediction to Anthropic's Message Batches API "
        "at once and exits; `collect --run DIR` (same other args) waits for those batches and writes the rows",
    )
    ap.add_argument("--run", default="", help="--batch collect: the run dir --batch submit made")
    args = ap.parse_args()
    ev = importlib.import_module(f"cfbench.evals.{args.name}.eval")
    versions = args.versions.split(",") if args.versions else list(getattr(ev, "DEFAULT_VERSIONS", ev.VERSIONS))
    if args.prep:
        run = new_run(
            "cfbench_evals",
            f"{args.name.replace('_', '-')}-inputs" + (f"-{args.tag}" if args.tag else ""),
            description=f"inputs for cfbench eval {args.name} ({', '.join(versions)}): cases + O-lens reads (cfbench/evals/{args.name})",
            models=[],
            params={"eval": args.name, "versions": versions, "placebo": args.placebo, "n_items": args.n_items, "strict": args.strict},
        )
        ev.prep(run, versions, Path(args.placebo), args.n_items or None, args.strict)
        finish(run)
        return
    assert args.slug or args.show or args.run, "--slug"
    conditions = args.conditions.split(",") if args.conditions else list(ev.CONDITIONS)
    check_agentic(conditions)
    inputs = Path(args.inputs) if args.inputs else latest_inputs(getattr(ev, "INPUTS_FROM", args.name), versions)
    task = getattr(ev, "TASK", "rate")
    ask_src = inspect.getsource(ev.ASK) if hasattr(ev, "ASK") else None
    notes = (EVALS / args.name / "eval.md").read_text() if (EVALS / args.name / "eval.md").exists() else ""
    base_src = inspect.getsource(ev.BASE_LINE) if hasattr(ev, "BASE_LINE") else None
    prompt_src = inspect.getsource(ev.PROMPT) if hasattr(ev, "PROMPT") else None
    assert not set(JLENS_CONDITIONS) & set(conditions), "cfbench.eval has no J-lens tools yet; use cfbench.run"
    inline = set(INLINE_CONDITIONS) & set(conditions)
    assert not inline or hasattr(ev, "INLINE"), f"{args.name} has no INLINE prompt for {sorted(inline)}"
    xml = (getattr(ev, "OLENS", None) or {}).get("context") == "xml"
    predictors = args.predictors.split(",")
    reads = olens.Reads(ev.reads(inputs)) if set(OLENS_CONDITIONS) & set(conditions) else None
    fve_note = olens.fve_note(reads) if reads else ""
    # interleaved by item, so a partial run still has every model on the items it reached
    cases = sorted(ev.cases(inputs, versions), key=lambda c: (c[1].get("version") or "", c[1].get("item") or "", c[1]["model"]))
    cases = [c for c in cases if not args.models or c[1]["model"] in args.models.split(",")][: args.limit or None]

    def placebo_sources(case: dict, base: list, seed: int) -> list[str] | None:
        """The eval's own placebo (PLACEBO(case, base, seed) -> per base run the reads source answering for it), else None:
        cfbench's other-env placebo."""
        return ev.PLACEBO(case, base, seed) if hasattr(ev, "PLACEBO") else None

    def prompt_of(env, case: dict, base: list, shown: list, condition: str, read) -> str:
        prompt = build_prompt(  # answer evals (TASK "answer", cfbench/answer.py) get the answer prompt; claim evals the rate one + ASK
            env,
            case["model"],
            base,
            shown,
            case["base_arm"],
            case["cf_arm"],
            condition,
            read,
            fve_note,
            args.max_calls,
            "answer" if task == "answer" else "rate",
        )
        prompt = base_line(prompt, ev.BASE_LINE(case | {"condition": condition}) if hasattr(ev, "BASE_LINE") else None)
        prompt = ask(prompt, ev.ASK(case) if hasattr(ev, "ASK") else None)
        prompt = with_notes(hide(prompt, case["model"], ev.MODEL_NAME) if ev.MODEL_NAME else prompt, notes)
        if xml and condition in OLENS_CONDITIONS:
            assert prompt.count(olens.MARKER_NOTE) == 1, "core.OLENS's read-marker sentence changed"
            prompt = prompt.replace(olens.MARKER_NOTE, olens.XML_NOTE)
        return ev.PROMPT(prompt, case) if hasattr(ev, "PROMPT") else prompt

    def inline_prompt(env, case: dict, base: list, condition: str, seed: int) -> tuple[str, dict]:
        prompt, extra = ev.INLINE(env, condition, case | {"inputs": str(inputs)}, base, seed)
        return with_notes(prompt, notes), extra

    if args.show:
        env, case, base = cases[0]
        print(f"# case {case['env']} {case['model']} (truth {case['k_cf']}/{case['n_cf']}), task {task}\n")
        shown = pick_examples(base, random.Random(args.seed), "answer" if task == "answer" else "rate")
        for condition in conditions:
            if condition in INLINE_CONDITIONS:
                prompt, extra = inline_prompt(env, case, base, condition, args.seed)
                print(f"===== {condition} {json.dumps(extra)} =====\n{prompt}\n")
                continue
            read = None
            if condition in OLENS_CONDITIONS and reads:
                src = placebo_sources(case, base, args.seed) if condition == "olens_placebo_agentic" else None
                placebo = condition == "olens_placebo_agentic" and src is None
                read = olens.tools(base, reads, placebo=placebo, seed=args.seed, sources=src, overrides=getattr(ev, "OLENS", None))[1]
            print(f"===== {condition} =====\n{prompt_of(env, case, base, shown, condition, read)}\n")
        return
    # --reuse: an earlier row of the same case whose prompt_sha equals this run's (the same system, prompt, tool schemas
    # and effort sent) counts as done here; its truth fields are re-taken from this run's case, since the investigator never
    # sees them. The hash doesn't see tool replies, so PLACEBO (which runs' reads the agentic lens tools serve) is still checked.
    inline_src = inspect.getsource(ev.INLINE) if hasattr(ev, "INLINE") else None
    key = lambda r, cond: (r.get("version"), r.get("item"), r["model"], r["base_arm"], r["cf_arm"], cond, r["predictor"], r["draw"])
    reuse, reuse_counts = {}, Counter()
    for path in filter(None, args.reuse.split(",")):
        old = json.loads((Path(path).parent / "meta.json").read_text())["params"]
        assert old.get("placebo_hook") == (inspect.getsource(ev.PLACEBO) if hasattr(ev, "PLACEBO") else None), f"{path}: PLACEBO differs"
        for r in map(json.loads, Path(path).read_text().splitlines()):
            reuse[key(r, r["condition"])] = r
    params = {
        "env": f"eval:{args.name}",
        "eval": args.name,
        "envs": ev.ENVS,
        "inputs": str(inputs.relative_to(ROOT)),
        "versions": versions,
        "conditions": ",".join(conditions),
        "predictors": args.predictors,
        "model_name": ev.MODEL_NAME,
        "models": sorted({c["model"] for _, c, _ in cases}),
        "task": task,
        "effort": args.effort or EFFORT,
        "inline_effort": (args.effort or getattr(ev, "INLINE_EFFORT", None)) if inline else None,
        "max_calls": args.max_calls,
        "draws": args.draws,
        "seed": args.seed,
        "limit": args.limit,
        "model_filter": args.models,
        "harness_version": HARNESS_VERSION,
        "eval_md": notes,
        "base_line": base_src,
        "ask": ask_src,
        "prompt_hook": prompt_src,
        "placebo_hook": inspect.getsource(ev.PLACEBO) if hasattr(ev, "PLACEBO") else None,
        "inline_hook": inline_src,
        "system": getattr(ev, "SYSTEM", None),
        "reuse": args.reuse,
        "batch": args.batch,
    }
    run = (
        Path(args.run)
        if args.batch == "collect"
        else new_run(
            "cfbench",
            args.slug,
            description=args.description or f"eval {args.name} ({', '.join(versions)}): {len(cases)} cases, {', '.join(conditions)}",
            models=[
                {"name": p, "served_by": "anthropic" if p.startswith((openrouter.CLAUDE, openrouter.BATCH)) else "openrouter"}
                for p in predictors
            ],
            params=params,
        )
    )
    out, slots = run / "predictions.jsonl", asyncio.Semaphore(args.concurrency)
    system = getattr(ev, "SYSTEM", None)

    def shown_of(base: list, draw: int) -> list:
        return pick_examples(base, random.Random(args.seed + draw), "answer" if task == "answer" else "rate")

    def reused(case: dict, condition: str, predictor: str, draw: int, sha: str) -> bool:
        """Writes the --reuse row for this prediction, if an earlier run has one with the same prompt_sha."""
        if not (old := reuse.get(key(case | {"predictor": predictor, "draw": draw}, condition))):
            return False
        old_sha = old.get("prompt_sha")
        reuse_counts["reused" if old_sha == sha else "prompt_sha differs" if old_sha else "no prompt_sha"] += 1
        if old_sha != sha:
            return False
        with out.open("a") as f:
            f.write(json.dumps(old | case | {"reused_from": args.reuse}) + "\n")
        return True

    def job(env, case: dict, base: list, condition: str, draw: int, shown: list) -> tuple[str, dict, dict, str]:
        """(prompt, tools, extra row fields (prompt_sha among them), effort) of one prediction."""
        tools, read, extra, effort = {}, None, {}, args.effort or EFFORT
        if condition in INLINE_CONDITIONS:
            prompt, extra = inline_prompt(env, case, base, condition, args.seed + draw)
            effort = args.effort or ev.INLINE_EFFORT  # --effort, when given, applies to every condition (Robbie, 2026-10-29)
        if condition in TOOL_CONDITIONS:
            tools = transcript_tools(base, chars=condition in OLENS_CONDITIONS)
        if condition in OLENS_CONDITIONS:
            assert reads is not None
            src = placebo_sources(case, base, args.seed + draw) if condition == "olens_placebo_agentic" else None
            placebo = condition == "olens_placebo_agentic" and src is None
            lens_tools, read = olens.tools(
                base, reads, placebo=placebo, seed=args.seed + draw, sources=src, overrides=getattr(ev, "OLENS", None)
            )
            if src is not None:
                extra["placebo_sources"] = src
            tools |= lens_tools
        if condition not in INLINE_CONDITIONS:
            prompt = prompt_of(env, case, base, shown, condition, read)
        return prompt, tools, extra | {"prompt_sha": prompt_sha(prompt, tools, effort, task, system)}, effort

    async def predicted(env, case: dict, condition: str, predictor: str, draw: int, prompt: str, tools: dict, effort: str, first=None):
        """predict() with --attempts tries, then None; `first` is the predictor's first reply when it came from a batch."""
        for attempt in range(args.attempts):
            try:
                budget = args.max_calls if tools else 0
                return await predict(
                    predictor, prompt, tools, budget, effort, task, system=system, grade=grader(task, case, env), first=first
                )
            except (openai.APIStatusError, anthropic.APIStatusError, openrouter.NoChoices, json.JSONDecodeError) as e:
                print(f"{case['env']} {case['model']} {condition} draw {draw}: attempt {attempt} failed: {e}")
                first = None  # a retry starts the conversation over

    def record(env, case: dict, condition: str, predictor: str, draw: int, shown: list, effort: str, extra: dict, result: dict):
        if ev.MODEL_NAME:  # the predictor must never have seen the model id
            for m in getattr(ev, "MODEL_IDS", [case["model"]]):
                assert not re.search(re.escape(m), json.dumps(result["messages"])), m
        if task == "answer":  # graded as cfbench.run does: the submission's canonical form vs the cf arm's modal answer
            result |= answer.grade(env, case, result["answer"]) if result["valid"] else {"canon": None, "correct": None}
        row = {**case, "condition": condition, "harness_version": HARNESS_VERSION, "eval": args.name, "predictor": predictor}
        row |= {"effort": effort} | extra
        row |= {"draw": draw, "shown": [r.source for r in shown], **result}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        got = f"p={result.get('p')}"
        if task == "answer":
            got = f"answer={result.get('canon')!r} ({'right' if result.get('correct') else 'wrong'})"
        print(f"{case['env']} {case['model']} {condition} draw {draw}: {got} ${result['cost']:.3f}")

    everything = [
        (env, case, base, c, p, d) for env, case, base in cases for c in conditions for p in predictors for d in range(args.draws)
    ]

    async def one(env, case: dict, base: list, condition: str, predictor: str, draw: int):
        shown = shown_of(base, draw)
        async with slots:  # the prompt is built inside the slot, so --concurrency also bounds how many are in memory
            prompt, tools, extra, effort = job(env, case, base, condition, draw, shown)
            if reused(case, condition, predictor, draw, extra["prompt_sha"]):
                return
            result = await predicted(env, case, condition, predictor, draw, prompt, tools, effort)
        if result:
            record(env, case, condition, predictor, draw, shown, effort, extra, result)

    async def submit(run: Path, everything: list):
        """Every prediction's first request, sent in batches of up to BATCH_MAX (or ~150 MB) as they're built, so memory
        holds one batch; run/batches.jsonl maps each batch's requests back to their predictions for `collect`."""
        api, chunk, keys = openrouter.anthropic_client(), [], []
        (run / "batches.jsonl").touch()  # a fully reused run has no batches, but can still be collected

        async def send():
            for attempt in range(5):
                try:
                    batch = await api.messages.batches.create(requests=[{"custom_id": str(i), "params": r} for i, r in enumerate(chunk)])
                    break
                except anthropic.APIConnectionError:
                    if attempt == 4:
                        raise
                    await asyncio.sleep(10 * 2**attempt)
            with (run / "batches.jsonl").open("a") as f:
                f.write(json.dumps({"batch": batch.id, "keys": keys, "request_hashes": [request_hash(r) for r in chunk]}) + "\n")
            print(f"anthropic batch {batch.id}: {len(chunk)} requests", flush=True)
            chunk.clear()
            keys.clear()

        size = 32  # outer {"requests": [...]} wrapper
        for env, case, base, condition, predictor, draw in everything:
            shown = shown_of(base, draw)
            prompt, tools, extra, effort = job(env, case, base, condition, draw, shown)
            if reused(case, condition, predictor, draw, extra["prompt_sha"]):
                continue
            request = batch_request(predictor, prompt, tools, effort)
            request_size = openrouter.batch_request_size(request)
            if chunk and (len(chunk) == openrouter.BATCH_MAX or size + request_size > openrouter.BATCH_BYTES):
                await send()
                size = 32
            chunk.append(request)
            keys.append(key(case | {"predictor": predictor, "draw": draw}, condition))
            size += request_size
        if chunk:
            await send()
        print(f"submitted; collect with the same command plus --batch collect --run {run}")

    def batch_request(predictor: str, prompt: str, tools: dict, effort: str) -> dict:
        messages, schemas = opening(prompt, tools, task, system)
        return openrouter.claude_request(predictor.removeprefix(openrouter.BATCH), messages, effort, tools=schemas, max_tokens=16000)

    def request_hash(request: dict) -> str:
        return sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

    async def collect(run: Path, everything: list):
        """Waits for submit's batches; as each ends, finishes its predictions (a reply without a valid submission gets its
        further turns, and an errored or expired request its whole conversation, from anthropic-direct/) and writes the rows,
        then finishes the run. Rerunnable: predictions already in predictions.jsonl are skipped."""
        api = openrouter.anthropic_client()
        jobs = {key(case | {"predictor": p, "draw": d}, c): (env, case, base, c, p, d) for env, case, base, c, p, d in everything}
        done = {key(r, r["condition"]) for r in map(json.loads, out.open())} if out.exists() else set()
        pending = [json.loads(line) for line in (run / "batches.jsonl").open()]
        while pending:
            for b in list(pending):
                if (await api.messages.batches.retrieve(b["batch"])).processing_status != "ended":
                    continue
                results = {e.custom_id: e.result async for e in await api.messages.batches.results(b["batch"])}

                async def finish_one(i: int, k: tuple, results: dict, hashes: list):
                    if k in done:
                        return
                    env, case, base, condition, predictor, draw = jobs[k]
                    shown = shown_of(base, draw)
                    async with slots:
                        prompt, tools, extra, effort = job(env, case, base, condition, draw, shown)
                        assert request_hash(batch_request(predictor, prompt, tools, effort)) == hashes[i], (
                            f"{k}: request differs from the one submitted (prompt, system, tools, model or effort)"
                        )
                        r = results.get(str(i))
                        first = openrouter.claude_reply(r.message) if r and r.type == "succeeded" else None
                        direct = predictor.replace(openrouter.BATCH, openrouter.CLAUDE)
                        result = await predicted(env, case, condition, direct, draw, prompt, tools, effort, first=first)
                    if result:
                        record(env, case, condition, predictor, draw, shown, effort, extra, result)
                        done.add(k)

                await asyncio.gather(*(finish_one(i, tuple(k), results, b["request_hashes"]) for i, k in enumerate(b["keys"])))
                pending.remove(b)
                print(f"collected {b['batch']}; {len(pending)} batches left", flush=True)
            if pending:
                await asyncio.sleep(60)
        missing = jobs.keys() - done
        if missing:
            raise RuntimeError(
                f"{len(missing)} predictions are missing; run marked abandoned. Retry collect for failed replies; "
                "if submission was interrupted, submit a new run with --reuse pointing to this run's predictions.jsonl."
            )
        finish(run)

    if args.batch:
        assert set(conditions) <= set(INLINE_CONDITIONS), "--batch: single-turn (inline) conditions only"
        assert all(p.startswith(openrouter.BATCH) for p in predictors), "--batch: anthropic-batch/ predictors only"
        if args.batch == "collect":  # a mistyped collection command must not change the original run's status
            submitted = json.loads((run / "meta.json").read_text())["params"]
            for name, value in params.items():
                if name not in {"batch", "reuse"}:
                    assert submitted.get(name) == value, f"{name} differs from submission; collect with the original configuration"

    try:
        if args.batch:
            await (submit if args.batch == "submit" else collect)(run, everything)
        else:
            await asyncio.gather(*(one(*j) for j in everything))
        if args.reuse and args.batch != "collect":  # collect never reuses: submit did
            print(
                f"--reuse: {reuse_counts['reused']} rows reused; not reused: {reuse_counts['prompt_sha differs']} whose prompt_sha "
                f"differs, {reuse_counts['no prompt_sha']} without one (older runs)"
            )
        if args.batch:
            return
    except BaseException:
        finish(run, status="abandoned")
        raise
    finish(run)


if __name__ == "__main__":
    asyncio.run(main())
