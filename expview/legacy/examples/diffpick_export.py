"""Export a diff-pick eval run (~/workspace/baseline-hyp/triples/diffpick) to an expview run.

    python examples/diffpick_export.py [run_dir] [out_dir]
    python expview.py build out_dir -o diffpick.html

Each trial is one item. Its environment is the whole setting directory (baseline/real/null arms, both diffs,
setting.json), including the grader files the model never saw. The prompt is rebuilt with diffpick.py's own
PROMPT / render_env / prompt_diff, so it is byte-identical to what was sent as long as settings/ is unchanged
since the run (checked below against the run's mtime).
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
TRIPLES = Path.home() / "workspace/baseline-hyp/triples"
sys.path.insert(0, str(TRIPLES / "diffpick"))
import diffpick as dp  # noqa: E402
from expview import Run, call, message  # noqa: E402

run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else TRIPLES / "diffpick/runs/v41flash_n16"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "runs" / f"diffpick_{run_dir.name}"

rows = [json.loads(l) for l in open(run_dir / "rows.jsonl")]
t_run = min((run_dir / "rows.jsonl").stat().st_mtime, (run_dir / "summary.json").stat().st_mtime)
newer = [p for p in dp.SETTINGS.rglob("*") if p.is_file() and p.stat().st_mtime > t_run]
if newer:
    print(f"warning: {len(newer)} setting files changed after the run, e.g. {newer[0]}; prompts may differ", file=sys.stderr)

summary = json.loads((run_dir / "summary.json").read_text())
pooled = summary["_pooled"]
model = rows[0]["model"]
run = Run(out, title=f"Diff-pick eval · {run_dir.name}", description=(
    f"Can **{model}** tell which of two edits to an environment lowers a behaviour? Each setting has a baseline "
    "environment, a *real* edit that measurably lowers the behaviour rate in Robbie's transcripts, and a *null* edit "
    "that reads as lowering but does not. The model sees the baseline (prompt-visible files only, so not meta.json "
    "or episode.json), the behaviour, and both diffs as A/B, counterbalanced 8/8. Correct = picks the real edit.\n\n"
    f"Pooled: {pooled['k']}/{pooled['n']} correct, posterior mean {pooled['mean']:.2f} "
    f"(95% interval {pooled['ci95'][0]:.2f}–{pooled['ci95'][1]:.2f}), cost ${summary['_cost_usd']}. "
    "Each item's environment is the full setting directory; the call shows the exact prompt, reasoning and reply."),
    label_order=["correct", "real_edit", "answer", "order", "confidence"],
    source=str(run_dir))
run.reset()

cache, seen = {}, {}
for r in sorted(rows, key=lambda r: (r["setting"], r["trial"])):
    s = r["setting"]
    if s not in cache:
        d = dp.SETTINGS / s
        meta = json.loads((d / "setting.json").read_text())
        env, real, null = dp.render_env(d / "baseline"), dp.prompt_diff(d, "real"), dp.prompt_diff(d, "null")
        beh = meta.get("behaviour_prompt", meta["behaviour"])
        cache[s] = dict(env=run.env(dir=d), prompts={
            "real_first": dp.PROMPT.format(behaviour=beh, env=env, diff_a=real, diff_b=null),
            "null_first": dp.PROMPT.format(behaviour=beh, env=env, diff_a=null, diff_b=real)})
    c = cache[s]
    subject = call("subject", r["model"],
                   messages=[message("user", c["prompts"][r["order"]])],
                   output=message("assistant", r.get("reply"), reasoning=r.get("reasoning")),
                   params={"max_tokens": 16000, "reasoning_effort": "high"},
                   parsed={"answer": r["answer"], "confidence": r["confidence"]},
                   provider=r.get("provider"), cost=r.get("cost"), error=r.get("error"))
    # a trial whose reply had no parsable answer was retried on resume; keep both attempts, count only parsed ones
    key = (s, r["trial"]); seen[key] = seen.get(key, 0) + 1
    tid = f"{s}/t{r['trial']:02d}" + (f" attempt {seen[key]}" if seen[key] > 1 or r["answer"] is None else "")
    run.add(id=tid, group=s, env=c["env"], calls=[subject],
            labels={"correct": r["correct"] if r["answer"] else None, "real_edit": "A" if r["order"] == "real_first" else "B", "answer": r["answer"], "order": r["order"], "confidence": r["confidence"]})
print(f"{len(rows)} items -> {out}")
