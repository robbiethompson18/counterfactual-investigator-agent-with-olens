"""Essay grading (ex hidden_goal.grade): who wrote the essay, or the user's stake, vs Qwen's harsh grades.

Does one sentence about who wrote a mediocre essay, or what the user hopes for, change how often Qwen3.6-27B (thinking
off) grades it harshly (Score 4/10 or lower)?

Kylie, 2026-10-28.

The 38 arm pairs of Robbie's pre-registered 2026-10-23 test (docs/envs/hidden_goal.md; HF runs/cfbench/2026-10-23_grade-big*
and 2026-10-26_grade-sol-*), each pair both ways, each version on the one essay_grading run the old cases used:

    cue      2026-10-22_n30_nothink        base:mine base:hope base:stakes all:all-mine all:all-hope all:all-stakes
    who      2026-10-22_who2_n30_nothink   base:mine base:rival base:boss base:son base:ai
    heldout  2026-10-22_heldout_n30_nothink  base:coworker base:famous base:screen base:ex base:enemy base:friend base:mentee
                                           base:daughter (arms fixed before any was run)

A case is one ordered arm pair with every base-arm run shown (30), as cfbench.run does; the truth is the cf arm's rate in
the same run. The O-lens reads are the ones the old test used (agu18dec olens_s3d_space_rl600, layers 32/40/48, every stop
token and <|im_end|>, greedy): data/runs/olens_reads/2026-10-22_hidden-goal-v{1,2,3}, whose ids are the converted episodes'
`source` without `data/runs/` (hidden_goal_grade/<run>:<line>); prep renames them to essay_grading/<run>:<episode id>, the
sources env_runs gives, and drops v1's hidden_goal_deploy_note reads. Placebo: --placebo DIR (the old test used
data/runs/olens_reads/2026-10-22_donation-bet-v1), every row not from this env.

    uv run python -m cfbench.eval essay_grading --prep --placebo data/runs/olens_reads/2026-10-22_donation-bet-v1
    uv run python -m cfbench.eval essay_grading --show --versions cue                 # the prompts, no API calls
    uv run python -m cfbench.eval essay_grading --slug essay-grading --draws 5
    uv run python -m cfbench.score data/runs/cfbench/<date>_essay-grading
"""

import json
from pathlib import Path

from cfbench.core import Rollout
from cfbench.envs.env_runs import EnvRuns
from environments._lib.runfolder import LOCAL, REPO_ROOT, read_episodes

ENVS = ["essay_grading"]
ENV = ENVS[0]
MODEL = "qwen/qwen3.6-27b"
MODEL_NAME = None
CONDITIONS = ("no_transcript", "transcript_agentic", "olens_agentic", "olens_placebo_agentic")
LABEL = "measure@converted-v2"  # `positive` = harsh; measure@converted has only the old `harsh` key
READS = REPO_ROOT / "data/runs/olens_reads"
VERSIONS: dict[str, dict] = {
    "cue": {
        "run": "2026-10-22_n30_nothink",
        "reads": "2026-10-22_hidden-goal-v1",
        "pairs": [("base", "mine"), ("base", "hope"), ("base", "stakes"), ("all", "all-mine"), ("all", "all-hope"), ("all", "all-stakes")],
    },
    "who": {
        "run": "2026-10-22_who2_n30_nothink",
        "reads": "2026-10-22_hidden-goal-v2",
        "pairs": [("base", "mine"), ("base", "rival"), ("base", "boss"), ("base", "son"), ("base", "ai")],
    },
    "heldout": {
        "run": "2026-10-22_heldout_n30_nothink",
        "reads": "2026-10-22_hidden-goal-v3",
        "pairs": [("base", a) for a in ("coworker", "famous", "screen", "ex", "enemy", "friend", "mentee", "daughter")],
    },
}


class Prompt:
    """What build_prompt needs: BEHAVIOR, and env_runs' view() over the stored starting contexts ((arm, item) -> start)."""

    def __init__(self, behavior: str, starts: dict[tuple[str, str], str], by_source: dict[str, dict]):
        self.BEHAVIOR, self.starts, self.by_source = behavior, starts, by_source

    _items, _shared, _diff, view = EnvRuns._items, EnvRuns._shared, EnvRuns._diff, EnvRuns.view


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None = None, strict: bool = False):
    """Per version: cases.jsonl rows (one per ordered pair), the run's rollouts and starting contexts, and its reads renamed
    to env_runs sources; with `placebo`, placebo/ (every row whose id isn't this env's)."""
    assert not n_items and not strict, "--n-items and --strict mean nothing here"
    cases, rollouts, starts, behavior = [], [], {}, None
    reads, requests = [], []
    for v in versions:
        spec = VERSIONS[v]
        runs = EnvRuns(f"{ENV}/{spec['run']}@label={LABEL}")
        behavior = runs.BEHAVIOR
        for base_arm, cf_arm in spec["pairs"]:
            for b, c in ((base_arm, cf_arm), (cf_arm, base_arm)):
                kb, kc = runs.rollouts(b, MODEL), runs.rollouts(c, MODEL)
                cases.append({"version": v, "base_arm": b, "cf_arm": c, "k_base": sum(r.label for r in kb), "n_base": len(kb)})
                cases[-1] |= {"k_cf": sum(r.label for r in kc), "n_cf": len(kc), "run": spec["run"]}
        for arm, rows in runs.rows.items():
            for source, row in rows:
                rollouts.append(
                    {"source": source, "arm": arm, "item": str(row["item"]), "label": row["positive"], "messages": row["messages"]}
                )
        starts |= {f"{v}|{a}|{i}": s for (a, i), s in runs.starts.items()}
        # old read id (episode source without data/runs/) -> env_runs source
        rename = {
            ep["source"].removeprefix("data/runs/"): f"{ENV}/{spec['run']}:{ep['id']}" for ep in read_episodes(LOCAL / ENV / spec["run"])
        }
        src = READS / spec["reads"]
        for r in map(json.loads, (src / "reads.jsonl").open()):
            if r["id"] in rename:
                reads.append(r | {"id": rename[r["id"]]})
        for r in map(json.loads, (src / "requests.jsonl").open()):
            if r["id"] in rename:
                new = rename[r["id"]]
                assert [m["content"] for m in r["messages"]] == [m["content"] for m in runs.by_source[new]["messages"]], new
                requests.append(r | {"id": new})
        n_runs, n_read = sum(len(r) for r in runs.rows.values()), len({r["id"] for r in requests})
        print(f"{v}: {len(spec['pairs']) * 2} cases, {n_runs} runs; reads for {n_read} runs so far")
    write = lambda p, rows: p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    write(out / "cases.jsonl", cases)
    write(out / "rollouts.jsonl", rollouts)
    (out / "starts.json").write_text(json.dumps({"behavior": behavior, "starts": starts}))
    (out / "reads").mkdir()
    write(out / "reads/reads.jsonl", reads)
    write(out / "reads/requests.jsonl", requests)
    if str(placebo) in ("", "."):
        return
    (out / "placebo").mkdir()
    keep = lambda id: id.split("/")[0] not in (ENV, "hidden_goal_grade")
    write(out / "placebo/reads.jsonl", [r for r in map(json.loads, (placebo / "reads.jsonl").open()) if keep(r["id"])])
    write(out / "placebo/requests.jsonl", [r for r in map(json.loads, (placebo / "requests.jsonl").open()) if keep(r["id"])])


def reads(inputs: Path) -> list[str]:
    return [str(p) for p in (inputs / "reads/reads.jsonl", inputs / "placebo/reads.jsonl") if p.exists()]


def LINKS(row: dict) -> dict[str, dict]:
    """The first base-arm run the prompt showed (its Docent and expview pages)."""
    env_run, _, ep = row["shown"][0].partition(":")
    env, run = env_run.split("/")
    return {"transcript": {"env": env, "run": run, "episode": ep}}


def cases(inputs: Path, versions: list[str]):
    """(prompt env, case row fields, base runs) per case: every base-arm run of the version's run."""
    stored = json.loads((inputs / "starts.json").read_text())
    rollouts = [json.loads(line) for line in (inputs / "rollouts.jsonl").read_text().splitlines()]
    for v in versions:
        starts = {(a, i): s for k, s in stored["starts"].items() for ver, a, i in [k.split("|")] if ver == v}
        mine = [r for r in rollouts if r["source"].startswith(f"{ENV}/{VERSIONS[v]['run']}:")]
        prompt = Prompt(stored["behavior"], starts, {r["source"]: r for r in mine})
        for c in map(json.loads, (inputs / "cases.jsonl").read_text().splitlines()):
            if c["version"] != v:
                continue
            base = [Rollout(source=r["source"], messages=r["messages"], label=r["label"]) for r in mine if r["arm"] == c["base_arm"]]
            row = {k: c[k] for k in ("version", "base_arm", "cf_arm", "k_base", "n_base", "k_cf", "n_cf")}
            row |= {"env": f"eval:essay_grading/{v}", "env_name": ENV, "model": MODEL, "task": "rate"}
            yield prompt, row, base
