"""Any environments/ env, read from its run folders (`data/envs/<env>/<run>/`, synced from HF `envs/<env>/`): the same
cases, pairs and tasks as evalcue_runs.py, over episodes instead of rows.

    uv run python -m cfbench.run env_runs:hint_mcq/2026-10-27_pause --task answer --items 6031,6598 --pairs pause/hint+d1:pause/nohint ...

The spec is `<env>/<run>[+<run>...][@<item>]` as for evalcue_runs. The label is the `positive` value of the run's `measure`
labels (`labels/measure@<ver>.jsonl`; a later file wins when a run was rescored); episodes with an error or no label are
skipped. Sources are `<env>/<run>:<episode id>`, the ids O-lens read requests use (scripts/olens_requests_from_runs.py's
`envs:` specs), so reads join to cases. BEHAVIOR, canon_answer and the answer-task attributes come from the env module.
"""

import json
from pathlib import Path

from cfbench.envs.evalcue_runs import EvalcueRuns
from environments._lib.core import load_env
from environments._lib.runfolder import LOCAL, read_episodes


def rows(env: str, run: str):
    """(source id, row) per episode of a run folder, rows shaped like evalcue's (model, arm, item, messages, positive, meta,
    files_before as text, cwd) so the rest of cfbench and the read-request builder need no second code path."""
    run_dir = Path(LOCAL) / env / run
    labels: dict[str, dict | None] = {}
    for path in sorted((run_dir / "labels").glob("measure@*.jsonl")):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            labels[rec["episode"]] = rec["values"]
    for ep in read_episodes(run_dir):
        values = labels.get(ep["id"]) or {}
        files = {p: (run_dir / "blobs" / h).read_text() for p, h in (ep.get("files_before") or {}).items()}
        row = {
            "model": ep["target"].split("@")[0],
            "arm": ep["arm"],
            "item": ep["item"],
            "sample": ep["sample"],
            "messages": ep["messages"],
            "tools": ep.get("tools") or None,
            "positive": values.get("positive"),
            "meta": ep.get("meta") or {},
            "files_before": files,
            "cwd": ep.get("workdir"),
            "thinking": ep.get("thinking"),
        }
        if ep.get("error"):
            row["error"] = ep["error"]
        yield f"{env}/{run}:{ep['id']}", row


class EnvRuns(EvalcueRuns):
    def __init__(self, spec: str):
        env = spec.split("/")[0]
        mod = load_env(env)
        super().__init__(spec)
        self.BEHAVIOR = mod.BEHAVIOR
        if hasattr(mod, "canon_answer"):
            self.ANSWER_FORMAT, self.ANSWER_MATCH, self.canon_answer = mod.ANSWER_FORMAT, mod.ANSWER_MATCH, mod.canon_answer

    def _iter_rows(self, env: str, run: str):
        yield from rows(env, run)


def load(spec: str) -> EnvRuns:
    return EnvRuns(spec)
