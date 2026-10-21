"""Run directories: every experiment's outputs go in data/runs/<experiment>/<YYYY-MM-DD>_<slug>/ with a meta.json saying
what produced them, and a run isn't finished until it's synced to the HF dataset under runs/ (same path).

<experiment> is the env name for agent-interp-envs screens (funding_email, sandbagging, ...) or the experiment family
otherwise (evalcue, identity, organisms). Models go in meta.json, not the path: one run often compares several.

From Python (scripts/ is on sys.path when you run `python scripts/<x>.py`):

    from runs import finish, new_run
    run = new_run("party_resample", "organism-vs-base", description="...",
                  models=[{"name": "Qwen/Qwen3.6-27B", "revision": "<hf sha>", "served_by": "runpod:qwen-shared, vLLM 0.11"}],
                  params={"temperature": 1.0, "top_p": 0.95, "n": 200})
    ...write outputs under `run`...
    finish(run)  # status done, then sync

From the shell, for runs a harness writes itself (point its results dir at the printed path):

    uv run python scripts/runs.py new funding_email arm-a-qwen -d "..." --model Qwen/Qwen3.6-27B@<sha> \\
        --served-by "runpod:qwen-envs via scripts/envs_gateway.py" --param n=10 --param reasoning_effort=high
    uv run python scripts/runs.py finish data/runs/funding_email/2026-10-21_arm-a-qwen   # done + sync
    uv run python scripts/runs.py finish RUN --status abandoned                         # still synced: failed runs are data
    uv run python scripts/runs.py sync RUN                                              # re-upload after adding files
    uv run python scripts/runs.py ls                                                    # every local run's meta, one line each

Syncing needs an HF token with write access to the dataset (`hf auth login`).
"""

import argparse
import json
import shlex
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi

REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
ROOT = Path("data/runs")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_info() -> dict | None:
    """The code that produced the run; None off a git checkout (e.g. a pod synced without .git)."""
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=False).stdout.strip()
    sha = git("rev-parse", "HEAD")
    if not sha:
        return None
    return {"sha": sha, "branch": git("branch", "--show-current"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def new_run(
    experiment: str, slug: str, description: str, models: list[dict], params: dict | None = None, command: str | None = None
) -> Path:
    """Create the run dir and its meta.json (status running). `models`: one dict per model, with at least `name`, and
    `revision` (HF commit) and `served_by` (which pod/endpoint/API and server version) whenever you know them."""
    run = ROOT / experiment / f"{datetime.now():%Y-%m-%d}_{slug}"
    run.mkdir(parents=True, exist_ok=False)  # a second run with the same slug on the same day needs its own slug
    meta = {
        "experiment": experiment,
        "slug": slug,
        "description": description,
        "status": "running",
        "started_at": now(),
        "finished_at": None,
        "machine": socket.gethostname().split(".")[0].lower(),
        "git": git_info(),
        "command": command or shlex.join(sys.argv),
        "models": models,
        "params": params or {},
    }
    (run / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return run


def finish(run: Path, status: str = "done", sync: bool = True):
    meta_path = run / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta |= {"status": status, "finished_at": now()}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    if sync:
        upload(run)


def upload(run: Path):
    """Upload the run dir to runs/<experiment>/<run> on HF. Unreadable files (e.g. root-owned ones a Docker harness wrote)
    raise rather than being skipped: chown them first."""
    meta = json.loads((run / "meta.json").read_text())
    path_in_repo = f"runs/{meta['experiment']}/{run.name}"
    HfApi().upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=str(run),
        path_in_repo=path_in_repo,
        commit_message=f"{path_in_repo} ({meta['status']}): {meta['description'][:120]}",
    )
    print(f"synced {run} -> https://huggingface.co/datasets/{REPO}/tree/main/{path_in_repo}")


def ls():
    for meta_path in sorted(ROOT.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        models = ",".join(m["name"] for m in meta["models"])
        print(f"{meta_path.parent}  {meta['status']:9}  {meta['machine']:12}  {models}  {meta['description'][:80]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", help="create a run dir and print its path")
    p.add_argument("experiment")
    p.add_argument("slug")
    p.add_argument("-d", "--description", required=True)
    p.add_argument("--model", action="append", required=True, help="NAME or NAME@HF_REVISION; repeat for several models")
    p.add_argument("--served-by", help="where the models ran, e.g. 'runpod:qwen-envs, vLLM 0.11' or 'openrouter'")
    p.add_argument("--param", action="append", default=[], help="KEY=VALUE, VALUE parsed as JSON if it can be; repeatable")
    p.add_argument("--command", help="the harness command this run wraps")
    p = sub.add_parser("finish", help="mark a run finished and sync it")
    p.add_argument("run", type=Path)
    p.add_argument("--status", default="done", choices=["done", "abandoned"])
    p.add_argument("--no-sync", action="store_true")
    sub.add_parser("sync", help="re-upload a run").add_argument("run", type=Path)
    sub.add_parser("ls", help="list local runs")
    args = ap.parse_args()

    if args.cmd == "new":
        models = []
        for spec in args.model:
            name, _, revision = spec.partition("@")
            models.append({"name": name, "revision": revision or None, "served_by": args.served_by})
        params = {}
        for kv in args.param:
            key, _, value = kv.partition("=")
            try:
                params[key] = json.loads(value)
            except json.JSONDecodeError:
                params[key] = value
        print(new_run(args.experiment, args.slug, args.description, models, params, args.command))
    elif args.cmd == "finish":
        finish(args.run, args.status, sync=not args.no_sync)
    elif args.cmd == "sync":
        upload(args.run)
    else:
        ls()


if __name__ == "__main__":
    main()
