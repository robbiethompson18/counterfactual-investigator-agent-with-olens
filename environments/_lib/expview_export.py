"""An env run as an expview dataset, so every rollout opens in expview (https://expview.pages.dev) as well as Docent:

    data/runs/env_<env>/<run>/expview/     items.jsonl (one item per episode: id = episode id, group = arm, labels = the
                                           latest measure@ labels, Docent link, the whole conversation as one call,
                                           the sandbox files at the start as the env), index.jsonl, envs/
    HF runs/env_<env>/<run>/expview/       with --upload; the dataset id is runs/env_<env>/<run>

    uv run python -m environments expview ENV/RUN|ENV [...] [--upload]     # ENV alone = every run of it

The env page (page.html) links a run's expview dataset and each episode's expview item once the export exists
(spec.expview_dataset). Re-exporting rewrites items; env snapshots are content-addressed.
"""

import json
import sys
from pathlib import Path

from .core import load_env
from .index import latest_labels
from .runfolder import LOCAL, REPO, REPO_ROOT, hf_token, read_episodes

EXPVIEW_ROOT = REPO_ROOT / "data" / "runs"


def out_dir(env: str, run: str) -> Path:
    return EXPVIEW_ROOT / f"env_{env}" / run / "expview"


def export(env: str, run: str) -> Path:
    sys.path.insert(0, str(REPO_ROOT / "expview"))
    from expview import Run, call  # ty: ignore[unresolved-import]

    run_dir = LOCAL / env / run
    meta = json.loads((run_dir / "meta.json").read_text())
    mod = load_env(env)
    _, labels = latest_labels(run_dir)
    target = meta.get("target") or {}
    ev = Run(
        out_dir(env, run),
        title=f"{env}: {run}",
        description=(
            f"Env run `envs/{env}/{run}` ({target.get('id')}, thinking {target.get('thinking')}). One item per episode, "
            f"grouped by arm; labels are `measure()` ({mod.PRIMARY} is the primary). {meta.get('description') or ''}"
        ),
        label_order=[mod.PRIMARY],
    )
    ev.reset()
    for ep in read_episodes(run_dir):
        blobs = {p: run_dir / "blobs" / h for p, h in (ep.get("files_before") or {}).items()}
        files = {p: b.read_text(errors="replace") for p, b in blobs.items() if b.exists()}
        vals = {k: v for k, v in (labels.get(ep["id"]) or {}).items() if v is None or isinstance(v, (bool, int, float, str))}
        ev.add(
            id=ep["id"],
            group=ep["arm"],
            labels=vals or ({"error": True} if ep.get("error") else {}),
            calls=[call("subject", target.get("model") or ep.get("target"), messages=ep.get("messages") or [], tools=ep.get("tools"))],
            env_files=files or None,
            links={"Docent": ep["docent_url"]} if ep.get("docent_url") else None,
            extra={"finish": ep.get("finish"), "item": ep.get("item"), "sample": ep.get("sample"), "error": ep.get("error")},
        )
    return out_dir(env, run)


def upload(paths: list[Path]) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    ops = [
        CommitOperationAdd(path_in_repo="runs/" + str(f.relative_to(EXPVIEW_ROOT)), path_or_fileobj=str(f))
        for p in paths
        for f in sorted(x for x in p.rglob("*") if x.is_file())
    ]
    info = HfApi(token=hf_token()).create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message="expview: env runs")
    return info.oid


def cli(args: list[str]):
    up = "--upload" in args
    outs = []
    for r in (a for a in args if a != "--upload"):
        env, _, one = r.partition("/")
        for run in [one] if one else sorted(p.name for p in (LOCAL / env).glob("*") if (p / "meta.json").exists()):
            outs.append(export(env, run))
            print(outs[-1], flush=True)
    if up and outs:
        print("uploaded", upload(outs)[:10])
