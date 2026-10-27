"""An env run as an expview dataset, so every rollout opens in expview (https://expview.pages.dev) as well as Docent:

    data/runs/env_<env>/<run>/expview/     items.jsonl (one item per episode: id = episode id, group = arm, labels = the
                                           latest measure@ labels, Docent link, the whole conversation as one call,
                                           the sandbox files at the start as the env), index.jsonl, envs/
    data/runs/env_<env>/by_arm/expview/<arm>/   every run's episodes of one arm (item id = <run>/<episode id>, grouped
                                           by target model), so one link shows the whole arm
    HF runs/env_<env>/...                  with --upload; dataset ids runs/env_<env>/<run> and runs/env_<env>/by_arm/<arm>

    uv run python -m environments expview ENV/RUN|ENV [...] [--upload]     # ENV alone = every run of it + the by-arm views

The env page (page.html) links a run's expview dataset and each episode's expview item once the export exists
(spec.expview_dataset). Re-exporting rewrites items; env snapshots are content-addressed.
"""

import json
import sys
from pathlib import Path

from .core import load_env, slug
from .index import latest_labels
from .runfolder import LOCAL, REPO, REPO_ROOT, hf_token, read_episodes

EXPVIEW_ROOT = REPO_ROOT / "data" / "runs"


def out_dir(env: str, run: str) -> Path:
    return EXPVIEW_ROOT / f"env_{env}" / run / "expview"


def _expview():
    sys.path.insert(0, str(REPO_ROOT / "expview"))
    import expview  # ty: ignore[unresolved-import]

    return expview


def _add(ev, ep: dict, run_dir: Path, labels: dict, target: dict, id: str, group: str):
    blobs = {p: run_dir / "blobs" / h for p, h in (ep.get("files_before") or {}).items()}
    files = {p: b.read_text(errors="replace") for p, b in blobs.items() if b.exists()}
    vals = {k: v for k, v in (labels.get(ep["id"]) or {}).items() if v is None or isinstance(v, (bool, int, float, str))}
    model = target.get("model") or ep.get("target")
    ev.add(
        id=id,
        group=group,
        labels=vals or ({"error": True} if ep.get("error") else {}),
        calls=[_expview().call("subject", model, messages=ep.get("messages") or [], tools=ep.get("tools"))],
        env_files=files or None,
        links={"Docent": ep["docent_url"]} if ep.get("docent_url") else None,
        extra={"finish": ep.get("finish"), "item": ep.get("item"), "sample": ep.get("sample"), "error": ep.get("error")},
    )


def export(env: str, run: str) -> Path:
    Run = _expview().Run
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
        _add(ev, ep, run_dir, labels, target, ep["id"], ep["arm"])
    return out_dir(env, run)


def arms_dir(env: str) -> Path:
    return EXPVIEW_ROOT / f"env_{env}" / "by_arm" / "expview"


def export_arms(env: str) -> Path:
    """One expview dataset per arm with every run's episodes of it, grouped by target model (and thinking mode)."""
    Run = _expview().Run
    mod = load_env(env)
    views = {}
    for run_dir in sorted(p for p in (LOCAL / env).glob("*") if (p / "meta.json").exists()):
        meta = json.loads((run_dir / "meta.json").read_text())
        _, labels = latest_labels(run_dir)
        target = meta.get("target") or {}
        model = f"{target.get('id')}" + {False: " · thinking off", True: " · thinking on"}.get(target.get("thinking"), "")
        for ep in read_episodes(run_dir):
            arm = ep["arm"]
            if arm not in views:
                views[arm] = Run(
                    arms_dir(env) / slug(arm),
                    title=f"{env}: arm {arm}, every run",
                    description=f"Every episode of arm `{arm}` across the runs of `envs/{env}/`, grouped by target model. Item ids are "
                    f"`<run>/<episode id>`; labels are `measure()` ({mod.PRIMARY} is the primary).",
                    label_order=[mod.PRIMARY],
                )
                views[arm].reset()
            _add(views[arm], ep, run_dir, labels, target, f"{run_dir.name}/{ep['id']}", model)
    return arms_dir(env)


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
        if not one:
            outs.append(export_arms(env))
            print(outs[-1], flush=True)
    if up and outs:
        print("uploaded", upload(outs)[:10])
