"""Generated indexes, so a browser never crawls HF (the tree API allows 500 calls / 5 min):

    envs/index.json         every env: title, status, primary measure, runs and episodes per target model
    envs/<env>/index.json   that env's runs: meta summary, label files, and per arm n / errors / unscorable / the
                            primary measure's distribution, with per-item k/n for a bool primary
    envs/<env>/evals.json   the cfbench runs on that env (evals.py), for the viewer's Evals tab

Built from the local data/envs/ folders; delete and rebuild any time.
"""

import datetime as dt
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .backends import canonical_id
from .core import ROOT, env_names, load_env, retired_arms, retired_run_arms, run_dirs
from .runfolder import LOCAL, read_episodes


def latest_labels(run_dir: Path, labeler="measure") -> tuple[str | None, dict]:
    files = sorted((run_dir / "labels").glob(f"{labeler}@*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None, {}
    return files[-1].name, {r["episode"]: r["values"] for r in map(json.loads, files[-1].read_text().splitlines())}


def summarize(vals: list, items: list) -> dict:
    present = [v for v in vals if v is not None]
    if all(isinstance(v, bool) for v in present):
        per_item = defaultdict(lambda: [0, 0])
        for v, it in zip(vals, items, strict=True):
            if v is not None:
                per_item[it][0] += int(v)
                per_item[it][1] += 1
        return {"type": "bool", "k": sum(map(int, present)), "n": len(present), "items": dict(per_item)}
    if all(isinstance(v, (int, float)) for v in present):
        return {"type": "number", "n": len(present), "mean": sum(present) / len(present) if present else None}
    return {"type": "category", "n": len(present), "counts": dict(Counter(map(str, present)))}


def env_index(env: str) -> dict:
    mod = load_env(env)
    runs = []
    for run_dir in run_dirs(env, mod, LOCAL):
        meta = json.loads((run_dir / "meta.json").read_text())
        label_file, labels = latest_labels(run_dir)
        by_arm, skip = defaultdict(list), retired_run_arms(mod, run_dir.name)
        for ep in read_episodes(run_dir):
            if ep["arm"] not in skip:
                by_arm[ep["arm"]].append(ep)
        arms = {}
        for arm, eps in by_arm.items():
            vals = [labels.get(e["id"], {}).get(mod.PRIMARY) if not e.get("error") else None for e in eps]
            arms[arm] = {
                "n": len(eps),
                "errors": sum(1 for e in eps if e.get("error")),
                "unscorable": sum(1 for e, v in zip(eps, vals, strict=True) if v is None and not e.get("error")),
                "primary": summarize(vals, [str(e["item"]) for e in eps]),
            }
        measures = sorted({k for v in labels.values() if v for k in v})
        runs.append(
            {
                "run": run_dir.name,
                "status": meta.get("status"),
                "created": meta.get("created"),
                "target": canonical_id(meta.get("target", {}).get("id")),
                "model": meta.get("target", {}).get("model"),
                "thinking": meta.get("target", {}).get("thinking"),
                "description": meta.get("description"),
                "converted_from": meta.get("converted_from"),
                "docent_collection": meta.get("docent_collection"),
                "n_episodes": sum(a["n"] for a in arms.values()) if skip else meta.get("n_episodes"),
                "label_files": sorted(p.name for p in (run_dir / "labels").glob("*.jsonl")),
                "primary_from": label_file,
                "measures": measures,
                "arms": arms,
            }
        )
    readme = (ROOT / env / "README.md").read_text() if (ROOT / env / "README.md").exists() else ""
    title = next((line.lstrip("# ").strip() for line in readme.splitlines() if line.startswith("# ")), env)
    status = re.search(r"\*\*Status[^*]*\*\*\s*([A-Z]+)", readme)
    return {
        "env": env,
        "title": title,
        "status": status.group(1) if status else None,
        "starred": getattr(mod, "STARRED", None),
        "primary": mod.PRIMARY,
        "pairs": {k: list(v) for k, v in getattr(mod, "PAIRS", {}).items()},
        "arms": list(mod.ARMS),
        "arm_factors": getattr(mod, "ARM_FACTORS", {}),
        "retired": sorted(retired_arms(mod)),
        "requires": getattr(mod, "REQUIRES", {}),
        "runs": runs,
    }


def build(envs: list[str] | None = None) -> list[Path]:
    """Rebuild envs/<env>/index.json for `envs` (default every env) and envs/index.json, plus each env's spec/ and links.json
    (spec.py; the spec only when the env's source changed). -> written paths."""
    from .evals import build as build_evals
    from .spec import build as build_spec

    written = []
    names = envs or env_names()
    for e in names:
        idx = env_index(e)
        p = LOCAL / e / "index.json"
        p.write_text(json.dumps(idx, indent=1) + "\n")
        written.append(p)
    written += build_spec(names)
    written += [p for e in names if (p := build_evals(e, export=False))]
    top = []
    for e in env_names():
        p = LOCAL / e / "index.json"
        if not p.exists():
            continue
        idx = json.loads(p.read_text())
        per_target = Counter()
        for r in idx["runs"]:
            per_target[r["target"]] += r["n_episodes"] or 0
        top.append(
            {
                "env": e,
                "title": idx["title"],
                "status": idx["status"],
                "starred": idx.get("starred"),
                "primary": idx["primary"],
                "n_runs": len(idx["runs"]),
                "n_episodes": sum(per_target.values()),
                "targets": dict(per_target),
            }
        )
    p = LOCAL / "index.json"
    p.write_text(json.dumps({"generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "envs": top}, indent=1) + "\n")
    written.append(p)
    return written
