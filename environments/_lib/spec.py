"""What every arm sends, built ahead of time so the static page can show and diff arms without running Python:

    envs/<env>/spec/env.json            the env's declarations (arms, factors, items per arm, pairs, comparisons, primary,
                                        requires, caps, defaults), its README, the source files it is built from (paths
                                        in the repo, readable on GitHub at `git_sha`) and a hash of them
    envs/<env>/spec/arms/<arm>.json     build(arm, item) for every item of the arm: {"items": {item: episode}, "blobs":
                                        {sha1: text}}; strings over 256 chars are stored once per arm as {"$b": sha1},
                                        and strings over 8 KB once per env as {"$f": sha1} -> spec/blobs/<sha1>.txt
                                        (overeagerness repeats MBs of sandbox logs in every arm)
    envs/<env>/links.json               per run: episode id -> [Docent agent_run id, primary value], so the page links a
                                        comparison's rollouts without reading episode shards; plus the run's expview dataset

`index` rebuilds the spec when the env's source hash changed (env folder + environments/_lib), so it costs nothing
after a run. Builds use sample 0 and the date the spec was made; the sample's rng and the date can change a prompt.
"""

import dataclasses
import datetime as dt
import hashlib
import json
from pathlib import Path

from .core import ROOT, Ctx, env_names, load_env, slug
from .runfolder import LOCAL, REPO_ROOT, git_sha

BLOB_MIN, FILE_MIN = 256, 8192
SOURCE_SUFFIXES = {".py", ".txt", ".json", ".jsonl", ".md", ".yaml", ".yml", ".toml", ".html", ".sh"}
EXPVIEW_BASE = "https://expview.pages.dev/"


def comparisons(mod) -> list[dict]:
    """PAIRS and the env's curated COMPARISONS as one list: {name, arms, items?, why?, kind}. Arms an env doesn't have
    are dropped (a comparison can name arms from a prompt version that isn't merged yet)."""
    out = []
    for name, p in getattr(mod, "PAIRS", {}).items():
        out.append({"name": name, "arms": list(p[:2]), "items": list(p[2]) if len(p) > 2 else None, "kind": "pair"})
    for name, c in getattr(mod, "COMPARISONS", {}).items():
        c = {"arms": c} if isinstance(c, (list, tuple)) else dict(c)
        arms = [a for a in c["arms"] if a in mod.ARMS]
        if len(arms) >= 2:
            out.append({"name": name, "arms": arms, "items": c.get("items"), "why": c.get("why"), "group": c.get("group"), "kind": "set"})
    return out


def retired(mod) -> list[str]:
    """Arms the env's RETIRED entries name (core.retired_arms where it exists; plain arm names otherwise)."""
    from . import core

    if hasattr(core, "retired_arms"):
        return sorted(core.retired_arms(mod))
    return sorted(a for a in getattr(mod, "RETIRED", []) if a in mod.ARMS)


def source_files(env: str) -> list[str]:
    """Repo-relative paths of what build() can read: the env folder and the shared _lib (git-tracked files only matter,
    but untracked ones are listed too so a local page shows what was built)."""
    dirs = [ROOT / env, ROOT / "_lib"]
    files = []
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix in SOURCE_SUFFIXES and "__pycache__" not in p.parts and p.name != "page.html":
                files.append(str(p.relative_to(REPO_ROOT)))
    return files


def source_hash(files: list[str]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode() + b"\0" + (REPO_ROOT / f).read_bytes() + b"\0")
    return h.hexdigest()[:16]


def _intern(x, blobs: dict, files: dict):
    if isinstance(x, str):
        if len(x) < BLOB_MIN:
            return x
        k = hashlib.sha1(x.encode()).hexdigest()[:20]
        if len(x) >= FILE_MIN:
            files[k] = x
            return {"$f": k, "len": len(x)}
        blobs[k] = x
        return {"$b": k}
    if isinstance(x, dict):
        return {k: _intern(v, blobs, files) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_intern(v, blobs, files) for v in x]
    if isinstance(x, bytes):
        return f"<{len(x)} bytes>"
    if x is None or isinstance(x, (bool, int, float)):
        return x
    return str(x)


def items_of(mod, arm) -> list[str]:
    return [str(i) for i in (mod.items_for(arm) if hasattr(mod, "items_for") else mod.items())]


def build_spec(env: str, force: bool = False) -> list[Path]:
    """Write envs/<env>/spec/ unless its source hash is unchanged. -> written paths (empty when up to date)."""
    out = LOCAL / env / "spec"
    files = source_files(env)
    shash = source_hash(files)
    head = out / "env.json"
    if not force and head.exists() and json.loads(head.read_text()).get("source_hash") == shash:
        return []
    mod = load_env(env)
    today = dt.date.today()
    for sub in ("arms", "blobs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old in (out / sub).glob("*"):
            old.unlink()
    written, arms_meta, errors, big = [], {}, {}, {}
    for arm in mod.ARMS:
        blobs, built = {}, {}
        its = items_of(mod, arm)
        for it in its:
            ctx = Ctx(env, arm, it, 0, today, dt.datetime.combine(today, dt.time(12)))
            try:
                ep = dataclasses.asdict(mod.build(arm, it, ctx))
            except Exception as e:  # noqa: BLE001  (shown on the page; one broken arm shouldn't hide the env)
                errors[f"{arm}~{it}"] = repr(e)[:500]
                continue
            built[it] = _intern(ep, blobs, big)
        p = out / "arms" / f"{slug(arm)}.json"
        p.write_text(json.dumps({"arm": arm, "items": built, "blobs": blobs}, separators=(",", ":"), default=str))
        written.append(p)
        arms_meta[arm] = {"file": f"arms/{slug(arm)}.json", "items": its, "bytes": p.stat().st_size}
    for k, text in big.items():
        (out / "blobs" / f"{k}.txt").write_text(text)
        written.append(out / "blobs" / f"{k}.txt")
    readme = (ROOT / env / "README.md").read_text() if (ROOT / env / "README.md").exists() else ""
    sha = git_sha()
    spec = {
        "env": env,
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "built_for": {"today": today.isoformat(), "sample": 0},
        "git_sha": sha,
        "source_hash": shash,
        "source_files": files,
        "doc": (mod.__doc__ or "").strip(),
        "readme": readme,
        "primary": mod.PRIMARY,
        "arms": list(mod.ARMS),
        "arm_factors": getattr(mod, "ARM_FACTORS", {}),
        "retired": retired(mod),
        "arm_specs": arms_meta,
        "comparisons": comparisons(mod),
        "requires": getattr(mod, "REQUIRES", {}),
        "caps": getattr(mod, "CAPS", {}),
        "defaults": getattr(mod, "DEFAULTS", {}),
        "behavior": getattr(mod, "BEHAVIOR", None),
        "build_errors": errors,
    }
    head.write_text(json.dumps(spec, indent=1, default=str) + "\n")
    return [*written, head]


def expview_dataset(env: str, run: str, meta: dict) -> str | None:
    """The run's expview dataset id (runs/<exp>/<run>[/<name>]): its own export (expview_export.py, items are episode
    ids) or else a converted run's old one (items named differently), if it exists locally (data/runs/.../items.jsonl)."""
    cands = [f"runs/env_{env}/{run}/expview"]
    src = meta.get("converted_from")
    for s in [src] if isinstance(src, str) else (src or []):
        cands.append(s.removeprefix("data/") + "/expview")
    for c in cands:
        d = REPO_ROOT / "data" / c
        if (d / "items.jsonl").exists():
            return c.removesuffix("/expview").replace("/expview/", "/")
    return None


def build_links(env: str) -> Path | None:
    """envs/<env>/links.json from the local run folders (labels from the latest measure@ file, as index.py uses)."""
    from .index import latest_labels
    from .runfolder import read_episodes

    if not (LOCAL / env).exists():
        return None
    mod = load_env(env)
    runs = {}
    for run_dir in sorted(p for p in (LOCAL / env).glob("*") if (p / "meta.json").exists()):
        meta = json.loads((run_dir / "meta.json").read_text())
        _, labels = latest_labels(run_dir)
        eps = {}
        for ep in read_episodes(run_dir):
            v = None if ep.get("error") else (labels.get(ep["id"]) or {}).get(mod.PRIMARY)
            eps[ep["id"]] = [ep.get("docent_run"), v]
        runs[run_dir.name] = {
            "docent_collection": meta.get("docent_collection"),
            "expview": expview_dataset(env, run_dir.name, meta),
            "expview_items": (REPO_ROOT / f"data/runs/env_{env}/{run_dir.name}/expview/items.jsonl").exists(),
            "target": (meta.get("target") or {}).get("id"),
            "thinking": (meta.get("target") or {}).get("thinking"),
            "episodes": eps,
        }
    p = LOCAL / env / "links.json"
    p.write_text(json.dumps({"env": env, "expview_base": EXPVIEW_BASE, "runs": runs}, separators=(",", ":")) + "\n")
    return p


def build(envs: list[str] | None = None, force: bool = False) -> list[Path]:
    """Spec (rebuilt only if the source changed) and links for each env. -> each env's spec/ dir (whole, so an upload
    always carries every arm) and links.json."""
    out = []
    for e in envs or env_names():
        rebuilt = build_spec(e, force)
        print(f"{e}: spec {'rebuilt' if rebuilt else 'up to date'}", flush=True)
        out.append(LOCAL / e / "spec")
        p = build_links(e)
        if p:
            out.append(p)
    return out
