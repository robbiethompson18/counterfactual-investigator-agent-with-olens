"""What every arm sends, built ahead of time so the static page can show and diff arms without running Python:

    envs/<env>/spec/env.json            the env's declarations (arms, factors, items per arm, pairs, comparisons, primary,
                                        requires, caps, defaults), its README, the source files it is built from (paths
                                        in the repo, readable on GitHub at `git_sha`) and a hash of them
    envs/<env>/spec/arms/<arm>.json     build(arm, item) for every item of the arm: {"items": {item: episode}, "blobs":
                                        {sha1: text}}; strings over 256 chars are stored once per arm as {"$b": sha1},
                                        and strings over 8 KB once per env as {"$f": sha1} -> spec/blobs/<sha1>.txt
                                        (overeagerness repeats MBs of sandbox logs in every arm)
    envs/<env>/links.json               per run: episode id -> [Docent agent_run id, primary value], so the page links a
                                        comparison's rollouts without reading episode shards; plus which episodes have O-lens reads

`index` rebuilds the spec when the env's source hash changed (env folder + environments/_lib), so it costs nothing
after a run. `spec --out DIR` builds every env's spec from the code alone into DIR/<env>/ plus DIR/index.json (the env
list with title, status, STARRED, primary): the GitHub Pages workflow does this on every push to main, so the hosted
page's arms, comparisons and stars always match main, and only runs and links come from HF.
Builds use sample 0 and the date the spec was made; the sample's rng and the date can change a prompt.
"""

import dataclasses
import datetime as dt
import hashlib
import json
from pathlib import Path

from .backends import canonical_id
from .core import ROOT, Ctx, env_names, load_env, retired_run_arms, run_dirs, slug
from .runfolder import LOCAL, REPO_ROOT, git_sha

BLOB_MIN, FILE_MIN = 256, 8192
SOURCE_SUFFIXES = {".py", ".txt", ".json", ".jsonl", ".md", ".yaml", ".yml", ".toml", ".html", ".sh"}


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


def build_spec(env: str, force: bool = False, out: Path | None = None) -> list[Path]:
    """Write envs/<env>/spec/ (or `out`) unless its source hash is unchanged. -> written paths (empty when up to date)."""
    out = out or LOCAL / env / "spec"
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
        "starred": getattr(mod, "STARRED", None),
        "title": title_status(env)[0],
        "status": title_status(env)[1],
        "build_errors": errors,
    }
    head.write_text(json.dumps(spec, indent=1, default=str) + "\n")
    return [*written, head]


def title_status(env: str) -> tuple[str, str | None]:
    """The env's README title and **Status** tag (as index.py reads them)."""
    import re

    readme = (ROOT / env / "README.md").read_text() if (ROOT / env / "README.md").exists() else ""
    title = next((line.lstrip("# ").strip() for line in readme.splitlines() if line.startswith("# ")), env)
    status = re.search(r"\*\*Status[^*]*\*\*\s*([A-Z]+)", readme)
    return title, status.group(1) if status else None


def build_site(out: Path, only: list[str] | None = None) -> Path:
    """Every env's spec from the code alone into out/<env>/, out/index.json listing them, and out/evals.json (every
    eval the code defines, evals.defined()). An env whose module doesn't load (e.g. data files not in git) gets
    {"env", "error"} so the page falls back to the spec on HF. With `only`, just those envs are rebuilt and the others
    keep their entries from the last build."""
    import shutil
    import traceback

    old = {}
    if only and (out / "index.json").exists():
        old = {x["env"]: x for x in json.loads((out / "index.json").read_text())["envs"]}
    envs = []
    for e in env_names():
        if only and e not in only and e in old:
            envs.append(old[e])
            continue
        d = out / e
        shutil.rmtree(d, ignore_errors=True)
        try:
            build_spec(e, force=True, out=d)
            spec = json.loads((d / "env.json").read_text())
            keep = ("env", "title", "status", "starred", "primary", "git_sha")
            envs.append({k: spec.get(k) for k in keep} | {"n_arms": len(spec["arms"])})
        except Exception as ex:  # noqa: BLE001
            d.mkdir(parents=True, exist_ok=True)
            err = {"env": e, "error": repr(ex)[:500], "traceback": traceback.format_exc()[-2000:]}
            (d / "env.json").write_text(json.dumps(err, indent=1) + "\n")
            envs.append({"env": e, "error": err["error"]})
        print(e, "error" if "error" in envs[-1] else "ok", flush=True)
    from .evals import defined

    (out / "evals.json").write_text(json.dumps({"evals": defined()}, indent=1) + "\n")
    p = out / "index.json"
    doc = {"generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git_sha": git_sha(), "envs": envs}
    p.write_text(json.dumps(doc, indent=1) + "\n")
    return p


def read_overlays(env: str, runs: set[str]) -> dict:
    """Metamodel-read datasets that cover this env's episodes: every expview run under data/runs/<env>/ whose items say
    {"extra": {"source": "envs/<env>/<run>", "episode": <episode id>}} (the O-lens read exports do). -> {run: {episode:
    [dataset id, item id]}}, preferring items that carry reads (tokens). A source naming a run that doesn't exist is
    matched to the longest existing run name it starts with (an export may tag a subset, e.g. `<run>-pp`)."""
    out: dict = {}
    for items in sorted((REPO_ROOT / "data/runs" / env).glob("*/expview/**/items.jsonl")):
        rel = items.parent.relative_to(REPO_ROOT / "data")
        ds = str(rel).replace("/expview/", "/").removesuffix("/expview")
        with open(items) as f:
            for line in f:
                it = json.loads(line)
                ex = it.get("extra") if isinstance(it.get("extra"), dict) else {}
                src, ep = ex.get("source"), ex.get("episode")
                if not (isinstance(src, str) and src.startswith(f"envs/{env}/") and ep and it.get("tokens")):
                    continue
                run = src.split("/", 2)[2]
                if run not in runs:
                    run = max((r for r in runs if run.startswith(r)), key=len, default=None)
                if run:
                    out.setdefault(run, {}).setdefault(ep, [ds, it["id"]])
    return out


def build_links(env: str) -> Path | None:
    """envs/<env>/links.json from the local run folders (each run's active measure labels, as index.py uses), merged with
    HF's links.json like env_index: runs only HF has are kept, local runs win."""
    from .index import hf_index
    from .runfolder import active_labels, read_episodes

    if not (LOCAL / env).exists():
        return None
    mod = load_env(env)
    runs = {}
    dirs = run_dirs(env, mod, LOCAL)
    reads = read_overlays(env, {p.name for p in dirs})
    for run_dir in dirs:
        meta = json.loads((run_dir / "meta.json").read_text())
        _, labels = active_labels(run_dir)
        eps, skip = {}, retired_run_arms(mod, run_dir.name)
        for ep in read_episodes(run_dir):
            if ep["arm"] in skip:
                continue
            v = None if ep.get("error") else (labels.get(ep["id"]) or {}).get(mod.PRIMARY)
            eps[ep["id"]] = [ep.get("docent_run"), v]
        runs[run_dir.name] = {
            "docent_collection": meta.get("docent_collection"),
            "reads": reads.get(run_dir.name, {}),  # episode -> [expview dataset, item] with metamodel (O-lens) reads
            "target": canonical_id((meta.get("target") or {}).get("id")),
            "thinking": (meta.get("target") or {}).get("thinking"),
            "episodes": eps,
        }
    retired = set(getattr(mod, "RETIRED_RUNS", []))
    runs = {r: v for r, v in hf_index(f"{env}/links.json").get("runs", {}).items() if r not in retired} | runs
    p = LOCAL / env / "links.json"
    p.write_text(json.dumps({"env": env, "runs": runs}, separators=(",", ":")) + "\n")
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
