"""cfbench evals for the env viewer: evals/index.json and evals/<name>.json (local data/evals/, HF evals/).

    uv run python -m environments evals [NAME ...] [--pull] [--upload]

An eval is a set of cfbench cases, not an env: it can span several envs, models and pairs, and an env can be in many
evals. Two kinds of cfbench run (data/runs/cfbench/<run>/) belong to an eval:
- `python -m cfbench.eval NAME` runs (meta params `eval`, `envs`): the eval NAME, defined in cfbench/evals/NAME/;
- plain `cfbench.run env_runs:<env>/...` runs: the eval `<env>-pairs`, every arm pair anyone ran on that env.

evals/<name>.json has the eval's title, description (its eval.md), envs, and per run: slug, date, harness version,
predictors, conditions, models, the RMSE per condition next to the "edit does nothing" null (over cases, as
cfbench.report.data), for two-model evals the paired comparison (mean p per model and the AUROC between the two models'
cases), and the cases grouped for charts. A group is (env, version, base arm, cf arm, model), pooling items: the observed
rate before (the pooled base-arm rate when the eval shows fewer runs than it has) and after, every draw's p with its item,
Docent run and expview item, the RMSE per condition, and Docent filters for both arms (envs/<env>/docent_filters.json).
No conversations: each run's are exported to data/runs/cfbench/<run>/expview/ (one item per draw).

--pull fetches the cfbench meta.json files missing locally from HF, then predictions.jsonl + docent.json of the runs in
an eval. --upload ships evals/ and the expview exports. `index` rebuilds evals/ from whatever is local.
"""

import ast
import datetime as dt
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .backends import canonical_id
from .core import load_env
from .runfolder import LOCAL, REPO, REPO_ROOT, hf_token

CFBENCH = REPO_ROOT / "data" / "runs" / "cfbench"
EVALS = REPO_ROOT / "data" / "evals"
EVAL_DIRS = REPO_ROOT / "cfbench" / "evals"
EXTRA = ("env", "model", "item", "base_arm", "cf_arm", "k_base", "n_base", "k_cf", "n_cf", "draw", "tool_calls", "turns", "shown")


def eval_of(meta: dict) -> tuple[str, list[str], list[str]] | None:
    """A cfbench run's (eval name, envs, env runs it read), or None if it belongs to no eval."""
    p = meta.get("params", {})
    if p.get("eval"):
        return p["eval"], p.get("envs") or [p["eval_env"]], []
    m = re.match(r"env_runs:([^/]+)/([^@]+)", p.get("env", ""))
    return (f"{m[1]}-pairs", [m[1]], m[2].split("+")) if m else None


def eval_runs() -> dict[str, list[Path]]:
    out = defaultdict(list)
    for p in sorted(CFBENCH.glob("*/meta.json")):
        e = eval_of(json.loads(p.read_text()))
        if e and (p.parent / "predictions.jsonl").exists() and not p.parent.name.endswith("-smoke"):  # smoke tests aren't evals
            out[e[0]].append(p.parent)
    return out


def _messages(msgs: list[dict]) -> list[dict]:
    """cfbench stored assistant tool_calls as Python reprs of OpenRouter's list; expview wants the list."""
    out = []
    for m in msgs:
        m = dict(m)
        if isinstance(m.get("tool_calls"), str):
            m["tool_calls"] = ast.literal_eval(m["tool_calls"])
        m.pop("reasoning_details", None)
        out.append(m)
    return out


def item_id(r: dict, cond: str) -> str:
    case = f"{r['base_arm']} -> {r['cf_arm']}" + (f"@{r['item']}" if r.get("item") else "")
    return f"{r['model'].split('/')[-1]}/{case}/{cond}/{r['predictor'].split('/')[-1]}/d{r['draw']}"


def export_expview(run_dir: Path, rows: list[dict], docent: dict) -> Path:
    """One expview item per prediction: the predictor's whole conversation as one call."""
    from cfbench.core import canon_condition

    sys.path.insert(0, str(REPO_ROOT / "expview"))
    from expview import Run, call  # ty: ignore[unresolved-import]

    meta = json.loads((run_dir / "meta.json").read_text())
    run = Run(
        run_dir / "expview",
        title=f"cfbench {meta['slug']}: investigator conversations",
        description=f"{meta.get('description') or ''}\n\nOne item per prediction: everything the investigator "
        f"({', '.join(meta['params']['predictors'].split(','))}) was sent, every tool call and result, and its submitted probability. "
        "`p` is the prediction, `truth` the counterfactual arm's observed rate.",
        label_order=["condition", "model", "p", "truth", "base rate"],
    )
    run.reset()
    cid, ids = docent.get("collection_id"), docent.get("runs") or [None] * len(rows)
    for r, did in zip(rows, ids, strict=True):
        if not r.get("messages"):
            continue
        cond = canon_condition(r["condition"])
        run.add(
            id=item_id(r, cond),
            group=f"{r['model'].split('/')[-1]} · {r['base_arm']} -> {r['cf_arm']} · {cond}",
            labels={
                "condition": cond,
                "model": r["model"],
                "p": r.get("p"),
                "truth": round(r["k_cf"] / r["n_cf"], 4),
                "base rate": round(r["k_base"] / r["n_base"], 4),
                "valid": r.get("valid", True),
                **({"item": r["item"]} if r.get("item") else {}),
            },
            links={"Docent": f"https://docent.transluce.org/dashboard/{cid}/agent_run/{did}"} if cid and did else {},
            calls=[
                call(
                    f"investigator ({cond})",
                    r["predictor"],
                    messages=_messages(r["messages"]),
                    tools=r.get("tools"),
                    parsed={"p": r.get("p"), "rationale": r.get("rationale")},
                    params={"effort": meta["params"].get("effort"), "max_calls": meta["params"].get("max_calls")},
                    cost=r.get("cost"),
                )
            ],
            extra={k: r[k] for k in EXTRA if k in r},
        )
    return run.path


class Envs:
    """Per env, loaded once: PAIRS, the run index (for model keys) and Docent arm filters."""

    def __init__(self):
        self.cache = {}

    def __getitem__(self, env: str) -> dict:
        if env not in self.cache:
            idx_p, filt_p = LOCAL / env / "index.json", LOCAL / env / "docent_filters.json"
            self.cache[env] = {
                "pairs": getattr(load_env(env), "PAIRS", {}),
                "index": json.loads(idx_p.read_text()) if idx_p.exists() else {},
                "filters": json.loads(filt_p.read_text()) if filt_p.exists() else {},
            }
        return self.cache[env]

    def target(self, env: str, model: str, arm: str, runs: list[str]) -> str:
        """The env viewer's model key (target · thinking) for `model`'s runs of `arm`: the given runs, else any run."""
        idx = self[env]["index"].get("runs", [])
        r = next((r for r in idx if r["run"] in runs), None) or next(
            (r for r in idx if (r.get("model") or "").lower() == model.lower() and arm in r["arms"]), {}
        )
        t = r.get("target") or canonical_id(model) or model
        return t + (" · thinking off" if r.get("thinking") is False else " · thinking on" if r.get("thinking") is True else "")

    def pair(self, env: str, b: str, c: str) -> tuple[str | None, bool]:
        for name, (x, y) in self[env]["pairs"].items():
            if (b, c) in ((x, y), (y, x)):
                return name, (b, c) == (y, x)
        return None, False


def auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def run_summary(run_dir: Path, envs: Envs, export=True) -> dict:
    from cfbench.core import canon_condition
    from cfbench.report.data import Case

    meta = json.loads((run_dir / "meta.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("task", "rate") == "rate"]
    docent = json.loads((run_dir / "docent.json").read_text()) if (run_dir / "docent.json").exists() else {}
    _, run_envs, env_runs = eval_of(meta) or (None, [], [])
    ids = docent.get("runs") or [None] * len(rows)

    cases: dict[tuple, dict] = {}  # (env, version, model, base, cf, item) -> rows by condition
    for r, did in zip(rows, ids, strict=False):
        env = r.get("env_name") or run_envs[0]
        key = (env, r.get("version"), r["model"], r["base_arm"], r["cf_arm"], r.get("item"))
        c = cases.setdefault(key, {"row": r, "draws": defaultdict(list)})
        cond = canon_condition(r["condition"])
        c["draws"][cond].append(
            {
                "p": r.get("p") if r.get("valid", True) else None,
                "draw": r["draw"],
                "item": r.get("item"),
                "predictor": r["predictor"],
                "calls": sum((r.get("tool_calls") or {}).values()),
                "docent": did,
                "ev": item_id(r, cond) if r.get("messages") else None,
            }
        )
    groups: dict[tuple, dict[str, Any]] = {}
    sq, sq_null = defaultdict(list), []
    for (env, version, model, b, cf, _item), c in cases.items():
        r = c["row"]
        case = Case(env, b, cf, r["k_base"], r["n_base"], r["k_cf"], r["n_cf"])
        case.p = {k: [d["p"] for d in ds if d["p"] is not None] for k, ds in c["draws"].items()}
        g: dict[str, Any] | None = groups.get((env, version, b, cf, model))
        if g is None:
            target = envs.target(env, model, b, env_runs)
            pair, rev = envs.pair(env, b, cf)
            filt = envs[env]["filters"]
            g = groups[env, version, b, cf, model] = {
                "env": env,
                "version": version,
                "model": model,
                "target": target,
                "base": b,
                "cf": cf,
                "pair": pair,
                "reverse": rev,
                "k_base": r.get("k_base_pool", r["k_base"]),
                "n_base": r.get("n_base_pool", r["n_base"]),
                "k_shown": r["k_base"],
                "n_shown": r["n_base"],
                "k_cf": r["k_cf"],
                "n_cf": r["n_cf"],
                "n_cases": 0,
                "draws": defaultdict(list),
                "_mse": defaultdict(list),
                "_null": [],
                "docent_arms": {a: filt.get(a + "\u0000" + target, [None])[0] for a in (b, cf)},
            }
        g["n_cases"] += 1
        for k, ds in c["draws"].items():
            g["draws"][k] += ds
            if case.p[k]:
                g["_mse"][k].append(case.mse(k))
                sq[k].append(case.mse(k))
        g["_null"].append(case.mse_null())
        sq_null.append(case.mse_null())
    out_groups = []
    for g in groups.values():
        g["rmse"] = {k: (sum(v) / len(v)) ** 0.5 for k, v in g.pop("_mse").items()}
        nul = g.pop("_null")
        g["rmse_null"] = (sum(nul) / len(nul)) ** 0.5
        g["draws"] = {k: sorted(ds, key=lambda d: (d["item"] or "", d["predictor"], d["draw"])) for k, ds in g["draws"].items()}
        out_groups.append(g)
    out_groups.sort(key=lambda g: (g["env"], g["version"] or "", g["pair"] or "~", g["reverse"], g["base"], g["cf"], g["model"]))

    # two models on the same (env, version, base, cf): mean p per model and the AUROC between their cases' mean p
    paired = []
    by = defaultdict(dict)
    for (env, version, model, b, cf, item), c in cases.items():
        by[env, version, b, cf].setdefault(model, []).append(c)
    for (env, version, b, cf), per_model in by.items():
        if len(per_model) != 2:
            continue
        (m1, c1), (m2, c2) = sorted(per_model.items(), key=lambda kv: -kv[1][0]["row"]["k_cf"] / kv[1][0]["row"]["n_cf"])
        conds = sorted({k for c in c1 + c2 for k in c["draws"]})
        mean = lambda c, k: sum(ps) / len(ps) if (ps := [d["p"] for d in c["draws"].get(k, []) if d["p"] is not None]) else None
        for k in conds:
            a = [x for c in c1 if (x := mean(c, k)) is not None]
            z = [x for c in c2 if (x := mean(c, k)) is not None]
            paired.append(
                {
                    "env": env,
                    "version": version,
                    "base": b,
                    "cf": cf,
                    "condition": k,
                    "models": [m1, m2],
                    "truth": [c1[0]["row"]["k_cf"] / c1[0]["row"]["n_cf"], c2[0]["row"]["k_cf"] / c2[0]["row"]["n_cf"]],
                    "mean_p": [sum(a) / len(a) if a else None, sum(z) / len(z) if z else None],
                    "n": [len(a), len(z)],
                    "auroc": auroc(a, z),
                }
            )
    if export and any(r.get("messages") for r in rows):
        export_expview(run_dir, rows, docent)
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(.*)", run_dir.name)
    return {
        "run": run_dir.name,
        "date": m[1] if m else None,
        "slug": meta.get("slug") or (m[2] if m else run_dir.name),
        "description": meta.get("description"),
        "status": meta.get("status"),
        "harness_version": meta["params"].get("harness_version", 1),
        "predictors": meta["params"].get("predictors", "").split(","),
        "conditions": sorted({k for g in out_groups for k in g["draws"]}),
        "models": sorted({g["model"] for g in out_groups}),
        "effort": meta["params"].get("effort"),
        "max_calls": meta["params"].get("max_calls"),
        "n_cases": len(cases),
        "docent_collection": docent.get("collection_id"),
        "expview": f"runs/cfbench/{run_dir.name}" if (run_dir / "expview" / "index.jsonl").exists() else None,
        # mean over cases of the per-case MSE (each case's draws averaged first), as cfbench.report.data.rmse
        "rmse": {k: (sum(v) / len(v)) ** 0.5 for k, v in sq.items()},
        "rmse_null": (sum(sq_null) / len(sq_null)) ** 0.5 if sq_null else None,
        "paired": paired,
        "groups": out_groups,
    }


def describe(name: str, envs: list[str]) -> tuple[str, str]:
    """(title, description markdown): an eval folder's docstring + eval.md, or the implicit env-pairs eval's."""
    d = EVAL_DIRS / name
    if (d / "eval.py").exists():
        doc = (d / "eval.py").read_text().split('"""')[1].strip()
        title, _, rest = doc.partition("\n")
        md = (d / "eval.md").read_text() if (d / "eval.md").exists() else ""
        return title.rstrip("."), rest.strip() + (f"\n\n**What the investigator is told about this eval (eval.md):**\n\n{md}" if md else "")
    return f"Arm pairs on {envs[0]}", f"Every `cfbench.run env_runs:{envs[0]}/...` run: base arm -> counterfactual arm, one case per pair."


def build(names: list[str] | None = None, export=True) -> list[Path]:
    """Write data/evals/<name>.json for `names` (default: every eval with a local run) and data/evals/index.json."""
    runs, envs, written = eval_runs(), Envs(), []
    EVALS.mkdir(parents=True, exist_ok=True)
    for name in names or list(runs):
        if name not in runs:
            continue
        summaries = sorted((run_summary(r, envs, export) for r in runs[name]), key=lambda s: s["run"], reverse=True)
        all_envs = sorted({e for r in runs[name] for e in (eval_of(json.loads((r / "meta.json").read_text())) or ("", [], []))[1]})
        title, description = describe(name, all_envs)
        p = EVALS / f"{name}.json"
        p.write_text(json.dumps({"name": name, "title": title, "description": description, "envs": all_envs, "runs": summaries}) + "\n")
        written.append(p)
    index = []
    for p in sorted(EVALS.glob("*.json")):
        if p.name == "index.json":
            continue
        e = json.loads(p.read_text())
        rs = e["runs"]
        index.append(
            {
                "name": e["name"],
                "title": e["title"],
                "envs": e["envs"],
                "n_runs": len(rs),
                "latest": max((r["date"] or "" for r in rs), default=""),
                "models": sorted({m for r in rs for m in r["models"]}),
                "conditions": sorted({c for r in rs for c in r["conditions"]}),
                "n_cases": sum(r["n_cases"] for r in rs),
            }
        )
    p = EVALS / "index.json"
    p.write_text(json.dumps({"generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "evals": index}, indent=1) + "\n")
    return [*written, p]


def pull():
    """Fetch every cfbench meta.json missing locally, then predictions + docent ids of the runs that belong to an eval."""
    from huggingface_hub import HfApi, hf_hub_download

    tok = hf_token()
    tree = HfApi(token=tok).list_repo_tree(REPO, repo_type="dataset", path_in_repo="runs/cfbench")
    dirs = [f.path for f in tree if "." not in f.path.rsplit("/", 1)[-1]]
    get = lambda path: hf_hub_download(REPO, path, repo_type="dataset", local_dir=REPO_ROOT / "data", token=tok)

    def meta(d):
        if not (REPO_ROOT / "data" / d / "meta.json").exists():
            _try(get, d + "/meta.json", quiet=True)  # old flat runs have none

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(meta, dirs))
    want = []
    for d in dirs:
        m = REPO_ROOT / "data" / d / "meta.json"
        if m.exists() and eval_of(json.loads(m.read_text())):
            want += [f"{d}/{f}" for f in ("predictions.jsonl", "docent.json") if not (REPO_ROOT / "data" / d / f).exists()]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda f: _try(get, f), want))
    print(f"pulled {len(want)} files")


def _try(f, x, quiet=False):
    try:
        return f(x)
    except Exception as e:  # noqa: BLE001
        if not quiet:
            print("skip", x, type(e).__name__)


def upload(names: list[str], written: list[Path]):
    from huggingface_hub import CommitOperationAdd, HfApi

    data = REPO_ROOT / "data"
    ops = [CommitOperationAdd(path_in_repo=str(p.relative_to(data)), path_or_fileobj=str(p)) for p in written]
    runs = eval_runs()
    for name in names:
        for run_dir in runs.get(name, []):
            for f in sorted((run_dir / "expview").rglob("*")) if (run_dir / "expview").is_dir() else []:
                if f.is_file():
                    ops.append(CommitOperationAdd(path_in_repo=str(f.relative_to(data)), path_or_fileobj=str(f)))
    info = HfApi(token=hf_token()).create_commit(
        repo_id=REPO, repo_type="dataset", operations=ops, commit_message="evals: " + ", ".join(names)
    )
    print(f"uploaded {len(ops)} files ({info.oid[:10]})")


def cli(args: list[str]):
    if "--pull" in args:
        pull()
    names = [a for a in args if not a.startswith("--")] or list(eval_runs())
    written = build(names)
    for p in written:
        print(p)
    if "--upload" in args:
        upload(names, written)
