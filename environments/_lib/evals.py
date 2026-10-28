"""cfbench evals of an env, for the env viewer's Evals tab: envs/<env>/evals.json.

    uv run python -m environments evals [ENV ...] [--pull] [--upload]

Every cfbench run (data/runs/cfbench/<run>/) whose env spec is `env_runs:<env>/...` is one eval of that env. Per run:
slug, date, harness version, predictors, conditions, the target (the env viewer's model key) and its cases. A case is
base arm -> cf arm (named after the env's PAIRS entry when it is one, either way round), with the k/n the run scored
against in both arms, every draw's predicted p per condition (with its Docent run and expview item) and the RMSE per
condition next to the "edit does nothing" null (cfbench.report.data). No conversations: those are in expview.

Each run's predictor conversations are exported to data/runs/cfbench/<run>/expview/ (one item per draw: the exact
prompt, tools, every tool call and result, the submit), so the page can link every point to what the investigator saw.

--pull fetches the cfbench meta.json files missing locally from HF, then predictions.jsonl + docent.json of the runs
that name an env. --upload ships evals.json (envs/<env>/) and the expview exports (runs/cfbench/<run>/expview/) to HF.
`index` rebuilds evals.json from whatever is local.
"""

import ast
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .backends import canonical_id
from .core import env_names, load_env
from .runfolder import LOCAL, REPO, REPO_ROOT, hf_token

CFBENCH = REPO_ROOT / "data" / "runs" / "cfbench"
CASE_KEYS = ("base_arm", "cf_arm", "k_base", "n_base", "k_cf", "n_cf")


def env_spec(meta: dict) -> tuple[str, list[str]] | None:
    """`env_runs:<env>/<run>[+<run>...][@...]` -> (env, runs)."""
    m = re.match(r"env_runs:([^/]+)/([^@]+)", meta.get("params", {}).get("env", ""))
    return (m[1], m[2].split("+")) if m else None


def cfbench_runs(env: str) -> list[Path]:
    out = []
    for p in sorted(CFBENCH.glob("*/meta.json")):
        spec = env_spec(json.loads(p.read_text()))
        if spec and spec[0] == env and (p.parent / "predictions.jsonl").exists():
            out.append(p.parent)
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
    return f"{r['base_arm']} -> {r['cf_arm']}/{cond}/{r['predictor'].split('/')[-1]}/d{r['draw']}"


def export_expview(run_dir: Path, rows: list[dict], docent: dict, case_of) -> Path:
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
        label_order=["condition", "p", "truth", "base rate"],
    )
    run.reset()
    cid = docent.get("collection_id")
    for i, r in enumerate(rows):
        if not r.get("messages"):
            continue
        cond, did = canon_condition(r["condition"]), (docent.get("runs") or [None] * len(rows))[i]
        pair = case_of(r["base_arm"], r["cf_arm"])
        run.add(
            id=item_id(r, cond),
            group=f"{r['base_arm']} -> {r['cf_arm']} · {cond}",
            labels={
                "condition": cond,
                "p": r.get("p"),
                "truth": round(r["k_cf"] / r["n_cf"], 4),
                "base rate": round(r["k_base"] / r["n_base"], 4),
                "valid": r.get("valid", True),
                **({"pair": pair[0]} if pair else {}),
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
            extra={k: r[k] for k in (*CASE_KEYS, "draw", "tool_calls", "turns", "shown") if k in r},
        )
    return run.path


def run_evals(run_dir: Path, mod, env_index: dict, docent_filters: dict, export=True) -> dict:
    from cfbench.core import canon_condition
    from cfbench.report.data import Case

    meta = json.loads((run_dir / "meta.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    docent = json.loads((run_dir / "docent.json").read_text()) if (run_dir / "docent.json").exists() else {}
    pairs = getattr(mod, "PAIRS", {})

    def case_of(b, c):
        for name, (x, y) in pairs.items():
            if (b, c) in ((x, y), (y, x)):
                return name, (b, c) == (y, x)
        return None

    # the env viewer's model key (target · thinking) of the env runs this eval read
    _, env_runs = env_spec(meta) or (None, [])
    er = next((r for r in env_index.get("runs", []) if r["run"] in env_runs), {})
    target = er.get("target") or canonical_id(meta["params"]["model"]) or meta["params"]["model"]
    target += " · thinking off" if er.get("thinking") is False else " · thinking on" if er.get("thinking") is True else ""

    cases: dict[tuple, dict] = {}
    for i, r in enumerate(rows):
        if r.get("task", "rate") != "rate":
            continue
        cond = canon_condition(r["condition"])
        key = (r["base_arm"], r["cf_arm"])
        c = cases.setdefault(key, {"base": key[0], "cf": key[1], **{k: r[k] for k in CASE_KEYS[2:]}, "draws": {}})
        c["draws"].setdefault(cond, []).append(
            {
                "p": r.get("p") if r.get("valid", True) else None,
                "draw": r["draw"],
                "predictor": r["predictor"],
                "calls": sum((r.get("tool_calls") or {}).values()),
                "cost": r.get("cost"),
                "docent": (docent.get("runs") or [None] * len(rows))[i],
                "ev": item_id(r, cond) if r.get("messages") else None,
            }
        )
    out_cases, sq, sq_null = [], defaultdict(list), []
    for c in cases.values():
        for ds in c["draws"].values():
            ds.sort(key=lambda d: (d["predictor"], d["draw"]))
        case = Case(mod.__name__, c["base"], c["cf"], c["k_base"], c["n_base"], c["k_cf"], c["n_cf"])
        case.p = {k: [d["p"] for d in ds if d["p"] is not None] for k, ds in c["draws"].items()}
        c["rmse"] = {k: case.mse(k) ** 0.5 for k, ps in case.p.items() if ps}
        c["rmse_null"] = case.mse_null() ** 0.5
        for k, ps in case.p.items():
            if ps:
                sq[k].append(case.mse(k))
        sq_null.append(case.mse_null())
        pc = case_of(c["base"], c["cf"])
        c["pair"], c["reverse"] = pc if pc else (None, False)
        c["docent_arms"] = {a: docent_filters.get(a + "\u0000" + target, [None])[0] for a in (c["base"], c["cf"])}
        out_cases.append(c)
    order = list(pairs)
    out_cases.sort(key=lambda c: (order.index(c["pair"]) if c["pair"] in order else len(order), c["reverse"], c["base"], c["cf"]))
    if export and any(r.get("messages") for r in rows):
        export_expview(run_dir, rows, docent, case_of)
    has_ev = (run_dir / "expview" / "index.jsonl").exists()
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(.*)", run_dir.name)
    return {
        "run": run_dir.name,
        "date": m[1] if m else None,
        "slug": meta.get("slug") or (m[2] if m else run_dir.name),
        "description": meta.get("description"),
        "status": meta.get("status"),
        "harness_version": meta["params"].get("harness_version", 1),
        "predictors": meta["params"].get("predictors", "").split(","),
        "conditions": sorted({k for c in out_cases for k in c["draws"]}),
        "effort": meta["params"].get("effort"),
        "max_calls": meta["params"].get("max_calls"),
        "target": target,
        "env_runs": env_runs,
        "docent_collection": docent.get("collection_id"),
        "expview": f"runs/cfbench/{run_dir.name}" if has_ev else None,
        # mean over cases of the per-case MSE (each case's draws averaged first), as cfbench.report.data.rmse
        "rmse": {k: (sum(v) / len(v)) ** 0.5 for k, v in sq.items()},
        "rmse_null": (sum(sq_null) / len(sq_null)) ** 0.5 if sq_null else None,
        "cases": out_cases,
    }


def build(env: str, export=True) -> Path | None:
    """Write data/envs/<env>/evals.json (None, and no file, when no cfbench run names the env)."""
    runs = cfbench_runs(env)
    p = LOCAL / env / "evals.json"
    if not runs:
        return None
    mod = load_env(env)
    idx_p, filt_p = LOCAL / env / "index.json", LOCAL / env / "docent_filters.json"
    idx = json.loads(idx_p.read_text()) if idx_p.exists() else {}
    filters = json.loads(filt_p.read_text()) if filt_p.exists() else {}
    evals = [run_evals(r, mod, idx, filters, export) for r in runs]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"env": env, "runs": sorted(evals, key=lambda e: e["run"], reverse=True)}, indent=1) + "\n")
    return p


def pull(envs: list[str]):
    """Fetch every cfbench meta.json missing locally, then predictions + docent ids of the runs naming one of `envs`."""
    from huggingface_hub import HfApi, hf_hub_download

    api, tok = HfApi(token=hf_token()), hf_token()
    tree = api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="runs/cfbench")
    dirs = [f.path for f in tree if "." not in f.path.rsplit("/", 1)[-1]]
    get = lambda path: hf_hub_download(REPO, path, repo_type="dataset", local_dir=REPO_ROOT / "data", token=tok)

    def meta(d):
        if not (REPO_ROOT / "data" / d / "meta.json").exists():
            try:
                get(d + "/meta.json")
            except Exception:  # noqa: BLE001 (old flat runs have no meta.json)
                return

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(meta, dirs))
    want = []
    for d in dirs:
        m = REPO_ROOT / "data" / d / "meta.json"
        spec = env_spec(json.loads(m.read_text())) if m.exists() else None
        if spec and spec[0] in envs:
            want += [f"{d}/{f}" for f in ("predictions.jsonl", "docent.json") if not (REPO_ROOT / "data" / d / f).exists()]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda f: _try(get, f), want))
    print(f"pulled {len(want)} files for {len([w for w in want if w.endswith('predictions.jsonl')])} runs")


def _try(f, x):
    try:
        return f(x)
    except Exception as e:  # noqa: BLE001
        print("skip", x, type(e).__name__)


def upload(envs: list[str], written: list[Path]):
    from huggingface_hub import CommitOperationAdd, HfApi

    ops = [CommitOperationAdd(path_in_repo="envs/" + str(p.relative_to(LOCAL)), path_or_fileobj=str(p)) for p in written]
    for env in envs:
        for run_dir in cfbench_runs(env):
            for f in sorted((run_dir / "expview").rglob("*")) if (run_dir / "expview").is_dir() else []:
                if f.is_file():
                    ops.append(CommitOperationAdd(path_in_repo=str(f.relative_to(REPO_ROOT / "data")), path_or_fileobj=str(f)))
    info = HfApi(token=hf_token()).create_commit(
        repo_id=REPO, repo_type="dataset", operations=ops, commit_message="envs: cfbench evals for " + ", ".join(envs)
    )
    print(f"uploaded {len(ops)} files ({info.oid[:10]})")


def cli(args: list[str]):
    envs = [a for a in args if not a.startswith("--")] or env_names()
    if "--pull" in args:
        pull(envs)
    written = [p for e in envs if (p := build(e))]
    for p in written:
        print(p)
    if "--upload" in args and written:
        upload(envs, written)
