"""cfbench evals for the env viewer: evals/index.json and evals/<name>.json (local data/evals/, HF evals/).

    uv run python -m environments evals [NAME ...] [--pull] [--upload]

An eval is a set of cfbench cases, not an env: it can span several envs, models and pairs, and an env can be in many
evals. Two kinds of cfbench run (data/runs/cfbench/<run>/) belong to an eval:
- `python -m cfbench.eval NAME` runs (meta params `eval`, `envs`): the eval NAME, defined in cfbench/evals/NAME/;
- plain `cfbench.run env_runs:<env>/...` runs: the eval `<env>-pairs`, every arm pair anyone ran on that env.

evals/<name>.json has the eval's title, description (its eval.md), envs, and per run: slug, date, harness version,
predictors, conditions, models, the RMSE per condition next to the "edit does nothing" null (over cases, as
cfbench.scoring) with 95% bootstrap CIs and paired differences between conditions (`boot`, resampling arm pairs), for
two-model evals the paired comparison (mean p per model and the AUROC between the two models' cases), and the cases
grouped for charts. A group is (env, version, base arm, cf arm, model), pooling items: the observed rate before (the
pooled base-arm rate when the eval shows fewer runs than it has) and after, every draw's p with its item, Docent run and
expview item, the RMSE per condition, and Docent filters for both arms (envs/<env>/docent_filters.json). No
conversations: each run's are exported to data/runs/cfbench/<run>/expview/ (one item per draw).

--pull fetches the cfbench meta.json files missing locally from HF, then predictions.jsonl + docent.json of the runs in
an eval. --upload ships evals/ and the expview exports. `index` rebuilds evals/ from whatever is local.
"""

import ast
import datetime as dt
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .backends import canonical_id
from .core import load_env
from .index import hf_index
from .runfolder import LOCAL, REPO, REPO_ROOT, hf_token

CFBENCH = REPO_ROOT / "data" / "runs" / "cfbench"
EVALS = REPO_ROOT / "data" / "evals"
EVAL_DIRS = REPO_ROOT / "cfbench" / "evals"
# envs with an `<env>-pairs` eval (every cfbench.run env_runs:<env>/... run): the viewer lists the evals defined here
PAIRS_ENVS = ("alignment_faking", "hint_mcq", "hinted_truthfulqa")
# evals set aside: off the viewer's eval lists (their own pages still open by URL); folders and HF files stay
RETIRED_EVALS = {"af_which", "af_refusals"}
# evalcue runs of envs since ported to environments/ under another name (the deleted cfbench/envs/evalcue_runs.py PORTS)
PORTS = {"hint_cf_wrong_mcq": "hint_mcq"}
EXTRA = ("env", "model", "item", "base_arm", "cf_arm", "k_base", "n_base", "k_cf", "n_cf", "draw", "tool_calls", "turns", "shown")


def eval_of(meta: dict) -> tuple[str, list[str], list[str]] | None:
    """A cfbench run's (eval name, envs, env runs it read), or None if it belongs to no eval."""
    p = meta.get("params", {})
    if p.get("eval"):
        return p["eval"], p.get("envs") or [p["eval_env"]], []
    m = re.match(r"(env|evalcue)_runs:([^/]+)/([^@]+)", p.get("env", ""))
    if not m:
        return None
    env = PORTS.get(m[2], m[2]) if m[1] == "evalcue" else m[2]
    return f"{env}-pairs", [env], m[3].split("+")


def eval_runs() -> dict[str, list[Path]]:
    """Eval name -> its runs, leaving out runs a later run copied in with --reuse (that run has their rows)."""
    out = defaultdict(list)
    metas = {p: json.loads(p.read_text()) for p in sorted(CFBENCH.glob("*/meta.json"))}
    reused = {Path(r).parent.name for m in metas.values() for r in (m.get("params", {}).get("reuse") or "").split(",") if r}
    for p, meta in metas.items():
        e = eval_of(meta)
        if p.parent.name in reused:
            continue
        # smoke tests and abandoned runs (their reusable rows live on in the run that reused them) aren't shown
        smoke_or_abandoned = p.parent.name.endswith("-smoke") or meta.get("status") == "abandoned"
        if e and (p.parent / "predictions.jsonl").exists() and not smoke_or_abandoned:
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


def seen_tokens(reads, row: dict) -> dict | None:
    """An expview tokens object of the base runs exactly as the investigator's tools gave them: each run's transcript as
    read_transcript renders it, split at the reads its O-lens tools could return there (the inputs' reads for olens_agentic;
    for olens_placebo_agentic the eval's PLACEBO sources, placed as olens.tools places them), every bullet of each read.
    None when the row's runs aren't in `reads`, or for a placebo it can't reproduce (other-env placebo, harness < 12)."""
    from cfbench import olens
    from cfbench.core import canon_condition, render_spans
    from expview import read

    cond, base = canon_condition(row["condition"]), row.get("sources") or row["shown"]
    if not all(s in reads for s in base):
        return None
    if cond == "olens_placebo_agentic" and (not row.get("placebo_sources") or row.get("harness_version", 1) < 12):
        return None
    given = row["placebo_sources"] if cond == "olens_placebo_agentic" else base if cond.startswith("olens") else [None] * len(base)
    toks, sections, out = [], [], {}
    for i, (src, by) in enumerate(zip(base, given, strict=True)):
        messages = reads.messages[src]
        text, start = render_spans(messages)[0], len(toks)
        pos = []
        if by is not None:
            m, rows = reads[by]
            pos = olens.positions(messages, rows if by == src else olens.onto(messages, m, rows))
        # a read sits at the start of the token after it (the tools' ⟨READ⟩ is just before its char)
        spans = list(itertools.pairwise(sorted({0, len(text), *(min(len(text), p.char) for p in pos)}))) or [(0, 0)]
        first = {a: start + k for k, (a, _) in enumerate(spans)}
        toks += [text[a:b] for a, b in spans]
        for p in pos:
            k = first.get(min(len(text), p.char), len(toks) - 1)
            for layer, r in sorted(p.layers.items()):
                lines = [[b["text"]] for b in r["bullets"]]
                out.setdefault(str(k), []).append(read("O-lens", lines, label=f"{p.where} · token {json.dumps(p.token)} · layer {layer}"))
        sections.append([f"run {i}" + (f" (model {'AB'[i]})" if len(base) == 2 else ""), start, len(toks)])
    return {"tokens": toks, "sections": sections, "reads": out}


def export_expview(run_dir: Path, rows: list[dict], docent: dict) -> Path:
    """One expview item per prediction: the predictor's whole conversation as one call, and (where the eval's inputs have
    them) the base runs with the reads its tools could return as the item's tokens (seen_tokens)."""
    from cfbench.core import canon_condition

    sys.path.insert(0, str(REPO_ROOT / "expview"))
    from expview import Run, call

    meta = json.loads((run_dir / "meta.json").read_text())
    run = Run(
        run_dir / "expview",
        title=f"cfbench {meta['slug']}: investigator conversations",
        description=f"{meta.get('description') or ''}\n\nOne item per prediction: everything the investigator "
        f"({', '.join(meta['params']['predictors'].split(','))}) was sent, every tool call and result, and its submission. "
        "Rate task: `p` is the prediction, `truth` the counterfactual arm's observed rate. Answer task: `answer` is the "
        "submitted answer (canonical), `truth` the counterfactual arm's most common answer.",
        label_order=["condition", "model", "p", "answer", "truth", "correct", "base rate"],
    )
    run.reset()
    reads = None
    params, ev = meta["params"], _eval_module(meta["params"].get("eval"))
    if ev is not None and hasattr(ev, "reads") and params.get("inputs"):
        from cfbench import olens

        reads = olens.Reads(ev.reads(REPO_ROOT / "data" / "runs" / params["inputs"]))
    cid, ids = docent.get("collection_id"), docent.get("runs") or [None] * len(rows)
    for r, did in zip(rows, ids, strict=True):
        if not r.get("messages"):
            continue
        cond = canon_condition(r["condition"])
        if r.get("task") == "answer":
            labels = {"condition": cond, "model": r["model"], "answer": r.get("canon"), "truth": r["modal_cf"], "correct": r.get("correct")}
            parsed = {"answer": r.get("answer"), "rationale": r.get("rationale")}
        else:
            labels = {"condition": cond, "model": r["model"], "p": r.get("p"), "truth": round(r["k_cf"] / r["n_cf"], 4)}
            labels["base rate"] = round(r["k_base"] / r["n_base"], 4)
            parsed = {"p": r.get("p"), "rationale": r.get("rationale")}
        run.add(
            id=item_id(r, cond),
            group=f"{r['model'].split('/')[-1]} · {r['base_arm']} -> {r['cf_arm']} · {cond}",
            labels=labels | {"valid": r.get("valid", True), **({"item": r["item"]} if r.get("item") else {})},
            links={"Docent": f"https://docent.transluce.org/dashboard/{cid}/agent_run/{did}"} if cid and did else {},
            calls=[
                call(
                    f"investigator ({cond})",
                    r["predictor"],
                    messages=_messages(r["messages"]),
                    tools=r.get("tools"),
                    parsed=parsed,
                    params={"effort": meta["params"].get("effort"), "max_calls": meta["params"].get("max_calls")},
                    cost=r.get("cost"),
                )
            ],
            extra={k: r[k] for k in EXTRA if k in r},
            tokens=seen_tokens(reads, r) if reads is not None else None,
        )
    return run.path


class Envs:
    """Per env, loaded once: PAIRS, the run index (for model keys) and Docent arm filters."""

    def __init__(self):
        self.cache = {}

    def __getitem__(self, env: str) -> dict:
        if env not in self.cache:
            idx_p, filt_p, links_p = LOCAL / env / "index.json", LOCAL / env / "docent_filters.json", LOCAL / env / "links.json"
            try:
                pairs = getattr(load_env(env), "PAIRS", {})
            except ModuleNotFoundError:  # a cfbench run over an old evalcue env (code deleted 2026-10-27) has no port
                pairs = {}
            self.cache[env] = {
                "pairs": pairs,
                "index": json.loads(idx_p.read_text()) if idx_p.exists() else {},
                "filters": json.loads(filt_p.read_text()) if filt_p.exists() else {},
                "links": json.loads(links_p.read_text()).get("runs", {}) if links_p.exists() else {},
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

    def docent(self, env: str, run: str, episode: str) -> str | None:
        """An env episode's Docent URL (envs/<env>/links.json)."""
        r = self[env]["links"].get(run) or {}
        did = (r.get("episodes", {}).get(episode) or [None])[0]
        if not did or not r.get("docent_collection"):
            return None
        return f"https://docent.transluce.org/dashboard/{r['docent_collection']}/agent_run/{did}"

    def pair(self, env: str, b: str, c: str) -> tuple[str | None, bool]:
        for name, (x, y, *_) in self[env]["pairs"].items():  # a pair can carry its items
            if (b, c) in ((x, y), (y, x)):
                return name, (b, c) == (y, x)
        return None, False


def _eval_module(name: str | None):
    if not name or not (EVAL_DIRS / name / "eval.py").exists():
        return None
    import importlib

    return importlib.import_module(f"cfbench.evals.{name}.eval")


def auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


CONDITION_ORDER = ("no_transcript", "transcript_agentic", "olens_agentic", "olens_placebo_agentic")


def rmse_ci(clusters: dict[tuple, dict[str, list[float]]], n_boot: int = 2000, seed: int = 0) -> dict:
    """95% bootstrap CIs for each condition's RMSE, the null's ("null"), and every paired difference (a - b over the same
    resample), resampling clusters: an arm pair (env, version, base, cf, model) with all its cases (items share a truth, so
    they aren't independent). `clusters` maps a cluster to {condition: [per-case MSE]} with "null" among the conditions;
    the RMSE pools the sampled clusters' cases, as the point estimate pools every case."""
    import random

    keys = list(clusters)
    order = lambda k: (k != "null", CONDITION_ORDER.index(k) if k in CONDITION_ORDER else 99, k)
    conds = sorted({k for c in clusters.values() for k in c}, key=order)

    def rmse(ks: list) -> dict[str, float]:
        out = {}
        for c in conds:
            v = [x for k in ks for x in clusters[k].get(c, [])]
            if v:
                out[c] = (sum(v) / len(v)) ** 0.5
        return out

    point, rng, boots = rmse(keys), random.Random(seed), []
    for _ in range(n_boot):
        boots.append(rmse([rng.choice(keys) for _ in keys]))
    q = lambda v: [sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]] if v else None
    pairs = [(a, b) for i, a in enumerate(conds) for b in conds[:i]]  # later condition minus earlier, each minus the null
    return {
        "n_clusters": len(keys),
        "n_boot": n_boot,
        "ci": {c: q([b[c] for b in boots if c in b]) for c in conds},
        "diffs": [
            {"a": a, "b": b, "est": point[a] - point[b], "ci": q([x[a] - x[b] for x in boots if a in x and b in x])}
            for a, b in pairs
            if a in point and b in point
        ],
    }


def run_summary(run_dir: Path, envs: Envs, export=True) -> dict:
    from cfbench.core import canon_condition
    from cfbench.scoring import Case

    meta = json.loads((run_dir / "meta.json").read_text())
    if meta.get("params", {}).get("task") == "answer":
        return answer_summary(run_dir, meta, envs, export)
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("task", "rate") in ("rate", "claim")]
    docent = json.loads((run_dir / "docent.json").read_text()) if (run_dir / "docent.json").exists() else {}
    _, run_envs, env_runs = eval_of(meta) or (None, [], [])
    ids = docent.get("runs") or [None] * len(rows)
    # the eval's LINKS(row) (cfbench/evals/<name>/eval.py): e.g. the base-arm episode the investigator was shown, as
    # {"env", "run", "episode"} (gets its Docent URL here) or an expview item {"dataset", "item"}
    hook = getattr(_eval_module(meta.get("params", {}).get("eval")), "LINKS", None)

    def links_of(r: dict) -> dict:
        out = hook(r) if hook else {}
        for spec in out.values():
            if "episode" in spec:
                spec["docent"] = envs.docent(spec["env"], spec["run"], spec["episode"])
        return out

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
                "rationale": r.get("rationale"),
                "links": links_of(r),
            }
        )
    groups: dict[tuple, dict[str, Any]] = {}
    sq, sq_null = defaultdict(list), []
    clusters: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))  # arm pair -> condition -> MSEs
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
                "k_cf_pool": r.get("k_cf_pool"),  # claim evals: the model's real pooled counterfactual rate, for context
                "n_cf_pool": r.get("n_cf_pool"),
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
                clusters[env, version, b, cf, model][k].append(case.mse(k))
        g["_null"].append(case.mse_null())
        sq_null.append(case.mse_null())
        clusters[env, version, b, cf, model]["null"].append(case.mse_null())
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
        "started": meta.get("started_at"),
        "task": meta["params"].get("task", "rate"),
        "inputs": Path(meta["params"]["inputs"]).name if meta["params"].get("inputs") else None,
        "versions": meta["params"].get("versions"),
        "harness_version": meta["params"].get("harness_version", 1),
        "predictors": meta["params"].get("predictors", "").split(","),
        "conditions": sorted({k for g in out_groups for k in g["draws"]}),
        "models": sorted({g["model"] for g in out_groups}),
        "effort": meta["params"].get("effort"),
        "max_calls": meta["params"].get("max_calls"),
        "n_cases": len(cases),
        "docent_collection": docent.get("collection_id"),
        "expview": f"runs/cfbench/{run_dir.name}" if (run_dir / "expview" / "index.jsonl").exists() else None,
        # mean over cases of the per-case MSE (each case's draws averaged first), as cfbench.scoring.rmse
        "rmse": {k: (sum(v) / len(v)) ** 0.5 for k, v in sq.items()},
        "rmse_null": (sum(sq_null) / len(sq_null)) ** 0.5 if sq_null else None,
        "boot": rmse_ci(clusters) if clusters else None,  # 95% bootstrap CIs over arm pairs
        "_sq": dict(sq),  # per-case MSEs, for the eval's pooled row (dropped before writing)
        "_clusters": {k: dict(v) for k, v in clusters.items()},
        "_sq_null": sq_null,
        "paired": paired,
        "groups": out_groups,
        "item_info": item_info(meta, rows),
    }


def run_header(run_dir: Path, meta: dict, docent: dict) -> dict:
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(.*)", run_dir.name)
    return {
        "run": run_dir.name,
        "date": m[1] if m else None,
        "slug": meta.get("slug") or (m[2] if m else run_dir.name),
        "description": meta.get("description"),
        "status": meta.get("status"),
        "harness_version": meta["params"].get("harness_version", 1),
        "predictors": meta["params"].get("predictors", "").split(","),
        "effort": meta["params"].get("effort"),
        "max_calls": meta["params"].get("max_calls"),
        "docent_collection": docent.get("collection_id"),
        "expview": f"runs/cfbench/{run_dir.name}" if (run_dir / "expview" / "index.jsonl").exists() else None,
    }


def answer_summary(run_dir: Path, meta: dict, envs: Envs, export=True) -> dict:
    """An answer-task run (cfbench/answer.py): % correct per condition next to "the edit does nothing" (the base arm's most
    common answer), over cases with equal weight as cfbench.answer, and per (env, version, base, cf, model) one row per
    item with its truth and every draw. Never mixed with the rate task's RMSE: the run says `task: answer`."""
    from cfbench.answer import modal
    from cfbench.core import canon_condition

    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    for r in rows:  # cfbench.run --items: the item is the env spec's @suffix
        r.setdefault("item", r["env"].rpartition("@")[2] if "@" in r["env"] else None)
    docent = json.loads((run_dir / "docent.json").read_text()) if (run_dir / "docent.json").exists() else {}
    _, run_envs, _ = eval_of(meta) or (None, [], [])
    ids = docent.get("runs") or [None] * len(rows)
    cases: dict[tuple, dict] = {}  # (env, version, model, base, cf, item) -> the case and its draws by condition
    for r, did in zip(rows, ids, strict=False):
        item = r["item"]
        key = (r.get("env_name") or run_envs[0], r.get("version"), r["model"], r["base_arm"], r["cf_arm"], item)
        c = cases.setdefault(key, {"row": r, "item": item, "draws": defaultdict(list)})
        cond = canon_condition(r["condition"])
        c["draws"][cond].append(
            {
                "answer": r.get("canon") if r.get("valid", True) else None,
                "raw": r.get("answer"),
                "correct": bool(r.get("correct")),
                "valid": r.get("valid", True),
                "draw": r["draw"],
                "item": item,
                "predictor": r["predictor"],
                "calls": sum((r.get("tool_calls") or {}).values()),
                "docent": did,
                "ev": item_id(r, cond) if r.get("messages") else None,
                "rationale": r.get("rationale"),
            }
        )
    groups: dict[tuple, dict[str, Any]] = {}
    per_cond, null = defaultdict(list), []
    for (env, version, model, b, cf, item), c in sorted(cases.items(), key=lambda kv: str(kv[0])):
        r = c["row"]
        g: dict[str, Any] | None = groups.get((env, version, b, cf, model))
        if g is None:
            pair, rev = envs.pair(env, b, cf)
            g = groups[env, version, b, cf, model] = {
                "env": env, "version": version, "model": model, "target": envs.target(env, model, b, []), "base": b, "cf": cf,
                "pair": pair, "reverse": rev, "items": [], "_ok": defaultdict(list), "_null": [],
            }  # fmt: skip
        base_modal = modal(r["answers_base"])
        g["items"].append(
            {"item": item, "truth": r["modal_cf"], "k_cf": r["k_cf"], "n_cf": r["n_cf"], "answers_base": r["answers_base"],
             "answers_cf": r["answers_cf"], "base_modal": base_modal, "draws": dict(c["draws"])}
        )  # fmt: skip
        for k, ds in c["draws"].items():
            if not (ds := [d for d in ds if d["valid"]]):  # invalid draws are dropped, as cfbench.answer and the evals' score.py do
                continue
            frac = sum(d["correct"] for d in ds) / len(ds)
            g["_ok"][k].append(frac)
            per_cond[k].append(frac)
        g["_null"].append(base_modal == r["modal_cf"])
        null.append(base_modal == r["modal_cf"])
    out_groups = []
    for g in groups.values():
        g["correct"] = {k: sum(v) / len(v) for k, v in g.pop("_ok").items()}
        nul = g.pop("_null")
        g["correct_null"] = sum(nul) / len(nul)
        out_groups.append(g)
    if export and any(r.get("messages") for r in rows):
        export_expview(run_dir, rows, docent)
    return run_header(run_dir, meta, docent) | {
        "task": "answer",
        "conditions": sorted({k for g in out_groups for k in g["correct"]}),
        "models": sorted({g["model"] for g in out_groups}),
        "n_cases": len(cases),
        "correct": {k: sum(v) / len(v) for k, v in per_cond.items()},
        "correct_null": sum(null) / len(null) if null else None,
        "groups": out_groups,
        "item_info": item_info(meta, rows),
    }


def paired_vs(right: dict[str, dict[Any, list[float]]], ref: str = "transcript_agentic", n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """Per condition, its % right minus `ref`'s over the items both have (an item's score = the mean over its draws): the
    mean difference, a 95% bootstrap CI over items, the sign-test counts (items better / worse) and p, two-sided and
    one-sided (condition > ref)."""
    import numpy as np

    if ref not in right:
        return []
    rng, out = np.random.default_rng(seed), []
    for k in sorted(right):
        items = sorted(set(right[k]) & set(right[ref]), key=str)
        if k == ref or not items:
            continue
        d = np.array([np.mean(right[k][i]) - np.mean(right[ref][i]) for i in items])
        boots = d[rng.integers(len(d), size=(n_boot, len(d)))].mean(1)
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        n = pos + neg
        tail = [sum(math.comb(n, i) for i in range(j, n + 1)) / 2**n for j in (pos, max(pos, neg))] if n else [1.0, 0.5]
        out.append({
            "condition": k, "ref": ref, "n": len(items), "diff": float(d.mean()),
            "ci": [float(x) for x in np.percentile(boots, [2.5, 97.5])], "better": pos, "worse": neg,
            "p_one": tail[0], "p_two": min(1.0, 2 * tail[1]),
        })  # fmt: skip
    return out


def item_info(meta: dict, rows: list[dict]) -> dict:
    """The eval's ITEM_INFO(row) (optional: what its question-level table shows, e.g. the question, gold, the model's own
    answer), once per item; {} for evals without it."""
    hook = getattr(_eval_module(meta.get("params", {}).get("eval")), "ITEM_INFO", None)
    return {r["item"]: hook(r) for r in {r["item"]: r for r in rows if r.get("item")}.values()} if hook else {}


def mark_current(name: str, summaries: list[dict]) -> dict:
    """Mark which conditions of each run are current (`current_conds`: the newest complete run per (version set,
    condition), so a later placebo-only rerun replaces just the placebo of an earlier run) and each run `current` if it
    has any (else superseded: the page folds those away), and return the eval-level pooled numbers over the current rows:
    claim evals get af_which's table per inputs run (cfbench/evals/af_which/score.py, plus pick accuracy and order
    consistency for paired-order evals), rate evals a pooled RMSE row and pooled paired comparison."""
    from cfbench.core import canon_condition

    seen: set = set()
    for s in summaries:  # newest first
        s["current_conds"] = []
        for c in s["conditions"] if s.get("status") in (None, "done") else []:
            if (key := (tuple(s.get("versions") or [s["run"]]), c)) not in seen:
                seen.add(key)
                s["current_conds"].append(c)
        s["current"] = bool(s["current_conds"])
    cur = [s for s in summaries if s["current"]]
    if any(s.get("task") == "claim" for s in cur):
        import importlib
        import importlib.util

        from cfbench.evals.af_which.score import stats

        if importlib.util.find_spec(f"cfbench.evals.{name}.score"):  # an eval's own claim scorer (same row schema), if any
            stats = getattr(importlib.import_module(f"cfbench.evals.{name}.score"), "stats", stats)
        from cfbench.evals.af_which_pair.score import pair_stats

        tables = []
        for inputs in dict.fromkeys(s["inputs"] for s in cur):
            runs = [s for s in cur if s["inputs"] == inputs]
            rows = [
                r
                for s in runs
                for line in (CFBENCH / s["run"] / "predictions.jsonl").read_text().splitlines()
                if canon_condition((r := json.loads(line))["condition"]) in s["current_conds"]
            ]
            table = stats(rows)
            if all("order" in r for r in rows):
                pair = pair_stats(rows)
                for t in table:
                    t |= pair.get((t["condition"], t["version"]), {})
            paired = {}
            if any(t.get("accuracy") is not None for t in table):  # scored as % right (guess = p >= 0.5): each condition vs transcript
                right: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
                for r in rows:
                    if r["valid"]:
                        right[r["version"]][canon_condition(r["condition"])][r["item"]].append(
                            float((r["p"] >= 0.5) == bool(r["k_cf"] // r["n_cf"]))
                        )
                paired = {v: paired_vs(by) for v, by in right.items()}
            tables.append({"inputs": inputs, "runs": [s["slug"] for s in runs], "rows": table, "paired": paired})
        return {"claim_tables": tables}
    if ans := [s for s in cur if s.get("task") == "answer"]:  # % correct per item (valid draws), pooled over the current runs
        right: dict[str, dict[tuple, list[float]]] = defaultdict(lambda: defaultdict(list))
        for s in ans:
            for g in s["groups"]:
                for it in g["items"]:
                    for k, ds in it["draws"].items():
                        if k in s["current_conds"]:
                            right[k][g["env"], g["version"], it["item"]] += [float(d["correct"]) for d in ds if d["valid"]]
        import numpy as np

        right = {k: {i: v for i, v in by.items() if v} for k, by in right.items()}
        rng, pooled = np.random.default_rng(0), []
        for k, by in right.items():  # % correct per condition over every item (an item's draws averaged), 95% bootstrap CI
            x = np.array([np.mean(v) for v in by.values()])
            boots = x[rng.integers(len(x), size=(2000, len(x)))].mean(1)
            pooled.append(
                {"condition": k, "n": len(x), "accuracy": float(x.mean()), "ci": [float(c) for c in np.percentile(boots, [2.5, 97.5])]}
            )
        return {"answer_paired": {"runs": [s["slug"] for s in ans], "rows": paired_vs(right), "pooled": pooled}}
    cur = [s for s in cur if s.get("task", "rate") == "rate"]
    if len(cur) < 2:
        return {}
    sq, sq_null = defaultdict(list), []
    clusters: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in cur:
        for k, v in s["_sq"].items():
            if k not in s["current_conds"]:
                continue
            sq[k] += v
        sq_null += s["_sq_null"]
        for key, by in s["_clusters"].items():
            for k, v in by.items():
                if k == "null" or k in s["current_conds"]:
                    clusters[key][k] += v
    paired: defaultdict[str, Any] = defaultdict(lambda: {"a": [], "z": [], "truth": defaultdict(list)})
    for s in cur:  # per-case mean p per model, from the groups' draws, pooled over the current runs
        models = sorted(
            {g["model"] for g in s["groups"]}, key=lambda m: -max(g["k_cf"] / g["n_cf"] for g in s["groups"] if g["model"] == m)
        )
        if len(models) != 2:
            continue
        for g in s["groups"]:
            for k, ds in g["draws"].items():
                if k not in s["current_conds"]:
                    continue
                per = defaultdict(list)
                for d in ds:
                    if d["p"] is not None:
                        per[d["item"]].append(d["p"])
                side = "a" if g["model"] == models[0] else "z"
                paired[k][side] += [sum(v) / len(v) for v in per.values()]
                paired[k]["truth"][side] += [g["k_cf"] / g["n_cf"]] * len(per)
                paired[k]["models"] = models
    return {
        "pooled": {
            "runs": [s["slug"] for s in cur],
            "rmse": {k: (sum(v) / len(v)) ** 0.5 for k, v in sq.items()},
            "rmse_null": (sum(sq_null) / len(sq_null)) ** 0.5,
            "n_cases": len(sq_null),
            "boot": rmse_ci(clusters),
            "paired": [
                {
                    "condition": k,
                    "models": v["models"],
                    "n": [len(v["a"]), len(v["z"])],
                    "auroc": auroc(v["a"], v["z"]),
                    "mean_p": [sum(v["a"]) / len(v["a"]), sum(v["z"]) / len(v["z"])],
                    "truth": [sum(v["truth"]["a"]) / len(v["truth"]["a"]), sum(v["truth"]["z"]) / len(v["truth"]["z"])],
                }
                for k, v in sorted(paired.items())
                if v["a"] and v["z"]
            ],
        }
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


def _summary(run_dir: Path, envs: Envs, export: bool) -> dict | None:
    """run_summary, or None (with a warning) for a run whose env no longer exists, e.g. a retired env's old runs."""
    try:
        return run_summary(run_dir, envs, export)
    except ModuleNotFoundError as e:
        print(f"skipping {run_dir.name}: {e}", file=sys.stderr)
        return None


def defined() -> list[dict]:
    """Every eval from the code alone (for the Pages spec/evals.json): each cfbench/evals/<name>/ with an eval.py, and
    `<env>-pairs` for each of PAIRS_ENVS. An eval module that doesn't import gets "error" and no envs; RETIRED_EVALS get
    "retired"."""
    import importlib

    out = []
    for d in sorted(EVAL_DIRS.iterdir()):
        if not (d / "eval.py").exists():
            continue
        try:
            envs, err = list(importlib.import_module(f"cfbench.evals.{d.name}.eval").ENVS), None
        except Exception as ex:  # noqa: BLE001
            envs, err = [], repr(ex)[:500]
        out.append({"name": d.name, "title": describe(d.name, envs)[0], "envs": envs} | ({"error": err} if err else {}))
    out += [{"name": f"{e}-pairs", "title": describe(f"{e}-pairs", [e])[0], "envs": [e]} for e in PAIRS_ENVS]
    return [e | {"retired": True} if e["name"] in RETIRED_EVALS else e for e in out]


def build(names: list[str] | None = None, export=True) -> list[Path]:
    """Write data/evals/<name>.json for `names` (default: every eval with a local run) and data/evals/index.json
    (HF's entries plus the local evals)."""
    runs, envs, written = eval_runs(), Envs(), []
    EVALS.mkdir(parents=True, exist_ok=True)
    for name in names or list(runs):
        if name not in runs:
            continue
        summaries = sorted(
            filter(None, (_summary(r, envs, export) for r in runs[name])), key=lambda s: s.get("started") or s["run"], reverse=True
        )
        summaries = [s for s in summaries if s.get("status") in (None, "done")]  # running ones show once they finish
        if only := getattr(_eval_module(name), "INPUTS_RUNS", None):  # the eval's broken inputs runs are left out entirely
            summaries = [s for s in summaries if s.get("inputs") in only]
        all_envs = sorted({e for r in runs[name] for e in (eval_of(json.loads((r / "meta.json").read_text())) or ("", [], []))[1]})
        title, description = describe(name, all_envs)
        extra = mark_current(name, summaries)
        for s in summaries:
            s.pop("_sq", None), s.pop("_sq_null", None), s.pop("_clusters", None)
        p = EVALS / f"{name}.json"
        claim = getattr(_eval_module(name), "CLAIM", None)  # a claim eval's claim in words, for the page's captions
        arm_labels = getattr(_eval_module(name), "ARM_LABELS", None)  # readable names for the eval's arms
        doc = {"name": name, "title": title, "description": description, "envs": all_envs, "runs": summaries, "claim": claim}
        doc |= {"arm_labels": arm_labels, "pair_labels": getattr(_eval_module(name), "PAIR_LABELS", None)} | extra
        p.write_text(json.dumps(doc) + "\n")
        written.append(p)
    # start from HF's index, so evals this machine has no runs of stay listed (an upload used to replace it with only the local ones)
    built = {p.stem for p in written}
    top = {e["name"]: e for e in hf_index("index.json", root="evals/").get("evals", [])}
    for p in sorted(EVALS.glob("*.json")):
        if p.name == "index.json" or (p.stem not in built and p.stem in top):  # an eval not rebuilt here: HF's entry is current
            continue
        e = json.loads(p.read_text())
        rs = e["runs"]
        top[e["name"]] = {
            "name": e["name"],
            "title": e["title"],
            "envs": e["envs"],
            "n_runs": len(rs),
            "latest": max((r["date"] or "" for r in rs), default=""),
            "models": sorted({m for r in rs for m in r["models"]}),
            "conditions": sorted({c for r in rs for c in r["conditions"]}),
            "n_cases": sum(r["n_cases"] for r in rs),
        }
    p = EVALS / "index.json"
    index = [top[n] for n in sorted(top)]
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
