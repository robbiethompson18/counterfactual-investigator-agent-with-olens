"""Loading and scoring behind the report charts (`graphs.cfbench_report`) and `cfbench.score`.

Which runs are read is cfbench/catalog.py. The method is written up in docs/cfbench-scoring.md. In short: a case is an
ordered arm pair; its score is the squared error of the predicted rate against the counterfactual arm's observed rate,
averaged over the predictor's draws; cases have equal weight; the headline is RMSE next to the RMSE of "the edit does
nothing" (predict the base arm's rate).
"""

import json
import math
import os
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
from scipy.stats import binom

from cfbench.catalog import AUTHOR_PROBE_RUN, AUTHOR_RUNS, ESSAY_LENS_RUNS, STANDARD_PROBE_RUN
from cfbench.core import P_MAX, P_MIN, canon_condition

PREDICTOR = "google/gemini-3.8-flash"
CONDS = ["no_transcript", "transcript_agentic", "olens_agentic", "olens_placebo_agentic"]
LABEL = {
    "no_transcript": "Env diff only",
    "transcript_agentic": "Transcript-only",
    "olens_agentic": "O-lens",
    "olens_placebo_agentic": "O-lens placebo",
    "jlens_agentic": "J-lens",
    "jlens_placebo_agentic": "J-lens placebo",
}
NULL_LABEL = "Edit does nothing"
OLD_LENS = "olens_s3d_rl600"  # the pre-space lens; its reads are never reused (CLAUDE.md, docs/olens-space-migration.md)

HF_REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"


def variant_of(row: dict) -> str:
    """env/rollout-run, e.g. hidden_goal_grade/2026-10-22_who2_n30_nothink: what cases are keyed by. One env's rollout runs
    reuse arm names (hidden_goal_grade's who2 and canon_off both have base:mine) with different rollouts and base rates, so
    keying by env alone merged them."""
    return row["env"].split(":")[1]


def env_of(variant: str) -> str:
    """The env of a case key's first element: `variant_of`'s env/run, maybe prefixed "<target>: " (the pooled chart). Envs,
    not rollout runs, are what the charts count and what ESS is over (docs/cfbench-scoring.md). Run names have no "/"."""
    return variant.rsplit("/", 1)[0]


def run_params(run_dir: str) -> dict:
    try:
        with open(os.path.join(run_dir, "meta.json")) as f:
            return json.load(f).get("params") or {}
    except FileNotFoundError:
        return {}


def uses_old_lens(run_dir: str) -> bool:
    params = run_params(run_dir)
    lenses = {s.get("lens") for s in params.get("olens_sources") or []} | {params.get("olens_lens")}
    return OLD_LENS in lenses or OLD_LENS in (params.get("olens_reads") or "")


def load_rows(dirs: list[str], conds=CONDS, predictor=PREDICTOR, allow_old_lens=False) -> dict[tuple, dict]:
    """(env/rollout run, base_arm, cf_arm, condition, draw) -> prediction row; valid rows only; later dirs win.

    Old-lens runs contribute no olens rows unless `allow_old_lens`. Stored condition names are mapped to today's (canon_condition).
    """
    rows = {}
    for d in dirs:
        path = os.path.join(d, "predictions.jsonl")
        if not os.path.exists(path):
            continue
        old = uses_old_lens(d) and not allow_old_lens
        with open(path) as f:
            lines = f.read().splitlines()
        for line in lines:
            r = json.loads(line)
            r["condition"] = canon_condition(r["condition"])
            if r["condition"] not in conds or r["predictor"] != predictor or not r.get("valid", True) or r.get("p") is None:
                continue
            if old and r["condition"].startswith("olens"):
                continue
            r["run"] = os.path.basename(d)
            rows[(variant_of(r), r["base_arm"], r["cf_arm"], r["condition"], r["draw"])] = r
    return rows


@dataclass
class Case:
    env: str
    base_arm: str
    cf_arm: str
    k_base: int
    n_base: int
    k_cf: int
    n_cf: int
    p: dict[str, list[float]] = field(default_factory=dict)  # condition -> predicted P(behavior), one per draw

    @property
    def q(self) -> float:
        return self.k_cf / self.n_cf

    @property
    def q_base(self) -> float:
        return self.k_base / self.n_base

    # squared error of the rate (docs/cfbench-scoring.md)
    def mse(self, cond: str) -> float:
        return float(np.mean([(p - self.q) ** 2 for p in self.p[cond]]))

    def mse_null(self) -> float:
        return (self.q_base - self.q) ** 2

    def noise(self) -> float:
        """Sampling variance of the observed rate: what a perfect predictor still scores, in expectation."""
        return self.q * (1 - self.q) / self.n_cf

    def bias_var(self, cond: str) -> tuple[float, float]:
        """mse = bias² + draw variance: is the predictor wrong, or just noisy across draws?"""
        ps = np.array(self.p[cond])
        return float((ps.mean() - self.q) ** 2), float(ps.var())

    # legacy binomial log-likelihood skill, kept for comparison with older numbers (docs/cfbench-v1-results.md)
    def _ll(self, p: float) -> float:
        return float(binom.logpmf(self.k_cf, self.n_cf, np.clip(p, P_MIN, P_MAX)))

    def log_gain(self, cond: str) -> float:
        return float(np.mean([self._ll(p) for p in self.p[cond]])) - self._ll((self.k_base + 0.5) / (self.n_base + 1))

    def log_oracle(self) -> float:
        return self._ll(self.q) - self._ll((self.k_base + 0.5) / (self.n_base + 1))


def build_cases(rows: dict[tuple, dict], conds=CONDS) -> dict[tuple, Case]:
    """(env/rollout run, base_arm, cf_arm) -> Case, keeping only cases with a prediction in every condition (same cases per bar)."""
    cases: dict[tuple, Case] = {}
    for (env, b, c, cond, _), r in sorted(rows.items(), key=lambda kv: kv[1]["draw"]):
        case = cases.setdefault((env, b, c), Case(env, b, c, r["k_base"], r["n_base"], r["k_cf"], r["n_cf"]))
        case.p.setdefault(cond, []).append(r["p"])
    return {k: v for k, v in cases.items() if all(c in v.p for c in conds)}


def rmse(cases: dict[tuple, Case], cond: str, keys=None) -> float:
    keys = list(cases) if keys is None else keys
    return float(np.sqrt(np.mean([cases[k].mse(cond) for k in keys])))


def rmse_null(cases: dict[tuple, Case], keys=None) -> float:
    keys = list(cases) if keys is None else keys
    return float(np.sqrt(np.mean([cases[k].mse_null() for k in keys])))


def noise_floor(cases: dict[tuple, Case], keys=None) -> float:
    keys = list(cases) if keys is None else keys
    return float(np.sqrt(np.mean([cases[k].noise() for k in keys])))


def brier_skill(cases: dict[tuple, Case], cond: str, keys=None) -> float:
    """1 − Σ mse / Σ mse_null: 0 = as good as 'the edit does nothing', 1 = perfect, negative = worse than the null."""
    keys = list(cases) if keys is None else keys
    return 1 - sum(cases[k].mse(cond) for k in keys) / sum(cases[k].mse_null() for k in keys)


# Dead (RMSE replaced it 2026-10-26, docs/cfbench-scoring.md); kept only to reproduce log-skill numbers in older docs.
def log_skill(cases: dict[tuple, Case], cond: str, keys=None) -> float:
    keys = list(cases) if keys is None else keys
    return sum(cases[k].log_gain(cond) for k in keys) / sum(cases[k].log_oracle() for k in keys)


def ess_envs(keys) -> float:
    """Effective number of envs behind an equal-weight-per-case mean: (Σ_e n_e)² / Σ_e n_e², n_e = cases in env e."""
    n = np.array(list(Counter(env_of(k[0]) for k in keys).values()), dtype=float)
    return float(n.sum() ** 2 / (n**2).sum())


def bootstrap(cases: dict[tuple, Case], conds=CONDS, ref="transcript_agentic", n_boot=2000, seed=0) -> dict:
    """95% case-bootstrap intervals for each condition's RMSE and for its RMSE reduction vs `ref` (paired: same resample)."""
    keys = list(cases)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        idx = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        s = {c: rmse(cases, c, idx) for c in conds}
        for c in conds:
            draws[c].append(s[c])
            draws["gain_" + c].append(s[ref] - s[c])
        draws["null"].append(rmse_null(cases, idx))

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {
        "rmse": {c: ci(draws[c]) for c in conds},
        "null": ci(draws["null"]),
        "gain": {c: ci(draws["gain_" + c]) for c in conds},
        "p_gain_le0": {c: float(np.mean(np.array(draws["gain_" + c]) <= 0)) for c in conds},
    }


def coverage(rows: dict[tuple, dict], cases: dict[tuple, Case]) -> dict:
    """What a chart is computed from: envs, arms, target-model rollouts and predictor transcripts behind its cases."""
    arms = {}
    n_pred = Counter()
    for (env, b, c, cond, _), r in rows.items():
        if (env, b, c) in cases:
            arms[(env, b)], arms[(env, c)] = r["n_base"], r["n_cf"]
            n_pred[cond] += 1
    return {
        "envs": len({env_of(e) for e, _ in arms}),
        "arms": len(arms),
        "rollouts": sum(arms.values()),
        "predictions": sum(n_pred.values()),
        "by_cond": dict(n_pred),
    }


# Standardized probe results. Charts share this loader instead of retaining the old selected-layer summaries.


def standard_probe_file(relative: str):
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    path = Path("data") / STANDARD_PROBE_RUN / relative
    if not path.exists():
        hf_hub_download(
            HF_REPO,
            f"{STANDARD_PROBE_RUN}/{relative}",
            repo_type="dataset",
            local_dir="data",
        )
    return path


def standard_probe_rows(family: str, site: str, target: str | None = None) -> list[dict]:
    path = standard_probe_file(f"fits/{family}/{site}/predictions.jsonl")
    with path.open() as stream:
        rows = [r for r in map(json.loads, stream) if target is None or r["family"] == target]
    assert len({r["id"] for r in rows}) == len(rows)
    return rows


def grouped_mean_ci(values, groups, reps: int = 2000) -> tuple[float, float, float]:
    """Mean over cases; percentile CI resampling question/essay groups, conditional on the fitted models."""
    values = np.asarray(values, dtype=float)
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.bincount(inverse, weights=values)
    rng = np.random.default_rng(0)
    boot = []
    for start in range(0, reps, 100):
        indices = rng.integers(len(counts), size=(min(100, reps - start), len(counts)))
        boot.extend((totals[indices].sum(1) / counts[indices].sum(1)).tolist())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def grouped_auc_ci(scores, truth, groups, reps: int = 2000) -> tuple[float, float, float]:
    from scipy.stats import rankdata

    scores, truth = np.asarray(scores), np.asarray(truth, dtype=bool)
    unique = np.unique(groups)
    rows = [np.flatnonzero(np.asarray(groups) == group) for group in unique]

    def auc(indices):
        y = truth[indices]
        n1 = y.sum()
        return float((rankdata(scores[indices])[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))

    rng = np.random.default_rng(0)
    boot = [auc(np.concatenate([rows[i] for i in rng.integers(len(rows), size=len(rows))])) for _ in range(reps)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return auc(np.arange(len(scores))), float(lo), float(hi)


def standard_mcq_comparison(runs, target: str) -> tuple[dict, dict]:
    """Historical lens draws and new within-benchmark OOF probes on exactly the same questions."""
    draws, truth = {}, {}
    for run in runs:
        with (run / "predictions.jsonl").open() as stream:
            for row in map(json.loads, stream):
                if row["condition"].startswith("probe") or row["condition"] == "qwen_logits":
                    continue
                item = row["item"]
                draws[row["condition"], item, row.get("draw", 0)] = float(bool(row["correct"]))
                labels = row["modal_cf"], row["gold"]
                if item in truth:
                    assert truth[item] == labels
                truth[item] = labels
    by = defaultdict(lambda: defaultdict(list))
    for (condition, item, _), value in draws.items():
        by[condition][item].append(value)
    data = {c: {i: (truth[i][0] == truth[i][1], float(np.mean(v))) for i, v in items.items()} for c, items in by.items()}
    groups = {}
    for site in ("user_end", "response_start"):
        probes = {}
        for row in standard_probe_rows(target, site, target):
            item = row["item"]
            if item not in truth:
                continue
            assert truth[item] == (chr(65 + row["label"]), chr(65 + row["gold"]))
            probes[item] = (row["label"] == row["gold"], float(row["predicted_label"] == row["label"]))
            groups[item] = row["group"]
        data[f"probe_{site}"] = probes
    common = set.intersection(*(set(items) for items in data.values()))
    assert common and all(set(items) == common for items in data.values()), "Conditions must cover identical comparison questions"
    return data, groups


def essay_probe_scores(site: str) -> dict:
    """Key standardized predictions by (opener arm, essay), as the investigator does."""
    return {(r["id"].split("/", 1)[1].split("~", 1)[0], r["item"]): r for r in standard_probe_rows("essay", site)}


def essay_lens_rows(run: str) -> list[dict]:
    """Read just scoring fields; stream the public run if it is not cached locally."""
    from pathlib import Path
    from urllib.request import urlopen

    cache = Path("data/plot_cache") / f"{run}.jsonl"
    if cache.exists():
        return [json.loads(line) for line in cache.open()]
    local = Path("data/runs/cfbench") / run / "predictions.jsonl"
    url = HF_URL + f"runs/cfbench/{run}/predictions.jsonl"
    fields = ("condition", "base_arm", "item", "draw", "valid", "p", "k_cf", "shift", "answers_base", "effort")
    with local.open() if local.exists() else urlopen(url) as stream:
        rows = [{k: r[k] for k in fields if k in r} for r in map(json.loads, stream)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def standard_essay_comparison() -> tuple[dict, dict]:
    """Same-case investigator and probe scores: condition -> case -> its draws' scores, each draw one prediction."""
    probes = {f"probe_{site}": essay_probe_scores(site) for site in ("user_end", "response_start")}
    reference = probes["probe_response_start"]
    draws = {}
    for run in ESSAY_LENS_RUNS:
        for r in essay_lens_rows(run):
            if not r.get("valid", True) or r["p"] is None:
                continue
            key = r["base_arm"], r["item"]
            assert reference[key]["truth"] == r["k_cf"]
            draws[r["condition"], key, r["draw"]] = r["p"]
    by = defaultdict(lambda: defaultdict(list))
    for (condition, key, _), value in draws.items():
        by[condition][key].append(value)
    order = (
        "transcript_inline",
        "olens_inline",
        "olens_placebo_inline",
        "nla_inline",
        "nla_placebo_inline",
        "jlens_inline",
        "jlens_placebo_inline",
    )
    scores = {c: dict(by[c]) for c in order}
    scores.update({c: {k: [r["predicted_shift"]] for k, r in rows.items()} for c, rows in probes.items()})
    assert all(set(values) == set(reference) for values in scores.values())
    return scores, reference


# ---- The deck's and headline charts' statistics (graphs.talk, graphs.headline_inline). Moved from slides/talk/figures.py on
# 2026-10-31 so that slides/ holds TeX only and graphs/ imports nothing from it.


def eval_json(name: str) -> dict:
    """The viewer's evals/NAME.json (data/evals/, or HF's evals/ when it isn't local)."""
    local = Path("data/evals") / f"{name}.json"
    if local.exists():
        return json.loads(local.read_text())
    with urllib.request.urlopen(HF_URL + f"evals/{name}.json") as r:
        return json.load(r)


def stars(p: float | None) -> str:
    return "" if p is None else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def sign_p(b: int, w: int) -> float:
    """Exact one-sided sign test: P(at least b better of b + w)."""
    n = b + w
    return 1.0 if not n else min(1.0, sum(math.comb(n, i) for i in range(b, n + 1)) / 2**n)


# The hint evals (hint_qwen_dsv4{,_before}{,_letter}_tail_inline), one panel per eval as on its Evals page (#eval=NAME):
# claim evals as % right (guess = p >= 0.5, CI from claim_tables, stars from its paired one-sided sign test against
# transcript-only) or as AUROC; answer evals as the page's headline answer panel recomputed as page.html's answerChartBlock
# does: every item's draws merged across runs, a bar is the mean over items of the item's fraction right, a 95% item bootstrap
# (2000 resamples; the page's own RNG differs, so a CI end can move by a point), the exact one-sided sign test against
# transcript-only over the items that differ, chance = 1 / the most options.
def claim_bars(name: str) -> tuple[dict, float, int]:
    """condition -> (% right, CI, stars) from the eval's claim table (pooled), chance 50%."""
    t = eval_json(name)["claim_tables"][0]
    rows = {r["condition"]: r for r in t["rows"] if r["version"] == "pooled"}
    paired = {p["condition"]: p for p in t["paired"]["pooled"] if p["ref"].startswith("transcript_")}
    bars = {c: (r["accuracy"], r["accuracy_ci"], stars(paired.get(c, {}).get("p_one"))) for c, r in rows.items()}
    return bars, 0.5, max(r["n_items"] for r in rows.values())


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a knew case's p > a didn't case's), ties half."""
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def auroc_bars(name: str, reps: int = 2000) -> tuple[dict, float, int]:
    """condition -> (AUROC of P(knew), 95% bootstrap CI over cases, stars), chance 0.5 (Kylie, 2026-10-30: in place of % right,
    which rewards a shift in the investigator's bias at the 0.5 threshold). An item's p is the mean over its draws, as
    score.py; stars: one-sided paired bootstrap that the condition's AUROC beats transcript-only's on the same cases."""
    p: dict[str, dict[tuple, list[float]]] = {}
    truth: dict[tuple, int] = {}
    for R in eval_json(name)["runs"]:
        for g in R["groups"]:
            for c, ds in g.get("draws", {}).items():
                for d in ds:
                    k = (g["env"], d["item"])
                    p.setdefault(c, {}).setdefault(k, []).append(d["p"])
                    truth[k] = g["k_cf"]
    keys = sorted(truth)
    y = np.array([truth[k] for k in keys])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keys), (reps, len(keys)))
    score = lambda x, i: auroc(x[i][y[i] == 1], x[i][y[i] == 0])
    xs = {c: np.array([np.mean(v[k]) for k in keys]) for c, v in p.items()}
    ref = np.array([score(xs["transcript_inline"], i) for i in idx])
    bars = {}
    for c, x in xs.items():
        boot = np.array([score(x, i) for i in idx])
        st = "" if c == "transcript_inline" else stars(float((boot - ref <= 0).mean()))
        bars[c] = (auroc(x[y == 1], x[y == 0]), np.quantile(boot, [0.025, 0.975]).tolist(), st)
    return bars, 0.5, len(keys)


def answer_bars(name: str, reps: int = 2000) -> tuple[dict, float, int]:
    """condition -> (% correct, CI, stars) over every item of every run, as the page's headline answer panel."""
    items: dict[tuple, dict[str, list[float]]] = {}
    letters = set()
    for R in eval_json(name)["runs"]:
        if R.get("task") != "answer":
            continue
        for g in R["groups"]:
            for it in g["items"]:
                I = items.setdefault((g["env"], g.get("version") or "", g["base"], g["cf"], it.get("item") or ""), {})
                for c, ds in it["draws"].items():
                    I.setdefault(c, []).extend(float(d["correct"]) for d in ds)
                letters |= {a for a in [it["truth"], *it.get("answers_base", {}), *it.get("answers_cf", {})] if a != "null"}
    per = {c: {k: sum(I[c]) / len(I[c]) for k, I in items.items() if I.get(c)} for c in {c for I in items.values() for c in I}}
    rng = np.random.default_rng(0)
    bars = {}
    for c, xs in per.items():
        x = np.array(list(xs.values()))
        boot = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
        ref = {k: v - per["transcript_inline"][k] for k, v in xs.items() if k in per["transcript_inline"]}  # this - transcript
        b, w = sum(d > 0 for d in ref.values()), sum(d < 0 for d in ref.values())
        p = None if c == "transcript_inline" else sign_p(b, w)
        bars[c] = (x.mean(), np.quantile(boot, [0.025, 0.975]).tolist(), stars(p))
    chance = 1 / (max(ord(a) for a in letters) - 64)
    return bars, chance, len(items)


# Essay grading (cfbench/evals/essay_grading_cf_inline): the investigator's draws from the runs' predictions (data/runs/cfbench/,
# or HF's runs/cfbench/), a case = (arm, intro), every draw one prediction, as score.py; standardized zero-counterfactual OOF
# probes, nested two-fold, both positions.
ESSAY_LENSES = ["olens_inline", "nla_inline", "jlens_inline"]
ESSAY_INVESTIGATORS = ["transcript_inline", *(c for x in ESSAY_LENSES for c in (x, x.replace("_inline", "_placebo_inline")))]


def rank_auroc(p: np.ndarray, y: np.ndarray) -> float:
    """AUROC by ranks, ties half (score.py's)."""
    from scipy.stats import rankdata

    n1 = y.sum()
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def _grade_mean(dist: dict) -> float:
    """Mean grade of a {score: count} distribution (cfbench.evals.essay_grading_cf_inline.eval.mean, without its imports)."""
    return sum(float(s) * k for s, k in dist.items()) / sum(dist.values())


@cache
def essay_cases() -> tuple[list[tuple], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """keys (arm, intro), truth (base mean > arm mean), condition -> scores (cases x draws; a probe has one per case), the
    observed arm mean."""
    ps: dict[str, dict[tuple, dict[int, float]]] = {}
    truth, arm_mean = {}, {}
    for run in ESSAY_LENS_RUNS:
        for r in essay_lens_rows(run):
            if r.get("valid", True) and r.get("p") is not None:
                k = (r["base_arm"], r["item"])
                ps.setdefault(r["condition"], {}).setdefault(k, {})[r["draw"]] = r["p"]  # rows are in completion order
                truth[k], arm_mean[k] = r["k_cf"], _grade_mean(r["answers_base"])  # answers_base: the observed (arm) prompt's scores
    keys = sorted(truth)
    xs = {c: np.array([[d[k][i] for i in sorted(d[k])] for k in keys]) for c, d in ps.items()}
    for site in ("user_end", "response_start"):
        pr = essay_probe_scores(site)
        assert all(pr[k]["truth"] == truth[k] for k in keys)
        xs[f"probe_{site}"] = np.array([[pr[k]["predicted_shift"]] for k in keys])
    return keys, np.array([truth[k] for k in keys]), xs, np.array([arm_mean[k] for k in keys])


def _logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    w = np.zeros(x.shape[1])
    for _ in range(50):  # Newton
        p = 1 / (1 + np.exp(-x @ w))
        w -= np.linalg.solve(x.T @ (x * (p * (1 - p))[:, None]) + l2 * np.eye(len(w)), x.T @ (p - y) + l2 * w)
    return w


def recalibrated(keys: list[tuple], y: np.ndarray, p: np.ndarray, am: np.ndarray) -> np.ndarray:
    """The investigator plus bet at the mean, as the essay eval page's combined() (branch claude/essay-grading-probe,
    score.py): out-of-fold logistic regression on [1, logit(p), the observed arm mean grade], 5 folds grouped by essay. `p`:
    cases x draws, one row per draw; returns the same shape."""
    d = p.shape[1]
    q = np.clip(p.ravel(), 0.01, 0.99)
    x = np.column_stack([np.ones(len(q)), np.log(q / (1 - q)), np.repeat(am, d)])
    yy = np.repeat(y, d).astype(float)
    essays = sorted({k[1] for k in keys})
    fold = {e: i % 5 for i, e in enumerate(np.random.default_rng(0).permutation(essays))}
    f = np.repeat([fold[k[1]] for k in keys], d)
    out = np.zeros(len(q))
    for i in range(5):
        out[f == i] = x[f == i] @ _logistic(x[f != i], yy[f != i])
    return out.reshape(p.shape)


def pooled_auroc(x: np.ndarray, y: np.ndarray) -> float:
    """AUROC over every draw (cases x draws): each draw is one prediction (docs/cfbench-scoring.md)."""
    return rank_auroc(x.ravel(), np.repeat(y, x.shape[1]))


@cache
def essay_auroc_bars(reps: int = 2000) -> tuple[dict, float, int, dict]:
    """condition -> (AUROC, 95% CI resampling essays as score.py does, stars: one-sided paired bootstrap vs transcript-only);
    the null (minus the observed arm mean: regression to the mean); n cases; paired differences -> (est, CI, one-sided p)."""
    keys, y, xs, am = essay_cases()
    xs = xs | {f"{c}+mean": recalibrated(keys, y, xs[c], am) for c in ESSAY_INVESTIGATORS}
    items = sorted({k[1] for k in keys})
    of = {i: np.array([n for n, k in enumerate(keys) if k[1] == i]) for i in items}
    rng = np.random.default_rng(0)
    idx = [np.concatenate([of[items[j]] for j in rng.integers(0, len(items), len(items))]) for _ in range(reps)]
    boot = {c: np.array([pooled_auroc(x[i], y[i]) for i in idx]) for c, x in xs.items()}
    bars = {}
    for c, x in xs.items():
        ref = "transcript_inline+mean" if c.endswith("+mean") else "transcript_inline"
        st = "" if c == ref else stars(float((boot[c] - boot[ref] <= 0).mean()))
        bars[c] = (pooled_auroc(x, y), np.quantile(boot[c], [0.025, 0.975]).tolist(), st)
    pairs = {}
    lens_pairs = [p for x in ESSAY_LENSES for p in ((x, x.replace("_inline", "_placebo_inline")), (x, "transcript_inline"))]
    for a, b in [*lens_pairs, ("olens_placebo_inline", "transcript_inline"), ("probe_user_end", "transcript_inline"),
                 ("probe_response_start", "transcript_inline"), *((f"{a}+mean", f"{b}+mean") for a, b in lens_pairs)]:  # fmt: skip
        if a in boot:
            d = boot[a] - boot[b]
            pairs[a, b] = (bars[a][0] - bars[b][0], np.quantile(d, [0.025, 0.975]).tolist(), float((d <= 0).mean()))
    return bars, rank_auroc(-am, y), len(keys), pairs


def essay_flips() -> tuple[int, int]:
    """Draws O-lens gets right (p >= 0.5 vs truth) and the same-numbered placebo draw wrong, and the reverse."""
    _, y, xs, _ = essay_cases()
    ok = {c: (xs[c] >= 0.5) == y[:, None] for c in ("olens_inline", "olens_placebo_inline")}
    return int((ok["olens_inline"] & ~ok["olens_placebo_inline"]).sum()), int((~ok["olens_inline"] & ok["olens_placebo_inline"]).sum())


# Part 3, blog authors (cfbench/evals/author_blog_drop_inline): does deleting this sentence make Qwen much less sure of its
# answer about the blogger? Per case the investigator's p (one draw) from the eval's two runs, the task-fit standard L2
# probe's out-of-fold score (scripts/author_probe/period_check.py, runs/author_mcq_blog/2026-10-31_period-probe/scores_l2.jsonl);
# CIs and stars resample bloggers.
AUTHOR_CONDS = ["transcript_inline", "olens_inline", "olens_placebo_inline", "nla_inline", "nla_placebo_inline"]
AUTHOR_GROUPS = [("transcript", ["transcript_inline"]), ("O-lens", ["olens_inline", "olens_placebo_inline"]),
                 ("NLA", ["nla_inline", "nla_placebo_inline"]), ("probe", ["probe"]),
                 ("probe +\ntranscript", ["transcript+probe"])]  # fmt: skip
AUTHOR_VERSIONS = {"pooled": "all cases"}  # pooled (Kylie, 2026-10-31: not the Qwen-wrong / Qwen-right split)


def author_rows(run: str) -> list[dict]:
    cache = Path("data/plot_cache") / f"{run}.jsonl"
    if cache.exists():
        return [json.loads(line) for line in cache.open()]
    local = Path("data/runs/cfbench") / run / "predictions.jsonl"
    url = HF_URL + f"runs/cfbench/{run}/predictions.jsonl"
    fields = ("condition", "item", "pair", "version", "valid", "p", "k_cf")
    with local.open() if local.exists() else urllib.request.urlopen(url) as stream:
        rows = [{k: r[k] for k in fields if k in r} for r in map(json.loads, stream)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def author_probe_file(name: str) -> Path:
    from huggingface_hub import hf_hub_download

    f = Path("data") / AUTHOR_PROBE_RUN / name
    if not f.exists():
        hf_hub_download(HF_REPO, f"{AUTHOR_PROBE_RUN}/{name}", repo_type="dataset", local_dir="data")
    return f


@cache
def author_cases() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """items, truth, version, blogger, condition -> score (the cases with every condition and a probe score)."""
    ps: dict[str, dict[str, float]] = {}
    meta = {}
    for run in AUTHOR_RUNS:
        for r in author_rows(run):
            if r.get("valid", True) and r.get("p") is not None:
                ps.setdefault(r["condition"], {})[r["item"]] = r["p"]
                meta[r["item"]] = (r["k_cf"], r["version"], r["pair"].split(":")[0])
    l2 = {r["item"]: r for r in map(json.loads, author_probe_file("scores_l2.jsonl").open())}
    probe = {i: r["probe_l2_h"] for i, r in l2.items()}
    items = sorted(i for i in meta if all(i in ps[c] for c in AUTHOR_CONDS) and i in probe)
    xs = {c: np.array([ps[c][i] for i in items]) for c in AUTHOR_CONDS} | {"probe": np.array([probe[i] for i in items])}
    xs["transcript+probe"] = np.array([l2[i]["haiku_plus_probe"] for i in items])  # the nested combiner, out of fold
    y, v, b = (np.array([meta[i][k] for i in items]) for k in range(3))
    return items, y, v, b, xs


@cache
def author_bars(reps: int = 2000) -> tuple[dict, dict, int]:
    """(version, condition) -> (AUROC, 95% CI, stars vs transcript-only, one-sided paired bootstrap over bloggers); paired
    differences (a, b) -> (AUROC est, 95% CI, one-sided p), pooled; n cases."""
    items, y, v, b, xs = author_cases()
    rng = np.random.default_rng(0)
    bars = {}
    for ver in AUTHOR_VERSIONS:
        m = np.ones(len(y), bool) if ver == "pooled" else v == ver
        us = sorted(set(b[m]))
        of = {u: np.flatnonzero(m & (b == u)) for u in us}
        idx = [np.concatenate([of[us[j]] for j in rng.integers(0, len(us), len(us))]) for _ in range(reps)]
        boot = {c: np.array([rank_auroc(x[i], y[i]) for i in idx]) for c, x in xs.items()}
        for c, x in xs.items():
            st = "" if c == "transcript_inline" else stars(float((boot[c] - boot["transcript_inline"] <= 0).mean()))
            bars[ver, c] = (rank_auroc(x[m], y[m]), np.quantile(boot[c], [0.025, 0.975]).tolist(), st)
    us = sorted(set(b))
    of = {u: np.flatnonzero(b == u) for u in us}
    idx = [np.concatenate([of[us[j]] for j in rng.integers(0, len(us), len(us))]) for _ in range(reps)]
    pairs = {}
    for a, c in [("olens_inline", "transcript_inline"), ("olens_inline", "olens_placebo_inline"), ("nla_inline", "transcript_inline"),
                 ("nla_inline", "nla_placebo_inline"), ("transcript+probe", "transcript_inline")]:  # fmt: skip
        da = np.array([rank_auroc(xs[a][i], y[i]) - rank_auroc(xs[c][i], y[i]) for i in idx])
        pairs[a, c] = (rank_auroc(xs[a], y) - rank_auroc(xs[c], y), np.quantile(da, [0.025, 0.975]).tolist(), float((da <= 0).mean()))
    return bars, pairs, len(items)


SIMULATED_RUN = "runs/harness_grid/2026-11-01_graderd-prompt-reads"
PROMPT_SECTIONS = ["tool block", "system prompt", "env block", "user"]


def simulated_by_section() -> dict:
    """harness_grid thinking off, every prompt position read (prefixes deduplicated): model -> section -> (draws saying
    simulated/mock/synthetic/fictional, draws), and the tool block split by plain vs MCP tool names. Qwen3.6-27B: one draw per
    position; GLM-5.3: five per position."""
    import csv

    def load(name):
        path = Path("data") / SIMULATED_RUN / name
        if not path.exists():
            from huggingface_hub import hf_hub_download

            hf_hub_download(HF_REPO, f"{SIMULATED_RUN}/{name}", repo_type="dataset", local_dir="data")
        return list(csv.DictReader(path.open()))

    out: dict = {m: {} for m in ("Qwen3.6-27B", "GLM-5.3")}
    for r in load("prompt_simulated_by_section.csv"):
        if r["seg"] in PROMPT_SECTIONS:
            out[r["model"]][r["seg"]] = (int(r["hits"]), int(r["draws"]))
    for r in load("prompt_simulated_tool_block_names.csv"):
        out[r["model"]][f"tool block, {r['names']}"] = (int(r["hits"]), int(r["draws"]))
    return out
