"""The critique's numbers on Adam Karvonen's released CHIVE data (docs/chive-critique.md). Each function prints one section."""

import collections
import functools
import json
import statistics

import numpy as np
import requests
from scipy.stats import fisher_exact

from chive_critique.data import ADAM_ARMS, ADAM_LABELS, ADAM_RUNS, RUN, adam, adam_url, band, jsonl, pct, run_file

KEEP = ("prompt_id", "behavior_summary", "question", "classifier_question", "structured_findings")


def extract():
    """Stream each run's investigations.jsonl (~1.5 GB) once; keep per-experiment counts and the structured findings (~10 MB)."""
    (RUN / "adam_rates").mkdir(parents=True, exist_ok=True)
    (RUN / "adam_findings").mkdir(parents=True, exist_ok=True)
    for run in ADAM_RUNS:
        with (
            requests.get(adam_url(f"{run}/investigations.jsonl"), stream=True, timeout=600) as r,
            open(RUN / "adam_rates" / f"rates_{run}.jsonl", "w") as rates_f,
            open(RUN / "adam_findings" / f"findings_{run}.jsonl", "w") as findings_f,
        ):
            r.raise_for_status()
            for line in r.iter_lines(chunk_size=1 << 20):
                if not line:
                    continue
                d = json.loads(line)
                exps = [{"exp": e["exp_num"], "k": e["grader_yes"], "n": e["n"], "t": e["temperature"]} for e in d["experiment_log"]]
                row = {"prompt_id": d["prompt_id"], "k0": d["baseline_grader_yes"], "n0": len(d["baseline_grader_verdicts"]), "exps": exps}
                rates_f.write(json.dumps(row) + "\n")
                findings_f.write(json.dumps({k: d.get(k) for k in KEEP}) + "\n")
        print("extracted", run)


def rates(run: str) -> list[dict]:
    return jsonl(run_file(f"adam_rates/rates_{run}.jsonl"))


def experiments(run: str) -> list[tuple[str, float, float, int, int]]:
    """(prompt_id, baseline rate, counterfactual rate, k, n) for every experiment of the run."""
    out = []
    for d in rates(run):
        out += [(d["prompt_id"], d["k0"] / d["n0"], e["k"] / e["n"], e["k"], e["n"]) for e in d["exps"] if e["k"] is not None and e["n"]]
    return out


def arm_rows(run: str, arm: str) -> dict[str, bool]:
    """claim_id -> was the answerer right, for one scored arm of Adam's tool eval (answerer Opus 4.8, effort high)."""
    return {c["claim_id"]: c["correct"] for c in adam(f"{run}/binary_evals/{ADAM_ARMS[run][arm]}_claude-opus-4-8_high_conc3.jsonl")}


def offpolicy():
    """§1: prompts whose earlier assistant turns came from another model, and each tool arm's accuracy on them vs transcript-only."""
    prompts = adam("gemma3_27b_it_wildchat_eval/prompts.jsonl")  # the same 4,714 prompts in every WildChat eval run
    prior = {r["id"]: sum(m["role"] == "assistant" for m in r["messages"]) for r in prompts}
    off = {p for p, n in prior.items() if n >= 1}
    single = [p for p in prior if p not in off]
    turns = sorted(collections.Counter(prior.values()).items())
    print(f"prompt pool: {pct(len(off), len(prior))} have a prior assistant turn; turns: {turns}")
    print("  prompts that produced an investigation:")
    for run in ADAM_RUNS:
        inv = {d["prompt_id"] for d in rates(run)}
        s, m = pct(sum(p in inv for p in single), len(single)), pct(sum(p in inv for p in off), len(off))
        print(f"    {ADAM_LABELS[run]:13s} single-turn {s}, multi-turn {m}")
    for run in ADAM_RUNS:
        inv = [d["prompt_id"] for d in rates(run)]
        ev = {c["claim_id"]: c for c in adam(f"{run}/eval_set_conc3_n2000.jsonl")}
        arms = {arm: arm_rows(run, arm) for arm in ADAM_ARMS[run]}
        # the paper's population: claims the counterfactual-tool arm got right, with a clean edit
        quality = [c for c in ev if arms["cf"].get(c) and ev[c]["edit_cleanliness"] != "confounded" and c in arms["plain"]]
        off_claims = sum(ev[c]["prompt_id"] in off for c in ev)
        print(f"== {ADAM_LABELS[run]}: investigations off-policy {pct(sum(p in off for p in inv), len(inv))};", end=" ")
        print(f"eval claims off-policy {pct(off_claims, len(ev))}; quality slice {len(quality)}")
        for label, want in [("off-policy", True), ("on-policy", False)]:
            ids = [c for c in quality if (ev[c]["prompt_id"] in off) == want]
            line = f"   {label:10s} n={len(ids):4d} transcript-only {statistics.mean(arms['plain'][c] for c in ids):.3f}"
            for arm in ("ao", "nla", "sae"):
                if arm in arms:
                    d = [arms[arm][c] - arms["plain"][c] for c in ids]
                    line += f" | {arm}-plain {100 * statistics.mean(d):+.1f}pp (SE {100 * statistics.stdev(d) / len(d) ** 0.5:.1f})"
            print(line)


def thresholds():
    """§2: experiments the 50pp / 15pp rule can't use, in both directions, and what the claim bank loses."""
    tot: collections.Counter = collections.Counter()
    for run in ADAM_RUNS:
        ex = experiments(run)
        bands = collections.Counter(band(b, r) for _, b, r, _, _ in ex)
        has_true = {p for p, b, r, _, _ in ex if band(b, r) == "true"}
        to_zero = collections.Counter(p for p, b, r, k, n in ex if k == 0 and 0.10 < b < 0.50)  # behaviour gone, baseline under 50%
        to_full = collections.Counter(p for p, b, r, k, n in ex if k == n and 0.50 < b < 0.90)  # behaviour saturated, baseline over 50%
        bank: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for c in adam(f"{run}/binary_counterfactual_questions.jsonl"):
            bank[c["prompt_id"]][c["is_true"]] += 1
        positive = {p for p, c in bank.items() if c[True]}
        gain = {p for p in bank if not bank[p][True] and to_zero[p]}  # verified, only false claims, but holds a stuck elimination
        row = {
            "investigations": len(rates(run)),
            "experiments": len(ex),
            "excluded": bands["excluded"],
            "to_zero_stuck": sum(to_zero.values()),
            "inv_to_zero": len(to_zero),
            "inv_to_zero_no_true": len(set(to_zero) - has_true),
            "to_full_stuck": sum(to_full.values()),
            "inv_to_full": len(to_full),
            "bank_investigations": len(bank),
            "bank_positive": len(positive),
            "bank_gain": len(gain),
            "true_claims": sum(c[True] for c in bank.values()),
            "true_gain": sum(min(2, to_zero[p]) for p in gain),
        }
        tot.update(row)
        print(ADAM_LABELS[run], row)
    print("TOTAL", dict(tot))
    n_ex, n_inv = tot["experiments"], tot["investigations"]
    print(f"  15-50pp band (never a claim): {pct(tot['excluded'], n_ex)} of experiments")
    print(f"  baseline 10-50% -> 0/n: {pct(tot['to_zero_stuck'], n_ex)} of experiments,", end=" ")
    print(f"in {pct(tot['inv_to_zero'], n_inv)} of investigations;", end=" ")
    print(f"{pct(tot['inv_to_zero_no_true'], n_inv)} of investigations have one and no >=50pp experiment")
    print(f"  mirror, baseline 50-90% -> n/n: {pct(tot['to_full_stuck'], n_ex)} of experiments,", end=" ")
    print(f"in {pct(tot['inv_to_full'], n_inv)} of investigations")
    pos, gain, true, true_gain = tot["bank_positive"], tot["bank_gain"], tot["true_claims"], tot["true_gain"]
    print(f"  claim bank: investigations with a true claim {pos:,} -> {pos + gain:,} (+{100 * gain / pos:.0f}%);", end=" ")
    print(f"true claims {true:,} -> +{true_gain:,} (+{100 * true_gain / true:.0f}%) at up to 2 per gained investigation")


@functools.cache
def fisher(k0: int, k: int, n: int = 30) -> float:
    return fisher_exact([[k0, n - k0], [k, n - k]]).pvalue


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def significance():
    """§2: positives under Fisher's exact test (BH within each run) vs the >=50pp rule, and how many <=15pp negatives are shown null.
    Every experiment and baseline in the three runs is 30 samples at T=1, so the 30-vs-30 table is the whole story."""
    rows = [(run, d["prompt_id"], d["k0"], e["k"]) for run in ADAM_RUNS for d in rates(run) for e in d["exps"]]
    assert all(d["n0"] == 30 and e["n"] == 30 for run in ADAM_RUNS for d in rates(run) for e in d["exps"])
    inv = np.array([f"{run}/{pid}" for run, pid, _, _ in rows])
    p = np.array([fisher(k0, k) for _, _, k0, k in rows])
    move = np.array([abs(k - k0) / 30 for _, _, k0, k in rows])
    q = np.empty_like(p)
    for run in ADAM_RUNS:
        m = np.array([r[0] == run for r in rows])
        q[m] = bh(p[m])
    chive = move >= 0.5
    pmax = p[chive].max()
    print(f"{len(rows):,} experiments in {len(set(inv)):,} investigations; weakest >=50pp positive has Fisher p = {pmax:.1g}")
    print(f"  share of experiments with no effect (Storey pi0, lambda 0.5): {min(1, 2 * (p > 0.5).mean()):.2f}")
    base = set(inv[chive])
    for name, m in [
        ("CHIVE >=50pp", chive),
        (f"Fisher p <= {pmax:.1g} (same stringency)", p <= pmax),
        ("Fisher p < 0.001", p < 0.001),
        ("BH q < 0.01", q < 0.01),
        ("BH q < 0.05", q < 0.05),
    ]:
        pos = set(inv[m])
        print(f"  {name:36s} {m.sum():6,d} experiments ({m.sum() / chive.sum():.2f}x), {len(pos):,} investigations (+{len(pos - base):,})")
    extra = (q < 0.05) & ~chive
    to_zero = (extra & (np.array([k for *_, k in rows]) == 0)).sum()
    print(f"  q < 0.05 but < 50pp: {extra.sum():,}, median move {100 * np.median(move[extra]):.0f}pp, to 0/30: {to_zero:,}")
    # TOST at alpha 0.05: the 90% Agresti-Caffo interval of the rate difference inside +-margin
    a = np.array([(k0 + 1) / 32 for _, _, k0, _ in rows])
    b = np.array([(k + 1) / 32 for _, _, _, k in rows])
    reach = np.abs(b - a) + 1.645 * np.sqrt((a * (1 - a) + b * (1 - b)) / 32)
    inert = move <= 0.15
    n_inert = int(inert.sum())
    print(f"  <=15pp 'inert': {n_inert:,}; shown within +-15pp {pct(int((inert & (reach <= 0.15)).sum()), n_inert)},", end=" ")
    print(f"within +-25pp {pct(int((inert & (reach <= 0.25)).sum()), n_inert)}")
    k0s = collections.Counter(d["k0"] for run in ADAM_RUNS for d in rates(run))
    print("  baseline k0 of investigations (0-12/30):", [k0s[k] for k in range(13)])


def contradictions():
    """§3 on Adam's data: hand labels of inert claims against the investigation's own findings (adam_inert_labels.jsonl)."""
    path = RUN / "adam_inert_labels.jsonl"
    if not path.exists():
        print("no adam_inert_labels.jsonl yet")
        return
    by_run = collections.defaultdict(list)
    for item in jsonl(path):
        by_run[item["run"]].append(item)
    for run, items in by_run.items():
        sampled = {i["prompt_id"] for i in items}
        contra = {i["prompt_id"] for i in items if i["label"] == "CONTRADICTS"}
        counts = dict(collections.Counter(i["label"] for i in items))
        print(f"{ADAM_LABELS[run]:13s} {len(items)} inert claims in {len(sampled)} sampled investigations: {counts}")
        print(f"{'':13s} investigations with a CONTRADICTS claim: {pct(len(contra), len(sampled))}")


def mistakes():
    """§4: Adam's own concreteness judge flags each investigated behaviour as a mistake or not (mechanism_concreteness.jsonl)."""
    for run in ADAM_RUNS:
        rows = adam(f"{run}/mechanism_concreteness.jsonl")
        n_mistake = sum(bool(r["is_mistake"]) for r in rows)
        cats = collections.Counter(c for r in rows for c in r["categories"]).most_common(6)
        print(f"{ADAM_LABELS[run]:13s} is_mistake {pct(n_mistake, len(rows))} of labelled investigations; top categories {cats}")
