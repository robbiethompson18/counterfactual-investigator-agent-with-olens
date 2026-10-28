"""Linear probe for Qwen's own MCQ answer at the option tokens (Kylie's idea, 2026-10-28).

Take hint_mcq questions Qwen answers consistently with no hint (modal no-hint answer in >= 7/8 samples), balanced between
consistently right and consistently wrong. Run Qwen3.6-27B (bf16, thinking off) on each no-hint prompt and save residual
activations at two sites per option: `end`, the option text's last token (usually its closing period; Kylie's site), and
`nl`, the newline token after the option. Label each option `is_answer` (= Qwen's modal
answer) and `is_gold`. Fit a logistic probe per layer, held out by question (GroupKFold), and ask whether it tracks Qwen's
answer or the truth: on wrong questions the two labels disagree.

    # GPU: select questions from the no-hint samples, forward pass, save activations + Qwen's letter logits
    uv run --with torch --with transformers --with accelerate python -m environments.hint_mcq.probe collect \\
        --runs 2026-10-27_scale2k 2026-10-27_scale2k-topup --slug scale2k
    # CPU: probes per layer and site, results.json + table.md (+ a PNG, gitignored)
    uv run --with scikit-learn python -m environments.hint_mcq.probe fit data/runs/mcq_answer_probe/<run>

RUN names are hint_mcq run folders under data/envs/hint_mcq/ (pull them from HF first). Activations (several GB) go to
data/probe_cache/<run name>/, outside the run folder, so the HF sync carries only the selection, logits and results.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from environments._lib.runfolder import LOCAL, REPO_ROOT, read_episodes
from environments.hint_mcq.env import LETTERS, build, first_letter

MODEL = "Qwen/Qwen3.6-27B"
CACHE = REPO_ROOT / "data" / "probe_cache"
SITES = ("end", "nl")


def select(runs: list[str], min_count: int, n_samples: int, seed: int) -> list[dict]:
    """Questions whose no-hint modal answer has >= min_count/n_samples, balanced right:wrong (the larger side subsampled)."""
    answers, gold = defaultdict(list), {}
    for run in runs:
        for ep in read_episodes(LOCAL / "hint_mcq" / run):
            if ep["arm"] != "nohint":
                continue
            a = first_letter((ep["messages"][-1].get("content") or "").strip())
            if a:
                answers[str(ep["item"])].append(a)
                gold[str(ep["item"])] = ep["meta"]["gold"]
    rows = []
    for item, xs in answers.items():
        own, k = Counter(xs).most_common(1)[0]
        if len(xs) >= n_samples and k / len(xs) >= min_count / n_samples:
            rows.append({"item": item, "own": own, "share": k / len(xs), "n": len(xs), "gold": gold[item], "right": own == gold[item]})
    right, wrong = [r for r in rows if r["right"]], [r for r in rows if not r["right"]]
    print(f"{len(answers)} items, {len(rows)} consistent: {len(right)} right, {len(wrong)} wrong")
    rng = random.Random(seed)
    n = min(len(right), len(wrong))
    return sorted(rng.sample(right, n) + rng.sample(wrong, n), key=lambda r: int(r["item"]))


def option_sites(text: str, start: int, n_opts: int) -> list[tuple[int, int]]:
    """(char of the option text's last character, char of the newline after it) for each option; `start` is where the user
    message begins in the rendered prompt. build() writes the options as "\\nA. text" lines ending at "\\n\\n"."""
    sites, pos = [], start
    for i in range(n_opts):
        pos = text.index(f"\n{LETTERS[i]}. ", pos)
        nxt = text.index(f"\n{LETTERS[i + 1]}. ", pos + 1) if i + 1 < n_opts else text.index("\n\n", pos + 1)
        sites.append((nxt - 1, nxt))
        pos += 1
    return sites


def collect(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    picked = select(a.runs, a.min_count, a.n_samples, a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    n_layers = model.config.get_text_config().num_hidden_layers
    layers = list(range(0, n_layers + 1, a.layer_stride)) + ([n_layers] if n_layers % a.layer_stride else [])
    run = new_run(
        "mcq_answer_probe",
        a.slug,
        description="Probe for Qwen3.6-27B's own MMLU-Pro answer at each option's last token and following newline (hint_mcq no-hint)",
        models=[{"name": MODEL, "served_by": "HF transformers bf16, output_hidden_states"}],
        params=vars(a) | {"layers": layers, "n_questions": len(picked)},
    )
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    acts = {s: [] for s in SITES}
    options, questions = [], []
    for qi, r in enumerate(picked):
        user = build("nohint", r["item"]).messages[0]["content"]
        msgs = [{"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        offs = enc["offset_mapping"]
        n_opts = len(_options(r["item"]))
        sites = option_sites(text, text.index(user), n_opts)
        pos = [next(i for i, (s, e) in enumerate(offs) if s <= c < e) for site in sites for c in site]  # end, nl, end, nl, ...
        with torch.no_grad():
            out = model(torch.tensor([enc["input_ids"]], device=model.device), output_hidden_states=True)
        hs = np.stack([out.hidden_states[layer][0, pos].float().cpu().numpy() for layer in layers], 1).astype(np.float16)  # [2*opts, L, d]
        logp = torch.log_softmax(out.logits[0, -1].float(), -1)[letter_ids[:n_opts]].tolist()
        questions.append(r | {"logp": logp, "logit_answer": LETTERS[int(np.argmax(logp))], "n_opts": n_opts})
        for i in range(n_opts):
            for j, s in enumerate(SITES):
                acts[s].append(hs[2 * i + j])
            options.append(
                {
                    "q": qi,
                    "item": r["item"],
                    "opt": i,
                    "letter": LETTERS[i],
                    "is_answer": LETTERS[i] == r["own"],
                    "is_gold": LETTERS[i] == r["gold"],
                    "right": r["right"],
                }
                | {f"{s}_token": tok.decode(enc["input_ids"][pos[2 * i + j]]) for j, s in enumerate(SITES)}
            )
        if qi % 50 == 0:
            print(qi, len(picked), flush=True)
    cache = CACHE / run.name
    cache.mkdir(parents=True, exist_ok=True)
    for s, xs in acts.items():
        np.save(cache / f"{s}.npy", np.stack(xs))  # [n_options, n_layers_saved, d]
    (run / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
    (run / "options.jsonl").write_text("".join(json.dumps(o) + "\n" for o in options))
    agree = np.mean([q["logit_answer"] == q["own"] for q in questions])
    print(f"saved {len(options)} options x {len(layers)} layers to {cache}; logit argmax = sampled modal answer on {agree:.1%}")


def _options(item: str) -> list[str]:
    from environments.hint_mcq.env import _table

    return list(_table().loc[str(item)].options)


def fit(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    run = Path(a.run)
    meta = json.loads((run / "meta.json").read_text())
    layers = meta["params"]["layers"]
    opts = [json.loads(line) for line in (run / "options.jsonl").read_text().splitlines()]
    q = np.array([o["q"] for o in opts])
    right = np.array([o["right"] for o in opts])
    # Kylie's site is the period ending each option; questions where every option's text ends in "." get their own split
    n_dot, n_opt = Counter(o["q"] for o in opts if o["end_token"].endswith(".")), Counter(o["q"] for o in opts)
    period = np.array([n_dot[o["q"]] == n_opt[o["q"]] for o in opts])
    print(
        f"end token ends in '.': {np.mean([o['end_token'].endswith('.') for o in opts]):.1%} of options; "
        f"every option does in {len(np.unique(q[period]))}/{len(n_opt)} questions"
    )
    y = {lab: np.array([o[lab] for o in opts]) for lab in ("is_answer", "is_gold")}
    folds = list(GroupKFold(n_splits=a.folds).split(q, groups=q))

    def cv_scores(X, lab):
        s = np.zeros(len(q))
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=a.C, max_iter=2000).fit(sc.transform(X[tr]), y[lab][tr])
            s[te] = clf.decision_function(sc.transform(X[te]))
        return s

    def metrics(s):
        """AUROC for is_answer, and per question whether the top-scored option is Qwen's answer or the gold, split by right/wrong."""
        top = {}
        for qi in np.unique(q):
            m = q == qi
            top[qi] = np.flatnonzero(m)[np.argmax(s[m])]
        tops = np.array(list(top.values()))
        out = {}
        splits = {"all": np.ones(len(q), bool), "right": right, "wrong": ~right}
        splits |= {f"{p}_{k}": m & pm for p, pm in (("period", period), ("noperiod", ~period)) for k, m in list(splits.items())}
        for name, mask in splits.items():
            t = tops[mask[tops]]
            out[name] = {
                "auroc_answer": roc_auc_score(y["is_answer"][mask], s[mask]),
                "auroc_gold": roc_auc_score(y["is_gold"][mask], s[mask]),
                "top1_answer": float(y["is_answer"][t].mean()),
                "top1_gold": float(y["is_gold"][t].mean()),
            }
        return out

    # baseline: the option's letter alone (Qwen's letter prior), same folds
    letter_onehot = np.eye(len(LETTERS))[[o["opt"] for o in opts]]
    results = {"letter_baseline": {lab: metrics(cv_scores(letter_onehot, lab)) for lab in y}, "layers": layers, "probes": {}}
    for site in a.sites:
        acts = np.load(CACHE / run.name / f"{site}.npy", mmap_mode="r")
        for li, layer in enumerate(layers):
            X = np.asarray(acts[:, li], dtype=np.float32)
            for lab in y:
                results["probes"][f"{site}/{layer}/{lab}"] = metrics(cv_scores(X, lab))
            r = results["probes"][f"{site}/{layer}/is_answer"]
            print(
                f"{site} L{layer}: is_answer probe top1 right {r['right']['top1_answer']:.2f} wrong {r['wrong']['top1_answer']:.2f} "
                f"(wrong: picks gold {r['wrong']['top1_gold']:.2f}), auroc {r['all']['auroc_answer']:.3f}",
                flush=True,
            )
    (run / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    (run / "table.md").write_text(table(results, a.sites))
    plot(results, a.sites, run)
    if a.finish:
        finish(run)


def table(res: dict, sites: list[str]) -> str:
    """One row per site and layer: the is_answer probe's top-1 (= Qwen's answer) on right and wrong questions, how often it
    picks the gold on wrong ones, and the is_gold probe's top-1 on wrong questions for comparison."""
    lines = [
        "| site | layer | AUROC is_answer | top-1 = answer (right q) | top-1 = answer (wrong q) | top-1 = gold (wrong q) "
        + "| is_gold probe: top-1 = gold (wrong q) | is_gold probe: top-1 = answer (wrong q) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    b = res["letter_baseline"]["is_answer"]
    lines.append(
        f"| letter only | | {b['all']['auroc_answer']:.3f} | {b['right']['top1_answer']:.2f} | {b['wrong']['top1_answer']:.2f} "
        f"| {b['wrong']['top1_gold']:.2f} | | |"
    )
    for site in sites:
        for layer in res["layers"]:
            p, g = res["probes"][f"{site}/{layer}/is_answer"], res["probes"][f"{site}/{layer}/is_gold"]
            cells = [p["all"]["auroc_answer"], p["right"]["top1_answer"], p["wrong"]["top1_answer"], p["wrong"]["top1_gold"]]
            cells += [g["wrong"]["top1_gold"], g["wrong"]["top1_answer"]]
            lines.append(f"| {site} | {layer} | " + " | ".join(f"{x:.3f}" for x in cells) + " |")
    lines += [
        "",
        '`end` site, is_answer probe, split by questions where every option\'s text ends in "." (period) or not (noperiod):',
        "",
        "| layer | period: top-1 = answer (right q) | period: top-1 = answer (wrong q) | period: top-1 = gold (wrong q) "
        + "| noperiod: top-1 = answer (right q) | noperiod: top-1 = answer (wrong q) | noperiod: top-1 = gold (wrong q) |",
        "|---|---|---|---|---|---|---|",
    ]
    for layer in res["layers"] if "end" in sites else []:
        p = res["probes"][f"end/{layer}/is_answer"]
        cells = [
            p[f"{s}_{k}"][m]
            for s in ("period", "noperiod")
            for k, m in (("right", "top1_answer"), ("wrong", "top1_answer"), ("wrong", "top1_gold"))
        ]
        lines.append(f"| {layer} | " + " | ".join(f"{x:.3f}" for x in cells) + " |")
    return "\n".join(lines) + "\n"


def plot(res: dict, sites: list[str], run: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(sites), figsize=(6 * len(sites), 4), sharey=True, squeeze=False)
    layers = res["layers"]
    for ax, site in zip(axes[0], sites, strict=True):
        get = lambda lab, split, k, site=site: [res["probes"][f"{site}/{layer}/{lab}"][split][k] for layer in layers]
        ax.plot(layers, get("is_answer", "right", "top1_answer"), color="#2a9d8f", label="is_answer probe, right q: picks answer")
        ax.plot(layers, get("is_answer", "wrong", "top1_answer"), color="#e76f51", label="is_answer probe, wrong q: picks answer")
        ax.plot(layers, get("is_answer", "wrong", "top1_gold"), color="#e76f51", ls="--", label="is_answer probe, wrong q: picks gold")
        ax.plot(layers, get("is_gold", "wrong", "top1_gold"), color="#8e44ad", ls=":", label="is_gold probe, wrong q: picks gold")
        ax.axhline(res["letter_baseline"]["is_answer"]["all"]["top1_answer"], color="#999", lw=1, label="letter-only baseline")
        ax.set(title=f"site: {site}", xlabel="layer", ylim=(0, 1))
    axes[0][0].set_ylabel("held-out questions, top-1 rate")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run / "probe_by_layer.png", dpi=130)


def _softmax_by_q(s: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Softmax of option scores within each question; options are stored grouped by question, `starts` = first index of each."""
    z = s - np.repeat(np.maximum.reduceat(s, starts), np.diff(np.append(starts, len(s))))
    e = np.exp(z)
    return e / np.repeat(np.add.reduceat(e, starts), np.diff(np.append(starts, len(s))))


def _fit_listwise(X: np.ndarray, t: np.ndarray, starts: np.ndarray, l2: float, n_free: int = 0) -> np.ndarray:
    """Conditional logit: p(i|q) = softmax_q(X w); minimise mean over questions of CE(t, p) + l2/2 |w|^2 with L-BFGS. The
    last `n_free` columns (per-letter biases) are not penalised."""
    from scipy.optimize import minimize

    n_q = len(starts)

    def f(w):
        p = _softmax_by_q(X @ w, starts)
        wp = w.copy()
        wp[len(w) - n_free :] = 0
        loss = -np.sum(t * np.log(p + 1e-12)) / n_q + 0.5 * l2 * wp @ wp
        return loss, X.T @ (p - t) / n_q + l2 * wp

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x


def pmf(a):
    """A distribution over each question's options from one (site, layer): listwise softmax probes trained on Qwen's modal
    answer (one-hot) or on its own letter pmf (its logits restricted to the question's letters), vs the binary is_answer
    probe's scores softmaxed per question, and uniform. Held out by question, same folds as `fit`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    run = Path(a.run)
    meta = json.loads((run / "meta.json").read_text())
    layers = meta["params"]["layers"]
    if a.site is None:
        res = json.loads((run / "results.json").read_text())["probes"]
        key = max((k for k in res if k.endswith("/is_answer")), key=lambda k: res[k]["all"]["auroc_answer"])
        a.site, a.layer = key.split("/")[0], int(key.split("/")[1])
    print(f"site {a.site}, layer {a.layer}")
    opts = [json.loads(line) for line in (run / "options.jsonl").read_text().splitlines()]
    qs = [json.loads(line) for line in (run / "questions.jsonl").read_text().splitlines()]
    q = np.array([o["q"] for o in opts])
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    is_answer = np.array([o["is_answer"] for o in opts], float)
    is_gold = np.array([o["is_gold"] for o in opts], float)
    right = np.array([r["right"] for r in qs])
    qwen = _softmax_by_q(np.concatenate([r["logp"] for r in qs]), starts)  # Qwen's pmf over the question's letters
    X = np.asarray(np.load(CACHE / run.name / f"{a.site}.npy", mmap_mode="r")[:, layers.index(a.layer)], dtype=np.float64)
    folds = list(GroupKFold(n_splits=a.folds).split(q, groups=q))

    def sub(idx):  # option indices -> (rows, question starts within them)
        qq = q[idx]
        return idx, np.flatnonzero(np.r_[True, qq[1:] != qq[:-1]])

    letter = np.eye(len(LETTERS))[[o["opt"] for o in opts]]

    def cv(target: np.ndarray, l2: float, folds, feats: str) -> np.ndarray:
        """feats: "act" (w.h), "act+letter" (w.h + b_letter, biases unpenalised) or "letter" (b_letter only)."""
        s = np.zeros(len(q))
        for tr, te in folds:
            if feats == "letter":
                w = _fit_listwise(letter[tr], target[tr], sub(tr)[1], 0.0, n_free=len(LETTERS))
                s[te] = letter[te] @ w
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            n_free = len(LETTERS) if feats == "act+letter" else 0
            if n_free:
                Xtr, Xte = np.hstack([Xtr, letter[tr]]), np.hstack([Xte, letter[te]])
            w = _fit_listwise(Xtr, target[tr], sub(tr)[1], l2, n_free)
            s[te] = Xte @ w
        return s

    def pick_l2(target, feats):  # inner GroupKFold on the first outer fold's training questions
        tr = folds[0][0]
        inner = [(tr[i], tr[j]) for i, j in GroupKFold(n_splits=4).split(tr, groups=q[tr])][:1]
        nll = {}
        for l2 in a.l2:
            s = cv(target, l2, inner, feats)
            te = inner[0][1]
            p = _softmax_by_q(s[te], sub(te)[1])
            nll[l2] = -np.sum(target[te] * np.log(p + 1e-12)) / len(sub(te)[1])
        print("  l2 grid (inner CE):", {k: round(v, 3) for k, v in nll.items()}, flush=True)
        return min(nll, key=lambda k: nll[k])

    def binary_scores():
        s = np.zeros(len(q))
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            s[te] = LogisticRegression(C=a.C, max_iter=2000).fit(sc.transform(X[tr]), is_answer[tr]).decision_function(sc.transform(X[te]))
        return s

    def metrics(p: np.ndarray) -> dict:
        ends = np.append(starts, len(q))
        top = np.array([starts[i] + np.argmax(p[starts[i] : ends[i + 1]]) for i in range(len(starts))])
        p_ans = np.add.reduceat(p * is_answer, starts)
        kl = np.add.reduceat(qwen * (np.log(qwen + 1e-12) - np.log(p + 1e-12)), starts)
        conf, hit = p[top], is_answer[top]
        out = {}
        for name, m in (("all", np.ones(len(starts), bool)), ("right", right), ("wrong", ~right)):
            bins = np.minimum((conf[m] * 10).astype(int), 9)
            ece = sum(abs(conf[m][bins == b].mean() - hit[m][bins == b].mean()) * (bins == b).mean() for b in np.unique(bins))
            out[name] = {
                "top1_answer": float(hit[m].mean()),
                "top1_gold": float(is_gold[top][m].mean()),
                "nll_answer": float(-np.log(p_ans[m] + 1e-12).mean()),
                "kl_qwen": float(kl[m].mean()),
                "mean_top_p": float(conf[m].mean()),
                "ece": float(ece),
            }
        return out

    out = {"site": a.site, "layer": a.layer, "methods": {}, "l2": {}}
    out["methods"]["uniform"] = metrics(_softmax_by_q(np.zeros(len(q)), starts))
    out["methods"]["binary probe, softmaxed"] = metrics(_softmax_by_q(binary_scores(), starts))
    for tname, target in (("answer one-hot", is_answer), ("Qwen letter pmf", qwen)):
        name = f"letter only, target {tname}"
        out["methods"][name] = metrics(_softmax_by_q(cv(target, 0.0, folds, "letter"), starts))
        for feats in ("act", "act+letter"):
            name = f"listwise {feats}, target {tname}"
            print(name, flush=True)
            out["l2"][name] = l2 = pick_l2(target, feats)
            out["methods"][name] = metrics(_softmax_by_q(cv(target, l2, folds, feats), starts))
    out["methods"]["Qwen's own letter pmf (ceiling)"] = metrics(qwen)
    (run / "pmf.json").write_text(json.dumps(out, indent=1) + "\n")
    cols = ("top1_answer", "top1_gold", "nll_answer", "kl_qwen", "mean_top_p", "ece")
    lines = [
        f"PMF over options from `{a.site}` layer {a.layer}, held-out questions (GroupKFold {a.folds}); l2 per listwise probe: {out['l2']}",
        "",
        "| method | split | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for name, r in out["methods"].items():
        for split in ("all", "right", "wrong"):
            lines.append(f"| {name} | {split} | " + " | ".join(f"{r[split][c]:.3f}" for c in cols) + " |")
    (run / "pmf.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--slug", required=True)
    c.add_argument("--min-count", type=int, default=7)
    c.add_argument("--n-samples", type=int, default=8)
    c.add_argument("--layer-stride", type=int, default=2)
    c.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("fit")
    f.add_argument("run")
    f.add_argument("--sites", nargs="+", default=list(SITES))
    f.add_argument("--folds", type=int, default=5)
    f.add_argument("--C", type=float, default=0.01)
    f.add_argument("--finish", action="store_true", help="mark the run done and sync it to HF")
    p = sub.add_parser("pmf")
    p.add_argument("run")
    p.add_argument("--site", help="default: the (site, layer) with the best is_answer AUROC in results.json")
    p.add_argument("--layer", type=int)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--l2", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    a = ap.parse_args()
    {"collect": collect, "fit": fit, "pmf": pmf}[a.cmd](a)


if __name__ == "__main__":
    main()
