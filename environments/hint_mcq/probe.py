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
from environments.hint_mcq.env import LETTERS, bare_letter, build

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
            a = bare_letter(ep["messages"][-1].get("content") or "")
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


def _center(X: np.ndarray, q: np.ndarray, within: bool, mu: np.ndarray | None) -> np.ndarray:
    """Subtract the training mean, or (within) each question's own mean option activation: the softmax over a question's
    options ignores per-question constants, so within-question residuals are the natural covariance for it."""
    if not within:
        return X - mu
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    n = np.diff(np.append(starts, len(q)))
    return X - np.repeat(np.add.reduceat(X, starts) / n[:, None], n, axis=0)


class Whitener:
    """Fit on training options: centering (global or within-question) plus the eigenbasis of the covariance. `transform`
    then whitens as `kind`: diag (per-dim std), shrink (eigenvalues (1-α)e + α·mean(e), α="lw" = Ledoit-Wolf's estimate) or
    pca (top k components). Rotation is left in the eigenbasis: with an isotropic L2 on w it doesn't change the fit."""

    def __init__(self, X: np.ndarray, q: np.ndarray, within: bool):
        from sklearn.covariance import ledoit_wolf_shrinkage

        self.within, self.mu = within, X.mean(0)
        Xc = _center(X, q, within, self.mu)
        self.std = Xc.std(0) + 1e-6
        self.e, self.V = np.linalg.eigh(Xc.T @ Xc / len(Xc))  # ascending
        self.e = np.maximum(self.e, 0)
        self.lw = float(ledoit_wolf_shrinkage(Xc, assume_centered=True))

    def transform(self, X: np.ndarray, q: np.ndarray, kind: str, param) -> np.ndarray:
        Xc = _center(X, q, self.within, self.mu)
        if kind == "diag":
            return Xc / self.std
        if kind == "none":  # de-meaned only: a global mean is a no-op under the per-question softmax, so this is raw h
            return Xc
        if kind == "pca":
            return Xc @ self.V[:, -param:] / np.sqrt(self.e[-param:] + 1e-12)
        alpha = self.lw if param == "lw" else param
        return Xc @ self.V / np.sqrt((1 - alpha) * self.e + alpha * self.e.mean())


WHITEN = {  # variant -> (within-question centering, kind, hyperparameter grid)
    "diag": (False, "diag", [None]),
    "demean": (False, "none", [None]),
    "shrink": (False, "shrink", ["lw", 0.01, 0.1, 0.3]),
    "pca": (False, "pca", [64, 128, 256, 512, 1024, 2048]),
    "within_diag": (True, "diag", [None]),
    "within_shrink": (True, "shrink", ["lw", 0.01, 0.1, 0.3]),
}


def whiten(a):
    """Listwise (one-hot target) pmf probe at one (site, layer) under different whitenings, nested CV: for each outer fold
    the whitening hyperparameter and L2 are chosen by a 4-fold inner GroupKFold on its training questions, then refit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    run = Path(a.run)
    out_dir = Path(a.out or run)
    layers = json.loads((run / "meta.json").read_text())["params"]["layers"]
    opts = [json.loads(line) for line in (run / "options.jsonl").read_text().splitlines()]
    qs = [json.loads(line) for line in (run / "questions.jsonl").read_text().splitlines()]
    q = np.array([o["q"] for o in opts])
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    is_answer = np.array([o["is_answer"] for o in opts], float)
    is_gold = np.array([o["is_gold"] for o in opts], float)
    right = np.array([r["right"] for r in qs])
    qwen = _softmax_by_q(np.concatenate([r["logp"] for r in qs]), starts)
    acts = np.load(CACHE / run.name / f"{a.site}.npy", mmap_mode="r")
    folds = list(GroupKFold(n_splits=a.folds).split(q, groups=q))
    sts = lambda idx: np.flatnonzero(np.r_[True, q[idx][1:] != q[idx][:-1]])

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
                "ece": float(ece),
            }
        return out

    def fit_eval(X, tr, te, within, kind, param, l2):
        w = Whitener(X[tr], q[tr], within) if isinstance(within, bool) else within  # a fitted Whitener can be passed
        coef = _fit_listwise(w.transform(X[tr], q[tr], kind, param), is_answer[tr], sts(tr), l2)
        return w.transform(X[te], q[te], kind, param) @ coef

    def nested(X, variant, fixed=None):
        """Held-out scores for every option; per outer fold the (param, l2) with the lowest inner CE, unless `fixed`."""
        within, kind, grid = WHITEN[variant]
        s, chosen = np.zeros(len(q)), []
        for tr, te in folds:
            if fixed is None:
                ce = defaultdict(float)
                for itr, ite in GroupKFold(n_splits=4).split(tr, groups=q[tr]):
                    itr, ite = tr[itr], tr[ite]
                    w = Whitener(X[itr], q[itr], within)
                    for param in grid:
                        for l2 in a.l2:
                            p = _softmax_by_q(fit_eval(X, itr, ite, w, kind, param, l2), sts(ite))
                            ce[(param, l2)] -= np.sum(is_answer[ite] * np.log(p + 1e-12))
                best = min(ce, key=lambda k: ce[k])
            else:
                best = fixed
            chosen.append(best)
            s[te] = fit_eval(X, tr, te, within, kind, *best)
        return s, chosen

    X = np.asarray(acts[:, layers.index(a.layer)], dtype=np.float64)
    out = {"site": a.site, "layer": a.layer, "variants": {}, "chosen": {}}
    for variant in a.variants:
        s, chosen = nested(X, variant)
        out["variants"][variant], out["chosen"][variant] = metrics(_softmax_by_q(s, starts)), [list(c) for c in chosen]
        r = out["variants"][variant]
        print(f"{variant}: chosen {chosen}; top1 {r['all']['top1_answer']:.3f}, nll {r['all']['nll_answer']:.3f}", flush=True)
    best = min(a.variants, key=lambda v: out["variants"][v]["all"]["nll_answer"])
    # the binary is_answer probe under the best whitening (its modal hyperparameter, no L2 retune: C as in `fit`)
    within, kind, _ = WHITEN[best]
    param = Counter(c[0] for c in out["chosen"][best]).most_common(1)[0][0]
    sb = np.zeros(len(q))
    for tr, te in folds:
        w = Whitener(X[tr], q[tr], within)
        clf = LogisticRegression(C=a.C, max_iter=2000).fit(w.transform(X[tr], q[tr], kind, param), is_answer[tr])
        sb[te] = clf.decision_function(w.transform(X[te], q[te], kind, param))
    out["binary_under_best"] = {"variant": best, "param": param, "auroc": roc_auc_score(is_answer, sb)} | metrics(_softmax_by_q(sb, starts))
    # quick layer check for the best variant at its modal (param, l2)
    fixed = Counter(tuple(c) for c in out["chosen"][best]).most_common(1)[0][0]
    out["layer_check"] = {"variant": best, "fixed": list(fixed), "layers": {}}
    for layer in a.check_layers:
        Xl = np.asarray(acts[:, layers.index(layer)], dtype=np.float64)
        out["layer_check"]["layers"][layer] = metrics(_softmax_by_q(nested(Xl, best, fixed)[0], starts))
        print(f"layer {layer}: {out['layer_check']['layers'][layer]['all']}", flush=True)
    (out_dir / f"{a.name}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    cols = ("top1_answer", "top1_gold", "nll_answer", "kl_qwen", "ece")
    lines = [
        f"Listwise pmf probe (one-hot target), `{a.site}` L{a.layer}, by whitening; nested CV (outer {a.folds}, inner 4 by question).",
        "",
        "| whitening | chosen (param, l2) per outer fold | split | " + " | ".join(cols) + " |",
        "|---|---|---|" + "---|" * len(cols),
    ]
    for v, r in out["variants"].items():
        for split in ("all", "right", "wrong"):
            ch = ", ".join(f"({p}, {l})" for p, l in out["chosen"][v]) if split == "all" else ""
            lines.append(f"| {v} | {ch} | {split} | " + " | ".join(f"{r[split][c]:.3f}" for c in cols) + " |")
    b = out["binary_under_best"]
    lines += [
        "",
        (
            f"Binary is_answer probe under `{best}` ({param}): AUROC {b['auroc']:.3f}, top-1 = answer {b['all']['top1_answer']:.3f} "
            f"(right {b['right']['top1_answer']:.3f}, wrong {b['wrong']['top1_answer']:.3f}), gold on wrong {b['wrong']['top1_gold']:.3f}, "
            f"NLL softmaxed {b['all']['nll_answer']:.3f}"
        ),
        "",
        f"Layer check, `{best}` at fixed {fixed}:",
        "",
        "| layer | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for layer, r in out["layer_check"]["layers"].items():
        lines.append(f"| {layer} | " + " | ".join(f"{r['all'][c]:.3f}" for c in cols) + " |")
    (out_dir / f"{a.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def dims(a):
    """Per-dimension spread of one (site, layer)'s activations: std and mean |h| over options, the top dims by std against
    the median, and whether those dims are large in every option (axis-aligned massive activations) or in a few."""
    run = Path(a.run)
    layers = json.loads((run / "meta.json").read_text())["params"]["layers"]
    X = np.asarray(np.load(CACHE / run.name / f"{a.site}.npy", mmap_mode="r")[:, layers.index(a.layer)], dtype=np.float64)
    q = np.array([json.loads(line)["q"] for line in (run / "options.jsonl").read_text().splitlines()])
    sd, mabs, mean = X.std(0), np.abs(X).mean(0), X.mean(0)
    n = np.bincount(q)
    within = (X - np.repeat(np.add.reduceat(X, np.flatnonzero(np.r_[True, q[1:] != q[:-1]])) / n[:, None], n, 0)).std(0)
    top = np.argsort(-sd)[: a.top]
    rank = np.argsort(np.argsort(-np.abs(X), 1), 1)  # per option, each dim's rank by |h|
    norm2 = (X**2).sum(1)
    lines = [
        (
            f"`{a.site}` hidden_states[{a.layer}], {len(X)} options of {len(set(q))} questions, d = {X.shape[1]}. "
            f"Median std {np.median(sd):.3g}, median mean |h| {np.median(mabs):.3g}; |h| norm median {np.median(np.sqrt(norm2)):.3g}."
        ),
        "",
        "| dim | std | std / median | within-question std | mean | mean abs | share of options with it in top-10 abs | share of norm^2 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in top:
        lines.append(
            f"| {d} | {sd[d]:.3g} | {sd[d] / np.median(sd):.1f} | {within[d]:.3g} | {mean[d]:.3g} | {mabs[d]:.3g} | "
            f"{(rank[:, d] < 10).mean():.2f} | {np.mean(X[:, d] ** 2 / norm2):.3f} |"
        )
    top_abs = np.argsort(-mabs)[: a.top]
    lines += ["", f"Top {a.top} dims by mean |h|: " + ", ".join(f"{d} ({mabs[d] / np.median(mabs):.0f}x)" for d in top_abs)]
    (Path(a.out) / "dims.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def dump(a):
    """One row per question with the diagonal listwise probe (one-hot target, nl L52 by default) over its options: raw
    scores s = w·z, log pmf (s minus the question's logsumexp) and pmf, from the outer fold that held the question out and
    from one fit on all questions (`_full`), next to Qwen's own letter log pmf. Written to pmf_options.jsonl."""
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    from environments.hint_mcq.env import _table

    run = Path(a.run)
    layers = json.loads((run / "meta.json").read_text())["params"]["layers"]
    opts = [json.loads(line) for line in (run / "options.jsonl").read_text().splitlines()]
    qs = [json.loads(line) for line in (run / "questions.jsonl").read_text().splitlines()]
    q = np.array([o["q"] for o in opts])
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    sts = lambda idx: np.flatnonzero(np.r_[True, q[idx][1:] != q[idx][:-1]])
    is_answer = np.array([o["is_answer"] for o in opts], float)
    X = np.asarray(np.load(CACHE / run.name / f"{a.site}.npy", mmap_mode="r")[:, layers.index(a.layer)], dtype=np.float64)

    def fit_score(tr, te):
        sc = StandardScaler().fit(X[tr])
        return sc.transform(X[te]) @ _fit_listwise(sc.transform(X[tr]), is_answer[tr], sts(tr), a.l2)

    held, fold = np.zeros(len(q)), np.zeros(len(q), int)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=a.folds).split(q, groups=q)):
        held[te], fold[te] = fit_score(tr, te), k
    every = np.arange(len(q))
    full = fit_score(every, every)
    logpmf = lambda s: np.log(_softmax_by_q(s, starts))
    qwen = np.concatenate([r["logp"] for r in qs])
    cols = {"qwen_logp": logpmf(qwen), "probe_s": held, "probe_logp": logpmf(held), "probe_s_full": full, "probe_logp_full": logpmf(full)}
    table = _table()
    ends = np.append(starts, len(q))
    with open(run / "pmf_options.jsonl", "w") as f:
        for qi, r in enumerate(qs):
            sl = slice(starts[qi], ends[qi + 1])
            row = table.loc[str(r["item"])]
            out = {"item": r["item"], "category": row.category, "question": row.question, "options": list(row.options)}
            out |= {"letters": list(LETTERS[: r["n_opts"]]), "own": r["own"], "gold": r["gold"], "right": r["right"]}
            out |= {"share": r["share"], "fold": int(fold[starts[qi]])}
            out |= {k: [round(float(x), 4) for x in v[sl]] for k, v in cols.items()}
            out["probe_p"] = [round(float(x), 4) for x in np.exp(cols["probe_logp"][sl])]
            out["probe_p_full"] = [round(float(x), 4) for x in np.exp(cols["probe_logp_full"][sl])]
            f.write(json.dumps(out) + "\n")
    print(f"wrote {run / 'pmf_options.jsonl'} ({len(qs)} questions; site {a.site} L{a.layer}, l2 {a.l2})")


def lens_requests(a):
    """O-lens read requests (scripts/olens_read_requests.py) at each option's `nl` token of the no-hint prompt: one row per
    question, `char_positions` = the newline after each option, as offsets into the user message. Also writes the saved
    `nl` activations of the first --check questions at hidden_states index --check-hidden (block check-hidden - 1) for
    scripts/olens_capture_check.py."""
    run = Path(a.run)
    layers = json.loads((run / "meta.json").read_text())["params"]["layers"]
    qs = [json.loads(line) for line in (run / "questions.jsonl").read_text().splitlines()]
    out = Path(a.out)
    rows, n = [], 0
    for r in qs:
        user = build("nohint", r["item"]).messages[0]["content"].strip()
        text = "\n" + user + "\n\n"  # option_sites searches "\nA. " lines that end at "\n\n"; offsets shift back by the "\n"
        offs = [nl - 1 for _, nl in option_sites(text, 0, r["n_opts"])]
        rows.append(
            {
                "id": f"hint_mcq/{r['item']}",
                "env": "hint_mcq",
                "arm": "nohint",
                "item": r["item"],
                "messages": [{"role": "user", "content": user}],
            }
        )
        rows[-1]["char_positions"] = [[0, c] for c in offs]
        n += len(offs)
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    k = sum(r["n_opts"] for r in qs[: a.check])
    ref = np.asarray(np.load(CACHE / run.name / "nl.npy", mmap_mode="r")[:k, layers.index(a.check_hidden)], dtype=np.float32)
    np.save(out.with_suffix(".check.npy"), ref)
    print(f"wrote {out}: {len(rows)} questions, {n} positions; check: {k} options at hidden_states[{a.check_hidden}]")


ENDORSE = r"correct (answer|term|choice|option)|is correct|the answer is|best (answer|fits|describes)|most (accurate|appropriate)"
REJECT = r"incorrect|is wrong|why (this|it) is wrong|not (the )?correct|distractor"


def lens_stats(a):
    """Crude regex reading of the O-lens reads at the option newlines (lens-requests -> olens_read_requests.py): the share of
    options whose read matches ENDORSE / REJECT per block, by option kind (Qwen's answer / gold / other) and right vs wrong
    question; the endorse rate at Qwen's option by quintile of Qwen's own p(answer) and of the probe's held-out p(answer);
    and a lens-only pick per question (highest endorse - reject count at --block, ties -> none) vs the probe."""
    import re

    run = Path(a.run)
    rows = {str(r["item"]): r for r in map(json.loads, (run / "pmf_options.jsonl").read_text().splitlines())}
    text = defaultdict(lambda: defaultdict(dict))  # item -> token_index -> block -> bullets joined
    for f in a.reads:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            text[r["id"].split("/", 1)[1]][r["token_index"]][r["layer"]] = " ".join(b["text"] for b in r["bullets"])
    blocks = sorted({b for pos in text.values() for v in pos.values() for b in v})
    rx = {"endorse": re.compile(ENDORSE, re.IGNORECASE), "reject": re.compile(REJECT, re.IGNORECASE)}
    opts = []  # (item, option index, kind, right, {block: text})
    for item, pos in text.items():
        r = rows[item]
        for j, p in enumerate(sorted(pos)):
            L = r["letters"][j]
            opts.append((item, j, "own" if L == r["own"] else "gold" if L == r["gold"] else "other", r["right"], pos[p]))
    lines = [f"O-lens regex stats over {len(text)} questions, {len(opts)} options. ENDORSE = `{ENDORSE}`, REJECT = `{REJECT}`.", ""]
    for name, pat in rx.items():
        lines += [
            f"Share of options whose read matches {name}:",
            "",
            "| block | own (right q) | own (wrong q) | gold (wrong q) | other (right q) | other (wrong q) |",
        ]
        lines.append("|---|---|---|---|---|---|")
        for b in blocks:
            cells = []
            for kind, right in (("own", True), ("own", False), ("gold", False), ("other", True), ("other", False)):
                hits = [bool(pat.search(t.get(b, ""))) for _, _, k, rt, t in opts if k == kind and rt == right]
                cells.append(f"{np.mean(hits):.2f}")
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        lines.append("")
    # endorse rate at Qwen's option vs confidence
    own = [(rows[i], t) for i, j, k, rt, t in opts if k == "own"]
    for conf in ("qwen", "probe"):
        p = np.array([np.exp(r["qwen_logp" if conf == "qwen" else "probe_logp"][r["letters"].index(r["own"])]) for r, _ in own])
        edges = np.quantile(p, [0.2, 0.4, 0.6, 0.8])
        lines += [f"Endorse rate at Qwen's option at block {a.block}, by quintile of {conf} p(Qwen's answer):", ""]
        lines += ["| quintile | p range | right q: endorse (n) | wrong q: endorse (n) |", "|---|---|---|---|"]
        qn = np.searchsorted(edges, p)
        for k in range(5):
            m = qn == k
            cell = []
            for right in (True, False):
                hit = [bool(rx["endorse"].search(t.get(a.block, ""))) for (r, t), mm in zip(own, m) if mm and r["right"] == right]
                cell.append(f"{np.mean(hit):.2f} ({len(hit)})" if hit else "–")
            lines.append(f"| {k + 1} | {p[m].min():.3f}–{p[m].max():.3f} | " + " | ".join(cell) + " |")
        lines.append("")
    # lens-only pick at --block vs the probe
    res = {True: defaultdict(int), False: defaultdict(int)}
    for item, pos in text.items():
        r = rows[item]
        ts = [pos[p].get(a.block, "") for p in sorted(pos)]
        sc = np.array([len(rx["endorse"].findall(t)) - len(rx["reject"].findall(t)) for t in ts])
        d = res[r["right"]]
        d["n"] += 1
        best = np.flatnonzero(sc == sc.max())
        if len(best) == 1:
            d["picked"] += 1
            d["lens_answer"] += r["letters"][best[0]] == r["own"]
            d["lens_gold"] += r["letters"][best[0]] == r["gold"]
        probe_top = r["letters"][int(np.argmax(r["probe_s"]))]
        d["probe_answer"] += probe_top == r["own"]
        d["probe_gold"] += probe_top == r["gold"]
    lines += [f"Lens-only pick at block {a.block} (highest endorse − reject match count; ties → no pick) vs the probe's top:", ""]
    lines += [
        "| questions | n | lens picks | lens = Qwen's answer | lens = gold | probe = Qwen's answer | probe = gold |",
        "|---|---|---|---|---|---|---|",
    ]
    for right, d in res.items():
        n = d["n"]
        lines.append(
            f"| {'right' if right else 'wrong'} | {n} | {d['picked'] / n:.2f} | {d['lens_answer'] / n:.2f} | {d['lens_gold'] / n:.2f} "
            f"| {d['probe_answer'] / n:.2f} | {d['probe_gold'] / n:.2f} |"
        )
    (run / "lens_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def tqa_prep(a):
    """The TruthfulQA out-of-sample set: hinted_truthfulqa `control` items whose no-hint modal answer has >= min_count/
    n_samples (all of them, not balanced), rendered as build("control", item) with the thinking-off generation prompt and
    --trail ("\\n\\n") inserted after the last option, so every option is followed by a newline token as in hint_mcq.
    Sites per option: `nl`, the token right after the option's text, and `end`, the text's last token. Writes the run
    (questions, options) and, in --work, the inputs for scripts/hs_at_positions.py (candidates "(A", "(B", ...: Qwen's
    replies open with "(X)" and "(X" is one token), the same for the unedited prompt (`_orig`, whose last nl site is
    <|im_end|>) and O-lens requests at the edited prompt's nl sites (`prompt_ids`)."""
    from transformers import AutoTokenizer

    from counterfactual_investigator_agent_with_olens.render import render_request
    from environments.hinted_truthfulqa import env as tqa

    answers, gold = defaultdict(list), {}
    for run in a.runs:
        for ep in read_episodes(LOCAL / "hinted_truthfulqa" / run):
            if ep["arm"] == "control" and str(ep["item"]).startswith("tqa-") and (ans := tqa.measure(ep)["answer"]):
                answers[str(ep["item"])].append(ans)
                gold[str(ep["item"])] = ep["meta"]["gold"]
    picked = []
    for item, xs in answers.items():
        own, k = Counter(xs).most_common(1)[0]
        if len(xs) >= a.n_samples and k / len(xs) >= a.min_count / a.n_samples:
            picked.append({"item": item, "own": own, "share": k / len(xs), "n": len(xs), "gold": gold[item], "right": own == gold[item]})
    picked.sort(key=lambda r: int(str(r["item"]).split("-")[1]))
    n_right = sum(r["right"] for r in picked)
    print(f"{len(answers)} items, {len(picked)} consistent: {n_right} right, {len(picked) - n_right} wrong")
    tok = AutoTokenizer.from_pretrained(MODEL)
    cand = [tok.encode(f"({L}", add_special_tokens=False) for L in tqa.LETTERS]
    assert all(len(c) == 1 for c in cand), "each (X must be one token"
    run = new_run(
        "mcq_answer_probe",
        a.slug,
        description="TruthfulQA out-of-sample set for the frozen MMLU-Pro answer probe (hinted_truthfulqa control, Qwen3.6-27B no-hint)",
        models=[{"name": MODEL, "served_by": "HF transformers bf16 (scripts/hs_at_positions.py)"}],
        params=vars(a) | {"n_questions": len(picked), "n_right": n_right},
    )
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)

    def render(msgs, n_opts, trail):
        """Token ids (thinking-off generation prompt) and, per option, the token after its text (nl) and its text's last
        token (end). `trail` goes into the rendered text right after the last option: Qwen's template trims message content,
        so it can't go through the message."""
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        start = text.index(msgs[-1]["content"])
        stop = start + len(msgs[-1]["content"])
        if not trail:
            _, rids, _ = render_request(tok, {"messages": msgs})  # the O-lens reader's rendering: the same tokens
            assert rids == tok(text, add_special_tokens=False)["input_ids"][: len(rids)], msgs
        text = text[:stop] + trail + text[stop:]
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        starts = [text.index(f"\n({tqa.LETTERS[i]}) ", start) for i in range(n_opts)]
        offs = enc["offset_mapping"]
        nl = [next(i for i, (s, e) in enumerate(offs) if s <= c < e) for c in [*starts[1:], stop]]
        end = [next(i for i, (s, e) in enumerate(offs) if s <= c - 1 < e) for c in [*starts[1:], stop]]
        return enc["input_ids"], nl, end

    questions, options, inputs, orig, requests = [], [], [], [], []
    for qi, r in enumerate(picked):
        ep = tqa.build("control", r["item"])
        opts = ep.meta["options"]
        ids, nl, end = render(ep.messages, len(opts), a.trail)
        oids, onl, oend = render(ep.messages, len(opts), "")
        questions.append(r | {"n_opts": len(opts), "question": tqa._items()[str(r["item"])]["question"], "options": opts})
        for i in range(len(opts)):
            L = tqa.LETTERS[i]
            options.append(
                {
                    "q": qi,
                    "item": r["item"],
                    "opt": i,
                    "letter": L,
                    "is_answer": L == r["own"],
                    "is_gold": L == r["gold"],
                    "right": r["right"],
                }
                | {
                    "nl_token": tok.decode([ids[nl[i]]]),
                    "end_token": tok.decode([ids[end[i]]]),
                    "orig_nl_token": tok.decode([oids[onl[i]]]),
                }
            )
        cands = [c[0] for c in cand[: len(opts)]]
        inputs.append({"id": r["item"], "ids": ids, "sites": {"nl": nl, "end": end}, "cand": cands})
        orig.append({"id": r["item"], "ids": oids, "sites": {"nl": onl, "end": oend}, "cand": cands})
        requests.append(
            {
                "id": f"hinted_truthfulqa/{r['item']}",
                "env": "hinted_truthfulqa",
                "arm": "control",
                "messages": ep.messages,
                "prompt_ids": ids,
                "positions": nl,
            }
        )
    (run / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
    (run / "options.jsonl").write_text("".join(json.dumps(o) + "\n" for o in options))
    (work / "hs_inputs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inputs))
    (work / "hs_inputs_orig.jsonl").write_text("".join(json.dumps(x) + "\n" for x in orig))
    (work / "lens_requests.jsonl").write_text("".join(json.dumps(x) + "\n" for x in requests))
    print(f"run {run}; {len(options)} options; nl tokens {Counter(o['nl_token'] for o in options).most_common(4)}; inputs in {work}")
    print(f"original prompt's nl tokens {Counter(o['orig_nl_token'] for o in options).most_common(4)}")


def oos(a):
    """The frozen probe on the TruthfulQA set: the diagonal listwise probe (one-hot target, nl, hidden_states[--hidden],
    l2 1) and the binary is_answer probe (C as in `fit`), each fit once on every question of the training run, applied
    unchanged to the test run's activations (scripts/hs_at_positions.py output in data/probe_cache/<test run>/). Variant
    `last_end`: the last option scored at its text's last token (its nl site is <|im_end|>). Writes oos.json, oos.md and a
    pmf_options.jsonl (as `dump`'s; fold -1, `_full` = the same frozen scores) to the test run."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from environments.hinted_truthfulqa.env import LETTERS as TQA_LETTERS

    train, test = Path(a.train), Path(a.test)
    tl = json.loads((train / "meta.json").read_text())["params"]["layers"]
    topts = [json.loads(line) for line in (train / "options.jsonl").read_text().splitlines()]
    tq = np.array([o["q"] for o in topts])
    ty = np.array([o["is_answer"] for o in topts], float)
    Xtr = np.asarray(np.load(CACHE / train.name / "nl.npy", mmap_mode="r")[:, tl.index(a.hidden)], dtype=np.float64)
    sc = StandardScaler().fit(Xtr)
    w = _fit_listwise(sc.transform(Xtr), ty, np.flatnonzero(np.r_[True, tq[1:] != tq[:-1]]), a.l2)
    clf = LogisticRegression(C=a.C, max_iter=2000).fit(sc.transform(Xtr), ty)

    qs = [json.loads(line) for line in (test / "questions.jsonl").read_text().splitlines()]
    opts = [json.loads(line) for line in (test / "options.jsonl").read_text().splitlines()]
    cache = CACHE / (a.acts or test.name)
    tag = f"_{a.acts.removeprefix(test.name).strip('_')}" if a.acts else ""
    layers = json.loads((cache / "layers.json").read_text())
    li = layers.index(a.hidden)
    logps = {r["id"]: r["logp"] for r in map(json.loads, (cache / "logp.jsonl").read_text().splitlines())}
    q = np.array([o["q"] for o in opts])
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.append(starts, len(q))
    is_answer = np.array([o["is_answer"] for o in opts], float)
    is_gold = np.array([o["is_gold"] for o in opts], float)
    right = np.array([r["right"] for r in qs])
    qwen = _softmax_by_q(np.concatenate([logps[r["item"]] for r in qs]), starts)
    agree = np.mean([TQA_LETTERS[int(np.argmax(logps[r["item"]]))] == r["own"] for r in qs])
    nl = np.asarray(np.load(cache / "nl.npy", mmap_mode="r")[:, li], dtype=np.float64)
    end = np.asarray(np.load(cache / "end.npy", mmap_mode="r")[:, li], dtype=np.float64)
    last_end = nl.copy()
    last_end[ends[1:] - 1] = end[ends[1:] - 1]

    def metrics(p: np.ndarray) -> dict:
        top = np.array([starts[i] + np.argmax(p[starts[i] : ends[i + 1]]) for i in range(len(starts))])
        p_ans = np.add.reduceat(p * is_answer, starts)
        kl = np.add.reduceat(qwen * (np.log(qwen + 1e-12) - np.log(p + 1e-12)), starts)
        conf, hit = p[top], is_answer[top]
        out = {}
        for name, m in (("all", np.ones(len(starts), bool)), ("right", right), ("wrong", ~right)):
            bins = np.minimum((conf[m] * 10).astype(int), 9)
            ece = sum(abs(conf[m][bins == b].mean() - hit[m][bins == b].mean()) * (bins == b).mean() for b in np.unique(bins))
            out[name] = {
                "n": int(m.sum()),
                "top1_answer": float(hit[m].mean()),
                "top1_gold": float(is_gold[top][m].mean()),
                "nll_answer": float(-np.log(p_ans[m] + 1e-12).mean()),
                "kl_qwen": float(kl[m].mean()),
                "ece": float(ece),
            }
        return out

    res = {"n_questions": len(qs), "n_right": int(right.sum()), "logit_agreement": float(agree)}
    res["chance_top1"] = float(np.mean(1 / np.diff(ends)))
    scores = {"listwise nl": sc.transform(nl) @ w, "listwise nl, last option at end": sc.transform(last_end) @ w}
    binary = {
        "binary nl": clf.decision_function(sc.transform(nl)),
        "binary nl, last option at end": clf.decision_function(sc.transform(last_end)),
    }
    res["methods"] = {"uniform": metrics(_softmax_by_q(np.zeros(len(q)), starts))}
    res["methods"] |= {k: metrics(_softmax_by_q(s, starts)) for k, s in (scores | binary).items()}
    res["methods"]["Qwen's own letter pmf"] = metrics(qwen)
    res["binary_auroc"] = {k: float(roc_auc_score(is_answer, s)) for k, s in binary.items()}
    (test / f"oos{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    cols = ("top1_answer", "top1_gold", "nll_answer", "kl_qwen", "ece")
    lines = [
        f"Frozen probe on TruthfulQA ({len(qs)} questions, {res['n_right']} right / {len(qs) - res['n_right']} wrong). Trained on all of "
        f"`{train.name}` (nl, hidden_states[{a.hidden}], diag, l2 {a.l2}). Chance top-1 {res['chance_top1']:.3f}; Qwen letter argmax = "
        f"sampled modal answer on {agree:.1%}. Binary probe AUROC: " + ", ".join(f"{k} {v:.3f}" for k, v in res["binary_auroc"].items()),
        "",
        "| method | split | n | " + " | ".join(cols) + " |",
        "|---|---|---|" + "---|" * len(cols),
    ]
    for name, r in res["methods"].items():
        for split in ("all", "right", "wrong"):
            lines.append(f"| {name} | {split} | {r[split]['n']} | " + " | ".join(f"{r[split][c]:.3f}" for c in cols) + " |")
    (test / f"oos{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if tag:  # a comparison set of activations: tables only
        return
    s, logpmf = scores["listwise nl"], lambda x: np.log(_softmax_by_q(x, starts))
    cols = {"qwen_logp": np.log(qwen), "probe_s": s, "probe_logp": logpmf(s)}
    with open(test / "pmf_options.jsonl", "w") as f:
        for qi, r in enumerate(qs):
            sl = slice(starts[qi], ends[qi + 1])
            row = {"item": r["item"], "category": "TruthfulQA", "question": r["question"], "options": r["options"]}
            row |= {"letters": list(TQA_LETTERS[: r["n_opts"]]), "own": r["own"], "gold": r["gold"], "right": r["right"]}
            row |= {"share": r["share"], "fold": -1} | {k: [round(float(x), 4) for x in v[sl]] for k, v in cols.items()}
            row["probe_p"] = [round(float(x), 4) for x in np.exp(cols["probe_logp"][sl])]
            row |= {"probe_s_full": row["probe_s"], "probe_logp_full": row["probe_logp"], "probe_p_full": row["probe_p"]}
            f.write(json.dumps(row) + "\n")


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
    p.add_argument("--l2", type=float, nargs="+", default=[1e-2, 1e-1, 1.0, 10.0, 100.0])
    w = sub.add_parser("whiten")
    w.add_argument("run")
    w.add_argument("--site", default="nl")
    w.add_argument("--layer", type=int, default=52)
    w.add_argument("--variants", nargs="+", default=list(WHITEN))
    w.add_argument("--check-layers", type=int, nargs="+", default=[44, 46, 48, 50, 52, 54, 56])
    w.add_argument("--folds", type=int, default=5)
    w.add_argument("--C", type=float, default=0.01)
    w.add_argument("--l2", type=float, nargs="+", default=[1e-2, 1e-1, 1.0, 10.0, 100.0])
    w.add_argument("--out", help="directory for the outputs (default: the run)")
    w.add_argument("--name", default="whiten", help="output file stem")
    d = sub.add_parser("dims")
    d.add_argument("run")
    d.add_argument("out")
    d.add_argument("--site", default="nl")
    d.add_argument("--layer", type=int, default=52)
    d.add_argument("--top", type=int, default=10)
    s = sub.add_parser("dump")
    s.add_argument("run")
    s.add_argument("--site", default="nl")
    s.add_argument("--layer", type=int, default=52)
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--l2", type=float, default=1.0)
    r = sub.add_parser("lens-requests")
    r.add_argument("run")
    r.add_argument("out")
    r.add_argument("--check", type=int, default=3, help="questions whose saved nl activations go to OUT.check.npy")
    r.add_argument("--check-hidden", type=int, default=52, help="hidden_states index of those activations (block index + 1)")
    ls = sub.add_parser("lens-stats")
    ls.add_argument("run")
    ls.add_argument("reads", nargs="+", help="olens_read_requests.py output(s) for lens-requests' requests")
    ls.add_argument("--block", type=int, default=52)
    t = sub.add_parser("tqa-prep")
    t.add_argument("--runs", nargs="+", default=["2026-10-27_control-n8-qwen36-nothink", "2026-10-27_control-n8-q400-816-qwen36-nothink"])
    t.add_argument("--slug", default="tqa-oos")
    t.add_argument("--work", default="data/tqa_oos", help="where the GPU inputs and O-lens requests go")
    t.add_argument("--min-count", type=int, default=7)
    t.add_argument("--n-samples", type=int, default=8)
    t.add_argument("--trail", default="\n\n", help="appended after the last option (the main prompt; the original goes to *_orig)")
    o = sub.add_parser("oos")
    o.add_argument("train", help="the MMLU-Pro run the probe is fit on")
    o.add_argument("test", help="the tqa-prep run, with scripts/hs_at_positions.py output in data/probe_cache/<run>/")
    o.add_argument("--hidden", type=int, default=52)
    o.add_argument("--l2", type=float, default=1.0)
    o.add_argument("--C", type=float, default=0.01)
    o.add_argument("--acts", help="data/probe_cache/ dir to score instead of the test run's (e.g. <run>_orig); writes oos_<suffix>.*")
    a = ap.parse_args()
    cmds = {
        "collect": collect,
        "fit": fit,
        "pmf": pmf,
        "whiten": whiten,
        "dims": dims,
        "dump": dump,
        "lens-requests": lens_requests,
        "lens-stats": lens_stats,
        "tqa-prep": tqa_prep,
        "oos": oos,
    }
    cmds[a.cmd](a)


if __name__ == "__main__":
    main()
