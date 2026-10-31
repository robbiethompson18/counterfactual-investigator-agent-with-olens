"""Linear probes on sentence-end activations (Kylie, 2026-10-31; exploratory, post hoc): does Qwen3.6-27B's residual
stream at a sentence's final token say whether deleting that sentence makes it much less sure of its answer?

Prompts are the eval's question-first prompts (blog.message, no options, thinking-off template), as the O-lens read.
Sites per prompt: every sentence end of the post (the token holding the sentence's last character, blog.py's split),
the question's "?" and the final "?" of "What do you think?". Eval: the 533 batch 1+2 pairs of author_blog_drop_inline;
train: batch 3's confident pairs (no shared bloggers), last sentence end + both "?" only.

    uv run --with transformers --with jinja2 python -m scripts.author_probe.period_probe inputs RUN   # RUN/hs_inputs.jsonl
    # GPU: scripts/sky/hs_positions_any.yaml, ARGS="--layers 16,20,24,28,32,36,40,44,48,52,56,60 --chunk 1000"
    uv run --with scikit-learn --with scipy python -m scripts.author_probe.period_probe analyze RUN
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.author_probe.blog import BANK, QUESTIONS, sentences
from scripts.author_probe.blog import message as qf_message

INPUTS = Path("data/runs/cfbench_evals/2026-10-30_author-blog-drop-inline-inputs-b2")
TRAIN_ANSWERS = Path("data/runs/author_mcq_blog/2026-10-30_qwen36-qfirst-c/qwen_answers.jsonl")
MODEL = "Qwen/Qwen3.6-27B"
N_TRAIN = 1500  # confident batch-3 pairs per attribute
LAYERS = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text().splitlines()]


def conf(r: dict) -> float:
    k = len(r["pmf"])
    return (max(r["pmf"]) - 1 / k) / (1 - 1 / k)


def sites(tok, text: str, attr: str) -> tuple[list[int], list[int], list[str]]:
    """(token ids, site token indices, site kinds: 'q', 's0', 's1', ..., 'end') for the question-first prompt."""
    user = qf_message(text, attr)  # no options: ends with "What do you think?"
    rendered = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, enable_thinking=False)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    s0 = rendered.index(user)
    tok_at = lambda c: next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= s0 + c < b)
    chars, kinds = [len(QUESTIONS[attr]) - 1], ["q"]
    cur = len(QUESTIONS[attr]) + 2
    for j, s in enumerate(sentences(text)):
        a = user.index(s, cur)
        cur = a + len(s)
        chars.append(cur - 1)
        kinds.append(f"s{j}")
    chars.append(len(user) - 1)
    kinds.append("end")
    return enc["input_ids"], [tok_at(c) for c in chars], kinds


def inputs(run: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    evals = sorted({c["pair"] for c in load(INPUTS / "cases.jsonl")})
    train = []
    for attr in ("sex", "age"):
        rs = sorted((r for r in load(TRAIN_ANSWERS) if r["attr"] == attr and r["order_agree"] and conf(r) >= 0.5), key=lambda r: r["id"])
        train += [r["id"] for r in np.random.default_rng(0).permutation(np.array(rs, dtype=object))[:N_TRAIN]]
    rows, index = [], []
    for split, pairs in (("eval", evals), ("train", train)):
        for pair in pairs:
            pid, attr = pair.split(":")
            ids, pos, kinds = sites(tok, passages[pid]["text"], attr)
            if split == "train":  # the question "?", the post's last sentence end, the final "?"
                keep = [0, len(pos) - 2, len(pos) - 1]
                pos, kinds = [pos[i] for i in keep], [kinds[i] for i in keep]
            rows.append({"id": pair, "ids": ids, "sites": {"h": pos}, "cand": []})
            index.append({"id": pair, "split": split, "kinds": kinds, "positions": pos})
    (run / "hs_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    n = sum(len(r["positions"]) for r in index)
    print(f"{len(evals)} eval + {len(train)} train prompts, {n} sites -> {run / 'hs_inputs.jsonl'}")


def activations(run: Path) -> tuple[np.ndarray, list[dict]]:
    """[sites, layers, d] fp16 in index order (the parts concatenated) and the index rows with each site's row offset."""
    parts = sorted((run / "hs").glob("part-*"), key=lambda p: int(p.name.split("-")[1]))
    A = np.concatenate([np.load(p / "h.npy") for p in parts])
    index, i = load(run / "index.jsonl"), 0
    for r in index:
        r["rows"] = list(range(i, i + len(r["positions"])))
        i += len(r["positions"])
    assert i == len(A), (i, A.shape)
    return A, index


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def analyze(run: Path):
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from scripts.author_probe.pronoun_probe import PRED, auc, boot, features, oof, paired

    A, index = activations(run)
    L = json.loads(next((run / "hs").glob("part-*/layers.json")).read_text())
    by = {r["id"]: r for r in index}
    answers = {r["id"]: r for r in load(TRAIN_ANSWERS)}
    out = ["# Linear probes on sentence-end activations (exploratory, post hoc)\n", f"Layers {L}.\n"]

    # eval cases: the case sentence's row and the one before it (the question "?" for the first sentence)
    preds: dict[str, dict] = {}
    for r in load(PRED):
        if r.get("valid", True) and r.get("p") is not None:
            preds.setdefault(r["item"], {})[r["condition"]] = r["p"]
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    cases: list[dict[str, Any]] = []
    for c in load(INPUTS / "cases.jsonl"):
        ix = by[c["pair"]]
        k = sentences(passages[c["pair"].split(":")[0]]["text"]).index(c["comment"])
        cases.append(c | {"k": k, "row": ix["rows"][ix["kinds"].index(f"s{k}")], "prev": ix["rows"][ix["kinds"].index(f"s{k}") - 1],
                          "post": c["pair"].split(":")[0], "haiku": preds.get(c["item"], {})})  # fmt: skip
    y = np.array([int(c["positive"]) for c in cases])
    post = np.array([c["post"] for c in cases])
    attr = np.array([c["attr"] for c in cases])
    have = np.array([all(k in c["haiku"] for k in ("transcript_inline", "olens_inline")) for c in cases])
    ht = np.array([c["haiku"].get("transcript_inline", 0.5) for c in cases])
    ho = np.array([c["haiku"].get("olens_inline", 0.5) for c in cases])

    def report(name: str, s: np.ndarray, rows: list):
        for sub, m in (("pooled", np.ones(len(y), bool)), ("sex", attr == "sex"), ("age", attr == "age")):
            (al, ah), (pl, ph) = boot(auc, y[m], s[m], post[m]), boot(paired, y[m], s[m], post[m])
            a, pa = auc(y[m], s[m]), paired(y[m], s[m], post[m])
            rows.append(f"| {name} | {sub} | {a:.3f} [{al:.3f}, {ah:.3f}] | {pa:.3f} [{pl:.3f}, {ph:.3f}] |")

    def increment(name: str, s: np.ndarray, rows: list, extra=None):
        """`extra`: more columns in the baseline (and kept in the + score model)."""
        m = have
        X0 = logit(ht[m])[:, None] if extra is None else np.column_stack([logit(ht[m]), extra[m]])
        base, p = oof(X0, y[m], post[m]), oof(np.column_stack([X0, s[m]]), y[m], post[m])
        rng, posts = np.random.default_rng(1), np.unique(post[m])
        idx = [np.flatnonzero(post[m] == q) for q in posts]
        bs = []
        for _ in range(2000):
            i = np.concatenate([idx[j] for j in rng.integers(0, len(posts), len(posts))])
            bs.append(auc(y[m][i], p[i]) - auc(y[m][i], base[i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        d = auc(y[m], p) - auc(y[m], base)
        pv = np.mean(np.array(bs) <= 0)
        rows.append(f"| {name} | {auc(y[m], base):.3f} | {auc(y[m], p):.3f} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {pv:.3f} |")

    head = ["| score | cases | AUROC [95% CI] | paired accuracy [95% CI] |", "|---|---|---|---|"]
    inc_head = ["| score | transcript p | + score | difference [95% CI] | one-sided p |", "|---|---|---|---|---|"]
    tab, inc = list(head), list(inc_head)
    report("Haiku transcript p", np.where(have, ht, 0.5), tab)
    report("Haiku O-lens p", np.where(have, ho, 0.5), tab)

    # probe 1: an off-the-shelf answer probe (batch 3), its P(Qwen's answer) trajectory over the eval posts' sentence ends
    tr = [r for r in index if r["split"] == "train"]
    out.append("\n## Probe 1: answer probe trained on batch 3, applied along each eval post\n")
    p1 = []
    for site, kind_of in (("last sentence end", lambda r: r["kinds"][1]), ('final "?"', lambda r: "end")):
        for target in ("answer", "label"):
            scores = {"step": np.zeros(len(cases)), "shift_w3": np.zeros(len(cases))}
            for at in ("sex", "age"):
                R = [r for r in tr if r["id"].endswith(at)]
                rows = [r["rows"][r["kinds"].index(kind_of(r))] for r in R]
                t = np.array([answers[r["id"]]["answer" if target == "answer" else "label"] for r in R])
                g = np.array([r["id"].split(":")[0] for r in R])
                best = (-1.0, None, None)
                for li in range(len(L)):
                    for C in (0.001, 0.01):
                        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
                        acc = []
                        for a, b in GroupKFold(n_splits=3).split(rows, groups=g):
                            pipe.fit(A[[rows[i] for i in a], li].astype(np.float32), t[a])
                            acc.append(pipe.score(A[[rows[i] for i in b], li].astype(np.float32), t[b]))
                        if np.mean(acc) > best[0]:
                            best = (float(np.mean(acc)), li, C)
                acc, li, C = best
                pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(A[rows, li].astype(np.float32), t)
                p1.append(f"- {at}, {target}, trained at the {site}: CV accuracy {acc:.3f} at layer {L[li]}, C {C}")
                for j, c in enumerate(cases):
                    if c["attr"] != at:
                        continue
                    ix = by[c["pair"]]
                    srows = [ix["rows"][i] for i, kd in enumerate(ix["kinds"]) if kd.startswith("s")]
                    pr = pipe.predict_proba(A[srows, li].astype(np.float32))[:, list(pipe.classes_).index("ABC".index(c["answer"]))]
                    f = features([[x] for x in pr], list(pr), c["k"])
                    for feat, arr in scores.items():
                        arr[j] = f[feat]
            for feat, sc in scores.items():
                name = f"probe 1 ({target}, {site}) P(answer) {feat}"
                report(name, sc, tab)
                increment(name, sc, inc)
    out += p1

    # probe 2: fit to the task, nested layer choice, 5-fold CV grouped by blogger
    def cv_probe(X_of_layer, n_comp=128, C=0.01):
        oofp = np.zeros(len(y))
        chosen = []
        for a, b in GroupKFold(n_splits=5).split(y, groups=post):
            best = (-1.0, 0)
            for li in range(len(L)):
                Xa = X_of_layer(li)[a]
                inner = []
                for ia, ib in GroupKFold(n_splits=3).split(Xa, groups=post[a]):
                    pipe = make_pipeline(StandardScaler(), PCA(n_comp, random_state=0), LogisticRegression(C=C, max_iter=2000))
                    pipe.fit(Xa[ia], y[a][ia])
                    inner.append(auc(y[a][ib], pipe.decision_function(Xa[ib])))
                if np.mean(inner) > best[0]:
                    best = (float(np.mean(inner)), li)
            li = best[1]
            chosen.append(L[li])
            pipe = make_pipeline(StandardScaler(), PCA(n_comp, random_state=0), LogisticRegression(C=C, max_iter=2000))
            pipe.fit(X_of_layer(li)[a], y[a])
            oofp[b] = pipe.decision_function(X_of_layer(li)[b])
        return oofp, chosen

    rows_k = np.array([c["row"] for c in cases])
    rows_prev = np.array([c["prev"] for c in cases])
    feats = {
        "h(sentence end)": lambda li: A[rows_k, li].astype(np.float32),
        "h(end k) - h(end k-1)": lambda li: A[rows_k, li].astype(np.float32) - A[rows_prev, li].astype(np.float32),
        "[h, diff]": lambda li: np.hstack([A[rows_k, li], A[rows_k, li] - A[rows_prev, li]]).astype(np.float32),
    }
    out.append("\n## Probe 2: fit to 'makes a difference' (StandardScaler, PCA 128, logistic C 0.01; layer chosen by inner CV)\n")
    for name, X in feats.items():
        s, chosen = cv_probe(X)
        out.append(f"- {name}: layers chosen per outer fold {chosen}")
        report(f"probe 2: {name}", s, tab)
        increment(f"probe 2: {name}", s, inc)
    texts = [c["comment"] for c in cases]
    s_txt = np.zeros(len(y))
    for a, b in GroupKFold(n_splits=5).split(y, groups=post):
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(C=1.0, max_iter=2000))
        pipe.fit([texts[i] for i in a], y[a])
        s_txt[b] = pipe.decision_function([texts[i] for i in b])
    report("control: TF-IDF of the sentence", s_txt, tab)
    increment("control: TF-IDF of the sentence", s_txt, inc)

    # structure control: what a sentence-end activation could carry besides content (its length and place in the post)
    n_sent = np.array([sum(kd.startswith("s") for kd in by[c["pair"]]["kinds"]) for c in cases])
    k_arr = np.array([c["k"] for c in cases])
    length = np.array([len(c["comment"]) for c in cases])
    struct = np.column_stack([np.log(length), length, k_arr, n_sent, k_arr / n_sent, k_arr == n_sent - 1, k_arr == 0]).astype(float)
    s_struct = oof(struct, y, post)
    report("control: sentence length + position in post", s_struct, tab)
    increment("control: sentence length + position in post", s_struct, inc)
    s_h, _ = cv_probe(feats["h(sentence end)"])
    increment("probe 2: h(sentence end), over Haiku + length/position", s_h, inc, extra=struct)
    s_hs = oof(np.column_stack([s_h, struct]), y, post)
    report("probe 2: h(sentence end) + length/position", s_hs, tab)

    out += ["\n## AUROC and within-post paired accuracy (95% CIs over bloggers)\n", *tab]
    out += ["\n## Increment over Haiku's transcript p (5-fold CV grouped by blogger; cases with Haiku p)\n", *inc]
    (run / "results.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    {"inputs": inputs, "analyze": analyze}[sys.argv[1]](Path(sys.argv[2]))
