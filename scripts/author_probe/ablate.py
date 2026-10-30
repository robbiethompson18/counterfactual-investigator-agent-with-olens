"""Leave-one-comment-out ablations on confident (passage, attribute) pairs: which comments move Qwen's answer, is the pair
stable otherwise, and do the frozen probes move with Qwen.

    uv run --with transformers --with jinja2 python -m scripts.author_probe.ablate prep RUN --src SRC_RUN
    # GPU: scripts/hs_at_positions.py on RUN/inputs_del_passage.jsonl and RUN/inputs_del_answer.jsonl (--layers LAYERS)
    uv run --with scikit-learn python -m scripts.author_probe.ablate analyze RUN --src SRC_RUN

SRC_RUN is the full-passage run (qwen_answers.jsonl, activations in data/probe_cache/<src>/). Candidates: Qwen's
chance-rescaled top probability c = (p_top - 1/K) / (1 - 1/K) >= 0.5 with both option orders agreeing, relationship
dropped (Qwen answers "Single" almost always); one passage per author, the one with the most candidate attributes.
For every comment i: Qwen's order-averaged pmf without it (d_i = TV to the full pmf), the passage-only <|im_end|>
activation (all candidate attributes share it) and the "?" activation (order 0) at the probes' layers.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from environments._lib.runfolder import REPO_ROOT
from scripts.author_probe.bank import ATTRS, BANK, LETTERS, MODEL, answer_message, orders

CACHE = REPO_ROOT / "data" / "probe_cache"
USE = ("sex", "age", "education", "income")
# probes of record: label-trained, layer = best out-of-sample AUROC (probes.md of the src run)
LAYERS = {"im_end": {"sex": 28, "age": 5, "education": 8, "income": 10}, "qmark": {"sex": 55, "age": 30, "education": 59, "income": 57}}


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text().splitlines()]


def conf(pmf: list[float]) -> float:
    K = len(pmf)
    return (max(pmf) - 1 / K) / (1 - 1 / K)


def comments_by_id() -> dict[str, dict]:
    from huggingface_hub import hf_hub_download

    raw = hf_hub_download("RobinSta/SynthPAI", "synthpai.jsonl", repo_type="dataset")
    return {c["id"]: c for c in map(json.loads, Path(raw).read_text().splitlines())}


def select(src: Path) -> dict[str, list[str]]:
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    ok: dict[str, list[str]] = {}
    for r in load(src / "qwen_answers.jsonl"):
        if r["attr"] in USE and conf(r["pmf"]) >= 0.5 and r["order_agree"]:
            ok.setdefault(r["passage"], []).append(r["attr"])
    best: dict[str, str] = {}
    for p, attrs in sorted(ok.items()):
        a = passages[p]["author"]
        if a not in best or len(attrs) > len(ok[best[a]]):
            best[a] = p
    return {p: ok[p] for p in sorted(best.values())}


def prep(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    run, src = Path(a.run), Path(a.src)
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    qs = {q["id"]: q for q in load(BANK / "questions.jsonl")}
    com = comments_by_id()
    picked = select(src)
    pairs = []
    with open(run / "inputs_del_passage.jsonl", "w") as fp, open(run / "inputs_del_answer.jsonl", "w") as fa:
        write_rows(tok, picked, passages, qs, com, pairs, fp, fa)
    (run / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    print(f"{len(picked)} passages, {len(pairs)} pairs; layers {sorted({v for d in LAYERS.values() for v in d.values()})}")


def write_rows(tok, picked, passages, qs, com, pairs, fp, fa):
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    for pid, attrs in picked.items():
        texts = [com[c]["text"].strip() for c in passages[pid]["comment_ids"]]
        assert "\n\n".join(texts) == passages[pid]["text"]
        for i in range(-1, len(texts)):  # -1 = the full passage, run through the same pipeline
            text = "\n\n".join(t for j, t in enumerate(texts) if j != i)
            rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, enable_thinking=False)
            ids = tok(rendered, add_special_tokens=False)["input_ids"]
            fp.write(json.dumps({"id": f"{pid}~{i}", "ids": ids, "sites": {"im_end": [ids.index(im_end)]}, "cand": letter_ids}) + "\n")
            for attr in attrs:
                q = qs[f"{pid}:{attr}"]
                for o, order in enumerate(orders(len(q["options"]))):
                    user = answer_message(text, q["question"], [q["options"][k] for k in order])
                    rendered = tok.apply_chat_template(
                        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
                    qchar = rendered.index(user) + len(text) + 2 + len(q["question"]) - 1
                    sites = {"qmark": [next(k for k, (s, e) in enumerate(enc["offset_mapping"]) if s <= qchar < e)]} if o == 0 else {}
                    row = {"id": f"{pid}:{attr}~{i}:o{o}", "ids": enc["input_ids"], "sites": sites, "cand": letter_ids[: len(order)]}
                    fa.write(json.dumps(row | {"order": order}) + "\n")
        pairs += [{"passage": pid, "attr": attr, "n_comments": len(texts)} for attr in attrs]


def frozen_probes(src: Path):
    """Label probes of record fit on all full passages of the src run: {(site, attr): (scaler, model)}, plus the
    out-of-fold probabilities on the full passages (5-fold GroupKFold by author) for the kept-set AUROCs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    cache = CACHE / src.name
    pids = [r["id"] for r in load(src / "passage_logp.jsonl")]
    authors = [p.rsplit("_", 1)[0] for p in pids]
    ans = load(src / "qwen_answers.jsonl")
    models, oof = {}, {}
    for site, by_attr in LAYERS.items():
        for attr, L in by_attr.items():
            k = list(ATTRS).index(attr)
            path = cache / "im_end.npy" if site == "im_end" else cache / "qsites" / f"part-{k:04d}" / "qmark.npy"
            X = np.asarray(np.load(path, mmap_mode="r")[:, L], dtype=np.float32)
            lab = {r["passage"]: r["label"] for r in ans if r["attr"] == attr}
            y = np.array([lab[p] for p in pids])

            def fit(Xt, yt):
                sc = StandardScaler().fit(Xt)
                return sc, LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Xt), yt)

            models[site, attr] = fit(X, y)
            o = np.zeros((len(pids), len(ATTRS[attr][1])))
            for tr, te in GroupKFold(n_splits=5).split(pids, groups=authors):
                s, m = fit(X[tr], y[tr])
                o[np.ix_(te, m.classes_)] = m.predict_proba(s.transform(X[te]))
            oof[site, attr] = dict(zip(pids, o, strict=True))
    return models, oof


def tfidf(src: Path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    text = {p["id"]: p["text"] for p in load(BANK / "passages.jsonl")}
    pids = [r["id"] for r in load(src / "passage_logp.jsonl")]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([text[p] for p in pids])
    out = {}
    for attr in USE:
        lab = {r["passage"]: r["label"] for r in load(src / "qwen_answers.jsonl") if r["attr"] == attr}
        out[attr] = LogisticRegression(C=10, max_iter=2000).fit(vec.transform([text[p] for p in pids]), [lab[p] for p in pids])
    return vec, out


def analyze(a):
    from scipy.stats import pearsonr
    from sklearn.metrics import roc_auc_score

    run, src = Path(a.run), Path(a.src)
    cache = CACHE / run.name
    pairs = load(run / "pairs.jsonl")
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    qs = {q["id"]: q for q in load(BANK / "questions.jsonl")}
    com = comments_by_id()
    layers = json.loads((cache / "passage" / "layers.json").read_text())
    # Qwen's pmf per (pair, deletion), averaged over orders in option space
    order_of = {r["id"]: r["order"] for r in load(run / "inputs_del_answer.jsonl")}
    pmf: dict[str, np.ndarray] = {}
    parts = sorted((cache / "answer").glob("part-*"))  # hs_at_positions --chunk
    for r in (r for part in parts for r in load(part / "logp.jsonl")):
        key = r["id"].rsplit(":o", 1)[0]
        lp = np.array(r["logp"])
        p = np.exp(lp - lp.max())
        canon = np.zeros(len(p))
        canon[order_of[r["id"]]] = p / p.sum()
        pmf[key] = pmf.get(key, 0) + canon / 2
    # activations: passage rows in input order (im_end), answer rows with a qmark site (order 0) in input order
    pid_rows = [r["id"] for r in load(run / "inputs_del_passage.jsonl")]
    q_rows = [r["id"].rsplit(":o", 1)[0] for r in load(run / "inputs_del_answer.jsonl") if r["sites"]]
    acts = {
        "im_end": dict(zip(pid_rows, np.load(cache / "passage" / "im_end.npy"), strict=True)),
        "qmark": dict(zip(q_rows, np.concatenate([np.load(part / "qmark.npy") for part in parts]), strict=True)),
    }
    models, oof = frozen_probes(src)
    vec, tf = tfidf(src)

    def probe(site: str, pid: str, attr: str, i: int) -> np.ndarray:
        s, m = models[site, attr]
        h = acts[site][f"{pid}~{i}" if site == "im_end" else f"{pid}:{attr}~{i}"][layers.index(LAYERS[site][attr])]
        out = np.zeros(len(ATTRS[attr][1]))
        out[m.classes_] = m.predict_proba(s.transform(h[None].astype(np.float32)))[0]
        return out

    rows: list[dict] = []
    for pr in pairs:
        pid, attr, n = pr["passage"], pr["attr"], pr["n_comments"]
        full = pmf[f"{pid}:{attr}~-1"]
        ans = int(full.argmax())
        texts = [com[c]["text"].strip() for c in passages[pid]["comment_ids"]]
        tf_full = tf[attr].predict_proba(vec.transform([passages[pid]["text"]]))[0]
        dels: list[dict] = []
        for i in range(n):
            p = pmf[f"{pid}:{attr}~{i}"]
            rest = "\n\n".join(t for j, t in enumerate(texts) if j != i)
            hum = com[passages[pid]["comment_ids"][i]]["reviews"]["human"][
                {"sex": "sex", "age": "age", "education": "education", "income": "income_level"}[attr]
            ]
            dels.append(
                {
                    "i": i,
                    "text": texts[i],
                    "tv": float(0.5 * np.abs(p - full).sum()),
                    "qwen_shift": float(p[ans] - full[ans]),
                    "argmax_kept": int(p.argmax()) == ans,
                    "probe_shift": {s: float(probe(s, pid, attr, i)[ans] - probe(s, pid, attr, -1)[ans]) for s in LAYERS},
                    "tfidf_shift": float(tf[attr].predict_proba(vec.transform([rest]))[0][ans] - tf_full[ans]),
                    "human": {"estimate": str(hum.get("estimate") or ""), "certainty": hum.get("certainty", 0)},
                    "pmf": p.round(4).tolist(),
                }
            )
        med = float(np.median([d["tv"] for d in dels]))
        thr = max(0.15, 5 * med)
        for d in dels:
            d["evidence"] = d["tv"] > thr
        label = qs[f"{pid}:{attr}"]["label"]
        rows.append(
            pr
            | {"label": label, "answer": ans, "pmf": full.round(4).tolist(), "conf": conf(full.tolist()), "median_tv": med}
            | {"stable": med < 0.05, "stable_argmax": bool(np.mean([d["argmax_kept"] for d in dels]) >= 0.9), "deletions": dels}
            | {"probe_full": {s: probe(s, pid, attr, -1).round(4).tolist() for s in LAYERS}}
            | {"probe_oof": {s: oof[s, attr][pid].round(4).tolist() for s in LAYERS}}
        )
    (run / "pairs_scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    rng = np.random.default_rng(0)
    lines = ["# Leave-one-comment-out ablations", ""]
    lines += [f"{len({r['passage'] for r in rows})} passages, {len(rows)} confident pairs (c >= 0.5, orders agree).", ""]
    lines += ["| attribute | pairs | stable (median TV < 0.05) | stable (argmax survives >= 90%) | stable with >= 1 evidence comment |"]
    lines += ["|---|---|---|---|---|"]
    for attr in USE:
        rs = [r for r in rows if r["attr"] == attr]
        lines.append(
            f"| {attr} | {len(rs)} | {sum(r['stable'] for r in rs)} | {sum(r['stable_argmax'] for r in rs)} | "
            f"{sum(r['stable'] and any(d['evidence'] for d in r['deletions']) for r in rs)} |"
        )
    lines += ["", "## Kept set (stable by median TV)", ""]
    lines += [
        (
            "Probe AUROC uses the out-of-fold full-passage predictions (author-grouped), restricted to the kept passages. Shift"
            " correlation: Pearson r over every deletion in the kept pairs between Qwen's shift (change in p of its original"
            " answer) and the scorer's shift in the same class; null = the scorer's shifts permuted within each pair (1,000 draws,"
            " 95th percentile shown). Evidence-human: share of evidence comments the SynthPAI human guessed this attribute from,"
            " vs all comments of the kept pairs."
        ),
        "",
        (
            "| attribute | kept | Qwen acc vs label | probe im_end AUROC vs label / vs Qwen | probe ? AUROC vs label / vs Qwen"
            " | r(Qwen, im_end) [null95] | r(Qwen, ?) [null95] | r(Qwen, TF-IDF) [null95] | evidence comments | human-guessed:"
            " evidence / all |"
        ),
        "|" + "---|" * 10,
    ]
    for attr in USE:
        ks = [r for r in rows if r["attr"] == attr and r["stable"]]
        if len(ks) < 5:
            lines.append(f"| {attr} | {len(ks)} | too few |" + " |" * 7)
            continue
        K = len(ATTRS[attr][1])
        acc = np.mean([r["answer"] == r["label"] for r in ks])

        def auc(site, key, ks=ks, K=K):
            P = np.array([r["probe_oof"][site] for r in ks])
            y = np.array([r[key] for r in ks])
            cls = [k for k in range(K) if 3 <= (y == k).sum() <= len(y) - 3]
            return np.mean([roc_auc_score(y == k, P[:, k]) for k in cls]) if cls else float("nan")

        def corr(get, ks=ks):
            q = np.concatenate([[d["qwen_shift"] for d in r["deletions"]] for r in ks])
            s = [np.array([get(d) for d in r["deletions"]]) for r in ks]
            r0 = pearsonr(q, np.concatenate(s))[0]
            null = [pearsonr(q, np.concatenate([rng.permutation(x) for x in s]))[0] for _ in range(1000)]
            return f"{r0:.2f} [{np.percentile(null, 95):.2f}]"

        ev = [d for r in ks for d in r["deletions"] if d["evidence"]]
        alld = [d for r in ks for d in r["deletions"]]
        hum = lambda ds: np.mean([bool(d["human"]["estimate"]) for d in ds]) if ds else float("nan")
        lines.append(
            f"| {attr} | {len(ks)} | {acc:.2f} | {auc('im_end', 'label'):.2f} / {auc('im_end', 'answer'):.2f} | "
            f"{auc('qmark', 'label'):.2f} / {auc('qmark', 'answer'):.2f} | {corr(lambda d: d['probe_shift']['im_end'])} | "
            f"{corr(lambda d: d['probe_shift']['qmark'])} | {corr(lambda d: d['tfidf_shift'])} | {len(ev)} | "
            f"{hum(ev):.2f} / {hum(alld):.2f} |"
        )
    lines += ["", "## Key passages (stable, clearest single evidence comment)", ""]
    for attr in USE:
        ks = [r for r in rows if r["attr"] == attr and r["stable"] and sum(d["evidence"] for d in r["deletions"]) == 1]
        ks.sort(key=lambda r: -max(d["tv"] for d in r["deletions"]))
        opts = ATTRS[attr][1]
        lines += [f"### {attr}", ""]
        for r in ks[: a.key]:
            d = max(r["deletions"], key=lambda d: d["tv"])
            text = d["text"].replace("\n", " ")
            lines.append(
                f"- `{r['passage']}`: Qwen {opts[r['answer']]} (p {max(r['pmf']):.2f}), label {opts[r['label']]}; without the comment"
                f" p({opts[r['answer']]}) {r['pmf'][r['answer']]:.2f} → {d['pmf'][r['answer']]:.2f} (TV {d['tv']:.2f}, median"
                f" {r['median_tv']:.3f}); probe shift im_end {d['probe_shift']['im_end']:+.2f}, ? {d['probe_shift']['qmark']:+.2f};"
                f' human guess from this comment: {d["human"]["estimate"] or "-"}. Comment: "{text[:300]}"'
            )
        lines.append("")
    (run / "ablate.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def view(a):
    """expview: one item per stable pair (all pairs with --all): the order-0 answer prompt, then a per-comment table of
    the deletion's TV, Qwen's shift, both probe shifts, the SynthPAI human's guess from that comment; evidence in bold."""
    import sys

    sys.path.insert(0, "expview")
    from expview import Run, call

    run = Path(a.run)
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    qs = {q["id"]: q for q in load(BANK / "questions.jsonl")}
    ex = Run(
        run / "expview",
        title="SynthPAI author MCQs: leave-one-comment-out",
        description=(
            "One item per stable confident (passage, attribute) pair. The reply panel is not a generation: it lists, per comment,"
            " TV(full pmf, pmf without the comment), the change in Qwen's probability of its original answer, the frozen label"
            " probes' change in the same class (passage-only <|im_end|> and the question's '?'), and the SynthPAI human's guess"
            " from that comment. **Bold** = evidence comment (TV > max(0.15, 5 x median))."
        ),
    )
    ex.reset()
    for r in load(run / "pairs_scored.jsonl"):
        if not (r["stable"] or a.all):
            continue
        q, opts = qs[f"{r['passage']}:{r['attr']}"], ATTRS[r["attr"]][1]
        rows = ["| # | TV | Qwen shift | probe <\\|im_end\\|> | probe ? | human guess | comment |", "|---|---|---|---|---|---|---|"]
        for d in r["deletions"]:
            b = "**" if d["evidence"] else ""
            text = d["text"].replace("\n", " ").replace("|", "/")
            rows.append(
                f"| {d['i']} | {b}{d['tv']:.3f}{b} | {d['qwen_shift']:+.3f} | {d['probe_shift']['im_end']:+.3f} | "
                f"{d['probe_shift']['qmark']:+.3f} | {d['human']['estimate'] or '-'} | {b}{text}{b} |"
            )
        head = (
            f"Qwen: {opts[r['answer']]} (pmf {dict(zip(opts, r['pmf'], strict=True))}); label: {opts[r['label']]}; "
            f"median TV {r['median_tv']:.3f}\n\n"
        )
        msg = answer_message(passages[r["passage"]]["text"], q["question"], opts)
        ex.add(
            id=f"{r['passage']}:{r['attr']}",
            group=r["attr"],
            labels={
                "label": opts[r["label"]],
                "qwen": opts[r["answer"]],
                "correct": r["answer"] == r["label"],
                "evidence_comments": sum(d["evidence"] for d in r["deletions"]),
                "median_tv": round(r["median_tv"], 3),
            },
            calls=[
                call(
                    "comment ablations (Qwen letter pmf, order 0 prompt shown)",
                    MODEL,
                    messages=[{"role": "user", "content": msg}],
                    output={"role": "assistant", "content": head + "\n".join(rows)},
                )
            ],
        )
    print(f"expview items in {ex.path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("run")
    p.add_argument("--src", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("run")
    z.add_argument("--src", required=True)
    z.add_argument("--key", type=int, default=10)
    v = sub.add_parser("view")
    v.add_argument("run")
    v.add_argument("--all", action="store_true")
    a = ap.parse_args()
    {"prep": prep, "analyze": analyze, "view": view}[a.cmd](a)


if __name__ == "__main__":
    main()
