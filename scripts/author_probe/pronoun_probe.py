"""Exploratory, post hoc (Kylie, 2026-10-31): do answer words in the O-lens reads say whether deleting a sentence makes
Qwen3.6-27B much less sure of its answer about the blogger? CPU only, existing reads. Sex first; age added after (its
words are a weaker proxy for the answer).

author_blog_drop_inline's cases (batches 1+2; each post has as many positives as negatives), per attribute. Per post, the
trajectory S over every read position of the blog post in order (bullets at layers 32/40/48, FVE clipped at 0):
  S = FVE-weighted count of the words of Qwen's answer: sex, its pronouns (he/him/his/himself or she/her/hers/herself);
      age, a short list per bucket (LEX), and
  S± = matching - the mean of the other answers' counts.
Sentences are blog.py's split of the post; a sentence's end is its final punctuation token (else its last read position).
Features of the case sentence k, for S and S±:
  level      S at the end of k
  step       S(end k) - S(end k-1)
  shift      mean S over positions after k - mean over positions before k (within the post); shift_w1..3: only the w
             sentences each side; shift_t: shift / (pooled per-position SD x sqrt(1/n_before + 1/n_after))
  slope      slope of the 3-sentence rolling mean of S(sentence end) at k
Baselines: the same features on the sentence-level counts in the text itself (each sentence one position), and on the
placebo reads Haiku saw (olens_placebo_inline's source post, mapped position by position as in its prompt); Haiku's p.
Metrics: AUROC (1:1, calibration-free) and within-post paired accuracy (every positive vs every negative of the same post,
ties 1/2), 95% CIs over bloggers; 5-fold CV grouped by blogger of logit(Haiku transcript p) vs that + lens features.

    uv run --with scikit-learn --with scipy python -m scripts.author_probe.pronoun_probe RUN
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.author_probe.blog import ATTRS, END, QUESTIONS, sentences

sys.path.insert(0, "expview")
from expview import Run, call

INPUTS = Path("data/runs/cfbench_evals/2026-10-30_author-blog-drop-inline-inputs-b2")
PRED = Path("data/runs/cfbench/2026-10-30_author-blog-drop-haiku-b2/predictions.jsonl")
CONDS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
LEX = {  # by Qwen's answer letter in the canonical order (blog.py ATTRS); age lists fixed before the age trajectory run
    "sex": {"A": r"he|him|his|himself", "B": r"she|her|hers|herself"},
    "age": {
        "A": r"school|teens?|teenagers?|parents|class|classes|homework|high school|grade",
        "B": r"college|university|job|boyfriend|girlfriend|roommates?|campus|dating",
        "C": r"kids|children|wife|husband|career|mortgage|married|son|daughter",
    },
}
RX = {a: {k: re.compile(rf"\b({v})\b", re.IGNORECASE) for k, v in d.items()} for a, d in LEX.items()}
BOOT = 2000
FEATS = ["level", "step", "shift", "shift_w1", "shift_w2", "shift_w3", "shift_t", "slope"]


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines()]


def counts(bullets: list, attr: str, answer: str) -> tuple[float, float]:
    """(matching, matching - mean of the other answers), FVE-weighted."""
    w = {k: sum(f * len(rx.findall(t)) for t, f in bullets) for k, rx in RX[attr].items()}
    return w[answer], w[answer] - float(np.mean([v for k, v in w.items() if k != answer]))


def positions(rows: list[dict]) -> list[tuple[int, str, list]]:
    """(char offset, token, [(text, fve)] over layers) per read position in the user message, in order."""
    by: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if r.get("char_offset") is not None:
            by[r["char_offset"], r["token"]] += [(b["text"], max(0.0, b["fve"])) for b in r["bullets"]]
    return [(o, t, bs) for (o, t), bs in sorted(by.items())]


def spans(msg: str, attr: str) -> list[tuple[int, int, str]]:
    """(start, end, text) of each sentence of the post inside the user message."""
    head = QUESTIONS[attr] + "\n\n"
    post = msg[len(head) : len(msg) - len(END) - 2]
    out, cur = [], len(head)
    for s in sentences(post):
        a = msg.index(s, cur)
        out.append((a, a + len(s), s))
        cur = a + len(s)
    return out


def features(seg: list[list[float]], ends: list[float], k: int) -> dict[str, float]:
    """seg: S at each read position, per sentence; ends: S at each sentence's end."""
    before, after = [x for s in seg[:k] for x in s], [x for s in seg[k + 1 :] for x in s]
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    out = {"level": ends[k], "step": ends[k] - (ends[k - 1] if k else 0.0), "shift": mean(after) - mean(before)}
    for w in (1, 2, 3):
        out[f"shift_w{w}"] = mean([x for s in seg[k + 1 : k + 1 + w] for x in s]) - mean([x for s in seg[max(0, k - w) : k] for x in s])
    if len(before) > 1 and len(after) > 1:
        nb, na = len(before), len(after)
        sd = np.sqrt(((nb - 1) * np.var(before, ddof=1) + (na - 1) * np.var(after, ddof=1)) / (nb + na - 2))
        out["shift_t"] = out["shift"] / (sd * np.sqrt(1 / len(before) + 1 / len(after))) if sd > 0 else 0.0
    else:
        out["shift_t"] = 0.0
    sm = np.convolve(np.pad(ends, 1, mode="edge"), np.ones(3) / 3, mode="valid")  # 3-sentence rolling mean, same length
    lo, hi = max(0, k - 1), min(len(sm) - 1, k + 1)
    out["slope"] = float((sm[hi] - sm[lo]) / max(1, hi - lo))
    return out


def series(P: list, sp: list, attr: str, answer: str, values=None) -> tuple[list, list, list, list]:
    """Per sentence: S and S± at its read positions, and at its end. `values`: per-position (S, S±) to use instead of P's."""
    v = values or [counts(bs, attr, answer) for _, _, bs in P]
    seg, seg2, end, end2 = [], [], [], []
    for a, b, _ in sp:
        idx = [i for i, (o, _, _) in enumerate(P) if a <= o < b]
        final = [i for i in idx if P[i][0] <= b - 1 < P[i][0] + len(P[i][1])]
        seg.append([v[i][0] for i in idx])
        seg2.append([v[i][1] for i in idx])
        j = (final or idx or [None])[-1]
        end.append(v[j][0] if j is not None else (end[-1] if end else 0.0))  # a sentence with no read: carry the last end
        end2.append(v[j][1] if j is not None else (end2[-1] if end2 else 0.0))
    return seg, seg2, end, end2


def auc(y, s) -> float:
    from scipy.stats import rankdata

    n1, n0 = y.sum(), len(y) - y.sum()
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def paired(y, s, post) -> float:
    num = den = 0.0
    for p in np.unique(post):
        m = post == p
        d = s[m & (y == 1)][:, None] - s[m & (y == 0)][None, :]
        num += (d > 0).sum() + 0.5 * (d == 0).sum()
        den += d.size
    return num / den if den else float("nan")


def boot(f, y, s, post, seed=0) -> tuple[float, float]:
    rng, posts = np.random.default_rng(seed), np.unique(post)
    idx = [np.flatnonzero(post == p) for p in posts]
    bs = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(posts), len(posts))
        i = np.concatenate([idx[j] for j in pick])
        relabel = np.concatenate([[n] * len(idx[j]) for n, j in enumerate(pick)])  # a post drawn twice counts twice
        bs.append(f(y[i], s[i], relabel) if f is paired else f(y[i], s[i]))
    return tuple(np.nanpercentile(bs, [2.5, 97.5]))


def oof(X, y, post) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, groups=post):
        sc = StandardScaler().fit(X[tr])
        out[te] = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    return out


def bar(x: float, top: float) -> str:
    return "█" * round(12 * max(0.0, x) / top) if top > 0 else ""


def main(run: Path):
    ex = Run(
        run / "expview",
        title="Answer words in O-lens reads along each post (exploratory)",
        description="Per case: S (FVE-weighted count of the words of Qwen's answer in the O-lens bullets: pronouns for sex, a "
        "short list per age bucket) along the post, per sentence: its end and the mean over its read positions, the case "
        "sentence marked ▶; then the bullets at its end. Per attribute, the top 5 positives and negatives by `lens shift_t`, "
        "plus 5 random of each.",
        label_order=["attr", "truth", "shift_t", "drop"],
    )
    out = [
        "# Answer words in the O-lens reads vs 'deleting this sentence makes a difference' (exploratory, post hoc)\n",
        (
            "AUROC (1:1, calibration-free) and within-post paired accuracy (each positive vs each negative of its post, ties ½), "
            "95% CIs over bloggers. ± = matching minus the mean of the other answers' words. Sex: pronouns; age: "
            + "; ".join(f"{k} {v}" for k, v in LEX["age"].items())
            + ".\n"
        ),
    ]
    scores = []
    for attr in ("sex", "age"):
        o, rows = analyze(attr, ex)
        out += o
        scores += [{"attr": attr} | {k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (run / "results.md").write_text("\n".join(out) + "\n")
    (run / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
    print("\n".join(x for x in out if "Sanity" not in x))


def analyze(attr: str, ex) -> tuple[list[str], list[dict]]:
    cases = {k: c for k, c in ((c["item"], c) for c in load(INPUTS / "cases.jsonl")) if c["attr"] == attr}
    text = {r["id"]: r["messages"][0]["content"] for r in load(INPUTS / "reads/requests.jsonl")}
    raw: dict[str, list] = defaultdict(list)
    for r in load(INPUTS / "reads/reads.jsonl"):
        raw[r["id"]].append(r)
    pos = {k: positions(v) for k, v in raw.items()}
    preds: dict[str, dict] = defaultdict(dict)
    placebo_src = {}
    for r in load(PRED):
        if r["item"] in cases and r.get("valid", True) and r.get("p") is not None:
            preds[r["item"]][r["condition"]] = r["p"]
            if r["condition"] == "olens_placebo_inline":
                placebo_src[r["item"]] = r["placebo_sources"][0]

    rows, fallback, skipped = [], 0, 0
    for item, c in cases.items():
        msg, P = text[c["pair"]], pos[c["pair"]]
        sp = spans(msg, attr)
        ks = [i for i, (_, _, s) in enumerate(sp) if s == c["comment"]]
        if not ks or not all(k in preds[item] for k in CONDS):
            skipped += 1
            continue
        k = ks[0]
        a, b, _ = sp[k]
        if not any(P[i][0] <= b - 1 < P[i][0] + len(P[i][1]) for i in range(len(P))):
            fallback += 1
        seg, seg2, end, end2 = series(P, sp, attr, c["answer"])
        Q = pos[placebo_src[item]]
        qv = [counts(Q[i % len(Q)][2], attr, c["answer"]) for i in range(len(P))]  # the placebo's i-th position at our i-th
        pseg, pseg2, pend, pend2 = series(P, sp, attr, c["answer"], qv)
        tv = [counts([(s, 1.0)], attr, c["answer"]) for _, _, s in sp]  # text: each sentence one position
        tseg, tseg2 = [[x[0]] for x in tv], [[x[1]] for x in tv]
        r = {"item": item, "post": c["pair"].split(":")[0], "version": c["version"], "y": int(c["positive"])}
        for name, (sg, en) in {
            "lens": (seg, end), "lens±": (seg2, end2), "placebo": (pseg, pend), "placebo±": (pseg2, pend2),
            "text": (tseg, [x[0] for x in tv]), "text±": (tseg2, [x[1] for x in tv]),
        }.items():  # fmt: skip
            r |= {f"{name} {f}": v for f, v in features(sg, en, k).items()}
        r |= {f"haiku {k2.removesuffix('_inline')}": preds[item][k2] for k2 in CONDS}
        r |= {"_k": k, "_sp": sp, "_seg": seg, "_end": end, "_c": c}
        rows.append(r)

    out = [
        f"\n# {attr}\n",
        (
            f"{len(rows)} of {len(cases)} {attr} cases ({skipped} skipped: no valid Haiku p in some condition, or the sentence "
            f"not found in blog.py's split), {len({r['post'] for r in rows})} bloggers. The sentence-final token is not a read "
            f"position in {fallback} cases ({fallback / len(rows):.0%}): the sentence's last read position stands in.\n"
        ),
    ]
    names = [k for k in rows[0] if not k.startswith("_") and k not in ("item", "post", "version", "y")]
    for ver in ("pooled", "qwen_wrong", "qwen_right"):
        R = [r for r in rows if ver == "pooled" or r["version"] == ver]
        y, post = np.array([r["y"] for r in R]), np.array([r["post"] for r in R])
        out.append(f"\n## {attr}, {ver}: {len(R)} cases ({y.sum()} positive), {len(set(post))} bloggers\n")
        out.append("| score | AUROC [95% CI] | paired accuracy [95% CI] |\n|---|---|---|")
        for n in names:
            s = np.array([r[n] for r in R], float)
            (al, ah), (pl, ph) = boot(auc, y, s, post), boot(paired, y, s, post)
            out.append(f"| {n} | {auc(y, s):.3f} [{al:.3f}, {ah:.3f}] | {paired(y, s, post):.3f} [{pl:.3f}, {ph:.3f}] |")

    out.append(f"\n## {attr}: incremental value over Haiku's transcript p (5-fold CV grouped by blogger, out-of-fold)\n")
    out.append("| model | AUROC | − transcript p [95% CI] | one-sided p | paired accuracy |\n|---|---|---|---|---|")
    y, post = np.array([r["y"] for r in rows]), np.array([r["post"] for r in rows])
    lp = np.array([r["haiku transcript"] for r in rows]).clip(0.01, 0.99)
    X0 = np.log(lp / (1 - lp))[:, None]
    base = oof(X0, y, post)
    for label, fs in {
        "transcript p": [],
        "+ lens level only (the one feature the first pass used)": ["lens level"],
        "+ lens S and S± features": [f"lens {f}" for f in FEATS] + [f"lens± {f}" for f in FEATS],
        "+ placebo S and S± features": [f"placebo {f}" for f in FEATS] + [f"placebo± {f}" for f in FEATS],
        "+ text S and S± features": [f"text {f}" for f in FEATS] + [f"text± {f}" for f in FEATS],
    }.items():
        p = oof(np.column_stack([X0] + [np.array([r[f] for r in rows]) for f in fs]), y, post) if fs else base
        rng, posts = np.random.default_rng(1), np.unique(post)
        idx = [np.flatnonzero(post == q) for q in posts]
        bs = []
        for _ in range(BOOT):
            i = np.concatenate([idx[j] for j in rng.integers(0, len(posts), len(posts))])
            bs.append(auc(y[i], p[i]) - auc(y[i], base[i]))
        d = auc(y, p) - auc(y, base)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append(
            f"| {label} | {auc(y, p):.3f} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {np.mean(np.array(bs) <= 0):.3f} | {paired(y, p, post):.3f} |"
        )

    out.append(f"\n## {attr}: sanity check, top 5 positives and negatives by lens shift_t\n")
    rng = np.random.default_rng(0)
    for y_ in (1, 0):
        R = [r for r in rows if r["y"] == y_]
        top = sorted(R, key=lambda r: -r["lens shift_t"])[:5]
        rand = [R[i] for i in rng.choice(len(R), 5, replace=False)]
        out.append(f"\n### {'makes a difference' if y_ else 'no difference'}\n")
        for pick, r in [("top", r) for r in top] + [("random", r) for r in rand]:
            c, k = r["_c"], r["_k"]
            hi = max(max(r["_end"]), 1e-9)
            lines = ["| # | S at end | mean S | | sentence |", "|---|---|---|---|---|"]
            for j, (_, _, s) in enumerate(r["_sp"]):
                m = float(np.mean(r["_seg"][j])) if r["_seg"][j] else 0.0
                mark = "▶" if j == k else ""
                lines.append(f"| {mark}{j} | {r['_end'][j]:.2f} | {m:.2f} | {bar(r['_end'][j], hi)} | {s[:110].replace('|', '/')} |")
            P = pos[c["pair"]]
            a, b, _ = r["_sp"][k]
            idx = [i for i, (o, _, _) in enumerate(P) if a <= o < b]
            bl = sorted(P[idx[-1]][2], key=lambda x: -x[1]) if idx else []
            body = f"Qwen answers {ATTRS[attr]['ABC'.index(c['answer'])]}: p {c['p0']:.3f} → {c['p1']:.3f} without ▶ "
            body += f"(drop {c['drop']:+.3f}, threshold {c['threshold']:.2f}). lens shift_t {r['lens shift_t']:+.2f}, step "
            body += f"{r['lens step']:+.2f}, shift {r['lens shift']:+.2f}.\n\n" + "\n".join(lines)
            body += "\n\nBullets at ▶'s last read position:\n\n" + "\n".join(f"- FVE {f:.3f}: {' '.join(t.split())}" for t, f in bl)
            ex.add(
                id=f"{pick}/{r['item']}",
                group=f"{attr}: {'positive' if y_ else 'negative'} ({pick})",
                labels={"attr": attr, "truth": y_, "shift_t": round(r["lens shift_t"], 2), "drop": round(c["drop"], 3)},
                calls=[
                    call(
                        "S along the post",
                        "agu18dec O-lens (Qwen3.6-27B)",
                        messages=[{"role": "user", "content": "(not a call: the reads along the post, as one message)"}],
                        output={"role": "assistant", "content": body},
                    )
                ],
            )
            if pick == "top":
                out.append(f"- `{r['item']}` shift_t {r['lens shift_t']:+.2f}, drop {c['drop']:+.2f}: _{c['comment'][:160]}_")
    return out, rows


if __name__ == "__main__":
    main(Path(sys.argv[1]))
