"""The env viewer's page for author_blog_drop_inline (Kylie, 2026-10-31: "make a good eval page"): AUROC per version and
condition, paired differences (AUROC, raw Brier, Platt-recalibrated Brier) with author-clustered CIs and one-sided p,
per-attribute AUROC, the calibration note and example cases. Statistics as scripts/author_probe/pvalues_drop.py.

    uv run --with scikit-learn python -m cfbench.evals.author_blog_drop_inline.score data/runs/cfbench/<run>   # page.html
"""

import json
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np

CONDS = ("transcript_inline", "olens_inline", "olens_placebo_inline", "nla_inline", "nla_placebo_inline")
PAIRS = (  # paired differences shown, A - B: positive = A better
    ("olens_inline", "transcript_inline"),
    ("olens_inline", "olens_placebo_inline"),
    ("nla_inline", "transcript_inline"),
    ("nla_inline", "nla_placebo_inline"),
    ("nla_inline", "olens_inline"),
)
VERSIONS = ("qwen_wrong", "qwen_right")
BOOT = 2000
EXPVIEW = "https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/"
REPO = "https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens"
HF = "https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/"
INPUTS = "runs/cfbench_evals/2026-10-30_author-blog-drop-inline-inputs-b2"


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from scipy.stats import rankdata

    n1 = y.sum()
    n0 = len(y) - n1
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def _platt(p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    q = np.clip(p, 1e-3, 1 - 1e-3)
    x = np.log(q / (1 - q))[:, None]
    out = np.zeros(len(p))
    for tr, te in GroupKFold(n_splits=5).split(x, groups=groups):
        out[te] = LogisticRegression().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return out


def _cases(rows: list[dict], conds=CONDS):
    """Per case (version, item): {condition: p}, truth, attribute, author, pair; the cases with every condition in `conds`."""
    by: dict = defaultdict(dict)
    meta = {}
    for r in rows:
        if r.get("valid", True) and r.get("p") is not None:
            k = (r["version"], r["item"])
            by[k][r["condition"]] = r["p"]
            meta[k] = (r["k_cf"], r["attr"], r["pair"].split(":")[0], r["pair"])
    keys = [k for k in by if all(c in by[k] for c in conds)]
    return keys, by, meta


def _boot_groups(authors: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(authors == a) for a in np.unique(authors)]


def _resample(groups, rng) -> np.ndarray:
    return np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def PAGE(rows: list[dict]) -> str:
    from cfbench.evals.author_blog_drop_inline.eval import datasets
    from cfbench.evals.page import bars, label, swatch, table

    present = {r["condition"] for r in rows}
    conds = tuple(c for c in CONDS if c in present)  # the NLA conditions only once their run is in
    keys, by, meta = _cases(rows, conds)
    y = np.array([meta[k][0] for k in keys])
    attr = np.array([meta[k][1] for k in keys])
    author = np.array([meta[k][2] for k in keys])
    version = np.array([k[0] for k in keys])
    P = {c: np.array([by[k][c] for k in keys]) for c in conds}
    R = {c: _platt(P[c], y, author) for c in conds}
    rng = np.random.default_rng(0)
    f3 = lambda x: f"{x:.3f}"
    pf = lambda p: "&lt; 0.001" if p < 0.001 else f"{p:.3f}"
    n_auth = len(set(author))
    out = [
        "<h2>Which sentence is Qwen's evidence about the blogger?</h2>",
        (
            '<p class="lede">Qwen3.6-27B (thinking off) reads a real blog post (Blog Authorship Corpus) and answers a first-person '
            'multiple-choice question about its author, asked before the post: "What is my sex / how old am I most likely to be, '
            'based on the blog post I wrote below? ... What do you think?". Deleting one sentence can make it much less sure. '
            "Truth: a case is positive when deleting that sentence lowers Qwen's probability on its answer (order-averaged) by "
            "more than max(0.15, 5 × the pair's median drop); each positive comes with one random non-significant sentence of the "
            "same post (base rate 0.50). Only pairs where Qwen is confident (c ≥ 0.5, both option orders agree) and stable "
            "(median deletion TV &lt; 0.05). Claude Haiku 5.5 (effort low, one draw) gives P(significant) from the prompt and the "
            "sentence alone (transcript), plus agu18dec's O-lens reads of Qwen3.6-27B at every punctuation and newline token of the "
            "question and post, layers 32/40/48 (O-lens, sampled at temperature 1), or another post's reads in their place "
            "(placebo); or, at the same tokens, ceselder's Qwen3.6-27B NLA (qwen3.6-27b-nla-rl, layer 42 only, greedy, each "
            "explanation with its critic FVE) (NLA), or another post's NLA explanations (NLA placebo, the O-lens placebo's post).</p>"
        ),
        (
            "<p><b>Verdict: neither the O-lens nor the NLA helps.</b> No gain over transcript-only on AUROC or on recalibrated "
            "Brier for either; the O-lens beats its placebo on raw Brier only (a calibration effect), and the NLA is slightly "
            "worse than transcript-only on raw Brier.</p>"
        ),
        (
            f'<p class="mute">{len(keys)} cases from {n_auth} bloggers (batches 1 + 2, 2 × 2,004 bloggers; a third batch was '
            "stopped). 95% CIs and one-sided p resample bloggers.</p>"
        ),
    ]
    _boot_groups(author)
    for v in VERSIONS:
        m = version == v
        gs = _boot_groups(author[m])
        res = []
        for c in conds:
            yy, pp = y[m], P[c][m]
            bs = [_auc(yy[i], pp[i]) for i in (_resample(gs, rng) for _ in range(BOOT))]
            res.append((c, _auc(yy, pp), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))))
        title = f"AUROC, {v.replace('_', ' ')} ({m.sum()} cases); axis from 0.5 (chance)"
        out.append(bars(res, 0.5, 1.0, title))

    def paired(m: np.ndarray, a: str, b: str) -> list[str]:
        gs = _boot_groups(author[m])
        ya = y[m]
        cells = []
        for name, f, better in (
            ("auroc", lambda i: _auc(ya[i], P[a][m][i]) - _auc(ya[i], P[b][m][i]), lambda x: x <= 0),
            ("raw", lambda i: np.mean((P[b][m][i] - ya[i]) ** 2 - (P[a][m][i] - ya[i]) ** 2), lambda x: x <= 0),
            ("recal", lambda i: np.mean((R[b][m][i] - ya[i]) ** 2 - (R[a][m][i] - ya[i]) ** 2), lambda x: x <= 0),
        ):
            obs = f(np.arange(m.sum()))
            bs = np.array([f(_resample(gs, rng)) for _ in range(BOOT)])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            cells.append(f'<b>{obs:+.3f}</b> [{lo:+.3f}, {hi:+.3f}] <span class="mute">p {pf(np.nanmean(better(bs)))}</span>')
        return cells

    body = []
    for a, b in (ab for ab in PAIRS if set(ab) <= set(conds)):
        for v in (None, *VERSIONS):
            m = np.ones(len(y), bool) if v is None else version == v
            body.append([f"{swatch(a)} − {swatch(b)}", v or "pooled", str(m.sum()), *paired(m, a, b)])
    out.append("<h3>Paired differences A − B (positive = A better; one-sided p for H1: A better)</h3>")
    out.append(table(["comparison", "version", "cases", "AUROC [95% CI]", "raw Brier gain", "recalibrated Brier gain"], body))

    body = []
    for v in VERSIONS:
        for at in ("sex", "age"):
            m = (version == v) & (attr == at)
            body.append([v, at, str(m.sum()), *[f3(_auc(y[m], P[c][m])) for c in conds]])
    out.append("<h3>AUROC by attribute</h3>")
    out.append(table(["version", "attribute", "cases", *[swatch(c) for c in conds]], body))

    means = " / ".join(f"{swatch(c)} {P[c].mean():.2f}" for c in conds)
    out.append(
        "<h3>Calibration</h3><p>Mean predicted probability: "
        + means
        + f" against a base rate of {y.mean():.2f}. Haiku's probabilities sit low; with O-lens reads in the prompt they sit "
        "a little higher, which alone lowers the raw Brier. After Platt-recalibrating each condition (logistic on logit p, "
        "5-fold CV grouped by blogger) the gains vanish, as does any AUROC difference. The raw-Brier result of the first "
        "batch alone (O-lens − transcript +0.010, p = 0.008) was this calibration effect.</p>"
    )

    gain = np.abs(P["transcript_inline"] - y) - np.abs(P["olens_inline"] - y)
    picks = [("O-lens helped most", int(np.argmax(gain))), ("O-lens hurt most", int(np.argmin(gain)))]
    picks += [("a female blogger opening with “Introducing my wife”", i) for i, k in enumerate(keys) if k[1] == "blog3344990:sex#0"][:1]
    body = []
    for why, i in picks:
        k = keys[i]
        pair = meta[k][3]
        ds = datasets(pair)
        links = [f'<a href="{EXPVIEW}#dataset={d}&item={escape(pair)}">{escape(n.split(":")[0])}</a>' for n, d in ds.items()]
        links.append(f'<a href="{EXPVIEW}#dataset={INPUTS}&item={escape(k[1])}%2Folens_inline">prompt with reads</a>')
        p3 = " / ".join(f"{P[c][i]:.2f}" for c in conds)
        body.append([escape(why), escape(k[1]), k[0], str(y[i]), p3, " · ".join(links)])
    out.append("<h3>Example cases</h3>")
    out.append(table(["", "case", "version", "truth", "p: " + " / ".join(label(c) for c in conds), "links"], body))
    out.append(
        f'<p class="mute">Write-up: <a href="{REPO}/blob/main/docs/author-mcq-probe.md">docs/author-mcq-probe.md</a>. Runs: '
        f'<a href="{HF}runs/cfbench/2026-10-30_author-blog-drop-haiku-b2">predictions</a>, '
        f'<a href="{HF}{INPUTS}">inputs</a>, <a href="{HF}runs/author_mcq_blog">Qwen answers and deletions</a>, '
        f'<a href="{HF}runs/olens_reads/2026-10-30_author-blog">reads</a>. Data: '
        '<a href="https://huggingface.co/datasets/tasksource/blog_authorship_corpus">Blog Authorship Corpus</a> '
        "(Schler et al. 2006).</p>"
    )
    return "\n".join(out)


if __name__ == "__main__":
    run = Path(sys.argv[1])
    html = PAGE([json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()])
    (run / "page.html").write_text(html)
    print(f"wrote {run / 'page.html'} ({len(html)} chars)")
