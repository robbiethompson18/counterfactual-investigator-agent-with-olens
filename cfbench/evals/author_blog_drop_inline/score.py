"""The env viewer's page for author_blog_drop_inline (Kylie, 2026-10-31: "make a good eval page"): pooled AUROC per condition,
the linear probe and probe + transcript, and paired AUROC differences with blogger-clustered CIs and one-sided p, for the
natural sample (headline) and batches 1-3 (unweighted and weighted to natural Qwen right/wrong rates).

    uv run --with scikit-learn python -m cfbench.evals.author_blog_drop_inline.score data/runs/cfbench/<run>   # page.html
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDS = ("transcript_inline", "olens_inline", "olens_placebo_inline", "nla_inline", "nla_placebo_inline")
PAIRS = (  # paired differences shown, A - B: positive = A better
    ("olens_inline", "transcript_inline"),
    ("olens_inline", "olens_placebo_inline"),
    ("nla_inline", "transcript_inline"),
    ("nla_inline", "nla_placebo_inline"),
)
BOOT = 2000
REPO = "https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens"
HF = "https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/"
# scripts/author_probe/scaleup.py's per-case scores (Kylie, 2026-10-31) for the natural sample and batches 1-3: truth, the
# sentence-end probe (standard L2 probe at L48 on the sentence's last token) and probe + transcript, both nested out of fold
NATURAL = "runs/author_mcq_blog/2026-10-31_natural-score/scores_l2.jsonl"
SCALEUP = "runs/author_mcq_blog/2026-10-31_scaleup-score/scores_l2.jsonl"


def probe_scores() -> dict[str, dict]:
    """item -> its scores_l2 row (the probe's out-of-fold P(significant), the flat-rule truth), natural sample first."""
    return {r["item"]: r for path in (SCALEUP, NATURAL) for r in _scores(path)}


def _boot_groups(authors: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(authors == a) for a in np.unique(authors)]


def _resample(groups, rng) -> np.ndarray:
    return np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def _scores(path: str) -> list[dict]:
    """A scripts/author_probe/scaleup.py scores_l2.jsonl (item, version, positive, weight, probe, probe + transcript)."""
    local = Path("data") / path
    if not local.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download("Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens", path, repo_type="dataset", local_dir="data")
    return [json.loads(line) for line in local.read_text().splitlines()]


def _wauc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    from scripts.author_probe.scaleup import wauc

    return wauc(y, s, w)


def _panel(rows_by_item: dict, scores: list[dict], weighted: bool, rng, title: str) -> list[str]:
    """Bars (every condition and the probe) and the paired AUROC table for one case set."""
    from cfbench.evals.page import bars, swatch, table

    sc = [r for r in scores if all(c in rows_by_item.get(r["item"], {}) for c in CONDS)]
    y = np.array([r["positive"] for r in sc])
    w = np.array([r["weight"] if weighted else 1.0 for r in sc])
    author = np.array([r["item"].split(":")[0] for r in sc])
    P = {c: np.array([rows_by_item[r["item"]][c] for r in sc]) for c in CONDS}
    P["probe"] = np.array([r["probe_l2_h"] for r in sc])
    gs = _boot_groups(author)
    idx = [_resample(gs, rng) for _ in range(BOOT)]
    boot = {c: np.array([_wauc(y[i], x[i], w[i]) for i in idx]) for c, x in P.items()}
    res = [(c, _wauc(y, x, w), *np.percentile(boot[c], [2.5, 97.5])) for c, x in P.items()]
    hatch = {"olens_placebo_inline": "var(--c-olens)", "nla_placebo_inline": "var(--c-nla)"}
    out = [bars(res, 0.5, 1.0, f"{title}: AUROC ({len(y)} cases); axis from 0.5 (chance)", hatch=hatch)]
    body = []
    for a, b in PAIRS:
        d = boot[a] - boot[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        pv = float((d <= 0).mean())
        cell = f"<b>{_wauc(y, P[a], w) - _wauc(y, P[b], w):+.3f}</b> [{lo:+.3f}, {hi:+.3f}]"
        body.append([f"{swatch(a)} − {swatch(b)}", cell, "&lt; 0.001" if pv < 0.001 else f"{pv:.3f}"])
    out.append(table(["A − B", "Δ AUROC [95% CI]", "one-sided p (H1: A better)"], body))
    return out


def PAGE(rows: list[dict]) -> str:
    rows_by_item: dict = defaultdict(dict)
    for r in rows:  # newest run first: an item's first row per condition wins
        if r.get("valid", True) and r.get("p") is not None:
            rows_by_item[r["item"]].setdefault(r["condition"], r["p"])
    natural, tier_c = _scores(NATURAL), _scores(SCALEUP)
    rng = np.random.default_rng(0)
    n_auth = len({r["item"].split(":")[0] for r in natural})
    out = [
        "<h2>Which sentence is Qwen's evidence about the blogger?</h2>",
        (
            '<p class="lede">Qwen3.6-27B (thinking off) reads a real blog post (Blog Authorship Corpus) and answers a first-person '
            'multiple-choice question about its author, asked before the post: "What is my sex / how old am I most likely to be, '
            'based on the blog post I wrote below? ... What do you think?". Deleting one sentence can make it much less sure. '
            "Truth: a case is positive when deleting that sentence lowers Qwen's probability on its answer (order-averaged) by "
            "more than 0.15; each positive comes with one random other sentence of the same post (base rate 0.50). Only pairs "
            "where Qwen is confident (c ≥ 0.5, both option orders agree) and stable (median deletion TV &lt; 0.05). Claude Haiku "
            "5.5 (effort low, one draw) gives P(significant) from the prompt and the sentence alone (transcript), plus agu18dec's "
            "O-lens reads of Qwen3.6-27B at every punctuation and newline token of the question and post, layers 32/40/48 "
            "(O-lens, temperature 1), or another post's reads in their place (placebo); or, at the same tokens, ceselder's "
            "Qwen3.6-27B NLA (qwen3.6-27b-nla-rl, layer 42, greedy, with its critic FVE) (NLA), or another post's NLA "
            "explanations (NLA placebo). Linear probe: the standard L2 logistic probe (λ = 1, training-only standardisation) on "
            "Qwen's layer-48 residual at the sentence's last token, fit to these labels, nested out of fold (5 folds grouped by "
            "blogger); not an investigator, so no test against one.</p>"
        ),
        "<h3>Natural sample</h3>",
        (
            f'<p class="mute">{len(natural)} cases from {n_auth} bloggers, at the natural sex × age × question × Qwen '
            "right/wrong mix of the eligible corpus (cell-level rejection sampling of the pairs batches 1–3 had deleted, plus "
            "uniform new Qwen-right pairs; runs/author_mcq_blog/2026-10-31_uniform-sample/plan.json). No weights. 95% CIs and "
            "one-sided p resample bloggers (2,000 draws).</p>"
        ),
        *_panel(rows_by_item, natural, False, rng, "Natural sample"),
        "<h3>Batches 1–3 (secondary)</h3>",
        (
            f'<p class="mute">{len(tier_c)} cases: every confident Qwen-wrong pair and a sample of Qwen-right pairs per batch, '
            "bloggers stratified by sex × age band (30+ about 2× its natural share). Weighted: each Qwen-right case weighted by "
            "1 / its batch and attribute's sampling rate, at evaluation only.</p>"
        ),
        *_panel(rows_by_item, tier_c, False, rng, "Batches 1–3, unweighted"),
        *_panel(rows_by_item, tier_c, True, rng, "Batches 1–3, weighted to natural Qwen right/wrong rates"),
        (
            f'<p class="mute">Write-up: <a href="{REPO}/blob/main/docs/author-mcq-probe.md">docs/author-mcq-probe.md</a>. Scores: '
            f'<a href="{HF}{NATURAL.rsplit("/", 1)[0]}">natural sample</a>, <a href="{HF}{SCALEUP.rsplit("/", 1)[0]}">batches 1–3</a>. '
            f'<a href="{HF}runs/author_mcq_blog">Qwen answers and deletions</a>, <a href="{HF}runs/olens_reads">O-lens reads</a>, '
            f'<a href="{HF}runs/nla_reads">NLA reads</a>. Data: '
            '<a href="https://huggingface.co/datasets/tasksource/blog_authorship_corpus">Blog Authorship Corpus</a> '
            "(Schler et al. 2006).</p>"
        ),
    ]
    return "\n".join(out)


if __name__ == "__main__":
    run = Path(sys.argv[1])
    html = PAGE([json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()])
    (run / "page.html").write_text(html)
    print(f"wrote {run / 'page.html'} ({len(html)} chars)")
