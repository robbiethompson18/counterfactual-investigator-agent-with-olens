"""Report a cutoff_qa_inline run: accuracy per condition (exact: `correct`; loose: whole-word containment either way), by
gate_class, the paired sign test of each condition vs transcript_inline over items (per-item mean over draws), and read-level
recall (items whose own reads have a bullet containing Qwen's modal answer), so lens misses and investigator misses are told apart.

    uv run python -m cfbench.evals.cutoff_qa_inline.report data/runs/cfbench/<date>_<slug>
"""

import json
import re
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

from counterfactual_investigator_agent_with_olens.runs import ROOT
from environments._lib.core import load_env

BASE = "transcript_inline"


def loose(a: str | None, b: str) -> bool:
    """Whole-word containment either way (both already normalized)."""
    if not a:
        return False
    return re.search(rf"(?<!\S){re.escape(b)}(?!\S)", a) is not None or re.search(rf"(?<!\S){re.escape(a)}(?!\S)", b) is not None


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0


def main(run: Path):
    rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    meta = json.loads((run / "meta.json").read_text())
    norm = load_env("cutoff_qa").norm
    per = defaultdict(lambda: defaultdict(list))  # condition -> item -> [(exact, loose)]
    cls = {}
    for r in rows:
        exact = bool(r.get("correct"))
        per[r["condition"]][r["item"]].append((exact, exact or loose(r.get("canon"), r["modal_cf"])))
        cls[r["item"]] = r["gate_class"]
    items = sorted({i for c in per.values() for i in c})
    print(f"{run.name}: {len(items)} items, {meta['params']['predictors']}, draws {meta['params']['draws']}, "
          f"gate_class {dict(sorted(__import__('collections').Counter(cls.values()).items()))}\n")  # fmt: skip
    print(f"{'condition':24} {'exact':>7} {'loose':>7} {'right':>7} {'wrong':>7} {'n':>5}   vs transcript (loose): win/loss p")
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    for cond, by_item in per.items():
        ex = [mean([e for e, _ in by_item[i]]) for i in by_item]
        lo = [mean([lo_ for _, lo_ in by_item[i]]) for i in by_item]
        by_cls = {c: mean([mean([lo_ for _, lo_ in by_item[i]]) for i in by_item if cls[i] == c]) for c in ("stable_right", "stable_wrong")}
        test = ""
        if cond != BASE and BASE in per:
            w = l = 0
            for i in by_item:
                if i in per[BASE]:
                    d = mean([lo_ for _, lo_ in by_item[i]]) - mean([lo_ for _, lo_ in per[BASE][i]])
                    w, l = w + (d > 0), l + (d < 0)
            test = f"{w}/{l} p={sign_test(w, l):.3f}"
        print(f"{cond:24} {100 * mean(ex):6.1f}% {100 * mean(lo):6.1f}% {100 * by_cls['stable_right']:6.1f}% "
              f"{100 * by_cls['stable_wrong']:6.1f}% {len(by_item):5}   {test}")  # fmt: skip
    reads = ROOT / meta["params"]["inputs"] / "reads/reads.jsonl"
    if reads.exists():
        hit, by_layer, by_tok = defaultdict(bool), defaultdict(lambda: defaultdict(bool)), defaultdict(lambda: defaultdict(bool))
        modal = {r["shown"][0]: r["modal_cf"] for r in rows}
        for line in reads.open():
            b = json.loads(line)
            if b["id"] not in modal:
                continue
            got = any(loose(norm(x["text"]), modal[b["id"]]) for x in b["bullets"])
            hit[b["id"]] |= got
            by_layer[b["layer"]][b["id"]] |= got
            by_tok[b["token"]][b["id"]] |= got
        n = len(modal)
        print(f"\nread-level recall (any bullet contains Qwen's modal answer): {sum(hit.values())}/{n} items")
        print("  by layer: " + ", ".join(f"L{L} {sum(v.values())}/{n}" for L, v in sorted(by_layer.items())))
        print("  by position: " + ", ".join(f"{json.dumps(t)} {sum(v.values())}/{n}" for t, v in by_tok.items()))
        for c in ("stable_right", "stable_wrong"):
            ids = [s for s, m in modal.items() if cls[next(r["item"] for r in rows if r["shown"][0] == s)] == c]
            print(f"  {c}: {sum(hit[s] for s in ids)}/{len(ids)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
