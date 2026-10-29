"""Per-item consistency of a no-CoT MCQ run: pooled accuracy, how many items have a consistent (modal share >= 0.8) answer,
how many of those are wrong (the items where "guess the answer Qwen gives without seeing it" is a real question), by
difficulty / category when the episode meta has them. Works on any env whose measure() gives `answer` and `correct`.

    uv run python -m environments.supergpqa.report ENV/RUN [ENV/RUN ...] [--arm base] [--min-consistency 0.8]

e.g. `supergpqa/2026-10-29_n8` vs `hint_mcq/2026-10-28_all12k-nohint --arm nohint`.
"""

import argparse
import json
from collections import Counter, defaultdict
from typing import NamedTuple

from environments._lib.runfolder import LOCAL, read_episodes


class Row(NamedTuple):
    item: str
    n: int
    share: float  # modal answer's share of the samples
    own_correct: bool  # modal answer == gold
    acc: float  # share of samples that are gold


def labeled_episodes(spec: str):
    env, run = spec.split("/", 1)
    run_dir = LOCAL / env / run
    files = sorted(run_dir.glob("labels/measure@*.jsonl"), key=lambda p: (p.stat().st_mtime, p.name))
    labels = {json.loads(line)["episode"]: json.loads(line)["values"] for line in files[-1].read_text().splitlines()} if files else {}
    for ep in read_episodes(run_dir):
        yield ep, labels.get(ep["id"]) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--arm", default="base")
    ap.add_argument("--min-consistency", type=float, default=0.8)
    ap.add_argument("--by", default="difficulty,discipline,category", help="meta keys to break down by, when present")
    a = ap.parse_args()
    answers: dict[str, list[str]] = defaultdict(list)
    meta: dict[str, dict] = {}
    n_eps = n_unscorable = 0
    for spec in a.runs:
        for ep, v in labeled_episodes(spec):
            if ep["arm"] != a.arm or ep.get("error"):
                continue
            n_eps += 1
            if v.get("answer") is None:
                n_unscorable += 1
                continue
            answers[str(ep["item"])].append(v["answer"])
            meta.setdefault(str(ep["item"]), ep["meta"])
    rows = []
    for item, ans in answers.items():
        own, k = Counter(ans).most_common(1)[0]
        gold = meta[item]["gold"]
        rows.append(Row(item, len(ans), k / len(ans), own == gold, sum(x == gold for x in ans) / len(ans)))
    print(f"episodes {n_eps}, unscorable (no lone letter) {n_unscorable} ({n_unscorable / max(n_eps, 1):.1%}), items {len(rows)}")

    def summarize(rs, label):
        if not rs:
            return
        cons = [r for r in rs if r.share >= a.min_consistency]
        wrong = [r for r in cons if not r.own_correct]
        unan = [r for r in rs if r.share == 1.0]
        acc = sum(r.acc for r in rs) / len(rs)
        print(
            f"{label:<28} items {len(rs):>5}  acc {acc:5.1%}  consistent(>={a.min_consistency:.0%}) {len(cons) / len(rs):5.1%}"
            f"  unanimous {len(unan) / len(rs):5.1%}  consistent&wrong {len(wrong) / len(rs):5.1%} ({len(wrong)})"
        )

    summarize(rows, "all")
    for key in a.by.split(","):
        groups = defaultdict(list)
        for r in rows:
            if key in meta[r.item]:
                groups[meta[r.item][key]].append(r)
        for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            summarize(rs, f"  {key}={g}")


if __name__ == "__main__":
    main()
