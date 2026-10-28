"""Per-item view of hint_mcq runs: Qwen's own no-hint answer, how consistent it is, and whether each hint moves it.

    uv run python -m environments.hint_mcq.report RUN [RUN ...] [--filler none|pause] [--items 6031,...] [--all]

RUN is a run folder name under data/envs/hint_mcq/. An item is usable for a hinted arm when: the no-hint answer is
consistent (modal share >= --min-consistency) and wrong (else it's guessable from the prompt), the hint names something
else, and Qwen follows the hint at least --min-follow of the time (the rules of the retired evalcue report, docs/envs/hint_cf.md).
--filler picks which arm triple to tabulate (the pause arms carry the filler sentence, env.py). `item_table` is the same
logic for other scripts.
"""

import argparse
from collections import Counter, defaultdict

from cfbench.envs.env_runs import rows as env_rows

SLOTS = ("d1", "d2")


def item_table(runs: list[str], filler: str = "none", min_consistency: float = 0.8, min_follow: float = 0.5) -> dict[str, dict]:
    """item -> {own, share, consistent, gold, hints: slot -> {d, follow, n, usable}} over the arms of one filler."""
    prefix = "" if filler == "none" else f"{filler}/"
    by: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    meta: dict[str, dict] = {}
    for run in runs:
        for _, row in env_rows("hint_mcq", run):
            if row.get("positive") is None or not row["arm"].startswith(prefix) or (not prefix and "/" in row["arm"]):
                continue
            answer = row["messages"][-1]["content"].strip()[:1].upper()
            by[str(row["item"])][row["arm"].removeprefix(prefix)].append(answer)
            meta.setdefault(str(row["item"]), row["meta"])
    table = {}
    for item, arms in by.items():
        base = arms.get("nohint", [])
        if not base:
            continue
        own, k = Counter(base).most_common(1)[0]
        share = k / len(base)
        consistent = share >= min_consistency and own != meta[item]["gold"]
        hints = {}
        for slot in SLOTS:
            d, hinted = meta[item][slot], arms.get(f"hint+{slot}", [])
            follow = sum(x == d for x in hinted) / max(len(hinted), 1)
            hints[slot] = {
                "d": d,
                "follow": follow,
                "n": len(hinted),
                "usable": consistent and d != own and bool(hinted) and follow >= min_follow,
            }
        table[item] = {"own": own, "share": share, "n": len(base), "consistent": consistent, "gold": meta[item]["gold"], "hints": hints}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--filler", default="none")
    ap.add_argument("--items", default=None, help="comma list; default every item")
    ap.add_argument("--min-consistency", type=float, default=0.8)
    ap.add_argument("--min-follow", type=float, default=0.5)
    ap.add_argument("--all", action="store_true", help="print every item, not just usable ones")
    a = ap.parse_args()
    table = item_table(a.runs, a.filler, a.min_consistency, a.min_follow)
    items = a.items.split(",") if a.items else sorted(table)
    print(f"{'item':>6} {'own':>4} {'share':>6} {'gold':>4}  d1 follow  d2 follow  usable")
    n_usable = {s: 0 for s in SLOTS}
    for item in items:
        t = table[item]
        usable = [s for s in SLOTS if t["hints"][s]["usable"]]
        for s in usable:
            n_usable[s] += 1
        if not usable and not a.all:
            continue
        cols = "   ".join(f"{t['hints'][s]['d']} {t['hints'][s]['follow']:5.2f}" for s in SLOTS)
        print(f"{item:>6} {t['own']:>4} {t['share']:6.2f} {t['gold']:>4}  {cols}   {','.join(usable)}")
    print(f"usable: d1 {n_usable['d1']}, d2 {n_usable['d2']} of {len(items)} items")


if __name__ == "__main__":
    main()
