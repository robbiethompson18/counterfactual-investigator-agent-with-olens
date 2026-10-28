"""No-hint census over all 12,032 MMLU-Pro test questions: Qwen3.6-27B's modal answer, how consistent it is and whether
it's right, from the `nohint` labels of the runs that together cover every item (docs/envs/hint_cf.md, "No-hint census").

    uv run python -m environments.hint_mcq.census

Reads the label files from HF, writes data/envs/hint_mcq/2026-10-28_all12k-nohint/census.csv (one row per item; a copy is
on HF at the same path under envs/, so a dataset pull gets it without rerunning this) and prints the per-category table.
Gold-hint candidates are the `consistently_wrong` rows: modal share >= 0.8 and the modal answer isn't gold.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from environments._lib.runfolder import LOCAL, REPO, hf_token
from environments.hint_mcq.env import _table

RUNS = ["2026-10-26_or_nothink2", "2026-10-27_scale2k", "2026-10-27_scale2k-topup", "2026-10-28_all12k-nohint"]
OUT = LOCAL / "hint_mcq/2026-10-28_all12k-nohint/census.csv"


def answers() -> dict[str, list[str]]:
    api, by = HfApi(token=hf_token()), defaultdict(list)
    for run in RUNS:
        for f in api.list_repo_tree(REPO, path_in_repo=f"envs/hint_mcq/{run}/labels", repo_type="dataset"):
            for line in Path(hf_hub_download(REPO, f.path, repo_type="dataset", token=hf_token())).read_text().splitlines():
                r = json.loads(line)
                arm, item, _ = r["episode"].split("~")
                if arm == "nohint":
                    by[item].append(r["values"]["answer"])
    return by


def census() -> pd.DataFrame:
    t = _table()
    rows = []
    for item, xs in answers().items():
        modal, k = Counter(xs).most_common(1)[0]
        row = t.loc[item]
        share = k / len(xs)
        rows.append(
            {
                "item": item,
                "category": row.category,
                "n": len(xs),
                "modal": modal,
                "modal_share": round(share, 4),
                "gold": row.answer,
                "d2": row.d2,
                "modal_correct": modal == row.answer,
                "consistent_80": share >= 0.8,
                "consistently_wrong": share >= 0.8 and modal != row.answer,
            }
        )
    order = {q: i for i, q in enumerate(t.index)}
    return pd.DataFrame(rows).sort_values("item", key=lambda s: s.map(order)).reset_index(drop=True)


def summary(d: pd.DataFrame) -> pd.DataFrame:
    def row(x: pd.DataFrame) -> dict:
        s = x.modal_share
        return {
            "n": len(x),
            "100%": (s == 1).mean(),
            ">=80%": x.consistent_80.mean(),
            ">=50%": (s >= 0.5).mean(),
            "right_80": int((x.consistent_80 & x.modal_correct).sum()),
            "wrong_80": int(x.consistently_wrong.sum()),
            "wrong_100": int(((s == 1) & ~x.modal_correct).sum()),
            "modal_acc": x.modal_correct.mean(),
        }

    by = sorted(d.groupby("category"), key=lambda kv: -len(kv[1]))
    return pd.DataFrame({"ALL": row(d)} | {c: row(g) for c, g in by}).T


def main() -> None:
    d = census()
    d.to_csv(OUT, index=False)
    print(f"{len(d)} items, {d.n.sum()} no-hint samples -> {OUT}")
    print(summary(d).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
