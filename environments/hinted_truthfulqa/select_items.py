"""Pick the hint arms' items: the first 100 TruthfulQA questions plus every question in the 8-sample no-hint runs that at
least one model gets wrong at least 7 times in 8 (Kylie, 2026-10-27). Writes hard_items.json beside this file.

    uv run python -m environments.hinted_truthfulqa.select_items
"""

import json
from collections import defaultdict
from pathlib import Path

from environments._lib.runfolder import LOCAL, active_labels, read_episodes

from . import env as E

BASE = [f"tqa-{i}" for i in range(100)]  # the hint pilot's items
MIN_WRONG = 7 / 8


def main():
    wrong = defaultdict(lambda: defaultdict(list))  # item -> model -> [wrong?]
    for d in sorted((LOCAL / "hinted_truthfulqa").glob("*_control-n8-*")):
        meta = json.loads((d / "meta.json").read_text())
        _, lab = active_labels(d)
        for ep in read_episodes(d):
            if not ep.get("error") and ep["arm"] == "control":
                wrong[ep["item"]][meta["target"]["model"]].append(not (lab.get(ep["id"]) or {}).get("correct"))
    hard = {i: sorted(m for m, w in by_model.items() if len(w) >= 8 and sum(w) / len(w) >= MIN_WRONG) for i, by_model in wrong.items()}
    hard = {i: ms for i, ms in hard.items() if ms}
    items = sorted(set(BASE) | set(hard), key=lambda i: int(i.split("-")[1]))
    out = Path(E.__file__).with_name("hard_items.json")
    by_item = dict(sorted(hard.items(), key=lambda kv: int(kv[0][4:])))
    out.write_text(json.dumps({"items": items, "reliably_wrong": by_item}, indent=1) + "\n")
    print(f"{len(items)} items ({len(hard)} reliably wrong for >=1 model, {len(set(hard) - set(BASE))} beyond the first 100) -> {out}")


if __name__ == "__main__":
    main()
