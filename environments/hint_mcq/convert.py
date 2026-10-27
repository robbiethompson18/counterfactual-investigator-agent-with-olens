"""data/runs/hint_cf_wrong_mcq/<run>/rows.jsonl (evalcue hint_cf.wrong_mcq) -> data/envs/hint_mcq/<run>/, after checking
build() against every stored row (`--dry` reports without writing)."""

import sys

from environments._lib.convert import convert_evalcue_runs
from environments._lib.core import load_env
from environments._lib.runfolder import REPO_ROOT

if __name__ == "__main__":
    convert_evalcue_runs(load_env("hint_mcq"), REPO_ROOT / "data/runs/hint_cf_wrong_mcq", dry="--dry" in sys.argv)
