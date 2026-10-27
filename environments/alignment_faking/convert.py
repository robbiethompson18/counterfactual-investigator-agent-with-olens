"""data/runs/alignment_faking/<run>/rows.jsonl (evalcue) -> data/envs/alignment_faking/<run>/, after checking build()."""

import sys

from environments._lib.convert import convert_evalcue_runs
from environments._lib.core import load_env
from environments._lib.runfolder import REPO_ROOT

if __name__ == "__main__":
    convert_evalcue_runs(load_env("alignment_faking"), REPO_ROOT / "data" / "runs" / "alignment_faking", dry="--dry" in sys.argv)
