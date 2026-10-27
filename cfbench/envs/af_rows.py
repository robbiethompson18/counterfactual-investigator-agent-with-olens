"""The alignment-faking env (evalcue/envs/alignment_faking), thinking off, as cfbench cases for Robbie's predictor:
`public` -> `preview` for DeepSeek V4 Flash and GLM 5.3, from `data/runs/cfbench_af/<model>/rows.jsonl` (made by
scripts/cfbench_af_prep.py). The public arm is the 100 runs with O-lens reads (one per item); preview is all its runs.

    uv run python -m cfbench.run af_rows:dsv4 --model deepseek/deepseek-v4-flash --pairs public:preview --slug ...
"""

from pathlib import Path

from cfbench.envs.overeager_rows import OvereagerRows
from evalcue.envs import alignment_faking as AF


class AfRows(OvereagerRows):
    RUNS = Path("data/runs/cfbench_af")

    def __init__(self, model: str):
        super().__init__(model)


def load(spec: str) -> AfRows:
    import cfbench.envs.overeager_rows as m

    m.BEHAVIOR.setdefault(spec, AF.BEHAVIOR)
    return AfRows(spec)
