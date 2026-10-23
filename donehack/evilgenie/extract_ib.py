"""Dump agent-interp-envs' ImpossibleBench (fjzzq2002/impossible_livecodebench) problems, workspace files exactly as the
repo's loader builds them, to evilgenie/data_ib/<split>__<task_id>.json; the unmodified tests (original_test) are kept so
grading can tell a real solve from a hack. Run once:
  uv run --no-project --with 'datasets==4.4.2' python evilgenie/extract_ib.py
"""

import json
import os
import sys
from pathlib import Path

AIE = Path(os.environ.get("AIE_DIR", Path.home() / "workspace/agent-interp-envs"))  # repo root (port.AIE is its environments/coding)
sys.path.insert(0, str(AIE / "src"))
from agent_interp_envs.dataloaders.impossiblebench import ImpossibleBenchLoader

OUT = Path(__file__).parent / "data_ib"
OUT.mkdir(exist_ok=True)
loader = ImpossibleBenchLoader(cache_dir=str(AIE / ".cache/coding"))
ds = loader.load_dataset(skip_validation=True)
for split in ds:
    for i in range(len(ds[split])):
        raw = ds[split][i]
        s = loader._convert_to_standard_format(raw, split)
        rec = {
            "task_id": s.task_id,
            "split": split,
            "index": i,
            "platform": "impossiblebench",
            "difficulty": None,
            "title": s.task_id,
            "entry_point": s.entry_point,
            "impossible_type": raw.get("impossible_type", split),
            "files": {"description.md": s.description, "solution.py": s.solution_template, "test_solution.py": s.test_code},
            "original_test_code": loader._create_test_code(raw["original_test"], s.entry_point) if raw.get("original_test") else None,
            "public_tests": [],
            "private_tests": [],
        }
        safe = s.task_id.replace("/", "_")
        (OUT / f"{split}__{safe}.json").write_text(json.dumps(rec))
    print(split, len(ds[split]))
