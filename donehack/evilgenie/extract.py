"""Dump agent-interp-envs' 154 EvilGenie problems (workspace files exactly as the repo's loader builds them) plus
LiveCodeBench's private tests (the holdout) to evilgenie/data/<task_id>.json. Run once:
  uv run --no-project --with 'datasets==4.4.2' python evilgenie/extract.py
"""

import base64
import json
import os
import pickle
import sys
import zlib
from pathlib import Path

AIE = Path(os.environ.get("AIE_DIR", Path.home() / "workspace/agent-interp-envs"))  # repo root (port.AIE is its environments/coding)
sys.path.insert(0, str(AIE / "src"))
from agent_interp_envs.dataloaders.evilgenie import EvilGenieLoader

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
loader = EvilGenieLoader(cache_dir=str(AIE / ".cache/coding"))
ds = loader.load_dataset(skip_validation=True)
for split in ds:
    for i in range(len(ds[split])):
        raw = ds[split][i]
        s = loader._convert_to_standard_format(raw, split)
        priv = raw["private_test_cases"]
        try:
            priv = json.loads(priv)
        except Exception:  # noqa: BLE001 - not JSON: the pickled/zlib/base64 form
            priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(priv.encode()))))
        rec = {
            "task_id": s.task_id,
            "split": split,
            "index": i,
            "platform": s.platform,
            "difficulty": s.difficulty,
            "title": raw["question_title"],
            "starter_code": s.starter_code,
            "metadata": s.metadata,
            "files": {"description.md": s.description, "solution.py": s.solution_template, "test_solution.py": s.test_code},
            "public_tests": json.loads(raw["public_test_cases"]),
            "private_tests": priv,
        }
        (OUT / f"{s.task_id}.json").write_text(json.dumps(rec))
        print(
            split,
            i,
            s.task_id,
            s.platform,
            len(rec["public_tests"]),
            len(priv),
            f"{sum(len(t['input']) + len(t['output']) for t in priv) / 1e6:.1f}MB",
        )
