"""Keep only the rows of a read.py states.pt whose prefix `key` is in a keys file (the shared pre-request positions).

python keep_keys.py STATES_IN KEYS_JSONL STATES_OUT
"""

import json
import sys
from pathlib import Path

import torch

src, keys_path, dst = sys.argv[1:]
keys = {json.loads(line)["key"] for line in Path(keys_path).read_text().splitlines()}
saved = torch.load(src)
idx = [i for i, r in enumerate(saved["rows"]) if r["key"] in keys]
missing = keys - {saved["rows"][i]["key"] for i in idx}
assert not missing, f"{len(missing)} keys not captured (capture differs from the 4-bullet run?)"
torch.save({"layers": saved["layers"], "states": saved["states"][idx], "rows": [saved["rows"][i] for i in idx]}, dst)
print(f"kept {len(idx)} of {len(saved['rows'])} rows")
