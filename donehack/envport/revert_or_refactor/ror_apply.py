"""Harness side of the apply_patch tool (run_step.py): apply_patch.apply_patch(text) in /agent as root, as the
original's root agent loop does in-process. Prints one JSON line: {"ok": bool, "out": summary-or-error}."""

import json
import os
import sys

sys.path.insert(0, "/opt")
from apply_patch import apply_patch

os.chdir("/agent")
text = sys.stdin.read()
try:
    print(json.dumps({"ok": True, "out": apply_patch(text)}))
except Exception as e:  # noqa: BLE001 - run_step.py reports any failure as "apply_patch failed: {e}"
    print(json.dumps({"ok": False, "out": str(e)}))
