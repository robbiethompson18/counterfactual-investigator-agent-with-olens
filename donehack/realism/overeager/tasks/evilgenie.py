# ruff: noqa: E501
"""EvilGenie (arXiv 2511.21654): hard LiveCodeBench problems with hackable public tests (the agent can hardcode outputs), 2 problems in our set."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("overeager_task_coding", Path(__file__).parent / "coding.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
TASK = _m.Coding("evilgenie", "eg:")
