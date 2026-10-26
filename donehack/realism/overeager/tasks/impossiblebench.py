# ruff: noqa: E501
"""ImpossibleBench (Zhong et al. 2025, arXiv 2510.20270): LiveCodeBench problems whose tests were mutated to contradict the spec (conflicting / one-off), 32 problems."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("overeager_task_coding", Path(__file__).parent / "coding.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
TASK = _m.Coding("impossiblebench", "ib:")
