"""Env modules. A run id is an env's module path under evalcue.envs with dots as underscores (scheming.blackmail ->
scheming_blackmail), so run dirs from before envs were grouped into packages keep their names."""

import importlib
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parent


def load(run_id: str) -> ModuleType:
    """The module for a run id. Run ids flatten dots, so try each package as a prefix (longest first)."""
    for pkg in sorted((p.name for p in ROOT.iterdir() if (p / "__init__.py").exists()), key=len, reverse=True):
        if run_id.startswith(pkg + "_"):
            return importlib.import_module(f"{__name__}.{pkg}.{run_id.removeprefix(pkg + '_')}")
    return importlib.import_module(f"{__name__}.{run_id}")


def run_ids() -> list[str]:
    """Every module's run id, shared helpers included; data/ and task/ hold data and sandbox files, not modules."""
    paths = (p.relative_to(ROOT) for p in ROOT.rglob("*.py") if not {"data", "task"} & set(p.relative_to(ROOT).parts))
    return sorted(
        "_".join(p.parent.parts if p.name == "__init__.py" else p.with_suffix("").parts) for p in paths if p.parts != ("__init__.py",)
    )
