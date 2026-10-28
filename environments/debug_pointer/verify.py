"""Reproduce every mini-repo's failing test and store the output the prompt shows.

    uv run python environments/debug_pointer/verify.py [ITEM ...]

For each repos/<item>/: refuse module names that shadow the stdlib or an installed package (the old tokenize.py shadowed
`tokenize` and failed with an ImportError instead of the shown assertion), copy the repo to a temp dir, run pytest there, and
require exactly one failing test that fails on an assertion in the test file (an exception raised inside a source module would
print that module's frame, which gives the answer away). The test-function frame and the E lines are written to
repos/<item>/output.txt, which build() shows verbatim after "Running the test suite gives:". Exits non-zero on any problem.
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOS = Path(__file__).parent / "repos"
PYTEST_DEPS = {"pytest", "pluggy", "iniconfig", "packaging", "py", "exceptiongroup", "tomli", "colorama", "pygments"}


def check_names(repo: Path) -> list[str]:
    problems = []
    for p in repo.glob("*.py"):
        mod = p.stem
        if mod in sys.stdlib_module_names or mod in sys.builtin_module_names or mod in PYTEST_DEPS:
            problems.append(f"{p.name} shadows the stdlib module {mod!r}")
        elif (spec := importlib.util.find_spec(mod)) is not None and not (spec.origin or "").startswith(str(REPOS.parent)):
            problems.append(f"{p.name} shadows an installed module {mod!r} ({spec.origin})")
    return problems


def run_pytest(repo: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(repo, work, ignore=shutil.ignore_patterns("__pycache__", "item.json", "output.txt"))
        # pytest is not a project dependency (no test suite here, CLAUDE.md), so run it from an ephemeral uv environment
        r = subprocess.run(
            ["uv", "run", "--no-project", "--with", "pytest", "python", "-m", "pytest"]
            + ["-q", "-p", "no:cacheprovider", "--tb=long", "--no-header"],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"},
        )
    return r.stdout + r.stderr


def extract(out: str, test_name: str) -> tuple[str | None, str | None]:
    """(the shown excerpt, problem). Excerpt = the test function's frame (def ... > ... E ...) then '1 failed'."""
    if not re.search(r"^1 failed in", out, re.MULTILINE):
        return None, "expected exactly one failing test; pytest said: " + (out.strip().splitlines() or ["(no output)"])[-1]
    m = re.search(r"^_{3,} \w+ _{3,}\n(.*?)^" + re.escape(test_name) + r":\d+: (\w+)", out, re.DOTALL | re.MULTILINE)
    if not m:
        return None, "could not find the test frame in the pytest output:\n" + out
    frame, exc = m.group(1), m.group(2)
    if exc != "AssertionError":
        return None, f"the test fails with {exc}, not an assertion (a source frame would name the buggy file):\n{out}"
    if re.search(r"^[a-z_0-9]+\.py:\d+: in ", frame, re.MULTILINE):
        return None, "the traceback shows a source-module frame:\n" + out
    return frame.strip("\n") + "\n\n1 failed", None


def main(items: list[str]) -> int:
    bad = 0
    for repo in sorted(REPOS.iterdir()):
        if not repo.is_dir() or (items and repo.name not in items):
            continue
        tests = list(repo.glob("test_*.py"))
        problems = check_names(repo)
        if len(tests) != 1:
            problems.append(f"expected one test_*.py, found {[t.name for t in tests]}")
        if not problems:
            excerpt, problem = extract(run_pytest(repo), tests[0].name)
            if problem:
                problems.append(problem)
            else:
                assert excerpt
                (repo / "output.txt").write_text(excerpt + "\n")
                first_e = next((line for line in excerpt.splitlines() if line.startswith("E ")), "")
                print(f"ok   {repo.name:12} {first_e[:100]}")
        for p in problems:
            print(f"FAIL {repo.name:12} {p}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
