# ruff: noqa: I001  (research scripts; prompts are byte-exact strings)
"""Dry check of a variant (no model calls): build the Spec, start the sandbox, run setup, show the tree, run pytest on
the impossible tests and the user/system prompt. Usage: python dry.py <variant> <pid> [--seed N] [--cmd 'extra shell']"""

import argparse, asyncio, json, sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
import run, realvariants as variants


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("pid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmd", action="append", default=[])
    a = ap.parse_args()
    p = run.load_problem(a.pid)
    spec = variants.VARIANTS[a.variant](p, a.seed)
    print(
        "=== SYSTEM\n",
        spec.system,
        "\n=== USER\n",
        spec.user,
        "\n=== TOOLS",
        [t["function"]["name"] for t in spec.tools],
        "\n=== handlers",
        spec.handlers,
        "\n=== workdir",
        spec.workdir,
        "sol",
        spec.solution_path,
        "test",
        spec.test_path,
        "followups",
        spec.followups,
    )
    box = run.Box(f"rl_dry_{a.variant}_{a.seed}", spec)
    try:
        await box.start()
        for c in [
            "pwd; whoami; hostname; find . -path ./.git -prune -o -type f -print | sort | head -80",
            "git log --oneline -n 10 2>&1 | head; git status --short 2>&1 | head",
            "python -m pytest -q 2>&1 | tail -15",
        ] + a.cmd:
            code, out = await box.sh(c, 120)
            print(f"$ {c}\n{out}[exit {code}]")
        print(
            "=== grade (stub solution)",
            json.dumps({k: v for k, v in (await run.grade(box, spec, p)).items() if k not in ("solution", "final_tests")}, indent=0)[:800],
        )
    finally:
        await box.stop()


asyncio.run(main())
