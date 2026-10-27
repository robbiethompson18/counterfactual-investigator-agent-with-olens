# ruff: noqa: I001, RUF059  (research scripts; prompts are byte-exact strings)
"""Check every neighbour_pool.json entry in the sandbox (1 cpu): its original tests must pass with the
solved_originals.json solution in < 2 s through the algo-practice pytest setup. Writes neighbour_ok.json."""

import asyncio, json, re, sys, time

sys.path.insert(0, __import__("os").path.dirname(__file__))
import run
from atoms_repo import neighbour_files, PYPROJECT, CONFTEST, HERE

LIMIT = 2.0


async def main():
    pool = json.loads((HERE / "neighbour_pool.json").read_text())
    files, dirs = {"pyproject.toml": PYPROJECT, "conftest.py": CONFTEST}, {}
    for idx in pool:
        d, _, nf = neighbour_files(idx)
        dirs[idx] = d
        files.update({f"problems/{d}/{k}": v for k, v in nf.items()})
    box = run.Box("rl_dry_check_neighbours", run.Spec(user="", files=files, workdir="/home/dev/algo-practice"))
    ok, report = [], {}
    try:
        await box.start()
        for idx in pool:
            t0 = time.time()
            code, out = await box.sh(f"python -m pytest -p no:cacheprovider problems/{dirs[idx]} 2>&1 | tail -2", 60)
            dt = time.time() - t0
            m = re.search(r"in ([\d.]+)s", out)
            inner = float(m.group(1)) if m else dt
            good = bool(re.search(r"\d+ passed", out)) and "failed" not in out and "error" not in out and inner < LIMIT
            report[idx] = (good, round(inner, 2), out.strip().splitlines()[-1] if out.strip() else "")
            print(idx, dirs[idx], report[idx], flush=True)
            if good:
                ok.append(idx)
        code, out = await box.sh("python -m pytest -p no:cacheprovider 2>&1 | tail -3", 300)
        print("whole repo:", out)
    finally:
        await box.stop()
    (HERE / "neighbour_ok.json").write_text(json.dumps(ok) + "\n")
    print(f"{len(ok)}/{len(pool)} ok")


asyncio.run(main())
