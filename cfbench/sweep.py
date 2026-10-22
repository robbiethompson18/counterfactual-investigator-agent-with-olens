"""Run cfbench (Haiku, transcript + no_transcript) on every evalcue run group that hasn't been scored yet, then print skills.

    uv run python -m cfbench.sweep            # pull runs from HF, run what's pending, print the table
    uv run python -m cfbench.sweep --dry      # just list groups and what would run

A group is a set of one env's run dirs that pool cleanly (cfbench/envs/evalcue_runs.py): runs join an earlier group when
they share at least one (arm, item) with it, every shared one has the same starting context, and `thinking` matches. A group
runs once all its run dirs have stopped (status not `running`, or rows untouched for STALE). When a new run joins a group
the bigger group gets a fresh cfbench run; the table shows the newest per group.
"""

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from cfbench.envs.evalcue_runs import MIN_RUNS, RUNS, _label, _start
from cfbench.score import ll
from evalcue import envs as evalcue_envs

STALE = 3600  # seconds; a `running` run dir untouched this long is treated as dead
REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"


def runs_of(env: str) -> list[Path]:
    started = lambda run: json.loads((run / "meta.json").read_text())["started_at"]
    return sorted((p.parent for p in (RUNS / env).glob("*/rows.jsonl")), key=started)


@dataclass
class Group:
    thinking: bool | None
    runs: list[str] = field(default_factory=list)
    starts: dict[tuple[str, str], str] = field(default_factory=dict)  # (arm, item) -> starting context
    arms: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))  # arm -> [k, n]
    stopped: bool = True

    def spec(self, env: str) -> str:
        return f"{env}/{'+'.join(self.runs)}"

    def usable(self) -> bool:
        """At least two arms with enough runs, and not every arm at the same rate (no case could move)."""
        big = [k / n for k, n in self.arms.values() if n >= MIN_RUNS]
        return len(big) >= 2 and len(set(big)) > 1


def groups(env: str) -> list[Group]:
    out: list[Group] = []
    for run in runs_of(env):
        meta = json.loads((run / "meta.json").read_text())
        g = Group(thinking=meta.get("params", {}).get("thinking"), runs=[run.name])
        for line in (run / "rows.jsonl").read_text().splitlines():
            row = json.loads(line)
            if (label := _label(row)) is not None:
                g.starts[row["arm"], str(row["item"])] = _start(row)
                g.arms[row["arm"]][0] += bool(label)
                g.arms[row["arm"]][1] += 1
        if not g.starts:
            continue
        g.stopped = meta["status"] != "running" or time.time() - (run / "rows.jsonl").stat().st_mtime > STALE
        for earlier in out:
            shared = g.starts.keys() & earlier.starts.keys()
            if earlier.thinking == g.thinking and shared and all(g.starts[key] == earlier.starts[key] for key in shared):
                earlier.runs += g.runs
                earlier.starts |= g.starts
                earlier.stopped &= g.stopped
                for arm, (k, n) in g.arms.items():
                    earlier.arms[arm][0] += k
                    earlier.arms[arm][1] += n
                break
        else:
            out.append(g)
    return out


def done_specs() -> dict[str, Path]:
    out = {}
    for meta_path in sorted((RUNS / "cfbench").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        env = meta.get("params", {}).get("env", "")
        if env.startswith("evalcue_runs:") and meta["status"] == "done":
            out[env.removeprefix("evalcue_runs:")] = meta_path.parent
    return out


def latest(done: dict[str, Path]) -> dict[str, Path]:
    """Drop groups that a later, bigger group of the same env contains (a top-up joined them)."""
    runs = {spec: (spec.split("/")[0], set(spec.split("/")[1].split("+"))) for spec in done}
    return {s: run for s, run in done.items() if not any(e == runs[s][0] and runs[s][1] < r for e, r in runs.values())}


def skill(run: Path) -> dict[str, tuple[float, int]]:
    """condition -> (skill, n cases)."""
    by_case = defaultdict(list)
    for r in map(json.loads, (run / "predictions.jsonl").read_text().splitlines()):
        by_case[r["condition"], r["base_arm"], r["cf_arm"]].append(r)
    totals: dict[str, list] = defaultdict(lambda: [0.0, 0.0, 0])
    for (condition, _, _), rows in by_case.items():
        r0, valid = rows[0], [r for r in rows if r["valid"]]
        k, n = r0["k_cf"], r0["n_cf"]
        null = ll(k, n, (r0["k_base"] + 0.5) / (r0["n_base"] + 1))
        t = totals[condition]
        t[0] += sum(ll(k, n, r["p"]) for r in valid) / max(len(valid), 1) - null
        t[1] += ll(k, n, k / n) - null
        t[2] += 1
    return {c: (g / o if o else float("nan"), m) for c, (g, o, m) in totals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--draws", type=int, default=3)
    args = ap.parse_args()
    envs = [env for env in evalcue_envs.run_ids() if hasattr(evalcue_envs.load(env), "BEHAVIOR")]
    if not args.no_pull:
        include = [f"runs/{env}/*" for env in envs]
        pull = ["hf", "download", REPO, "--repo-type", "dataset", "--include", *include, "--local-dir", "data"]
        subprocess.run(pull, check=True, capture_output=True)

    done = done_specs()
    for env in envs:
        for g in groups(env):
            spec = g.spec(env)
            status = "done" if spec in done else "skip" if not g.usable() else "wait" if not g.stopped else "RUN"
            print(f"{status:4} {spec}  ({len(g.arms)} arms)")
            if status != "RUN" or args.dry:
                continue
            # first run + how many joined it + a hash of the run list, since a group can grow by different runs on the same day
            tag = g.runs[0].split("_", 1)[1] + (
                f"+{len(g.runs) - 1}-{hashlib.sha1(spec.encode()).hexdigest()[:6]}" if len(g.runs) > 1 else ""
            )
            cmd = ["uv", "run", "python", "-m", "cfbench.run", f"evalcue_runs:{spec}", "--pairs", "hub", "--draws", str(args.draws)]
            cmd += ["--concurrency", "32", "--slug", f"{env}_{tag}", "--description", f"overnight sweep: Haiku on {spec}"]
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL)
    print(f"\n{'group':90} {'cases':>5} {'transcript':>10} {'no_transcript':>13}")
    for spec, run in sorted(latest(done_specs()).items()):
        s = skill(run)
        t, nt = s.get("transcript", (float("nan"), 0)), s.get("no_transcript", (float("nan"), 0))
        print(f"{spec[:90]:90} {t[1]:>5} {t[0]:>10.2f} {nt[0]:>13.2f}")


if __name__ == "__main__":
    main()
