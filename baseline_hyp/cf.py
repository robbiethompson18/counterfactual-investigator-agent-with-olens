"""Counterfactual edits available for each baseline view: every other arm of the same env that has runs
for the same variant, with its rate and a unified diff of environment/ (system prompt, user turns,
starting files) against the baseline. Edits that live outside the inputs (tool outputs scripted by the
harness) show up only in the harness-config (meta) diff. Writes cf/<view>/<arm>.diff and cf_index.json.
For the browser only; never shown to the investigator.
"""

import collections
import difflib
import json
import shutil
import tempfile
from pathlib import Path

from build import ENVS, RAW, write_environment

OUT = Path("cf")


def single_change(env, base_arm, arm):
    """Does this arm change exactly one thing relative to the baseline arm? (hand rules per family)"""
    if env == "whistleblow_single":  # arms are '+'-joined knob sets
        return len(set(arm.split("+")) ^ set(base_arm.split("+"))) == 1
    if env == "shutdown_killswitch":  # baseline is may_edit; most other arms are built on base (no may_edit line)
        return arm == "base" or arm.startswith("may_edit_")
    if env == "scheming_exfil":
        return arm not in ("nudge_mild_evalcue", "nudge_mild_realcue")
    if env == "dumb_user_report_inline_b":
        return not arm.startswith("stack_")
    return True  # hiring: one name swap per arm


def load_all(env):
    rows = []
    for f in sorted(RAW.glob(f"{env}/*/rows.jsonl")):
        rows += [r for r in map(json.loads, f.read_text().splitlines()) if not r.get("error")]
    return rows


def snapshot(rows, tmp):
    d = Path(tempfile.mkdtemp(dir=tmp))
    write_environment(rows, d)
    return {str(p.relative_to(d)): p.read_text() for p in sorted(d.rglob("*")) if p.is_file()}


def udiff(a, b, label_a, label_b):
    out = []
    for f in sorted(set(a) | set(b)):
        if a.get(f) != b.get(f):
            out += difflib.unified_diff(
                (a.get(f) or "").splitlines(),
                (b.get(f) or "").splitlines(),
                f"{label_a}/{f}" if f in a else "/dev/null",
                f"{label_b}/{f}" if f in b else "/dev/null",
                n=2,
                lineterm="",
            )
    return "\n".join(out)


def meta_lines(r):
    return {"meta.json": json.dumps(r.get("meta") or {}, indent=1, sort_keys=True)}


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    tmp = tempfile.mkdtemp()
    index = {}
    for env, (base_arm, _) in ENVS.items():
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in load_all(env):
            by[r["arm"]][r["item"]].append(r)
        pooled = {a: [sum(bool(r["positive"]) for v in vs.values() for r in v), sum(len(v) for v in vs.values())] for a, vs in by.items()}
        index[env] = {"arms": pooled, "views": {}}
        for variant, brows in sorted(by[base_arm].items()):
            if not any(r["positive"] for r in brows):
                continue
            view = f"{env}__{variant}"
            base_snap, base_meta = snapshot(brows, tmp), meta_lines(brows[0])
            cfs = []
            for arm, vs in sorted(by.items()):
                if arm == base_arm or variant not in vs:
                    continue
                rows = vs[variant]
                d_env = udiff(base_snap, snapshot(rows, tmp), "baseline", arm)
                d_meta = udiff(base_meta, meta_lines(rows[0]), "baseline", arm)
                text = (d_env or "(no change to environment/: the edit is not in the inputs)") + (
                    "\n\n# harness config (not visible to the subject model directly)\n" + d_meta if d_meta else ""
                )
                p = OUT / view / f"{arm}.diff"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text + "\n")
                k, n = sum(bool(r["positive"]) for r in rows), len(rows)
                cfs.append(
                    {
                        "arm": arm,
                        "k": k,
                        "n": n,
                        "diff": str(p),
                        "env_changed": bool(d_env),
                        "single": single_change(env, base_arm, arm),
                        "lines": sum(1 for l in d_env.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---")),
                    }
                )
            index[env]["views"][view] = cfs
            print(view, len(cfs), "arms,", sum(not c["env_changed"] for c in cfs), "with no environment/ change")
    Path("cf_index.json").write_text(json.dumps(index, indent=1))
    shutil.rmtree(tmp)
