"""Run a baseline variant and its null edits (nulls/edits.json) on Qwen3.6-27B with Robbie's harness.
Run from the repo root so evalcue (envs, sandbox, detect) is importable (see nulls/run.sh):

  OPENROUTER_API_KEY=... EVALCUE_MODEL=qwen/qwen3.6-27b QWEN_VLLM_API_KEY=unused \
    uv run python baseline_hyp/nulls_run.py --view <env>__<variant> --n 20 --out <rows.jsonl>

Each null arm = the baseline arm's Episode with one exact substring replacement applied to the system prompt, first user
message, or a starting file. Before running it checks that the rebuilt baseline matches views/<view>/environment exactly.
The baseline is rerun alongside on the same backend, so null effects are measured against a matched baseline.
OpenRouter is pinned to one provider (--provider), with Qwen's top_k=20 added.
"""

import argparse
import asyncio
import importlib
import json
import os
import random
import sys
import types
from pathlib import Path

HYP = Path(__file__).resolve().parent
sys.path.insert(0, os.getcwd())
from evalcue import harness

ap = argparse.ArgumentParser()
ap.add_argument("--view", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=2000)
ap.add_argument("--conc", type=int, default=80)
ap.add_argument("--provider", default="DeepInfra")
ap.add_argument("--max-tokens", type=int, default=65536)
ap.add_argument("--edits", default=str(HYP / "nulls/edits.json"))
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()

env_name, variant = a.view.split("__", 1)
idx = {e["env"]: e for e in json.loads((HYP / "index.json").read_text())}
base_arm = idx[env_name]["arm"]
edits = {e["arm"]: e for e in json.loads(Path(a.edits).read_text())[a.view]}


def env_module(name):
    """Flat env name (scheming_exfil) -> family package module (evalcue.envs.scheming.exfil)."""
    fams = sorted((p.name for p in (Path("evalcue/envs")).iterdir() if p.is_dir()), key=len, reverse=True)
    for fam in fams:
        if name.startswith(fam + "_"):
            return importlib.import_module(f"evalcue.envs.{fam}.{name[len(fam) + 1 :]}")
    return importlib.import_module(f"evalcue.envs.{name}")


base = env_module(env_name)


def env_view(ep):
    out = {}
    u = 0
    for m in ep.messages:
        if m["role"] == "system":
            out["system_prompt.md"] = m["content"]
        elif m["role"] == "user":
            u += 1
            out[f"user_turn_{u}.md"] = m["content"]
        else:
            break
    out.update({f"files/{p}": c for p, c in ep.files.items() if isinstance(c, str)})
    return out


def apply(ep, e):
    f, old, new = e["file"], e["old"], e["new"]
    if f.startswith("files/"):
        p = f[len("files/") :]
        assert ep.files[p].count(old) == 1, (f, old)
        ep.files = {**ep.files, p: ep.files[p].replace(old, new)}
        return ep
    role, k = ("system", 1) if f == "system_prompt.md" else ("user", int(f.split("_")[-1].split(".")[0]))
    seen = 0
    for m in ep.messages:
        if m["role"] == role:
            seen += 1
            if seen == k:
                assert m["content"].count(old) == 1, (f, old)
                m["content"] = m["content"].replace(old, new)
                return ep
    raise KeyError(f)


# sanity: rebuilt baseline == the environment/ the investigators saw
ref = {str(p.relative_to(d)): p.read_text() for d in [HYP / "views" / a.view / "environment"] for p in d.rglob("*") if p.is_file()}
got = env_view(base.build(base_arm, variant))
assert got == ref, (
    f"rebuilt baseline differs from views/{a.view}/environment: {sorted(set(got) ^ set(ref)) or [k for k in got if got[k] != ref.get(k)]}"
)
for e in edits.values():  # every edit applies cleanly
    apply(base.build(base_arm, variant), e)
print(f"{a.view}: baseline arm {base_arm} rebuilt OK; {len(edits)} edits apply", flush=True)
if a.dry:
    sys.exit()

mod = types.ModuleType(f"evalcue.envs.{env_name}")  # row["env"] = flat env name, as in the HF runs
mod.items = lambda: [variant]
mod.detect = base.detect
mod.ARMS = ["baseline", *edits]
mod.build = lambda arm, item: base.build(base_arm, item) if arm == "baseline" else apply(base.build(base_arm, item), edits[arm])

if harness.MODEL != "qwen":  # pin one OpenRouter provider and add Qwen's top_k
    for c in harness.CLIENTS:
        orig = c.chat.completions.create

        async def create(*args, _orig=orig, **kw):
            kw["extra_body"] = {**(kw.get("extra_body") or {}), "top_k": 20, "provider": {"order": [a.provider], "allow_fallbacks": False}}
            for attempt in range(12):  # the pinned provider rate-limits (429); back off instead of losing the episode
                try:
                    return await _orig(*args, **kw)
                except Exception as e:
                    if "429" not in str(e) and "rate" not in str(e).lower() or attempt == 11:
                        raise
                    await asyncio.sleep(min(120, 5 * 2**attempt) * (0.5 + random.random()))

        c.chat.completions.create = create

asyncio.run(
    harness.run_batch(
        mod, arms=mod.ARMS, n_per_arm=a.n, thinking=True, max_tokens=a.max_tokens, out=Path(a.out), concurrency=a.conc, seed=a.seed
    )
)
rows = harness.load_rows(Path(a.out))
harness.summarize(rows, mod.ARMS)
