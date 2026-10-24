"""All envport/<env> GLM 5.3 runs on one expview page: env = the port's README/Containerfile/upstream config + the episode's final
files (whatever the record keeps), calls = the full conversation, labels = the detector's scalars.
  uv run python analysis/glm53_dash/expview_envport.py        # -> $EXPVIEW_DIR/runs/glm53_envport
  python $EXPVIEW_DIR/expview.py build $EXPVIEW_DIR/runs/glm53_envport"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPVIEW = Path(os.environ.get("EXPVIEW_DIR", Path.home() / "workspace/expview"))
sys.path.insert(0, str(EXPVIEW))
from expview import Run, call

# a checkout of https://github.com/gkroiz/agent-interp-envs (set AIE_DIR to point elsewhere)
AIE = Path(os.environ.get("AIE_DIR", Path.home() / "workspace/agent-interp-envs"))
ENVS = [
    ("norvane", "norvane", ["norvane_glm53"]),
    ("tables_coverage_tests", "tables_coverage_tests", ["tables_coverage_tests_glm53"]),
    ("optimize_cache_c", "optimize_cache_c", ["optimize_cache_c_glm53"]),
    ("lucky_seed", "lucky_seed", ["lucky_seed_glm53"]),
    ("lucky_seed_readable_grader", "lucky_seed", ["lucky_seed_readable_grader_glm53"]),
    ("grade_your_own", "grade_your_own", ["grade_your_own_glm53"]),
]
KEEP = ("final", "agent_py_files", "workspace", "report", "solution", "cache", "test")


def files_of(r, prefix="final/"):
    out = {}
    for k, v in r.items():
        if isinstance(v, str) and any(t in k for t in KEEP) and len(v) > 0:
            out[f"{prefix}{k}.txt"] = v
        elif isinstance(v, dict) and k in ("agent_py_files", "workspace_files", "final_files"):
            for n, t in v.items():
                if isinstance(t, str):
                    out[f"{prefix}{k}/{n}"] = t
    for k in ("telemetry", "detector"):
        if isinstance(r.get(k), dict):
            out[f"{prefix}{k}.json"] = json.dumps(r[k], indent=1, default=str)[:200_000]
    return out


def main():
    run = Run(
        EXPVIEW / "runs/glm53_envport",
        title="GLM 5.3 ported envs",
        description="""`z-ai/glm-5.3` (OpenRouter, reasoning medium, no max_tokens) on agent-interp-envs environments ported to the donehack
podman sandbox (donehack/envport/<env>/, see each port's README in the file tree for what's faithful and what deviates).
Episode cutoff: 32k context or $0.10 → end=cut_long. No LLM judge was run; labels are each port's mechanical detector fields.
One group per env × config.""",
    )
    run.reset()
    n = 0
    for env, port_dir, epdirs in ENVS:
        pdir = ROOT / "envport" / port_dir
        base = {
            f"port/{p.name}": p.read_text()
            for p in sorted(pdir.glob("*"))
            if p.is_file() and p.suffix in (".md", ".py", ".json", "") and p.stat().st_size < 400_000 and p.name != "Containerfile"
        }
        base.update({f"port/{p.name}": p.read_text() for p in pdir.glob("Containerfile*")})
        cfgdir = AIE / "configs" / env
        for epdir in epdirs:
            for p in sorted((ROOT / "runs" / epdir).glob("*.json")):
                d = json.loads(p.read_text())
                r = d["record"]
                cfg = cfgdir / f"{r.get('config', '')}.yaml"
                files = {**base, **({f"upstream_config/{cfg.name}": cfg.read_text()} if cfg.exists() else {}), **files_of(r)}
                labels = {
                    k: v
                    for k, v in (r.get("detector") or {}).items()
                    if isinstance(v, (bool, int, float, str)) and not isinstance(v, str) or isinstance(v, str) and len(v) < 40
                }
                labels.update(end=r.get("end"), steps=r.get("steps"), cost=round(r.get("cost") or 0, 4), cut_reason=r.get("cut_reason"))
                run.add(
                    id=f"{env}/{r.get('config')}/seed{r.get('seed')}",
                    group=f"{env} · {r.get('config')}",
                    labels={k: v for k, v in labels.items() if v is not None},
                    env_files=files,
                    calls=[
                        call(
                            "agent",
                            r.get("model"),
                            messages=d["messages"],
                            params={"reasoning": {"effort": "medium"}, "max_tokens": None},
                            cost=r.get("cost"),
                            tokens=r.get("tokens"),
                        )
                    ],
                )
                n += 1
    print(n, "items")


if __name__ == "__main__":
    main()
