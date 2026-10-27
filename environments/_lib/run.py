"""envrun: run an env against a model given by flags, into a run folder.

    uv run python -m environments run ENV --tag dev --arms a,b --items 0:4 --n 2 \\
        --model z-ai/glm-5.3 --backend openrouter --provider fireworks --effort medium --thinking on
    uv run python -m environments run ENV --pair NAME --model Qwen/Qwen3.6-27B --served-name qwen --backend vllm --url $QWEN_VLLM_URL
    uv run python -m environments index [ENV ...]          # rebuild envs/<env>/index.json + envs/index.json
    uv run python -m environments upload ENV/RUN [...]      # one HF commit: the run dirs + their indexes

Arms interleave (sample-major, then item, then arm), so a stopped run still has every arm. Defaults follow the repo rules:
no max_tokens unless --max-tokens is given (recorded), reasoning effort always sent (medium) unless the env REQUIRES
otherwise, episode caps in turns and prompt tokens, never dollars.

A run finishes the way scripts/runs.py did: its transcripts go to Docent (scripts/docent_upload.py, when DOCENT_API_KEY is
set; a failure only prints the retry command), then the run folder and its indexes go to HF unless --no-upload. Ctrl-C
finishes the run as `abandoned` and still uploads it: failed runs are data.
"""

import argparse
import asyncio
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import traceback
from typing import Any

import httpx

from .backends import Target
from .core import Ctx, episode_id, load_env
from .index import build as build_index
from .loop import run_episode
from .runfolder import LOCAL, REPO_ROOT, RunWriter, git_sha, upload


def parse_items(spec: str | None, all_items: list) -> list:
    if not spec:
        return list(all_items)
    if ":" in spec and all(p.isdigit() or p == "" for p in spec.split(":")):
        a, b = spec.split(":")
        return list(all_items)[int(a or 0) : int(b) if b else None]
    return spec.split(",")


def select_arms(env, spec: str | None) -> list[str]:
    """--arms: comma-separated arm names, or factor filters `key=value` (ANDed; repeat a key to OR its values), e.g.
    `task=calendar,variant=core` or `variant=base,variant=core`. Factors come from the env's ARM_FACTORS."""
    if not spec:
        return list(env.ARMS)
    parts = spec.split(",")
    if not any("=" in p for p in parts):
        return parts
    factors = getattr(env, "ARM_FACTORS", {})
    want: dict[str, set] = {}
    for p in parts:
        k, _, v = p.partition("=")
        want.setdefault(k, set()).add(v)
    picked = [arm for arm, f in factors.items() if all(f.get(k) in vs for k, vs in want.items())]
    if not picked:
        raise SystemExit(f"no arm matches {spec!r} (factors: {sorted({k for f in factors.values() for k in f})})")
    return picked


def add_target_args(ap):
    """Model flags. Each defaults to the env's DEFAULTS (the settings its stored runs used), so a bare `run ENV` reproduces
    the old requests; anything given here that differs from DEFAULTS is recorded as an override."""
    ap.add_argument("--model")
    ap.add_argument("--backend", choices=["vllm", "openrouter", "openrouter-raw"])
    ap.add_argument("--url", help="vllm base URL")
    ap.add_argument("--served-name", help="model name the server expects, if not --model (the shared Qwen server: qwen)")
    ap.add_argument("--provider", help="OpenRouter provider slug, pinned with no fallbacks ('any' = unpinned)")
    ap.add_argument("--effort", help="reasoning effort when thinking is on ('none' = don't send)")
    ap.add_argument("--thinking", choices=["on", "off"])
    ap.add_argument("--max-tokens", type=int, default=None, help="per call; default no cap")
    ap.add_argument("--body", choices=["openrouter", "realism"], help="request body layout (see backends.Target)")
    for k in ("temperature", "top_p", "top_k", "presence_penalty"):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=None)


# Qwen3.6 model-card sampling per thinking mode: the default for any Qwen target whose env declares no sampling
QWEN_SAMPLING = {
    "on": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
    "off": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
}


def make_target(env, a) -> tuple[Target, dict]:
    """The env's DEFAULTS overlaid with the flags given. -> (target, {field: {"default", "used"}} for every flag that differs)."""
    d = dict(getattr(env, "DEFAULTS", {}))
    flags = {
        "model": a.model,
        "backend": a.backend,
        "url": a.url,
        "served_name": a.served_name,
        "provider": None if a.provider == "any" else a.provider,
        "effort": None if a.effort == "none" else a.effort,
        "thinking": None if a.thinking is None else a.thinking == "on",
        "max_tokens": a.max_tokens,
        "body": a.body,
    }
    given = {k: v for k, v in flags.items() if v is not None or (k in ("provider", "effort") and getattr(a, k) is not None)}
    base: dict[str, Any] = {"effort": "medium", "thinking": True, "body": "openrouter"}
    cfg: dict[str, Any] = {**base, **{k: v for k, v in d.items() if k != "sampling"}, **given}
    # the env's stored-run sampling applies to its own default model; any other Qwen gets the model card, others nothing
    own = (cfg.get("model") == d.get("model") or d.get("sampling_all_models")) and d.get("sampling")
    sd = own or (QWEN_SAMPLING if "qwen" in str(cfg.get("model")).lower() else {})
    sampling = dict(sd.get("on" if cfg["thinking"] else "off", {})) if set(sd) <= {"on", "off"} else dict(sd)
    flag_sampling = {k: getattr(a, k) for k in ("temperature", "top_p", "top_k", "presence_penalty") if getattr(a, k) is not None}
    sampling |= flag_sampling
    if not cfg.get("model") or not cfg.get("backend"):
        raise SystemExit("no model: pass --model and --backend (this env declares no DEFAULTS)")
    t = Target(
        model=cfg["model"],
        backend=cfg["backend"],
        url=cfg.get("url"),
        served_name=cfg.get("served_name"),
        provider=cfg.get("provider"),
        effort=cfg.get("effort"),
        thinking=bool(cfg.get("thinking")),
        max_tokens=cfg.get("max_tokens"),
        sampling=sampling,
        body=cfg.get("body") or "openrouter",
    )
    overrides = {k: {"default": d.get(k), "used": v} for k, v in given.items() if k in d and d[k] != v}
    overrides |= {f"sampling.{k}": {"default": sampling.get(k), "used": v} for k, v in flag_sampling.items()}
    return t, overrides


def main(argv):
    ap = argparse.ArgumentParser(prog="envrun")
    ap.add_argument("env")
    ap.add_argument("--tag", required=True, help="run slug; the run is data/envs/ENV/<date>_<tag>")
    ap.add_argument("--arms", help="comma list of arms, or factor filters like task=calendar,variant=core (default: all ARMS)")
    ap.add_argument("--pair", help="run a PAIRS entry's two arms (and its item subset)")
    ap.add_argument("--items", help="slice a:b or comma list (default: all)")
    ap.add_argument("--n", type=int, default=1, help="samples per (arm, item)")
    ap.add_argument("--today", help="frozen date for the env's clock (default: today)")
    ap.add_argument("--max-turns", type=int)
    ap.add_argument("--max-prompt-tokens", type=int)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--override", action="append", default=[], help="KEY: accept a flag that conflicts with the env's REQUIRES")
    ap.add_argument("-d", "--description", default="")
    ap.add_argument("--no-upload", action="store_true", help="don't upload the run + indexes to HF when done")
    add_target_args(ap)
    a = ap.parse_args(argv)

    env = load_env(a.env)
    target, default_overrides = make_target(env, a)
    requires = getattr(env, "REQUIRES", {})
    overrides = {}
    for k, want in requires.items():
        have = getattr(target, k, None)
        if have != want:
            if k not in a.override:
                raise SystemExit(f"{a.env} REQUIRES {k}={want!r} but the flags give {have!r}; pass --override {k} to run anyway")
            overrides[k] = {"required": want, "used": have}

    arms, items = select_arms(env, a.arms), parse_items(a.items, env.items())
    if a.pair:
        p = env.PAIRS[a.pair]
        arms = list(p[:2])
        if len(p) > 2:
            items = [str(i) for i in p[2]]
    unknown = [x for x in arms if x not in env.ARMS]
    if unknown:
        print(f"note: arms not in ARMS (build() must accept them): {unknown}", file=sys.stderr)

    # an env whose items depend on the arm (items_for) runs each arm over its own items, sliced the same way
    def arm_items(arm):
        if not hasattr(env, "items_for") or (a.pair and len(env.PAIRS[a.pair]) > 2):
            return items
        own = env.items_for(arm)
        return [i for i in parse_items(a.items, own) if i in own]

    per_arm = {arm: arm_items(arm) for arm in arms}
    today = dt.date.fromisoformat(a.today) if a.today else dt.date.today()
    caps = {
        **getattr(env, "CAPS", {}),
        **{k: v for k, v in {"max_turns": a.max_turns, "max_prompt_tokens": a.max_prompt_tokens}.items() if v},
    }
    run = f"{today.isoformat()}_{a.tag}"
    w = RunWriter(
        a.env,
        run,
        {
            "description": a.description,
            "command": "python -m environments run " + shlex.join(argv),
            "git": git_sha(),
            "target": target.record(),
            "caps": caps,
            "today": today.isoformat(),
            "overrides": overrides,
            "defaults_overridden": default_overrides,
            "arms": arms,
            "items": items if not hasattr(env, "items_for") else {arm: per_arm[arm] for arm in arms},
            "n_per_cell": a.n,
            "primary": env.PRIMARY,
            "docent_collection": None,
        },
    )
    width = max(map(len, per_arm.values()), default=0)
    jobs = [(arm, per_arm[arm][i], s) for s in range(a.n) for i in range(width) for arm in arms if i < len(per_arm[arm])]
    print(f"{a.env}: {len(jobs)} episodes -> {w.dir}", flush=True)
    results: list[dict] = []
    status = "done"
    try:
        asyncio.run(_run_all(env, jobs, target, today, caps, a.conc, w, results))
    except KeyboardInterrupt:
        status = "abandoned"
        print(f"interrupted after {len(results)}/{len(jobs)} episodes; finishing as abandoned", file=sys.stderr)
    labels = []
    for ep in results:
        try:
            labels.append((ep["id"], None if ep.get("error") else env.measure(ep)))
        except Exception as e:  # noqa: BLE001
            labels.append((ep["id"], {"_error": repr(e)[:300]}))
    w.labels("measure", (git_sha() or "unknown")[:12], labels)
    cost = sum((ep.get("usage") or {}).get("cost") or 0 for ep in results)
    w.finish(status, cost_usd=round(cost, 6), errors=sum(1 for ep in results if ep.get("error")))
    print(f"{status}: {len(results)} episodes, {w.meta['errors']} errors, ${cost:.4f}; {w.dir}")
    docent_upload(w.dir)
    written = build_index([a.env])
    if not a.no_upload:
        n, oid = upload([w.dir, *written], f"envrun {a.env} {run} ({status})")
        print(f"uploaded {n} files to envs/ ({oid[:10]})")


def docent_upload(run_dir):
    """Mirror the run's transcripts to Docent before the HF upload, so the Docent ids ship with the run. Never blocks it."""
    cmd = ["uv", "run", "-q", "--script", str(REPO_ROOT / "scripts" / "docent_upload.py"), "upload", str(run_dir)]
    if not os.environ.get("DOCENT_API_KEY"):
        print(f"DOCENT_API_KEY unset; not uploading to Docent. Later: {shlex.join(cmd)} (or `backfill`)")
    elif subprocess.run(cmd, check=False).returncode:
        print(f"Docent upload failed; HF upload continues. Retry: {shlex.join(cmd)}")


async def _run_all(env, jobs, target, today, caps, conc, w, out: list):
    sem = asyncio.Semaphore(conc)

    async def one(arm, item, s):
        ctx = Ctx(env=env.NAME, arm=arm, item=str(item), sample=s, today=today, now=dt.datetime.combine(today, dt.time(12, 0)))
        rec = {"id": episode_id(arm, item, s), "arm": arm, "item": str(item), "sample": s, "target": target.id, "today": today.isoformat()}
        async with sem:
            try:
                ep = env.build(arm, item, ctx)
                rec |= await run_episode(env, ep, ctx, target, client, caps)
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)[:1000]
                rec["traceback"] = traceback.format_exc()[-3000:]
        w.add(rec)
        out.append(rec)
        print(rec["id"], rec.get("finish") or "ERROR " + rec.get("error", "")[:200], flush=True)

    async with httpx.AsyncClient(timeout=httpx.Timeout(6 * 3600, connect=60)) as client:
        await asyncio.gather(*(one(*j) for j in jobs))


def cli():
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("-h", [])
    if cmd == "run":
        main(rest)
    elif cmd == "index":
        for p in build_index(rest or None):
            print(p)
    elif cmd == "upload":
        dirs = [LOCAL / r for r in rest]
        envs = sorted({r.split("/")[0] for r in rest})
        written = build_index(envs)
        n, oid = upload([*dirs, *written], "envs: upload " + ", ".join(rest))
        print(f"uploaded {n} files ({oid[:10]})")
    else:
        print(__doc__)
        print("envs:", json.dumps(sorted(p.name for p in LOCAL.glob("*") if p.is_dir())))
