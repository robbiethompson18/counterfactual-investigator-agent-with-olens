"""Live dashboard for an evolve run: the provenance tree with every node's scores, and each node opened into its
rates, diffs, files, Target transcripts, Forecaster trials, Rate audit and agent job, with every model's exact prompt.

    evolve dash <run> [--port 8815] [--host ADDR ...]

Read-only over data/<run>/ (index.json, events.jsonl, nodes/, agent_jobs/), so it can run beside the loop. The page
(dashboard.html) polls /api/state and re-renders whatever changed.

    /api/state?run=              index + per-node posteriors, latest epoch (front, layers), agent jobs, event count,
                                 job progress (cap, per-job stage) and the Qwen vLLM queue
    /api/events?run=&since=N     events.jsonl lines from N on
    /api/node?run=&id=           one node: info, texts, diffs, file tree, scores, forecaster trials, audits, agent job
    /api/rows?run=&id=&batch=&arm=           one batch of Target runs, summarised
    /api/record?run=&id=&batch=&arm=&sample= one Target run in full, with the exact prompt it was given
    /api/file?run=&id=&path=     a text file inside the node dir
    /api/nodediff?run=&id=&other=   how node id differs from other (default: its parent), file by file
    /api/job?run=&job=           an agent job: job.json, INSTRUCTIONS.md, transcript, inputs
    /api/comparisons?run=&node=  innocuousness comparisons (all, or those involving node), newest first
    /api/comparison?run=&id=     one comparison: the judge's exact messages, tool calls and verdict
"""

import argparse
import difflib
import functools
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .measure import batches, load
from .node import ARMS, Node, unified_diff
from .store import ROOT

QWEN_METRICS = os.environ.get("QWEN_VLLM_URL", "http://127.0.0.1:18000/v1").rsplit("/v1", 1)[0] + "/metrics"

PAGE = Path(__file__).resolve().parent / "dashboard.html"
FORECASTER_DIRS = ("forecaster", "picker")  # the Forecaster was called the Picker; old runs keep picker/
GENOME_FILES = ("behaviour.txt", "grader.py", "hypothesis.md", "node.json")
MAX_FILE = 400_000
DRAWS = 4000


@functools.lru_cache(maxsize=4096)
def beta_summary(k: int, n: int) -> dict:
    """Uniform-prior posterior Beta(1+k, 1+n-k): mean, 95% interval, P(p > 0.5)."""
    rng = random.Random(f"{k}/{n}")
    xs = sorted(rng.betavariate(1 + k, 1 + n - k) for _ in range(DRAWS))
    return {
        "k": k,
        "n": n,
        "mean": (1 + k) / (2 + n),
        "ci95": [xs[int(0.025 * DRAWS)], xs[int(0.975 * DRAWS) - 1]],
        "p_gt_half": sum(x > 0.5 for x in xs) / DRAWS,
    }


def _json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _text(p: Path, limit=MAX_FILE):
    try:
        t = p.read_text(errors="replace")
    except OSError:
        return None
    return t if len(t) <= limit else t[:limit] + f"\n...[{len(t) - limit} more chars]"


def _jsonl(p: Path) -> list[dict]:
    out = []
    try:
        with open(p) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass  # a line being appended right now
    except OSError:
        pass
    return out


def _inside(base: Path, rel: str) -> Path | None:
    p = (base / rel).resolve()
    return p if p.is_relative_to(base.resolve()) and p.is_file() else None


def _tree(d: Path, skip=("__pycache__",)) -> list[dict]:
    """[{path, size}] for every file under d (relative paths), skipping caches."""
    if not d.is_dir():
        return []
    return [
        {"path": str(p.relative_to(d)), "size": p.stat().st_size}
        for p in sorted(d.rglob("*"))
        if p.is_file() and not any(s in p.parts for s in skip)
    ]


def _arm_prompt(node: Node, arm: str) -> dict:
    a = node.arm(arm)
    return {"messages": a.messages, "tools": a.tools, "episode": a.episode, "files": a.files}


def _forecaster_scores(s: dict) -> dict:
    return s.get("forecaster") or s.get("picker") or {}


def _normalise_scores(s: dict) -> dict:
    """scores.json with the old Picker keys renamed to Forecaster ones, so the page reads one name."""
    for k in ("rank", "pooled"):
        iv = (s.get(k) or {}).get("intervals80") or {}
        if "picker_p" in iv:
            iv.setdefault("forecaster_p", iv.pop("picker_p"))
    if "picker" in s:
        s.setdefault("forecaster", s.pop("picker"))
    return s


class Run:
    def __init__(self, name: str):
        self.name = name
        self.dir = ROOT / name
        if not (self.dir / "nodes").is_dir():
            raise FileNotFoundError(name)

    def node(self, nid: str) -> Node:
        n = Node(self.dir / "nodes" / nid)
        if not n.dir.resolve().is_relative_to((self.dir / "nodes").resolve()) or not n.dir.is_dir():
            raise FileNotFoundError(nid)
        return n

    def events(self, since=0) -> list[dict]:
        return _jsonl(self.dir / "events.jsonl")[since:]

    # --- state -------------------------------------------------------------------------------------------------
    def state(self) -> dict:
        idx = _json(self.dir / "index.json", {}) or {}
        evs = self.events()
        epoch = next((e for e in reversed(evs) if e["event"] == "epoch"), None)
        ratings = _json(self.dir / "innocuous_ratings.json", {}) or {}
        layer_of = {nid: i + 1 for i, layer in enumerate((epoch or {}).get("layers", [])) for nid in layer}
        nodes = {}
        for nid, e in idx.items():
            s = _json(self.dir / "nodes" / nid / "scores.json", {}) or {}
            info = _json(self.dir / "nodes" / nid / "node.json", {}) or {}
            rates = s.get("pooled_rates") or e.get("rates") or {}
            fc = _forecaster_scores(s)
            inv = _json(self.dir / "nodes" / nid / "investigator" / "scores.json")
            ino = ratings.get(nid)
            pick = fc.get("all")
            nodes[nid] = {
                **e,
                "rank": _normalise_scores({"rank": s.get("rank") or e.get("scores")})["rank"],
                "pooled": _normalise_scores({"pooled": s.get("pooled")})["pooled"],
                "rank_batch": s.get("rank_batch"),
                "arm_post": {a: beta_summary(v[0], v[1]) | {"unscorable": v[2]} for a, v in rates.items()},
                "forecaster_post": beta_summary(*pick) if pick and pick[1] else None,
                "forecaster_latest": fc.get("latest"),
                "investigator": inv | {"post": beta_summary(*inv["all"]) if inv.get("all") and inv["all"][1] else None} if inv else None,
                "innocuous": ino,
                "layer": layer_of.get(nid),
                "front_rounds": info.get("front_rounds", 0),
                "ref_count": info.get("ref_count", 0),
                "batches": len(batches(Node(self.dir / "nodes" / nid))),
                "job": info.get("job"),
                "rewrite": bool(info.get("rewrite")),  # older jobs: a separate rewrite mode
                "suggestion": info.get("suggestion"),
            }
        last = evs[-1]["t"] if evs else None
        return {
            "run": self.name,
            "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_events": len(evs),
            "last_event": last,
            "epoch": epoch,
            "nodes": nodes,
            "jobs": self.jobs(),
            "in_flight": self._in_flight(evs),
            "progress": self.progress(evs),
            "n_comparisons": _count_lines(self.dir / "innocuous.jsonl"),
            "vllm": vllm_metrics(),
        }

    def _in_flight(self, evs) -> list[dict]:
        """Work started and not yet finished, from the event log: measurements and agent jobs."""
        measuring, agents = {}, {}
        for e in evs:
            ev = e["event"]
            if ev == "measure_start":
                measuring[e["node"]] = e
            elif ev == "audited":
                measuring.pop(e.get("node"), None)
            elif ev in ("mutation_start", "creation_start"):
                agents[e.get("round")] = e
            elif ev in ("mutation_failed", "creation_failed", "novelty") or (ev == "node_added" and e.get("round") is not None):
                agents.pop(e.get("round"), None)
        live = {n for n in measuring if (_json(self.dir / "nodes" / n / "node.json") or {}).get("stage") in ("measuring", "submitted")}
        return [{"what": "measuring", **e} for n, e in measuring.items() if n in live] + [{"what": "agent", **e} for e in agents.values()]

    def progress(self, evs) -> dict:
        """Per loop job (numbered by job_start), its kind and current stage, replayed from the event log:
        agent (queued / working, from the job dir) -> measuring -> scoring (summariser + audit) -> done | failed."""
        jobs, node_job, measure_s, started = {}, {}, [], {}
        dirs = {}
        for d in (self.dir / "agent_jobs").glob("*/loop.json"):
            r = (_json(d) or {}).get("round")
            if r is not None:
                dirs[r] = d.parent.name
        states = {j["job"]: j["state"] for j in self.jobs()}

        def at(j, stage, t, **kw):
            jobs.setdefault(j, {"job": j, "t0": t}).update(stage=stage, since=t, **kw)

        for e in evs:
            ev, t, r = e["event"], e["t"], e.get("round")
            if ev == "job_start":
                at(e["job"], "starting", t, surprise=e.get("surprise"))
            elif ev in ("mutation_start", "creation_start"):
                kind = ev.split("_")[0] + (" (rewrite)" if e.get("rewrite") else "")
                if e.get("suggestion"):
                    kind += f" ({suggestion_tag(e['suggestion'])})"
                at(r, "agent", t, kind=kind, reference=e.get("reference"), examples=e.get("examples"))
            elif ev == "topup_start":
                at(r, "measuring", t, kind="topup", node=e.get("node"))
                node_job[e.get("node")] = r
            elif ev in ("mutation_failed", "creation_failed"):
                at(r, "failed", t, outcome="; ".join(map(str, e.get("errors") or []))[:300])
            elif ev == "novelty" and not e.get("new"):
                at(r, "failed", t, outcome=f"duplicate of {e.get('duplicate_of')}")
            elif ev == "node_added" and r is not None and r in jobs:
                at(r, "admitted", t, node=e["node"])
                node_job[e["node"]] = r
            elif ev in ("measure_start", "measure_resumed"):
                started[e["node"]] = t
                if e["node"] in node_job:
                    at(node_job[e["node"]], "measuring", t)
            elif ev == "measured":
                if e["node"] in started:
                    measure_s.append(_secs(e["t"]) - _secs(started.pop(e["node"])))
                if e["node"] in node_job:
                    at(node_job[e["node"]], "scoring", t)
            elif ev == "audited" and e["node"] in node_job:
                at(node_job[e["node"]], "done", t, outcome=f"audit {e.get('score')}")
        # A loop restart abandons in-flight top-ups and re-measures children under their old job, so the log alone
        # can leave ghosts: a job is live only if it is its node's latest job and the node is still being measured.
        stage = {nid: (_json(self.dir / "nodes" / nid / "node.json") or {}).get("stage") for nid in set(node_job) | set(started)}
        latest = {}
        for j in sorted(jobs.values(), key=lambda j: j["job"]):
            if j.get("node"):
                latest[j["node"]] = j["job"]
        for j in jobs.values():
            j["dir"] = dirs.get(j["job"])
            if j["stage"] == "agent":
                j["agent_state"] = states.get(j["dir"], "posting")
                if j["agent_state"] in ("done", "failed"):
                    j.update(stage="interrupted", outcome="agent finished but the loop never picked up the result")
            elif j["stage"] in ("measuring", "scoring", "admitted") and (
                latest.get(j.get("node")) != j["job"] or stage.get(j.get("node")) not in ("measuring", "submitted")
            ):
                j.update(stage="interrupted", outcome="abandoned by a loop restart")
        return {
            "max_jobs": (_json(self.dir / "loop.json") or {}).get("max_jobs"),
            "started": sum(e["event"] == "job_start" for e in evs),
            "finished": bool(evs) and evs[-1]["event"] == "loop_finished",
            "jobs": sorted(jobs.values(), key=lambda j: j["job"]),
            "measure_median_s": statistics.median(measure_s[-30:]) if measure_s else None,
            "seed_measuring": [{"node": n, "since": t} for n, t in started.items() if n not in node_job and stage.get(n) == "measuring"],
            "recent": evs[-25:][::-1],
        }

    def jobs(self) -> list[dict]:
        d = self.dir / "agent_jobs"
        if not d.is_dir():
            return []
        out = []
        for j in sorted((p for p in d.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime):
            st = next(
                (s for f, s in (("result.json", "done"), ("failed.json", "failed"), ("claimed.json", "claimed")) if (j / f).exists()), None
            )
            job = _json(j / "job.json", {}) or {}
            out.append(
                {
                    "job": j.name,
                    "kind": job.get("kind"),
                    "state": st or ("pending" if job else "incomplete"),
                    "created": job.get("created"),
                    "claimed": _json(j / "claimed.json"),
                    "result": _json(j / "result.json"),
                    "failed": _json(j / "failed.json"),
                    "reference": (job.get("locked") or {}).get("parent"),
                }
            )
        return out

    # --- one node ----------------------------------------------------------------------------------------------
    def node_detail(self, nid: str) -> dict:
        n = self.node(nid)
        info = n.info
        texts = {f: _text(n.dir / f) for f in GENOME_FILES if (n.dir / f).exists()}
        diffs = {}
        if (n.dir / "baseline").is_dir():
            for a in ("real", "null"):
                if (n.dir / a).is_dir():
                    diffs[a] = unified_diff(n.dir / "baseline", n.dir / a)
        bs = batches(n)
        per_batch = {}
        for b in bs:
            per_batch[b] = {}
            for a in ARMS:
                rows = load(n, b, a)
                k = sum(r["verdict"] is True for r in rows)
                m = sum(r["verdict"] in (True, False) for r in rows)
                per_batch[b][a] = {"k": k, "n": m, "unscorable": len(rows) - m, "post": beta_summary(k, m)}
        forecaster = self._forecaster(n)
        investigator = self._trials(n.dir / "investigator")
        audits = {p.stem: _json(p) for p in sorted((n.dir / "audit").glob("*.json"))} if (n.dir / "audit").is_dir() else {}
        agent_files = {p.name: _text(p) for p in sorted(n.dir.glob("agent_*")) if p.is_file() and p.suffix != ".jsonl"}
        return {
            "id": nid,
            "info": info,
            "texts": texts,
            "diffs": diffs,
            "tree": _tree(n.dir, skip=("__pycache__", "runs", "audit", *FORECASTER_DIRS, "picker_superseded", "investigator")),
            "scores": _normalise_scores(n.scores()),
            "batches": per_batch,
            "forecaster": forecaster,
            "investigator": investigator | {"scores": _json(n.dir / "investigator" / "scores.json")} if investigator else {},
            "audits": audits,
            "novelty": _json(n.dir / "novelty.json"),
            "agent_files": agent_files,
            "job": self.job(info["job"], brief=True) if info.get("job") else None,
        }

    def _trials(self, d: Path) -> dict:
        """{trials: {batch: rows}, summary: {batch: posterior on the correct rate}} for an A/B pick dir."""
        if not d.is_dir():
            return {}
        trials = {p.stem: _jsonl(p) for p in sorted(d.glob("b*.jsonl"), key=lambda p: int(p.stem[1:]))}
        summ = {}
        for b, rows in trials.items():
            k = sum(r.get("correct") is True for r in rows)
            m = sum(r.get("correct") is not None for r in rows)
            summ[b] = beta_summary(k, m) | {"unparsed": len(rows) - m}
        return {"trials": trials, "summary": summ}

    def _forecaster(self, n: Node) -> dict:
        d = next((n.dir / x for x in FORECASTER_DIRS if (n.dir / x).is_dir()), None)
        if d is None:
            return {}
        return self._trials(d) | {
            "prompts": _json(d / "prompts.json"),
            "summariser": _json(d / "summary.json"),
        }

    # --- innocuousness comparisons -----------------------------------------------------------------------------
    def comparisons(self, node: str | None = None) -> list[dict]:
        out = []
        for r in _jsonl(self.dir / "innocuous.jsonl"):
            if node and node not in (r.get("a"), r.get("b")):
                continue
            w = r.get("winner")
            out.append(r | {"winner_id": {"A": r.get("a"), "B": r.get("b")}.get(w, w)})
        return out[::-1]

    def comparison(self, cid: str) -> dict:
        base = self.dir / "innocuous"
        p = _inside(base, f"{cid}.json")
        if not p:
            raise FileNotFoundError(cid)
        d = _json(p) or {}
        w = d.get("winner")
        try:
            from .innocuous import BASH, VERDICT  # what the judge is offered (current code, not stored per comparison)

            tools = [BASH, VERDICT]
        except ImportError:
            tools = None
        return d | {"winner_id": {"A": d.get("a"), "B": d.get("b")}.get(w, w), "tools_current": tools}

    def rows(self, nid, batch, arm) -> list[dict]:
        out = []
        for r in load(self.node(nid), batch, arm):
            rec = r.get("record") or {}
            out.append(
                {
                    "sample": r["sample"],
                    "verdict": r["verdict"],
                    "grader_error": r.get("grader_error"),
                    "error": rec.get("error"),
                    "finish": rec.get("finish"),
                    "n_turns": rec.get("n_turns"),
                    "tokens": rec.get("completion_tokens"),
                    "final": (rec.get("final_response") or "")[-200:],
                }
            )
        return out

    def record(self, nid, batch, arm, sample: int) -> dict:
        n = self.node(nid)
        row = next((r for r in load(n, batch, arm) if r["sample"] == sample), None)
        if row is None:
            raise FileNotFoundError(f"{nid}/{batch}/{arm}/{sample}")
        prompt = _arm_prompt(n, arm)
        rec = row.get("record") or {}
        before = prompt.pop("files")
        rec_files = rec.pop("files_after", {}) or {}
        changed = {p: c for p, c in rec_files.items() if before.get(p) != c}
        deleted = [p for p in before if p not in rec_files] if rec_files else []
        rec.pop("transcript", None)  # same content as turns, in chat format
        return {**row, "record": rec, "prompt": prompt, "files_changed": changed, "files_deleted": deleted}

    def node_diff(self, nid: str, other: str | None = None) -> dict:
        """Genome diff other -> nid: behaviour, grader, each arm dir, plus both nodes' REAL / NULL edits side by side."""
        n = self.node(nid)
        other = other or n.info.get("parent")
        if not other:
            return {"id": nid, "other": None}
        o = self.node(other)
        files = {}
        for f in ("behaviour.txt", "grader.py", "hypothesis.md"):
            a, b = _text(o.dir / f) or "", _text(n.dir / f) or ""
            if a != b:
                files[f] = "\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(), f"{other}/{f}", f"{nid}/{f}", lineterm="", n=2))
        for arm in ("baseline", "null", "real"):
            if (o.dir / arm).is_dir() and (n.dir / arm).is_dir():
                d = unified_diff(o.dir / arm, n.dir / arm)
                if d:
                    files[arm + "/"] = d
        pair = (("other", o), ("node", n))
        edits = {who: {a: unified_diff(x.dir / "baseline", x.dir / a) for a in ("real", "null") if (x.dir / a).is_dir()} for who, x in pair}
        return {"id": nid, "other": other, "files": files, "edits": edits}

    def file(self, nid, rel) -> dict:
        n = self.node(nid)
        p = _inside(n.dir, rel)
        if not p:
            raise FileNotFoundError(rel)
        return {"path": rel, "text": _text(p)}

    # --- agent jobs --------------------------------------------------------------------------------------------
    def job(self, name: str, brief=False) -> dict | None:
        j = self.dir / "agent_jobs" / name
        if not j.resolve().is_relative_to((self.dir / "agent_jobs").resolve()) or not j.is_dir():
            return None
        out = {
            "job": name,
            "meta": _json(j / "job.json"),
            "claimed": _json(j / "claimed.json"),
            "result": _json(j / "result.json"),
            "failed": _json(j / "failed.json"),
            "manager": _json(j / "manager.json"),
            "has_transcript": (j / "transcript.jsonl").exists(),
        }
        if brief:
            return out
        return out | {
            "instructions": _text(j / "INSTRUCTIONS.md"),
            "report": _text(j / "transcript.md"),
            "transcript": cc_transcript(j / "transcript.jsonl"),
            "ro": _tree(j / "ro"),
        }

    def job_file(self, name, rel) -> dict:
        base = self.dir / "agent_jobs" / name
        p = _inside(base, rel)
        if not p:
            raise FileNotFoundError(rel)
        return {"path": rel, "text": _text(p)}


def suggestion_tag(text: str) -> str:
    """Short name for an Evolver suggestion (loop.SUGGESTIONS), by its wording."""
    t = text.lower()
    for key, tag in (("simplif", "simplify"), ("rewrite", "rewrite"), ("reward hacking", "hack-tool"), ("realistic", "realism")):
        if key in t:
            return tag
    return "suggestion"


def _count_lines(p: Path) -> int:
    try:
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _secs(t: str) -> float:
    return time.mktime(time.strptime(t, "%Y-%m-%dT%H:%M:%SZ"))


_vllm = {"t": 0.0, "out": None, "tok": None}
_vllm_lock = threading.Lock()


def vllm_metrics() -> dict | None:
    """Requests running / waiting, KV-cache use and generation tokens/s on the Qwen server (cached 2 s)."""
    with _vllm_lock:
        now = time.time()
        if now - _vllm["t"] < 2:
            return _vllm["out"]
        try:
            text = urllib.request.urlopen(QWEN_METRICS, timeout=1.5).read().decode()
        except OSError as e:
            _vllm.update(t=now, out={"error": str(e)[:200]})
            return _vllm["out"]

        def total(name):
            return sum(float(v) for v in re.findall(rf"^{name}{{[^}}]*}} ([0-9.eE+-]+)$", text, re.MULTILINE))

        kv = [float(v) for v in re.findall(r"^vllm:kv_cache_usage_perc{[^}]*} ([0-9.eE+-]+)$", text, re.MULTILINE)]
        tok = total("vllm:generation_tokens_total")
        prev = _vllm["tok"]
        rate = (tok - prev[1]) / (now - prev[0]) if prev and now > prev[0] else None
        _vllm.update(
            t=now,
            tok=(now, tok),
            out={
                "running": total("vllm:num_requests_running"),
                "waiting": total("vllm:num_requests_waiting"),
                "kv": max(kv) if kv else None,
                "gen_tok_s": rate,
                "engines": len(kv),
            },
        )
        return _vllm["out"]


def cc_transcript(p: Path) -> list[dict]:
    """A Claude Code (sub)agent transcript as [{role, kind, text, name?, input?}] in order."""
    out = []
    for line in _jsonl(p):
        msg = line.get("message")
        if line.get("type") not in ("user", "assistant") or not isinstance(msg, dict):
            continue
        role = msg.get("role", line["type"])
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "kind": "text", "text": content})
            continue
        for c in content or []:
            t = c.get("type")
            if t == "text":
                out.append({"role": role, "kind": "text", "text": c.get("text", "")})
            elif t == "thinking":
                out.append({"role": role, "kind": "thinking", "text": c.get("thinking", "")})
            elif t == "tool_use":
                out.append({"role": role, "kind": "tool_use", "name": c.get("name"), "input": c.get("input")})
            elif t == "tool_result":
                body = c.get("content")
                if isinstance(body, list):
                    body = "\n".join(x.get("text", "") for x in body if isinstance(x, dict))
                out.append({"role": role, "kind": "tool_result", "text": body or "", "error": bool(c.get("is_error"))})
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/runs":
                return self._json(sorted(p.name for p in ROOT.iterdir() if (p / "nodes").is_dir()))
            run = Run(q.get("run", self.server.default_run))
            match u.path:
                case "/api/state":
                    return self._json(run.state())
                case "/api/events":
                    return self._json(run.events(int(q.get("since", 0))))
                case "/api/node":
                    return self._json(run.node_detail(q["id"]))
                case "/api/rows":
                    return self._json(run.rows(q["id"], q["batch"], q["arm"]))
                case "/api/record":
                    return self._json(run.record(q["id"], q["batch"], q["arm"], int(q["sample"])))
                case "/api/nodediff":
                    return self._json(run.node_diff(q["id"], q.get("other")))
                case "/api/file":
                    return self._json(run.file(q["id"], q["path"]))
                case "/api/job":
                    j = run.job(q["job"])
                    return self._json(j) if j else self._send(404, b'{"error":"no such job"}')
                case "/api/comparisons":
                    return self._json(run.comparisons(q.get("node")))
                case "/api/comparison":
                    return self._json(run.comparison(q["id"]))
                case "/api/jobfile":
                    return self._json(run.job_file(q["job"], q["path"]))
            self._send(404, b'{"error":"not found"}')
        except (FileNotFoundError, KeyError) as e:
            self._send(404, json.dumps({"error": f"not found: {e}"}).encode())
        except Exception as e:  # noqa: BLE001 (keep serving; report the failure to the page)
            self._send(500, json.dumps({"error": repr(e)}).encode())

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode())


def serve(run: str, hosts: list[str], port: int):
    import socket
    import threading

    servers = []
    for h in hosts:
        cls = type("S", (ThreadingHTTPServer,), {"address_family": socket.AF_INET6 if ":" in h else socket.AF_INET, "daemon_threads": True})
        s = cls((h, port), Handler)
        s.default_run = run
        servers.append(s)
        print(f"evolve dashboard for {run}: http://{'[' + h + ']' if ':' in h else h}:{port}/", flush=True)
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    servers[0].serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evolve dash")
    ap.add_argument("run")
    ap.add_argument("--port", type=int, default=8815)
    ap.add_argument("--host", action="append", help="address to bind (repeatable); default 127.0.0.1")
    a = ap.parse_args(argv)
    serve(a.run, a.host or ["127.0.0.1"], a.port)


if __name__ == "__main__":
    main()
