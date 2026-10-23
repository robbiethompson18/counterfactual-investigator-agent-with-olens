"""Live dashboard for an evolve run: the provenance tree with every node's scores, and each node opened into its
rates, diffs, files, Target transcripts, Forecaster trials, Rate audit and agent job, with every model's exact prompt.

    evolve dash <run> [--port 8815] [--host ADDR ...]

Read-only over data/<run>/ (index.json, events.jsonl, nodes/, agent_jobs/), so it can run beside the loop. The page
(dashboard.html) polls /api/state and re-renders whatever changed.

    /api/state?run=              index + per-node posteriors, latest epoch (front, layers), agent jobs, event count
    /api/events?run=&since=N     events.jsonl lines from N on
    /api/node?run=&id=           one node: info, texts, diffs, file tree, scores, forecaster trials, audits, agent job
    /api/rows?run=&id=&batch=&arm=           one batch of Target runs, summarised
    /api/record?run=&id=&batch=&arm=&sample= one Target run in full, with the exact prompt it was given
    /api/file?run=&id=&path=     a text file inside the node dir
    /api/job?run=&job=           an agent job: job.json, INSTRUCTIONS.md, transcript, inputs
"""

import argparse
import functools
import json
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .measure import batches, load
from .node import ARMS, Node, unified_diff
from .store import ROOT

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
        layer_of = {nid: i + 1 for i, layer in enumerate((epoch or {}).get("layers", [])) for nid in layer}
        nodes = {}
        for nid, e in idx.items():
            s = _json(self.dir / "nodes" / nid / "scores.json", {}) or {}
            info = _json(self.dir / "nodes" / nid / "node.json", {}) or {}
            rates = s.get("pooled_rates") or e.get("rates") or {}
            fc = _forecaster_scores(s)
            pick = fc.get("all")
            nodes[nid] = {
                **e,
                "rank": _normalise_scores({"rank": s.get("rank") or e.get("scores")})["rank"],
                "pooled": _normalise_scores({"pooled": s.get("pooled")})["pooled"],
                "rank_batch": s.get("rank_batch"),
                "arm_post": {a: beta_summary(v[0], v[1]) | {"unscorable": v[2]} for a, v in rates.items()},
                "forecaster_post": beta_summary(*pick) if pick and pick[1] else None,
                "forecaster_latest": fc.get("latest"),
                "layer": layer_of.get(nid),
                "front_rounds": info.get("front_rounds", 0),
                "ref_count": info.get("ref_count", 0),
                "batches": len(batches(Node(self.dir / "nodes" / nid))),
                "job": info.get("job"),
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
        return [{"what": "measuring", **e} for e in measuring.values()] + [{"what": "agent", **e} for e in agents.values()]

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
        audits = {p.stem: _json(p) for p in sorted((n.dir / "audit").glob("*.json"))} if (n.dir / "audit").is_dir() else {}
        agent_files = {p.name: _text(p) for p in sorted(n.dir.glob("agent_*")) if p.is_file() and p.suffix != ".jsonl"}
        return {
            "id": nid,
            "info": info,
            "texts": texts,
            "diffs": diffs,
            "tree": _tree(n.dir, skip=("__pycache__", "runs", "audit", *FORECASTER_DIRS, "picker_superseded")),
            "scores": _normalise_scores(n.scores()),
            "batches": per_batch,
            "forecaster": forecaster,
            "audits": audits,
            "novelty": _json(n.dir / "novelty.json"),
            "agent_files": agent_files,
            "job": self.job(info["job"], brief=True) if info.get("job") else None,
        }

    def _forecaster(self, n: Node) -> dict:
        d = next((n.dir / x for x in FORECASTER_DIRS if (n.dir / x).is_dir()), None)
        if d is None:
            return {}
        trials = {p.stem: _jsonl(p) for p in sorted(d.glob("b*.jsonl"), key=lambda p: int(p.stem[1:]))}
        summ = {}
        for b, rows in trials.items():
            k = sum(r.get("correct") is True for r in rows)
            m = sum(r.get("correct") is not None for r in rows)
            summ[b] = beta_summary(k, m) | {"unparsed": len(rows) - m}
        return {
            "trials": trials,
            "summary": summ,
            "prompts": _json(d / "prompts.json"),
            "summariser": _json(d / "summary.json"),
        }

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
                case "/api/file":
                    return self._json(run.file(q["id"], q["path"]))
                case "/api/job":
                    j = run.job(q["job"])
                    return self._json(j) if j else self._send(404, b'{"error":"no such job"}')
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
