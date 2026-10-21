"""Read-only viewer for CHIVE-on-WeirdChat run directories (data/runs/*), embedded as a tab in the
olens dashboard. Stdlib HTTP; reads the run files on each request (cached by mtime).

    uv run python dashboard/server.py            # http://bramble:8805/ (tailnet only)
"""

import argparse
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from claims import label

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "data" / "runs"
_cache: dict[Path, tuple[float, object]] = {}


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    m = path.stat().st_mtime
    hit = _cache.get(path)
    if hit and hit[0] == m:
        return hit[1]  # type: ignore[return-value]
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    _cache[path] = (m, recs)
    return recs


def jload(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def tail(path: Path, n: int = 80) -> list[str]:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 200_000))
        lines = f.read().decode(errors="replace").splitlines()
    return [x[:400] for x in lines if x.strip()][-n:]


def run_list() -> list[dict]:
    out = []
    for d in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        if not (d / "config.json").exists():
            continue
        cfg = jload(d / "config.json", {})
        log = d / "log.txt"
        out.append(dict(
            name=d.name,
            status=jload(d / "status.json", {}),
            n_prompts=len(jload(d / "wc_meta.json", {})),
            target=cfg.get("target_llm"),
            n_completions=cfg.get("n_completions"),
            claude=cfg.get("stage3_llm"),
            grader=cfg.get("grader_llm"),
            log_mtime=log.stat().st_mtime if log.exists() else None,
            verifier=[v["score"] for v in jsonl(d / "verification.jsonl")],
            has_trace=(d / "investigator.jsonl").exists(),
        ))
    return out


def run_detail(name: str) -> dict:
    d = RUNS / name
    assert d.parent == RUNS and (d / "config.json").exists(), f"no run {name}"
    meta = jload(d / "wc_meta.json", {})
    pool = {r["id"]: r for r in jsonl(d / "prompts.jsonl")}
    ver = {v["prompt_id"]: v for v in jsonl(d / "verification.jsonl")}
    scr = {s["prompt_id"]: s for s in jsonl(d / "screening.jsonl")}
    prompts = {}
    for pid, m in meta.items():
        prompts[pid] = dict(
            {k: v for k, v in m.items() if k not in ("rubric", "example_reply")},
            user_message=pool.get(pid, {}).get("user_message", ""),
            screening=dict((k, scr[pid].get(k)) for k in ("interest_score", "question", "hypotheses", "counterfactual"))
            if pid in scr
            else None,
        )
    invs = []
    inv_recs = jsonl(d / "investigations.jsonl")
    base = jload(d / "baseline.json")
    if not inv_recs and base and meta:  # wctool.py case in progress: show experiments as they land
        pid = next(iter(meta))
        inv_recs = [dict(prompt_id=pid, baseline_grader_yes=sum(base["verdicts"]), baseline_grader_verdicts=base["verdicts"],
                         completions=base["completions"], experiment_log=jsonl(d / "experiments.jsonl"),
                         question="(investigation in progress)", investigation_report="(in progress; not submitted yet)")]
    for inv in inv_recs:
        pid = inv["prompt_id"]
        k0, n0 = inv["baseline_grader_yes"], len(inv["baseline_grader_verdicts"])
        exps = []
        for e in inv["experiment_log"]:
            k1, n1 = e["grader_yes"], len(e["grader_verdicts"])
            exps.append(dict(
                exp_num=e["exp_num"], k=k1, n=n1, **label(k0, n0, k1, n1),
                messages=e["messages"], edit_record=e.get("edit_record"),
            ))
        v = ver.get(pid)
        sf = inv.get("structured_findings") or {}
        invs.append(dict(
            prompt_id=pid, k0=k0, n0=n0, experiments=exps,
            n_agent_turns=inv.get("n_agent_turns"),
            tokens=dict(input=inv.get("input_tokens"), cache_read=inv.get("cache_read_tokens"), output=inv.get("output_tokens")),
            question=inv.get("question"),
            structured_answer=sf.get("structured_answer") or inv.get("investigation_report", "")[:20000],
            commentary=sf.get("commentary"),
            confidence=sf.get("confidence"),
            verification=dict(score=v["score"], explanation=v.get("explanation_assessment"), issues=v.get("issues"),
                              experiments=v.get("experiment_assessments")) if v else None,
        ))
    calib = {}
    for f in sorted(d.glob("calibration_*.json")):
        c = jload(f)
        calib[f.stem.removeprefix("calibration_")] = {  # reply texts are fetched one at a time (/api/calib_text)
            pid: dict(x, disagreements=[{k: v for k, v in g.items() if k != "text"} for g in x.get("disagreements", [])])
            for pid, x in c.items()
        }
    return dict(name=name, config=jload(d / "config.json"), status=jload(d / "status.json", {}),
                prompts=prompts, investigations=invs, calibration=calib, log=tail(d / "log.txt"))


def trace(name: str) -> dict:
    """The headless investigator's stream (wctool.py investigate) as a list of steps."""
    d = RUNS / name
    assert d.parent == RUNS
    steps, summary = [], None
    f = d / "investigator.jsonl"
    for line in f.read_text().splitlines() if f.exists() else []:
        if not line.startswith("{"):
            continue
        e = json.loads(line)
        if e.get("type") == "assistant":
            for c in e["message"].get("content", []):
                if c.get("type") == "text" and c["text"].strip():
                    steps.append(dict(kind="text", text=c["text"]))
                elif c.get("type") == "thinking" and c.get("thinking", "").strip():
                    steps.append(dict(kind="thinking", text=c["thinking"]))
                elif c.get("type") == "tool_use":
                    inp = c.get("input", {})
                    steps.append(dict(kind="call", tool=c.get("name"), text=inp.get("command") or json.dumps(inp)[:20000]))
        elif e.get("type") == "user":
            for c in e["message"].get("content", []) if isinstance(e["message"].get("content"), list) else []:
                if c.get("type") == "tool_result":
                    body = c.get("content")
                    if isinstance(body, list):
                        body = "\n".join(b.get("text", "") for b in body if isinstance(b, dict))
                    steps.append(dict(kind="error" if c.get("is_error") else "result", text=str(body)))
        elif e.get("type") == "result":
            summary = {k: e.get(k) for k in ("num_turns", "duration_ms", "total_cost_usd", "stop_reason", "subtype")}
    return dict(steps=steps, summary=summary)


def calib_text(name: str, grader: str, pid: str, i: int) -> dict:
    """One calibration disagreement in full: the prompt, the reply, the rubric and both judges' reasons."""
    d = RUNS / name
    assert d.parent == RUNS and "/" not in grader
    g = jload(d / f"calibration_{grader}.json")[pid]["disagreements"][i]
    m = jload(d / "wc_meta.json")[pid]
    prompt = {r["id"]: r for r in jsonl(d / "prompts.jsonl")}[pid]["messages"]
    return dict(g, prompt=prompt, rubric=m["rubric"], rubric_id=m["rubric_id"])


def samples(name: str, pid: str, exp: int) -> dict:
    d = RUNS / name
    assert d.parent == RUNS
    base = jload(d / "baseline.json")
    recs = jsonl(d / "investigations.jsonl") or ([dict(prompt_id=pid, completions=base["completions"],
        baseline_grader_verdicts=base["verdicts"], experiment_log=jsonl(d / "experiments.jsonl"))] if base else [])
    for inv in recs:
        if inv["prompt_id"] != pid:
            continue
        if exp == 0:
            return dict(completions=inv["completions"], verdicts=inv["baseline_grader_verdicts"])
        for e in inv["experiment_log"]:
            if e["exp_num"] == exp:
                return dict(completions=e["completions"], verdicts=e["grader_verdicts"])
    raise KeyError(f"{pid} exp {exp}")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p, _, q = self.path.partition("?")
        args = dict(kv.split("=", 1) for kv in q.split("&") if "=" in kv)
        try:
            if p in ("/", "/index.html"):
                return self.send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if p == "/api/runs":
                return self.send(json.dumps(run_list()))
            if p == "/api/run":
                return self.send(json.dumps(run_detail(args["name"])))
            if p == "/api/trace":
                return self.send(json.dumps(trace(args["name"])))
            if p == "/api/calib_text":
                return self.send(json.dumps(calib_text(args["name"], args["grader"], args["pid"], int(args["i"]))))
            if p == "/api/samples":
                return self.send(json.dumps(samples(args["name"], args["pid"], int(args["exp"]))))
            self.send('{"error":"not found"}', code=404)
        except Exception as e:
            self.send(json.dumps(dict(error=f"{type(e).__name__}: {e}")), code=500)


class Server6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8805)
    ap.add_argument("--bind", nargs="+", default=["100.86.34.70", "fd7a:115c:a1e0::e934:2247"], help="tailnet addresses")
    args = ap.parse_args()
    servers = [(Server6 if ":" in b else ThreadingHTTPServer)((b, args.port), H) for b in args.bind]
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f"listening on {args.bind} port {args.port}", flush=True)
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
