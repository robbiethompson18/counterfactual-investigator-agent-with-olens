"""File browser for baseline-hyp: the agent's evidence folders (views/) and its investigation runs (runs/).

python server.py --port 8812 --bind <addr> [--bind <addr6>]
"""

import argparse
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
TOPS = ["views", "views_single", "runs", "cf", "forks", "judge"]
INDEX = (Path(__file__).parent / "index.html").read_bytes()


RUNS = {"runs/v41": "DSv4.1 Flash", "runs/v4f": "DSv4 Flash"}
SINGLE_RUNS = {"runs/v41_single": "DSv4.1 Flash", "runs/v4f_single": "DSv4 Flash"}
JUDGE = {
    "judge/v41": "DSv4.1 Flash, 8 traces + rate",
    "judge/v41_single": "DSv4.1 Flash, one transcript",
    "judge/v4f": "DSv4 Flash, 8 traces + rate",
    "judge/v4f_single": "DSv4 Flash, one transcript",
}
FORKS = {
    "forks/v41": "after investigation (8 traces + rate)",
    "forks/v41_control": "control: README only, no investigation",
    "forks/v41_env": "control: behaviour + rate + environment, no transcripts",
    "forks/v41_single": "after single-transcript investigation",
}


def index():
    idx = json.loads((ROOT / "index.json").read_text())
    cfp = ROOT / "cf_index.json"
    cf = json.loads(cfp.read_text()) if cfp.exists() else {}
    sp = ROOT / "single_index.json"
    single = json.loads(sp.read_text()) if sp.exists() else {}
    for e in idx:
        e["arms"] = cf.get(e["env"], {}).get("arms", {})
        for v in e["variants"]:
            v["singles"] = []
            for name in single.get(v["view"] or "", []):
                runs = []
                for rd, lab in SINGLE_RUNS.items():
                    rp = ROOT / rd / f"{name}__0.json"
                    if rp.exists():
                        rec = json.loads(rp.read_text())["record"]
                        runs.append(
                            {"path": str(rp.relative_to(ROOT)), "model": lab, "nhyp": len((rec["result"] or {}).get("hypotheses") or [])}
                        )
                v["singles"].append({"name": name, "transcript": f"views_single/{name}/transcript.md", "runs": runs})
            v["cfs"] = cf.get(e["env"], {}).get("views", {}).get(v["view"] or "", [])
            v["runs"] = []
            if v["view"]:
                for rd, lab in RUNS.items():
                    for p in sorted((ROOT / rd).glob(f"{v['view']}__[0-9]*.json")):
                        rec = json.loads(p.read_text())["record"]
                        v["runs"].append(
                            {
                                "path": str(p.relative_to(ROOT)),
                                "seed": rec["seed"],
                                "turns": rec["turns"],
                                "model": lab,
                                "nhyp": len((rec["result"] or {}).get("hypotheses") or []),
                            }
                        )
    return idx


def tree(view):
    d = (ROOT / "views" / view).resolve()
    if not d.is_relative_to(ROOT / "views") or not d.is_dir():
        return []
    return [str(p.relative_to(d)) for p in sorted(d.rglob("*")) if p.is_file()]


def render_run(p):
    d = json.loads(p.read_text())
    rec, msgs = d["record"], d["messages"]
    res = rec.get("result") or {}
    md = [
        f"# Investigation: {rec['env']} seed {rec['seed']}",
        f"{rec['model']} · {rec['turns']} turns · ${rec['cost']:.4f}",
        "",
        "## Submitted hypotheses",
        "",
        res.get("summary", ""),
    ]
    for i, h in enumerate(res.get("hypotheses", []), 1):
        md += [
            "",
            f"**{i}. {h['hypothesis']}** (confidence {h.get('confidence')})",
            "",
            f"- *Evidence:* {h['evidence']}",
            f"- *Test:* {h['test']}",
        ]
    md += ["", "## Full transcript"]
    for m in msgs:
        if m["role"] == "tool":
            c = m.get("content") or ""
            md += ["", "### [tool result]", "```", c if len(c) < 6000 else c[:6000] + f"\n[... {len(c) - 6000} more chars]", "```"]
            continue
        if m.get("reasoning"):
            md += ["", f"### [{m['role']} reasoning]", m["reasoning"].strip()]
        if (m.get("content") or "").strip():
            md += ["", f"### [{m['role']}]", m["content"].strip()]
        for tc in m.get("tool_calls") or []:
            f = tc["function"]
            try:
                a = json.dumps(json.loads(f.get("arguments") or "{}"), indent=1)
            except json.JSONDecodeError:
                a = f.get("arguments")
            md += ["", f"### [tool call: {f['name']}]", "```json", a, "```"]
    return "\n".join(md)


def named():
    import importlib
    import os
    import sys

    sys.path.insert(0, str(ROOT))
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        sc = importlib.reload(importlib.import_module("score"))
        return {
            "dirs": sc.DIRS,
            "known": {e: {k: v[1] for k, v in ks.items()} for e, ks in sc.KNOWN.items()},
            "scores": {d: sc.score(d) for d in sc.DIRS},
        }
    finally:
        os.chdir(cwd)


_fs_cache = {}


def fork_scores():
    import importlib
    import os
    import sys
    import time

    if _fs_cache.get("t", 0) > time.time() - 120:
        return _fs_cache["v"]
    sys.path.insert(0, str(ROOT))
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        sf = importlib.import_module("score_forks")
        out = {}
        for d, lab in FORKS.items():
            if (ROOT / d).exists():
                rows = sf.load(d)
                out[d] = {
                    "label": lab,
                    "all": sf.score(rows),
                    "min8": sf.score(rows, min_n=8),
                    "pooled": sf.score(rows, pooled=True),
                    "single": sf.score(rows, single=True),
                    "single_min8": sf.score(rows, min_n=8, single=True),
                }
    finally:
        os.chdir(cwd)
    _fs_cache.update(t=time.time(), v=out)
    return out


_js_cache = {}


def judge_scores():
    import importlib
    import os
    import sys
    import time

    if _js_cache.get("t", 0) > time.time() - 120:
        return _js_cache["v"]
    sys.path.insert(0, str(ROOT))
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        sj = importlib.import_module("score_judge")
        out = {}
        for d, lab in JUDGE.items():
            if (ROOT / d).exists():
                rows = sj.load(d)
                out[d] = {"label": lab, "all": sj.score(rows), "min8": sj.score(rows, min_n=8), "per_edit": sj.per_edit(rows)}
    finally:
        os.chdir(cwd)
    _js_cache.update(t=time.time(), v=out)
    return out


def preds(view, arm):
    out = []
    for d, lab in FORKS.items():
        for p in sorted((ROOT / d).glob(f"{view}__*/*__{glob_escape(arm)}.json")):
            if p.name.split("__", 1)[1] != f"{arm}.json":
                continue
            r = json.loads(p.read_text())
            out.append({"cond": lab, "inv": Path(r["inv"]).stem, "fmt": r["fmt"], "parsed": r["parsed"], "path": str(p.relative_to(ROOT))})
    return out


def glob_escape(s):
    return "".join(f"[{c}]" if c in "*?[" else c for c in s)


def render_fork(p):
    r = json.loads(p.read_text())
    return "\n".join(
        [
            f"# Fork: {r['fmt']} on edit `{r['arm']}`",
            f"Investigation {r['inv']} · {r['model']} · context {r['context']}",
            (
                f"Observed: baseline {r['base_k']}/{r['base_n']} → edit {r['arm_k']}/{r['arm_n']} ({r['delta_pp']:+.0f} pp)"
                f" · parsed answer {r['parsed']}"
            ),
            "",
            "## Question",
            "",
            r["question"],
            "",
            "## Reasoning",
            "",
            (r.get("reasoning") or "").strip(),
            "",
            "## Reply",
            "",
            (r.get("reply") or "").strip(),
        ]
    )


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="text/plain; charset=utf-8", code=200):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self.send(INDEX, "text/html; charset=utf-8")
        if u.path == "/api/index":
            return self.send(json.dumps(index()), "application/json")
        if u.path == "/api/named":
            return self.send(json.dumps(named()), "application/json")
        if u.path == "/api/fork_scores":
            return self.send(json.dumps(fork_scores()), "application/json")
        if u.path == "/api/preds":
            v, a = q.get("view", [""])[0], q.get("arm", [""])[0]
            js = {d: {"label": o["label"], "kn": o["per_edit"].get(f"{v}|{a}")} for d, o in judge_scores().items()}
            return self.send(json.dumps({"forks": preds(v, a), "judge": js}), "application/json")
        if u.path == "/api/judge_scores":
            return self.send(
                json.dumps({d: {k: o[k] for k in ("label", "all", "min8")} for d, o in judge_scores().items()}), "application/json"
            )
        if u.path == "/api/tree":
            return self.send(json.dumps(tree(q.get("view", [""])[0])), "application/json")
        if u.path == "/api/file":
            rel = q.get("path", [""])[0]
            p = (ROOT / rel).resolve()
            if not any(p.is_relative_to(ROOT / t) for t in TOPS) or not p.is_file():
                return self.send("not found", code=404)
            if p.suffix == ".json" and p.is_relative_to(ROOT / "forks"):
                return self.send(json.dumps({"kind": "md", "text": render_fork(p)}), "application/json")
            if p.suffix == ".json" and p.parent.parent.name == "runs":
                return self.send(json.dumps({"kind": "md", "text": render_run(p)}), "application/json")
            kind = {".md": "md", ".diff": "diff"}.get(p.suffix, "text")
            return self.send(json.dumps({"kind": kind, "text": p.read_text(errors="replace")}), "application/json")
        self.send("not found", code=404)


class V6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8812)
    ap.add_argument("--bind", action="append", required=True)
    a = ap.parse_args()
    servers = [(V6 if ":" in b else ThreadingHTTPServer)((b, a.port), H) for b in a.bind]
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print("serving", a.bind, a.port, flush=True)
    servers[0].serve_forever()
