"""expview: one standard viewer for experiment runs.

A run is a directory:

    <run>/run.json        {"title", "description" (markdown), ...}            (optional)
    <run>/items.jsonl     one item per line (format in README.md)
    <run>/envs/<id>.json  {"files": {"rel/path": "text", ...}}                (environment snapshots)

Emit from Python:

    from expview import Run, call
    run = Run("runs/my_eval", title="My eval", description="...")
    run.add(id="s1/t0", group="s1", labels={"correct": True},
            env_dir="settings/s1",                       # snapshotted into envs/, deduplicated
            calls=[call("subject", model, messages=prompt_msgs, output=reply_msg)])

Build a single self-contained HTML page (publishable as an Artifact, or open locally):

    python expview.py build <run> [-o page.html]

Only the standard library is used.
"""
import argparse, base64, gzip, hashlib, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_FILE = 512 * 1024  # bytes kept per environment file
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules"}


# ---------------------------------------------------------------- emitting

def read_env_dir(d, skip=()):
    """Snapshot every file under d as {relpath: text}. Binary files and oversize files are stubbed, never dropped."""
    d = Path(d)
    files = {}
    for p in sorted(d.rglob("*")):
        if not p.is_file() or SKIP_DIRS & set(p.relative_to(d).parts):
            continue
        rel = str(p.relative_to(d))
        if rel in skip:
            continue
        raw = p.read_bytes()
        try:
            text = raw[:MAX_FILE].decode("utf-8")
        except UnicodeDecodeError:
            files[rel] = f"[binary file, {len(raw)} bytes]"
            continue
        files[rel] = text if len(raw) <= MAX_FILE else text + f"\n[... truncated, {len(raw) - MAX_FILE} more bytes]"
    return files


def message(role, content=None, reasoning=None, tool_calls=None, **extra):
    """An OpenAI-style chat message. tool_calls: [{"name", "arguments"}] or OpenAI's {"function": {...}} form."""
    m = {"role": role}
    if content is not None: m["content"] = content
    if reasoning: m["reasoning"] = reasoning
    if tool_calls: m["tool_calls"] = [_tc(t) for t in tool_calls]
    m.update({k: v for k, v in extra.items() if v is not None})
    return m


def _tc(t):
    f = t.get("function", t)
    args = f.get("arguments")
    return {"id": t.get("id"), "name": f.get("name"), "arguments": args if isinstance(args, str) else json.dumps(args)}


def normalize_messages(msgs):
    """Accept OpenAI / OpenRouter response messages as-is: reasoning_content -> reasoning, tool_calls flattened."""
    out = []
    for m in msgs or []:
        m = dict(m)
        if "reasoning_content" in m:
            m["reasoning"] = m.pop("reasoning_content") or m.get("reasoning")
        if not m.get("reasoning"):
            m.pop("reasoning", None)
        m.pop("reasoning_details", None)
        if m.get("tool_calls"):
            m["tool_calls"] = [_tc(t) for t in m["tool_calls"]]
        out.append(m)
    return out


def call(name, model, messages, output=None, tools=None, params=None, parsed=None, **extra):
    """One LLM call: `messages` is exactly what was sent; `output` (a message or list of messages) is what came back.
    For a multi-turn agent episode pass the whole conversation as `messages` and leave `output` None:
    every assistant message is shown as model output, everything else as model input."""
    msgs = normalize_messages(messages)
    if output is not None:
        msgs += normalize_messages(output if isinstance(output, list) else [output])
    c = {"name": name, "model": model, "messages": msgs}
    if tools: c["tools"] = tools
    if params: c["params"] = params
    if parsed is not None: c["parsed"] = parsed
    c.update({k: v for k, v in extra.items() if v is not None})
    return c


class Run:
    def __init__(self, path, title=None, description="", **meta):
        self.path = Path(path)
        (self.path / "envs").mkdir(parents=True, exist_ok=True)
        rj = self.path / "run.json"
        info = json.loads(rj.read_text()) if rj.exists() else {}
        info.update({k: v for k, v in dict(title=title, description=description, **meta).items() if v})
        info.setdefault("title", self.path.name)
        rj.write_text(json.dumps(info, indent=1) + "\n")
        self._envs = {}

    def env(self, files=None, dir=None, skip=()):
        """Store an environment snapshot; returns its id (content hash, so identical envs are stored once)."""
        files = dict(files or {})
        if dir is not None:
            files = {**read_env_dir(dir, skip), **files}
        blob = json.dumps({"files": files}, sort_keys=True)
        eid = hashlib.sha1(blob.encode()).hexdigest()[:12]
        if eid not in self._envs:
            p = self.path / "envs" / f"{eid}.json"
            if not p.exists():
                p.write_text(blob)
            self._envs[eid] = True
        return eid

    def tokens(self, obj, key):
        """Store a token-reads object (see README) under tokens/, named by the item id so a re-export (e.g. as reads land)
        overwrites the same published file; returns the reference an item carries."""
        (self.path / "tokens").mkdir(exist_ok=True)
        tid = hashlib.sha1(str(key).encode()).hexdigest()[:16]
        (self.path / "tokens" / f"{tid}.json").write_text(json.dumps(obj, separators=(",", ":")))
        return {"ref": tid, "n": len(obj["tokens"]), "reads": len(obj.get("reads", {}))}

    def add(self, id, calls, group=None, labels=None, env=None, env_dir=None, env_files=None, extra=None, tokens=None, mode="a"):
        if env is None and (env_dir is not None or env_files):
            env = self.env(files=env_files, dir=env_dir)
        item = {"id": str(id), "group": group, "labels": labels or {}, "env": env, "calls": calls}
        if extra is not None: item["extra"] = extra
        if tokens is not None: item["tokens"] = tokens if "ref" in tokens else self.tokens(tokens, id)
        with open(self.path / "items.jsonl", mode) as f:
            f.write(json.dumps({k: v for k, v in item.items() if v is not None}) + "\n")

    def reset(self):
        """Clear items (keeps env snapshots, which are content-addressed)."""
        (self.path / "items.jsonl").write_text("")


# ---------------------------------------------------------------- building

class Interner:
    """Long strings are stored once in a table and referenced as {"$": index}; prompts repeat a lot across items."""
    def __init__(self, min_len=80):
        self.tab, self.idx, self.min_len = [], {}, min_len

    def __call__(self, x):
        if isinstance(x, str):
            if len(x) < self.min_len:
                return x
            if x not in self.idx:
                self.idx[x] = len(self.tab)
                self.tab.append(x)
            return {"$": self.idx[x]}
        if isinstance(x, list):
            return [self(v) for v in x]
        if isinstance(x, dict):
            return {k: self(v) for k, v in x.items()}
        return x


def load_run(path):
    path = Path(path)
    info = json.loads((path / "run.json").read_text()) if (path / "run.json").exists() else {"title": path.name}
    items = [json.loads(l) for l in open(path / "items.jsonl") if l.strip()]
    envs = {}
    for it in items:
        e = it.get("env")
        if isinstance(e, dict):  # inline {"files": {...}} or {"dir": path}
            files = read_env_dir(e["dir"]) if "dir" in e else e.get("files", {})
            eid = hashlib.sha1(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12]
            envs[eid] = files
            it["env"] = eid
        elif e and e not in envs:
            envs[e] = json.loads((path / "envs" / f"{e}.json").read_text())["files"]
    return info, items, envs


def build(path, out=None):
    info, items, envs = load_run(path)
    I = Interner()
    data = {"run": info, "envs": I(envs), "items": I(items)}
    data["blobs"] = I.tab
    payload = base64.b64encode(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), 9)).decode()
    page = (HERE / "template.html").read_text()
    title = (info.get("title") or "Experiment run").replace("<", "&lt;")
    page = page.replace("{{TITLE}}", title).replace("{{DATA}}", payload)
    out = Path(out or Path(path) / "index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    refs = sorted({it["tokens"]["ref"] for it in items if it.get("tokens")})
    if refs:  # token reads ship next to the page as base64(gzip) text, fetched when an item's token view opens
        (out.parent / "tokens").mkdir(exist_ok=True)
        for r in refs:
            raw = (Path(path) / "tokens" / f"{r}.json").read_bytes()
            (out.parent / "tokens" / f"{r}.gz.txt").write_text(base64.b64encode(gzip.compress(raw, 9)).decode())
        tot = sum((out.parent / "tokens" / f"{r}.gz.txt").stat().st_size for r in refs)
        print(f"{len(refs)} token files in {out.parent / 'tokens'} ({tot / 1e6:.1f} MB); publish them with the page", file=sys.stderr)
    print(f"{out}: {len(items)} items, {len(envs)} environments, {len(I.tab)} unique long strings, "
          f"{len(page) / 1e6:.2f} MB", file=sys.stderr)
    if len(page) > 15e6:
        print("warning: page is over 15 MB; Artifacts cap at 16 MB", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="render a run directory to one HTML page")
    b.add_argument("run"); b.add_argument("-o", "--out")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.run, a.out)


if __name__ == "__main__":
    main()
