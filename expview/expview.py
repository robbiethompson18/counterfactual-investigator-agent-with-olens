"""expview: the project's one transcript viewer. This module writes runs, builds static pages and makes links.

A run is a directory (format in expview/CLAUDE.md):

    <run>/run.json          {"title", "description" (markdown), "label_order", "readers", ...}   (optional)
    <run>/items.jsonl       one item per line: id, group, labels, calls (exact model/judge I/O), env, extra, links, tokens
    <run>/index.jsonl       one line per item: its summary and byte span in items.jsonl (the viewer reads items by range)
    <run>/envs/<id>.json    {"files": {"rel/path": "text", ...}}   environment snapshots, content-addressed
    <run>/tokens/<ref>.json {"tokens", "sections", "reads", "legend"}  per-token metamodel reads (O-lens, J-lens, NLA, ...)

Emit from Python (scripts/ style: put expview/ on sys.path):

    sys.path.insert(0, "expview"); from expview import Run, call, link
    run = Run(run_dir / "expview", title="My eval", description="What one item is, what the labels mean.")
    run.add(id="s1/t0", group="s1", labels={"correct": True}, links={"Docent": url},
            calls=[call("subject", model, messages=prompt_msgs, output=reply_msg),
                   call("judge", judge_model, messages=judge_msgs, output=judge_reply, parsed={"correct": True})],
            tokens={"tokens": toks, "sections": [["user", 0, 40], ["reasoning", 40, 900]], "reads": {"812": [...]}})
    print(link(run.path, item="s1/t0", token=812))  # https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/#dataset=...&item=s1/t0&token=812

The viewer (GitHub Pages, BASE_URL; source expview/static/index.html) is a static page that reads these directories from
the public HF dataset once `scripts/runs.py finish` has synced the run. It also opens env runs (envs/<env>/<run>/)
directly, with every run that points at an episode (item.episode / extra.source + extra.episode) stitched in.

CLI:

    python expview/expview.py build RUN [-o page.html]   # one self-contained page (Artifact), token files next to it
    python expview/expview.py check RUN                  # validate the format
    python expview/expview.py index PATH...              # (re)write index.jsonl for every run under PATH whose index is stale
    python expview/expview.py link RUN [--item ID] [--token N] [--call N] [--q REGEX]
    python expview/expview.py ls                         # every run under the roots, with its dataset id

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import sys
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MAX_FILE = 512 * 1024  # bytes kept per environment file
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules"}
BASE_URL = os.environ.get("EXPVIEW_URL", "https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview")


# ---------------------------------------------------------------- where runs live


def roots() -> dict[str, Path]:
    """Dataset roots, alias -> dir. EXPVIEW_ROOTS="alias=path,alias=path" overrides. The alias is the first part of a
    dataset id, so `data/runs/eval_gaming/2026-10-24_glm53/expview/mc` is `runs/eval_gaming/2026-10-24_glm53/mc`.
    runs: experiment runs (synced to HF by scripts/runs.py, which is what the public viewer shows); imports: views derived
    from other datasets (re-made by a script in expview/importers, not synced; viewable only through `build`)."""
    env = os.environ.get("EXPVIEW_ROOTS")
    if env:
        return {a: Path(p).expanduser() for a, p in (x.split("=", 1) for x in env.split(",") if x)}
    return {"runs": REPO / "data/runs", "imports": REPO / "data/expview/imports"}


def is_run(d: Path) -> bool:
    return (d / "items.jsonl").is_file()


def discover(rts: dict[str, Path] | None = None, max_depth: int = 6) -> dict[str, Path]:
    """dataset id -> run dir, for every run dir (a dir with items.jsonl) under the roots."""
    out = {}
    for alias, root in (rts or roots()).items():
        if not root.is_dir():
            continue
        for dirpath, dirnames, _ in os.walk(root, followlinks=True):  # a symlinked run dir counts (depth-limited)
            d = Path(dirpath)
            depth = len(d.relative_to(root).parts)
            if is_run(d):
                try:
                    out[dataset_id(d, {alias: root})] = d
                except ValueError as e:  # one odd dir must not stop the rescan
                    print(f"expview: skipping {d}: {e}", file=sys.stderr)
            keep = [n for n in sorted(dirnames) if n not in SKIP_DIRS | {"envs", "tokens"} and not n.endswith((".new", ".old"))]
            dirnames[:] = [] if depth >= max_depth else keep  # *.new / *.old: an export being swapped in
    return out


def dataset_id(run: str | Path, rts: dict[str, Path] | None = None) -> str:
    """The id the viewer shows a run dir under: <alias>/<path under the root>, with `expview` path parts dropped."""
    p = Path(os.path.abspath(Path(run).expanduser()))  # lexical first, so a symlinked run dir under a root keeps its link path
    for q, r in ((p, lambda r: Path(os.path.abspath(r))), (p.resolve(), Path.resolve)):
        for alias, root in (rts or roots()).items():
            try:
                rel = q.relative_to(r(root))
            except ValueError:
                continue
            return "/".join([alias, *[x for x in rel.parts if x != "expview"]])
    raise ValueError(f"{run} is not under any expview root ({', '.join(f'{a}={r}' for a, r in (rts or roots()).items())})")


def link(
    run: str | Path, item: str | None = None, token: int | None = None, call: int | None = None, q: str | None = None, base: str = BASE_URL
) -> str:
    """Deep link into the viewer: a run (dir or dataset id), an item, a token of its transcript, or one of its calls."""
    d = str(run) if not Path(run).expanduser().exists() else dataset_id(run)
    params = {"dataset": d, "item": item, "token": token, "call": call, "query": q}
    kept = {k: v for k, v in params.items() if v is not None}
    return f"{base}/#" + urllib.parse.urlencode(kept, quote_via=lambda s, *_: urllib.parse.quote(s, safe="/:"))


def episode_link(env: str, run: str, episode: str | None = None, base: str = BASE_URL, **keys) -> str:
    """Deep link to an env run or one of its episodes (envs/<env>/<run>/ on HF), with every attached run's reads and judges;
    keys: view, token, call, reads (the attached dataset whose token reads to show), query."""
    params = {"env": env, "run": run, "episode": episode, **keys}
    kept = {k: v for k, v in params.items() if v is not None}
    return f"{base}/#" + urllib.parse.urlencode(kept, quote_via=lambda s, *_: urllib.parse.quote(s, safe="/:~"))


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
    if content is not None:
        m["content"] = content
    if reasoning:
        m["reasoning"] = reasoning
    if tool_calls:
        m["tool_calls"] = [_tc(t) for t in tool_calls]
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
    if tools:
        c["tools"] = tools
    if params:
        c["params"] = params
    if parsed is not None:
        c["parsed"] = parsed
    c.update({k: v for k, v in extra.items() if v is not None})
    return c


def read(tool, lines, label=None, score=None, score_name=None, inline=None, **extra):
    """One metamodel readout at a token: `lines` is a list of strings or [text, score] pairs (bullets, top-k tokens...).
    inline=True draws its first line in the transcript next to the token."""
    r = {"tool": tool, "lines": [list(x) if isinstance(x, (list, tuple)) else [x] for x in lines]}
    for k, v in dict(label=label, score=score, score_name=score_name, inline=inline, **extra).items():
        if v is not None:
            r[k] = v
    return r


def source_of(it: dict) -> str | None:
    """ENV/RUN:LINE of the rows.jsonl row an item shows, when the item says so (extra.source, or run_dir + row_index); the
    viewer turns it into a Docent link through that run's docent.json."""
    ex = it.get("extra") if isinstance(it.get("extra"), dict) else {}
    m = re.search(r"(?:^|/)([^/:]+/[^/:]+):(\d+)$", str(ex.get("source") or ""))
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    rd, row, sf = ex.get("run_dir"), ex.get("row_index"), ex.get("source_file")
    if isinstance(rd, str) and isinstance(row, int) and sf in (None, "rows.jsonl"):
        return f"{rd.removeprefix('runs/').strip('/')}:{row}"
    return None


def links_of(it: dict) -> dict:
    out = dict(it.get("links") or {})
    ex = it.get("extra") if isinstance(it.get("extra"), dict) else {}
    for k in ("docent", "docent_url"):
        if isinstance(ex.get(k), str) and ex[k].startswith("http"):
            out.setdefault("Docent", ex[k])
    return out


def episode_of(it: dict, info: dict | None = None) -> str | None:
    """The env episode an item shows or reads, as `envs/<env>/<run>/<episode id>`: the item's own `episode`, else
    extra.source (or extra.env_run) = envs/<env>/<run> + extra.episode, else run.json `episodes_from` = envs/<env>/<run>
    with the item id as the episode id. The viewer opens that episode with this item's reads, judge calls and labels."""
    ep = it.get("episode")
    if isinstance(ep, str) and ep.startswith("envs/"):
        return ep
    ex = it.get("extra") if isinstance(it.get("extra"), dict) else {}
    src = next((ex[k] for k in ("source", "env_run") if isinstance(ex.get(k), str) and ex[k].startswith("envs/")), None)
    if src and ex.get("episode"):
        return f"{src.rstrip('/')}/{ex['episode']}"
    frm = (info or {}).get("episodes_from")
    if isinstance(frm, str) and frm.startswith("envs/"):
        return f"{frm.rstrip('/')}/{it['id']}"
    return None


def summary(it: dict, off: int, n: int, info: dict | None = None) -> dict:
    """An item's index.jsonl line: what the run page and search need, plus the byte span of its line in items.jsonl."""
    s = {"id": str(it["id"]), "off": off, "len": n, "ep": episode_of(it, info)}
    for k in ("group", "labels", "tokens", "base"):
        if it.get(k) is not None:
            s[k] = it[k]
    s["links"] = links_of(it)
    s["src"] = source_of(it)
    s["calls"] = [{"name": c.get("name"), "model": c.get("model"), "n": len(c.get("messages") or [])} for c in it.get("calls") or []]
    return {k: v for k, v in s.items() if v is not None}


def run_info(run: Path) -> dict:
    p = Path(run) / "run.json"
    return json.loads(p.read_text()) if p.exists() else {}


def episode_line(s: dict) -> str:
    """An episodes.jsonl line: which env episode an item belongs to and where the item is, so the viewer can stitch this
    run's reads and judge calls into that episode without loading index.jsonl."""
    return json.dumps({"ep": s["ep"], "id": s["id"], "off": s["off"], "len": s["len"], "tokens": bool(s.get("tokens"))}) + "\n"


def write_index(run: Path) -> int:
    """(Re)write index.jsonl (and episodes.jsonl, for items that point at env episodes) from items.jsonl; returns the
    number of items."""
    run = Path(run)
    info = run_info(run)
    n, off, eps = 0, 0, 0
    with open(run / "items.jsonl", "rb") as f, open(run / "index.jsonl.tmp", "w") as out, open(run / "episodes.jsonl.tmp", "w") as epf:
        for raw in f:
            start, off = off, off + len(raw)
            if raw.strip():
                s = summary(json.loads(raw), start, len(raw), info)
                out.write(json.dumps(s) + "\n")
                if s.get("ep"):
                    epf.write(episode_line(s))
                    eps += 1
                n += 1
    os.replace(run / "index.jsonl.tmp", run / "index.jsonl")
    if eps:
        os.replace(run / "episodes.jsonl.tmp", run / "episodes.jsonl")
    else:
        (run / "episodes.jsonl.tmp").unlink()
        (run / "episodes.jsonl").unlink(missing_ok=True)
    return n


def index_fresh(run: Path) -> bool:
    """index.jsonl exists and covers items.jsonl exactly (its last span ends at the file's end, same number of lines)."""
    run = Path(run)
    ix, items = run / "index.jsonl", run / "items.jsonl"
    if not ix.exists():
        return items.stat().st_size == 0
    with open(ix, "rb") as f:
        lines = [x for x in f if x.strip()]
    last = json.loads(lines[-1]) if lines else {"off": 0, "len": 0}
    with open(items, "rb") as f:
        n_items = sum(1 for x in f if x.strip())
    return last["off"] + last["len"] == items.stat().st_size and len(lines) == n_items


class Run:
    def __init__(self, path, title=None, description="", **meta):
        """meta keys go into run.json: notes (markdown), label_order, tokens_section / tokens_button (nav names),
        readers ({tool: {"about": markdown, "color": "#hex"}}), source (where the data came from)."""
        self.path = Path(path)
        (self.path / "envs").mkdir(parents=True, exist_ok=True)
        rj = self.path / "run.json"
        info = json.loads(rj.read_text()) if rj.exists() else {}
        info.update({k: v for k, v in dict(title=title, description=description, **meta).items() if v})
        info.setdefault("title", self.path.name)
        rj.write_text(json.dumps(info, indent=1) + "\n")
        self._info = info
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
        """Store a token-reads object (see CLAUDE.md) under tokens/, named by the item id so a re-export (e.g. as reads land)
        overwrites the same file; returns the reference an item carries."""
        (self.path / "tokens").mkdir(exist_ok=True)
        tid = hashlib.sha1(str(key).encode()).hexdigest()[:16]
        (self.path / "tokens" / f"{tid}.json").write_text(json.dumps(obj, separators=(",", ":")))
        return {"ref": tid, "n": len(obj["tokens"]), "reads": len(obj.get("reads", {}))}

    def add(
        self,
        id,
        calls,
        group=None,
        labels=None,
        env=None,
        env_dir=None,
        env_files=None,
        extra=None,
        tokens=None,
        links=None,
        base=None,
        notes=None,
        episode=None,
        mode="a",
    ):
        """Append one item. links: {"Docent": url, ...} shown as buttons on the item.
        base=(dataset id, item id) makes this an overlay on that item (CLAUDE.md "Overlays"): it shows the base item's transcript,
        reads, env and calls, plus this item's calls (in the side panel), labels, notes [{"pos": int | None, "text": md}] and
        tokens={"reads": {pos: [read, ...]}} merged into the base transcript.
        episode="envs/<env>/<run>/<episode id>" attaches this item (its reads, judge calls, labels) to that env episode, so
        the episode's page shows them (CLAUDE.md "Attaching to env episodes"); extra.source + extra.episode works too."""
        if env is None and (env_dir is not None or env_files):
            env = self.env(files=env_files, dir=env_dir)
        item = {"id": str(id), "group": group, "labels": labels or {}, "env": env, "calls": calls, "extra": extra, "links": links}
        item["episode"] = episode
        if base is not None:
            item["base"] = {"d": str(base[0]), "i": str(base[1])} if isinstance(base, (tuple, list)) else base
            item["notes"] = notes or None
            item["reads"] = (tokens or {}).get("reads") or None
        elif tokens is not None:
            item["tokens"] = tokens if "ref" in tokens else self.tokens(tokens, id)
        line = (json.dumps({k: v for k, v in item.items() if v is not None}) + "\n").encode()
        with open(self.path / "items.jsonl", mode + "b") as f:
            off = f.tell()
            f.write(line)
        s = summary(item, off, len(line), self._info)
        with open(self.path / "index.jsonl", mode) as f:  # the viewer reads items by byte range, so the index goes with them
            f.write(json.dumps(s) + "\n")
        if s.get("ep"):
            with open(self.path / "episodes.jsonl", mode) as f:
                f.write(episode_line(s))

    def reset(self):
        """Clear items (keeps env snapshots, which are content-addressed)."""
        for name in ("items.jsonl", "index.jsonl"):
            (self.path / name).write_text("")
        (self.path / "episodes.jsonl").unlink(missing_ok=True)


# ---------------------------------------------------------------- reading


def load_tokens(run: Path, ref: str) -> dict:
    p = Path(run) / "tokens" / f"{ref}.json"
    if p.exists():
        return json.loads(p.read_bytes())
    gz = Path(run) / "tokens" / f"{ref}.gz.txt"  # a built page's copy
    return json.loads(gzip.decompress(base64.b64decode(gz.read_text())))


def points_at(url, d: str, i: str) -> bool:
    """A viewer link (any host, or a bare #hash) to item i of dataset d: on an overlay it would open a second copy of its page."""
    h = urllib.parse.parse_qs(str(url).partition("#")[2])
    return h.get("d") == [d] and h.get("i") == [str(i)]


def merge_overlay(ov: dict, base: dict, base_d: str) -> dict:
    """An overlay item over its base item, as the viewer shows it: the base's transcript / tokens (with the dataset they live
    in), env and calls, then the overlay's calls (flagged "overlay", shown in the side panel), labels, links, notes and reads."""
    m = dict(base)
    m.update(id=ov["id"], group=ov.get("group", base.get("group")), extra=ov.get("extra", base.get("extra")))
    m["base"] = {"d": base_d, "i": base["id"]}
    m["labels"] = {**(base.get("labels") or {}), **(ov.get("labels") or {})}
    m["links"] = {k: v for k, v in {**(base.get("links") or {}), **(ov.get("links") or {})}.items() if not points_at(v, base_d, base["id"])}
    m["calls"] = list(base.get("calls") or []) + [{**c, "overlay": True} for c in ov.get("calls") or []]
    if m.get("tokens"):
        m["tokens"] = {**m["tokens"], "d": base_d}
    m["env_d"] = base_d
    if not base.get("env") and ov.get("env"):  # a base without files (e.g. Celeste's imports) shows the overlay's
        m["env"], m["env_d"] = ov["env"], None
    for k in ("reads", "notes"):
        if ov.get(k):
            m["extra_reads" if k == "reads" else k] = ov[k]
    return m


def find_item(d: str, i: str) -> tuple[dict, Path] | None:
    """Item `i` of dataset `d` and its run dir, by scanning items.jsonl."""
    run = discover().get(d)
    if run:
        for line in (run / "items.jsonl").read_text().splitlines():
            if line.strip() and f'"{i}"' in line:
                it = json.loads(line)
                if str(it.get("id")) == str(i):
                    return it, run
    return None


_REFS: dict[str, dict[str, str]] = {}


def item_tokens(d: str, i: str) -> dict | None:
    """The tokens object (transcript + metamodel reads) of item `i` of dataset `d`, or None. For a run of judge calls over
    transcripts another dataset already shows (e.g. Celeste's imports): pass it as run.add(tokens=...) so each judge item
    opens on the transcript it judged, reads inline, judge in the side panel."""
    if d not in _REFS:
        run = discover().get(d)
        _REFS[d] = {"": str(run)} if run else {}
        if run:
            for line in (run / "items.jsonl").read_text().splitlines():
                it = json.loads(line) if line.strip() else {}
                if (it.get("tokens") or {}).get("ref"):
                    _REFS[d][str(it["id"])] = it["tokens"]["ref"]
    ref = _REFS[d].get(str(i))
    return load_tokens(Path(_REFS[d][""]), ref) if ref else None


def resolve_env(run: Path, e) -> tuple[str | None, dict | None]:
    """An item's env field -> (id, files): an envs/<id>.json id, inline {"files": {...}}, or {"dir": path} read now."""
    if not e:
        return None, None
    if isinstance(e, dict):
        files = read_env_dir(e["dir"]) if "dir" in e else e.get("files", {})
        return hashlib.sha1(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12], files
    return e, json.loads((Path(run) / "envs" / f"{e}.json").read_text())["files"]


def iter_items(run: Path):
    with open(Path(run) / "items.jsonl", "rb") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_run(path):
    path = Path(path)
    info = json.loads((path / "run.json").read_text()) if (path / "run.json").exists() else {"title": path.name}
    items, envs = list(iter_items(path)), {}
    for it in items:
        eid, files = resolve_env(path, it.get("env"))
        if eid:
            envs.setdefault(eid, files)
            it["env"] = eid
    return info, items, envs


def check(path) -> list[str]:
    """Format problems in a run dir (empty list = fine)."""
    path, probs, ids = Path(path), [], set()
    if not is_run(path):
        return [f"{path}: no items.jsonl"]
    if not index_fresh(path):
        probs.append("index.jsonl missing or stale (the public viewer needs it): python expview/expview.py index RUN")
    info, envs_named = run_info(path), set()
    for n, it in enumerate(iter_items(path), 1):
        where = f"items.jsonl:{n} ({it.get('id')})"
        if "id" not in it:
            probs.append(f"{where}: no id")
        elif it["id"] in ids:
            probs.append(f"{where}: duplicate id")
        ids.add(it.get("id"))
        ep = episode_of(it, info) if "id" in it else None
        if ep:
            if len(ep.split("/")) != 4:
                probs.append(f"{where}: episode {ep} is not envs/<env>/<run>/<episode id>")
            envs_named.add(ep.split("/")[1])
        for k, v in (it.get("labels") or {}).items():
            if isinstance(v, (dict, list)):
                probs.append(f"{where}: label {k} is not a scalar")
        for j, c in enumerate(it.get("calls") or []):
            if not isinstance(c.get("messages"), list):
                probs.append(f"{where}: call {j} has no messages list")
        b = it.get("base")
        if b and not (isinstance(b, dict) and find_item(b.get("d"), b.get("i"))):
            probs.append(f"{where}: base {b} not found under the roots")
        e = it.get("env")
        if isinstance(e, str) and not (path / "envs" / f"{e}.json").exists():
            probs.append(f"{where}: envs/{e}.json missing")
        tk = it.get("tokens")
        if tk:
            try:
                t = load_tokens(path, tk["ref"])
            except (OSError, KeyError, ValueError) as err:
                probs.append(f"{where}: tokens {tk.get('ref')}: {err}")
                continue
            n_tok = len(t.get("tokens", []))
            for s in t.get("sections") or []:
                if not (len(s) == 3 and 0 <= s[1] <= s[2] <= n_tok):
                    probs.append(f"{where}: bad section {s}")
            for p, rs in (t.get("reads") or {}).items():
                if not (str(p).isdigit() and int(p) < n_tok):
                    probs.append(f"{where}: read position {p} outside 0..{n_tok - 1}")
                for r in rs:
                    if "tool" not in r or not isinstance(r.get("lines"), list) or any(not isinstance(x, list) for x in r["lines"]):
                        probs.append(f"{where}: read at {p} needs tool and lines [[text, score?], ...]")
                        break
    try:
        exp = dataset_id(path).split("/")[1]
    except (ValueError, IndexError):
        exp = None
    for e in sorted(envs_named - {exp}):
        probs.append(f"items point at envs/{e} episodes, but the viewer only stitches runs under runs/{e}/ into them (this is {exp})")
    return probs


# ---------------------------------------------------------------- building a static page


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


def build(path, out=None, ids=None):
    """One self-contained HTML page for a run (the same UI as the public page, data embedded); token files go next to it as
    tokens/<ref>.gz.txt and are fetched on demand. ids: only these items."""
    info, items, envs = load_run(path)
    if ids:
        keep = set(ids)
        items = [it for it in items if it["id"] in keep]
        envs = {k: v for k, v in envs.items() if any(it.get("env") == k for it in items)}
    src = {}  # token ref -> run dir it lives in (an overlay's transcript is its base's)
    for n, it in enumerate(items):
        b = it.get("base")
        hit = find_item(b["d"], b["i"]) if b else None
        if hit:
            base, brun = hit
            eid, files = resolve_env(brun, base.get("env"))
            if eid:
                envs.setdefault(eid, files)
                base = {**base, "env": eid}
            items[n] = it = merge_overlay(it, base, b["d"])
            if it.get("tokens"):
                src[it["tokens"]["ref"]] = brun
    intern = Interner()
    data = {"id": Path(path).name, "run": info, "envs": intern(envs), "items": intern(items)}
    data["blobs"] = intern.tab
    payload = base64.b64encode(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), 9)).decode()
    page = (HERE / "static/index.html").read_text()
    title = (info.get("title") or "Experiment run").replace("<", "&lt;")
    page = page.replace("{{TITLE}}", title).replace("{{DATA}}", payload)
    out = Path(out or Path(path) / "index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    refs = sorted({it["tokens"]["ref"] for it in items if it.get("tokens")})
    if refs:  # token reads ship next to the page as base64(gzip) text, fetched when an item's transcript opens
        (out.parent / "tokens").mkdir(exist_ok=True)
        for r in refs:
            raw = json.dumps(load_tokens(src.get(r, Path(path)), r), separators=(",", ":")).encode()
            (out.parent / "tokens" / f"{r}.gz.txt").write_text(base64.b64encode(gzip.compress(raw, 9)).decode())
        tot = sum((out.parent / "tokens" / f"{r}.gz.txt").stat().st_size for r in refs)
        print(f"{len(refs)} token files in {out.parent / 'tokens'} ({tot / 1e6:.1f} MB); publish them with the page", file=sys.stderr)
    print(f"{out}: {len(items)} items, {len(envs)} environments, {len(intern.tab)} long strings, {len(page) / 1e6:.2f} MB", file=sys.stderr)
    if len(page) > 15e6:
        print("warning: page is over 15 MB; Artifacts cap at 16 MB (build fewer items with --ids)", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="render a run directory to one HTML page")
    b.add_argument("run")
    b.add_argument("-o", "--out")
    b.add_argument("--ids", help="file with one item id per line: build only those")
    c = sub.add_parser("check", help="validate a run directory")
    c.add_argument("run", nargs="+")
    ix = sub.add_parser("index", help="write index.jsonl for every run dir under PATH whose index is missing or stale")
    ix.add_argument("path", nargs="+")
    ix.add_argument("--force", action="store_true", help="rewrite fresh ones too")
    lk = sub.add_parser("link", help="print a deep link into the viewer")
    lk.add_argument("run", help="run dir or dataset id")
    lk.add_argument("--item")
    lk.add_argument("--token", type=int, help="token position in the item's transcript")
    lk.add_argument("--call", type=int, help="call index (0-based)")
    lk.add_argument("--q", help="regex to highlight")
    sub.add_parser("ls", help="list every run under the roots with its dataset id")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.run, a.out, Path(a.ids).read_text().split() if a.ids else None)
    elif a.cmd == "check":
        bad = 0
        for r in a.run:
            probs = check(r)
            bad += bool(probs)
            print(f"{r}: {'ok' if not probs else f'{len(probs)} problems'}", *probs[:50], sep="\n  ")
        sys.exit(1 if bad else 0)
    elif a.cmd == "index":
        for root in a.path:
            for d in sorted({p.parent for p in Path(root).rglob("items.jsonl")} | ({Path(root)} if is_run(Path(root)) else set())):
                if a.force or not index_fresh(d):
                    print(f"{d}: {write_index(d)} items indexed")
    elif a.cmd == "link":
        print(link(a.run, a.item, a.token, a.call, a.q))
    elif a.cmd == "ls":
        for d, p in sorted(discover().items()):
            print(f"{d}\t{p}")


if __name__ == "__main__":
    main()
