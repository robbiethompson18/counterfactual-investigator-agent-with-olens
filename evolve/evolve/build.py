"""Build check for a node dir (what an agent's submit runs) and the one-paragraph text card of a node.

check(node_dir, sample_records) -> list of error strings; [] means the node can be run.
"""

import ast
import asyncio
import io
import json
import re
import threading
import tokenize
from pathlib import Path

from . import grading
from .node import ARMS, Arm, Node, changed_chars, unified_diff

EDIT_LIMIT = 400  # max changed characters baseline -> real and baseline -> null
GRADER_MAX_LINES = 15  # non-blank, non-comment lines in grader.py
IMPL_KINDS = ("bash", "script", "record")
STOP_KINDS = ("reply", "tool", "command", "max_turns", "file_exists")


def grader_lines(src: str) -> int:
    """Non-blank lines that are not only a comment (docstrings count: keep them short)."""
    comment_only = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT and not tok.line[: tok.start[1]].strip():
                comment_only.add(tok.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return sum(1 for i, ln in enumerate(src.splitlines(), 1) if ln.strip() and i not in comment_only)


def _check_stop(rule, i, where) -> list[str]:
    if not isinstance(rule, str):
        return [f"{where}: stop[{i}] must be a string, got {rule!r}"]
    kind, sep, arg = rule.partition(":")
    if kind not in STOP_KINDS:
        return [f"{where}: stop rule {rule!r}: unknown kind {kind!r} (use one of {', '.join(STOP_KINDS)})"]
    if kind == "reply":
        return [] if not sep else [f"{where}: stop rule {rule!r}: 'reply' takes no argument"]
    if not arg:
        return [f"{where}: stop rule {rule!r} needs an argument ({kind}:<...>)"]
    if kind == "command":
        try:
            re.compile(arg)
        except re.error as e:
            return [f"{where}: stop rule {rule!r}: bad regex: {e}"]
    if kind == "max_turns" and not (arg.isdigit() and int(arg) > 0):
        return [f"{where}: stop rule {rule!r}: max_turns needs a positive integer"]
    return []


def _check_arm(d: Path, name: str) -> tuple[list[str], Arm | None]:
    errs = []
    if not d.is_dir():
        return [f"{name}/ is missing"], None
    for f in ("tools.json", "episode.json", "meta.json", "prefill.json"):
        if (d / f).exists():
            try:
                json.loads((d / f).read_text())
            except json.JSONDecodeError as e:
                errs.append(f"{name}/{f}: invalid JSON: {e}")
    if errs:
        return errs, None
    try:
        arm = Arm.load(d)
    except Exception as e:  # noqa: BLE001
        return [f"{name}/: does not load: {type(e).__name__}: {e}"], None
    if not any(m["role"] == "user" for m in arm.messages):
        errs.append(f"{name}/: no user turn (need user_turn_1.md)")
    for i, m in enumerate(arm.messages):
        if not isinstance(m, dict) or "role" not in m:
            errs.append(f"{name}/: message {i} is not a {{role, content}} object")
        elif m["role"] in ("system", "user") and not str(m.get("content") or "").strip():
            errs.append(
                f"{name}/: {m['role']} message {i} is empty"
                + (" (delete system_prompt.md for no system prompt)" if m["role"] == "system" else "")
            )
    if not isinstance(arm.tools, list):
        errs.append(f"{name}/tools.json must be a JSON list")
        arm.tools = []
    seen = set()
    for i, t in enumerate(arm.tools):
        w = f"{name}/tools.json[{i}]"
        if not isinstance(t, dict):
            errs.append(f"{w}: must be an object")
            continue
        miss = [k for k in ("name", "description", "parameters", "impl") if k not in t]
        if miss:
            errs.append(f"{w}: missing {', '.join(miss)}")
            continue
        w = f"{name}/tools.json tool {t['name']!r}"
        if not isinstance(t["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", t["name"]):
            errs.append(f"{w}: name must match [A-Za-z0-9_-]{{1,64}}")
        if t["name"] in seen:
            errs.append(f"{w}: duplicate tool name")
        seen.add(t["name"])
        if not isinstance(t["parameters"], dict) or t["parameters"].get("type") != "object":
            errs.append(f'{w}: parameters must be a JSON Schema object with "type": "object"')
        impl = t["impl"]
        kind = impl.get("kind") if isinstance(impl, dict) else None
        if kind not in IMPL_KINDS:
            errs.append(f"{w}: impl.kind must be one of {', '.join(IMPL_KINDS)}, got {kind!r}")
        elif kind == "script":
            p = impl.get("path")
            if not p or p not in arm.files:
                errs.append(f"{w}: script path {p!r} not found in {name}/files/")
            elif not arm.files[p].startswith("#!"):
                errs.append(f"{w}: script {name}/files/{p} must start with a #! line")
            elif p in arm.episode.get("readonly", []):
                errs.append(f"{w}: script {p} is listed as readonly, which removes its execute bit")
        elif kind == "bash" and "command" not in (t["parameters"].get("properties") or {}):
            errs.append(f"{w}: bash-kind tools need a 'command' parameter")
    ep = arm.episode
    stop = ep.get("stop", [])
    if not isinstance(stop, list):
        errs.append(f"{name}/episode.json: stop must be a list")
    else:
        for i, r in enumerate(stop):
            errs += _check_stop(r, i, f"{name}/episode.json")
            if isinstance(r, str) and r.startswith("tool:") and r[5:] not in seen and arm.tools:
                errs.append(f"{name}/episode.json: stop rule {r!r} names no tool in {name}/tools.json")
    ro = ep.get("readonly", [])
    if not isinstance(ro, list):
        errs.append(f"{name}/episode.json: readonly must be a list")
    else:
        errs += [f"{name}/episode.json: readonly path {p!r} not in {name}/files/" for p in ro if p not in arm.files]
    if not isinstance(arm.meta, dict):
        errs.append(f"{name}/meta.json must be a JSON object")
    return errs, arm


def _strings(tree) -> set[str]:
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _empty_record(meta) -> dict:
    return {
        "final_response": "",
        "turns": [],
        "tool_calls": [],
        "transcript": [],
        "files_after": {},
        "finish": "reply",
        "n_turns": 0,
        "readonly_tampered": [],
        "meta": meta,
    }


async def check_async(node_dir, sample_records: list[dict] | None = None) -> list[str]:
    d = Path(node_dir)
    errs = []
    for f in ("node.json", "behaviour.txt", "grader.py"):
        if not (d / f).is_file():
            errs.append(f"{f} is missing")
    info = {}
    if (d / "node.json").is_file():
        try:
            info = json.loads((d / "node.json").read_text())
            if not isinstance(info, dict):
                errs.append("node.json must be a JSON object")
                info = {}
        except json.JSONDecodeError as e:
            errs.append(f"node.json: invalid JSON: {e}")
    if (d / "behaviour.txt").is_file() and not (d / "behaviour.txt").read_text().strip():
        errs.append("behaviour.txt is empty (one sentence: what the grader counts)")

    arms = {}
    for a in ARMS:
        e, arm = _check_arm(d / a, a)
        errs += e
        if arm is not None:
            arms[a] = arm

    if all((d / a).is_dir() for a in ARMS) and not (info.get("edit_exempt") or info.get("seed")):
        for a in ("real", "null"):
            n = changed_chars(d / "baseline", d / a)
            limit = info.get("edit_limit", EDIT_LIMIT)
            if n > limit:
                errs.append(f"{a}/ differs from baseline/ by {n} changed characters (limit {limit})")
            elif n == 0:
                errs.append(f"{a}/ is identical to baseline/ (the edit is empty)")

    src = (d / "grader.py").read_text() if (d / "grader.py").is_file() else None
    grader_ok = False
    if src is not None:
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            errs.append(f"grader.py: syntax error line {e.lineno}: {e.msg}")
            tree = None
        if tree is not None:
            if not any(isinstance(n, ast.FunctionDef) and n.name == "grade" for n in tree.body):
                errs.append("grader.py must define a top-level function grade(record)")
            else:
                grader_ok = True
            n = grader_lines(src)
            if n > GRADER_MAX_LINES and not info.get("grader_exempt"):
                errs.append(f"grader.py has {n} non-blank non-comment lines (limit {GRADER_MAX_LINES})")
            names = {a: {t.get("name") for t in arm.tools if isinstance(t, dict)} for a, arm in arms.items()}
            all_tools = set().union(*names.values()) if names else set()
            for s in sorted(_strings(tree) & all_tools):
                missing = [a for a in ARMS if a in names and s not in names[a]]
                if missing:
                    errs.append(f"grader.py refers to tool {s!r}, which is missing from {', '.join(m + '/tools.json' for m in missing)}")

    if grader_ok and not errs:
        # every arm's empty run (a target that just replies "") must grade without an exception; plus samples
        recs = [_empty_record(arms[a].meta) for a in ARMS]
        for r in sample_records or []:
            arm = arms.get(r.get("_arm"))
            recs.append({**r, "meta": arm.meta} if arm else r)
        recs = [{k: v for k, v in r.items() if not k.startswith("_")} for r in recs]
        try:
            verdicts = await grading.grade_records(src, recs)
        except Exception as e:  # noqa: BLE001
            verdicts = [{"error": f"could not run grader: {e}"}]
        bad = [(i, v["error"]) for i, v in enumerate(verdicts) if isinstance(v, dict) and "error" in v]
        for i, e in bad[:3]:
            what = f"an empty run of {ARMS[i]}/ (final_response '', no turns, no tool calls)" if i < 3 else f"sample record {i - 3}"
            errs.append(f"grader.py failed on {what}: {e.strip()[-500:]}")
        if len(bad) > 3:
            errs.append(f"... grader.py failed on {len(bad)} records in total")
    return errs


def check(node_dir, sample_records: list[dict] | None = None) -> list[str]:
    """Sync wrapper (safe to call with or without a running event loop). Records may carry "_arm" to get that
    arm's current meta.json substituted in."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(check_async(node_dir, sample_records))
    out = []
    t = threading.Thread(target=lambda: out.append(asyncio.run(check_async(node_dir, sample_records))))
    t.start()
    t.join()
    return out[0]


def _clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n].rstrip() + " ..."


def card(node_dir, diff_chars=700) -> str:
    """A short text card: behaviour, scenario summary, tools, and both diffs (truncated)."""
    d = Path(node_dir)
    node = Node(d)
    info = node.info
    beh = (d / "behaviour.txt").read_text().strip() if (d / "behaviour.txt").exists() else "(none)"
    try:
        arm = node.arm("baseline")
        sys_p = next((m["content"] for m in arm.messages if m["role"] == "system"), "")
        user = next((m["content"] for m in arm.messages if m["role"] == "user"), "")
        tools = ", ".join(t.get("name", "?") for t in arm.tools) or "none (plain chat)"
        files = ", ".join(list(arm.files)[:8]) + (" ..." if len(arm.files) > 8 else "")
    except Exception:  # noqa: BLE001
        sys_p = user = files = ""
        tools = "?"
    lines = [f"[{node.id}]" + (f" {info['title']}" if info.get("title") else ""), f"Behaviour: {beh}"]
    if sys_p:
        lines.append(f"System prompt: {_clip(sys_p, 300)}")
    lines.append(f"User: {_clip(user, 300)}")
    lines.append(f"Tools: {tools}" + (f"; files: {files}" if files else ""))
    for a in ("real", "null"):
        try:
            diff = unified_diff(d / "baseline", d / a)
        except Exception:  # noqa: BLE001
            diff = "(unavailable)"
        lines.append(f"{a.upper()} edit:\n{_clip(diff, diff_chars) or '(none)'}")
    return "\n".join(lines)
