"""Dump one evalcue env's episodes and a self-contained grader source, as JSON on stdout.

Runs inside a family checkout's venv (cwd = ~/workspace/env-explorer/replication/co/<family>):
    QWEN_VLLM_API_KEY=unused .venv/bin/python <this> <env> <item> <arm> [<arm> ...]
Output: {"episodes": {arm: Episode-as-dict}, "items": env.items(), "grader": "<python source>",
         "prefill": {arm: bool}}   (prefill = the episode has non system/user prompt messages)

The grader source is detect() plus every evalcue helper/constant it reaches (same walk as
baseline-hyp/triples/auditor/extract_grader.py, but nothing truncated and module aliases flattened: `L.final_text` ->
`final_text`, with a module prefix only where two modules define the same name). The entry point is renamed `detect`.
"""

import ast
import inspect
import json
import re
import sys
import textwrap
import types
from dataclasses import asdict

sys.path.insert(0, ".")
import importlib

ENV = sys.argv[1]
mod = importlib.import_module(f"evalcue.envs.{ENV}")


def ours(o) -> bool:
    m = o.__name__ if isinstance(o, types.ModuleType) else getattr(o, "__module__", None)
    return str(m).startswith("evalcue")


def const_src(name, v) -> str:
    if isinstance(v, re.Pattern):
        fl = v.flags & ~re.UNICODE
        return f"{name} = re.compile({v.pattern!r}" + (f", flags={re.RegexFlag(fl)!s}".replace("RegexFlag.", "re.") if fl else "") + ")"
    if isinstance(v, dict) and "variants" in v and "detect" in v:  # dumb_user_exploit scenario: only grader-relevant fields
        keep = {k: v[k] for k in ("name", "detect", "tag", "frac") if k in v}
        keep["variants"] = [{k: x[k] for k in ("product", "defect", "keywords", "truth", "bad", "good") if k in x} for x in v["variants"]]
        return f"{name} = {keep!r}  # scenario: prompt-text fields omitted"
    return f"{name} = {v!r}"


units = {}  # key (module, name) -> {"kind", "src" | "value", "renames": {old_text_pattern: key}}
imports = set()
order = []


def add_func(fn, key):
    if key in units:
        return
    src = textwrap.dedent(inspect.getsource(fn))
    unit = {"kind": "func", "src": src, "attr": {}, "names": {}}
    units[key] = unit
    order.append(key)
    g = vars(sys.modules[fn.__module__])
    cv = dict(zip(fn.__code__.co_freevars, [c.cell_contents for c in fn.__closure__])) if getattr(fn, "__closure__", None) else {}
    local = set(fn.__code__.co_varnames)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base = cv.get(node.value.id, g.get(node.value.id))
            if isinstance(base, types.ModuleType) and ours(base) and hasattr(base, node.attr):
                k = resolve(base.__name__, node.attr, getattr(base, node.attr))
                if k:
                    unit["attr"][f"{node.value.id}.{node.attr}"] = k
        if isinstance(node, ast.Name) and node.id not in local:
            if node.id in cv:
                k = resolve(fn.__module__, node.id, cv[node.id])
            elif node.id in g:
                k = resolve(fn.__module__, node.id, g[node.id])
            else:
                continue
            if k:
                unit["names"][node.id] = k


def resolve(modname, name, v):
    """Register what `name` (as seen from module `modname`) refers to; return its unit key, or None if it's external."""
    if isinstance(v, types.ModuleType):
        if not ours(v):
            imports.add(f"import {v.__name__}" + (f" as {name}" if name != v.__name__ else ""))
        return None
    if inspect.isfunction(v) or inspect.isclass(v):
        if ours(v):
            key = (v.__module__, v.__name__)
            add_func(v, key)
            return key
        imports.add(f"from {v.__module__} import {v.__name__}" + (f" as {name}" if name != v.__name__ else ""))
        return None
    if callable(v):
        return None
    key = (modname, name)
    if key not in units:
        units[key] = {"kind": "const", "value": v}
        order.append(key)
    return key


entry = mod.detect
entry_key = (entry.__module__, entry.__name__)
add_func(entry, entry_key)

# final names: the entry is `detect`; otherwise a unit keeps its name unless another module defines the same name
by_name = {}
for k in order:
    by_name.setdefault(k[1], []).append(k)
final = {}
for k in order:
    final[k] = "detect" if k == entry_key else k[1] if len(by_name[k[1]]) == 1 and k[1] != "detect" else f"{k[0].split('.')[-1]}_{k[1]}"

consts, funcs = [], []
for k in order:
    u = units[k]
    if u["kind"] == "const":
        consts.append(f"# {k[0]}.{k[1]}\n" + const_src(final[k], u["value"]))
        continue
    src = u["src"]
    for pat, tk in u["attr"].items():
        src = re.sub(rf"\b{re.escape(pat)}\b", final[tk], src)
    names = {**u["names"], k[1]: k}
    for old, tk in names.items():
        if final[tk] != old:
            src = re.sub(rf"(?<![\w.]){re.escape(old)}\b(?!\s*=[^=])", final[tk], src)
    funcs.append(f"# {k[0]}.{k[1]}\n{src}")

for name in ("re", "json"):
    imports.add(f"import {name}")
grader = "\n".join(sorted(imports)) + "\n\n" + "\n".join(consts) + "\n\n\n" + "\n\n".join(funcs)

item, arms = sys.argv[2], sys.argv[3:]
eps = {a: asdict(mod.build(a, item)) for a in arms}
# prompt prefix check over ALL items of these arms (the validator assumes prompt = leading system/user messages)
prefill = {a: any(any(m["role"] not in ("system", "user") for m in mod.build(a, it).messages) for it in mod.items()) for a in arms}
json.dump({"episodes": eps, "items": list(mod.items()), "grader": grader, "prefill": prefill}, sys.stdout)
