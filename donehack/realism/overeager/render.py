# ruff: noqa: E501
"""Write every task × variant (item 0) to envs/<task>/<variant>/ and build one grid page.

  uv run python realism/overeager/render.py --out <html> [--runs <runs/overeager/NAME>]
Each env folder: system_prompt.md, user_turn_1.md, tools.json, files/ (the working directory), app_data.json (what the
app tools serve), setup.sh. With --runs, the page also shows per-cell outcome counts and links one transcript per cell.
"""

import argparse
import html
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from harness_variants import ACTIVE
from oe_run import TASKS, build_spec, load

ENVS = HERE / "envs"


def write_env(task, variant, i=0):
    spec, _t, inst = build_spec(task, variant, i)
    d = ENVS / task / variant
    if d.exists():
        shutil.rmtree(d)
    (d / "files").mkdir(parents=True)
    (d / "system_prompt.md").write_text(spec.system)
    (d / "user_turn_1.md").write_text(spec.user)
    (d / "tools.json").write_text(json.dumps(spec.tools, indent=1))
    (d / "setup.sh").write_text("\n".join(spec.setup) + "\n" if spec.setup else "")
    if spec.meta.get("prefill"):  # eval-cue arms: the prefilled first turn (the output is the sandbox's real `ls -la`)
        (d / "prefilled_turn.md").write_text(
            f"assistant -> execute_command: {spec.meta['prefill']}\ntool -> (real output of `{spec.meta['prefill']}` in the workdir, which lists .task.json)\n"
        )
    (d / "README.md").write_text(f"{task} / {variant}: workdir {spec.workdir}\n")
    for k, v in spec.files.items():
        p = d / "files" / k
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(v) if isinstance(v, bytes) else p.write_text(v)
    app = {k: v for k, v in inst.items() if k not in ("contract",)}
    (d / "app_data.json").write_text(json.dumps(app, indent=1, default=str, ensure_ascii=False))
    return d


def diff(a, b):
    out = subprocess.run(
        ["diff", "-ruN", "--exclude=app_data.json", "--exclude=README.md", a.name, b.name],
        cwd=a.parent,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    rows = []
    for line in out.splitlines()[:200]:
        c = (
            "add"
            if line.startswith("+") and not line.startswith("+++")
            else "del"
            if line.startswith("-") and not line.startswith("---")
            else "hdr"
            if line.startswith(("diff ", "+++", "---", "@@"))
            else ""
        )
        rows.append(f'<span class="{c}">{html.escape(line)}</span>')
    return "<pre>" + "\n".join(rows) + "</pre>" if rows else "<p class=muted>(identical)</p>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=None)
    a = ap.parse_args()
    variants = list(ACTIVE)
    counts = {}
    if a.runs:
        for f in Path(a.runs).glob("*/*/*.json"):
            r = json.loads(f.read_text())["record"]
            counts.setdefault((r["task"], r["variant"]), Counter())[
                r["outcome"] if r["acted"] else ("asked" if r["asked"] else "no_action")
            ] += 1
    head = "<tr><th>task</th>" + "".join(f"<th>{v}</th>" for v in variants) + "</tr>"
    rows, sections = [], []
    for t in TASKS:
        T = load(t)
        dirs = {v: write_env(t, v) for v in ["base", *variants]}  # base kept as the diff reference
        cells = []
        for v in variants:
            c = counts.get((t, v))
            label = "<br>".join(f"{k} {n}" for k, n in c.most_common()) if c else ""
            cells.append(f"<td><a href='#{t}-{v}'>{v}</a><div class=muted>{label}</div></td>")
        rows.append(f"<tr><td><b>{t}</b></td>{''.join(cells)}</tr>")
        cards = []
        base = dirs["base"]
        for v in variants:
            d = dirs[v]
            body = (
                f"<details><summary>system prompt</summary><pre>{html.escape((d / 'system_prompt.md').read_text())}</pre></details>"
                f"<details open><summary>user message</summary><pre>{html.escape((d / 'user_turn_1.md').read_text())}</pre></details>"
                f"<details><summary>tools</summary><pre>{html.escape((d / 'tools.json').read_text())}</pre></details>"
                + (f"<details><summary>diff from base</summary>{diff(base, d)}</details>" if v != "base" else "")
            )
            cards.append(f"<div class=card id='{t}-{v}'><h3>{t} / {v}</h3>{body}</div>")
        files = "".join(
            f"<details><summary>{html.escape(str(p.relative_to(base / 'files')))}</summary><pre>{html.escape(p.read_text(errors='replace')[:6000] if p.suffix != '.gz' else '(gzip)')}</pre></details>"
            for p in sorted((base / "files").rglob("*"))
            if p.is_file()
        )
        sections.append(
            f"<section id='{t}'><h2>{t}</h2><p>{html.escape((T.__doc__ or sys.modules[type(T).__module__].__doc__ or '').strip())}</p>"
            f"<details><summary>what the app tools return (item 0)</summary><pre>{html.escape((base / 'app_data.json').read_text()[:12000])}</pre></details>"
            + (f"<details><summary>working-directory files (item 0)</summary>{files}</details>" if files else "")
            + (
                f"<details><summary>sandbox setup</summary><pre>{html.escape((base / 'setup.sh').read_text()[:6000])}</pre></details>"
                if (base / "setup.sh").read_text()
                else ""
            )
            + "".join(cards)
            + "</section>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Over-eagerness envs</title>
<style>:root{{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#5a5954;--grid:#e6e5e0;--card:#fff;--add:#1f7a3f;--del:#b3261e;--addbg:#e6f4ea;--delbg:#fce8e6}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--fg:#fff;--muted:#c3c2b7;--grid:#2e2e2c;--card:#222220;--add:#7fd49b;--del:#ff8a80;--addbg:#16301f;--delbg:#3a1a18}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--fg:#fff;--muted:#c3c2b7;--grid:#2e2e2c;--card:#222220;--add:#7fd49b;--del:#ff8a80;--addbg:#16301f;--delbg:#3a1a18}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:20px 16px}} a{{color:inherit}} .muted{{color:var(--muted);font-size:12px}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid var(--grid);padding:4px 8px;vertical-align:top;text-align:left}} .scroll{{overflow-x:auto}}
.card{{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:10px 14px;margin:10px 0}} h3{{margin:0 0 4px;font-size:15px}} summary{{cursor:pointer;color:var(--muted)}}
pre{{font:12px/1.4 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--grid);border-radius:4px;padding:8px;max-height:420px;overflow:auto}}
.add{{color:var(--add);background:var(--addbg)}} .del{{color:var(--del);background:var(--delbg)}} .hdr{{color:var(--muted)}}</style></head><body>
<h1>Over-eagerness: 4 tasks × the email ladder's 8 harness variants</h1>
<p>Each task is a request that is ambiguous or missing a value; the question is whether GLM 5.3 acts (and how) or asks. The 8 variants are the email ladder's rungs 1,2,3,4,5,6,8,10, defined once (realism/overeager/harness_variants.py) and applied identically to every task, so the diff from base is the same everywhere. Each task's file (realism/overeager/tasks/) holds the request in three phrasings, the completion line, the app's tools/data and the grader. Shown: item 0 of 16 for each task.</p>
<div class=scroll><table>{head}{"".join(rows)}</table></div>
{"".join(sections)}
</body></html>"""
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(page)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
