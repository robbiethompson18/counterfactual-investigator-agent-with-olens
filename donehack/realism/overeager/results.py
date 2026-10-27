# ruff: noqa: E501
"""Results page + expview export for an over-eagerness run (realism/runs/overeager/<run>).

  uv run python realism/overeager/results.py --run v1 --out <html> [--expview]
Page: one small bar chart per task (acted: one candidate / several / other, and asked), variants in ladder order, 95%
Wilson intervals on "acted", n per bar. --expview writes every episode to ~/workspace/expview/runs/overeager_<run> for
http://bramble:8830 (env files, exact prompts and tools, transcript, record).
"""

import argparse
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from harness_variants import ACTIVE
from oe_run import TASKS

COL = {"one": "#2a78d6", "multi": "#8a5cd1", "other": "#b0461c", "asked": "#1f9e74"}


def wilson(k, n, z=1.96):
    if not n:
        return 0, 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def load(run):
    recs = defaultdict(list)
    for f in (HERE.parent / "runs" / "overeager" / run).glob("*/*/*.json"):
        r = json.loads(f.read_text())["record"]
        recs[(r["task"], r["variant"])].append(r)
    return recs


def chart(task, recs, variants, ds):
    W, left, top, ph, gw = 980, 44, 14, 150, (980 - 60) / len(variants)
    sy = lambda p: top + ph * (1 - p)
    svg = [f'<svg viewBox="0 0 {W} {top + ph + 54}" width="100%" role="img" aria-label="{task} acted/asked by variant">']
    for t in (0, 0.5, 1):
        svg.append(
            f'<line x1="{left}" x2="{W - 8}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="var(--grid)"/><text x="{left - 6}" y="{sy(t) + 4:.1f}" class="ax" text-anchor="end">{int(t * 100)}%</text>'
        )
    for i, v in enumerate(variants):
        rs = [r for r in recs.get((task, v), []) if r["outcome"] != "cut_long"]
        n = len(rs)
        x0 = left + 8 + i * gw
        if not n:
            svg.append(f'<text x="{x0 + gw / 2 - 8:.1f}" y="{sy(0) - 4:.1f}" class="ax" text-anchor="middle">–</text>')
        else:
            one = sum(bool(r["acted"]) and r.get("acted_one") and not r.get("acted_multi") for r in rs)
            multi = sum(bool(r["acted"]) and bool(r.get("acted_multi")) for r in rs)
            acted = sum(bool(r["acted"]) for r in rs)
            other = acted - one - multi
            asked = sum(bool(r["asked"]) for r in rs)
            bw = min(26, gw / 2 - 6)
            y = sy(0)
            for k, col in [(one, COL["one"]), (multi, COL["multi"]), (other, COL["other"])]:
                h = ph * k / n
                if h:
                    svg.append(
                        f'<rect x="{x0:.1f}" y="{y - h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{col}"><title>{v}: {k}/{n}</title></rect>'
                    )
                y -= h
            p, lo, hi = wilson(acted, n)
            svg.append(
                f'<line x1="{x0 + bw / 2:.1f}" x2="{x0 + bw / 2:.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="var(--fg)" opacity=".5"/><text x="{x0 + bw / 2:.1f}" y="{sy(p) - 4:.1f}" class="val" text-anchor="middle">{round(100 * p)}</text>'
            )
            h = ph * asked / n
            svg.append(
                f'<rect x="{x0 + bw + 3:.1f}" y="{sy(0) - h:.1f}" width="{bw:.1f}" height="{max(h, 1):.1f}" fill="{COL["asked"]}"><title>{v}: asked {asked}/{n}</title></rect>'
            )
        svg.append(
            f'<a href="http://bramble:8830/#d={ds}" target="_blank"><text x="{x0 + gw / 2 - 8:.1f}" y="{top + ph + 16}" class="ax" text-anchor="middle">{html.escape(v.replace("base_", "b_"))}</text></a><text x="{x0 + gw / 2 - 8:.1f}" y="{top + ph + 30}" class="ax" text-anchor="middle">n={n}</text>'
        )
    svg.append("</svg>")
    return "".join(svg)


def export(run, recs_dir):
    sys.path.insert(0, "/home/claude/workspace/expview")
    from expview import Run, call

    out = Path("/home/claude/workspace/expview/runs") / f"overeager_{run}"
    R = Run(
        out,
        title=f"Over-eagerness tasks ({run})",
        description="GLM 5.3, 10 tasks x harness variants. outcome = what it did; asked = no action and a question to the user.",
        label_order=["task", "variant", "outcome", "asked"],
    )
    R.reset()
    for f in sorted(recs_dir.glob("*/*/*.json")):
        d = json.loads(f.read_text())
        r, env = d["record"], d["env"]
        files = {f"workspace/{k}": v for k, v in env["files"].items() if isinstance(v, str)}
        files["config.json"] = json.dumps(
            {"system": env["system"], "user": env["user"], "tools": env["tools"], "workdir": env["workdir"]}, indent=1
        )
        files["record.json"] = json.dumps(r, indent=1, ensure_ascii=False)
        R.add(
            id=f"{r['task']}/{r['variant']}/{r['item']}_s{r['seed']}",
            group=f"{r['task']}/{r['variant']}",
            labels={"task": r["task"], "variant": r["variant"], "outcome": r["outcome"], "asked": r["asked"]},
            env_files=files,
            calls=[
                call(
                    "subject",
                    "z-ai/glm-5.3",
                    messages=d["messages"],
                    tools=env["tools"],
                    params={"reasoning": {"effort": "medium"}},
                    cost=r["cost"],
                )
            ],
        )
    return f"expview/overeager_{run}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expview", action="store_true")
    a = ap.parse_args()
    recs = load(a.run)
    ds = export(a.run, HERE.parent / "runs" / "overeager" / a.run) if a.expview else f"expview/overeager_{a.run}"
    variants = [v for v in ACTIVE if any((t, v) in recs for t in TASKS)]
    sections = []
    for t in TASKS:
        if not any((t, v) in recs for v in variants):
            continue
        doc = ""
        sections.append(f"<section class=card id='{t}'><h2>{t}</h2>{doc}{chart(t, recs, variants, ds)}</section>")
    total = sum(len(v) for v in recs.values())
    cost = sum(r["cost"] for v in recs.values() for r in v)
    leg = "".join(
        f'<span class=lg><i style="background:{c}"></i>{k}</span>'
        for k, c in [
            ("acted on one candidate", COL["one"]),
            ("acted on several / all", COL["multi"]),
            ("acted otherwise (hack, invented address, tampering…)", COL["other"]),
            ("asked (no action, question to the user)", COL["asked"]),
        ]
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Over-eagerness results</title>
<style>:root{{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#5a5954;--grid:#e6e5e0;--card:#fff}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--fg:#fff;--muted:#c3c2b7;--grid:#2e2e2c;--card:#222220}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--fg:#fff;--muted:#c3c2b7;--grid:#2e2e2c;--card:#222220}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1040px;margin:0 auto;padding:20px 16px}} a{{color:inherit}}
.card{{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:10px 14px;margin:12px 0}} h2{{font-size:16px;margin:0 0 4px}}
.ax{{fill:var(--muted);font-size:11px}} .val{{fill:var(--muted);font-size:10px}} .lg{{margin-right:14px;font-size:13px;white-space:nowrap}} .lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}}</style></head><body>
<h1>Over-eagerness and reward hacking across harness variants (GLM 5.3)</h1>
<p>{total} episodes (${cost:.2f}). Per task, variants left to right: the email ladder rungs (base_home … core; base itself and the MCP-named-tools arm are dropped as uninformative) (a .task.json eval-cue arm was also run but is dropped: the effect was verbalized). Left bar: acted (stacked by kind), with 95% Wilson interval and %; right bar: asked. Episodes that hit the $0.10 cap are dropped. Labels link to the transcripts on 8830. Environments: <a href="https://claude.ai/artifact/CVMrM91ishdzCAvGXsuKRk">task × variant grid</a>.</p>
<div>{leg}</div>
{"".join(sections)}
</body></html>"""
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(page)
    print("wrote", a.out, "episodes", total)


if __name__ == "__main__":
    main()
