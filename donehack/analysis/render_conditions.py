"""Render the model-visible inputs of every condition in port.CONDS (with and without the iterate line), plus the
unified diff of every single-factor pair. Writes analysis/conditions_rendered.md. No model calls."""

import difflib
from itertools import combinations
from pathlib import Path

from pairs import envname, factors, port, render_inputs

OUT = Path(__file__).parent
envs = [(c, it) for it in (True, False) for c in port.CONDS]
md = [
    "# Rendered conditions",
    "",
    (
        "Everything the model receives (neutral system prompt, user prompt, tool schemas, workspace listing, "
        "submit_solution.py), built by analysis/pairs.render_inputs exactly as port.episode does. "
        "`/noit` = the 'Iterate on your solution by running pytest' line is removed. "
        "description.md / solution.py / test_solution.py are identical in all environments."
    ),
    "",
    "## Conditions",
    "",
]
for c, it in envs:
    f = factors(c, it)
    md += [
        f"### {envname(c, it)}",
        "",
        *[f"- {k}: {v}" for k, v in f.items()],
        "",
        "```",
        "".join(render_inputs(c, it)).rstrip("\n"),
        "```",
        "",
    ]
md += ["## Single-factor pairs", ""]
n = 0
for a, b in combinations(envs, 2):
    fa, fb = factors(*a), factors(*b)
    diff = [f"{k}: {fa[k]} -> {fb[k]}" for k in fa if fa[k] != fb[k]]
    if len(diff) != 1:
        continue
    n += 1
    d = difflib.unified_diff(render_inputs(*a), render_inputs(*b), fromfile=envname(*a), tofile=envname(*b), n=2)
    md += [f"### {envname(*a)} → {envname(*b)}", "", f"_{diff[0]}_", "", "```diff", "".join(d).rstrip("\n"), "```", ""]
(OUT / "conditions_rendered.md").write_text("\n".join(md) + "\n")
print(len(envs), "environments,", n, "single-factor pairs")
