"""Helpers for an eval's own page section (Kylie, 2026-10-30: evals should mostly write their own page).

An eval's score.py may define `PAGE(rows) -> str`: HTML built from the current runs' prediction rows, which
`python -m environments evals NAME` stores as the eval's `page_html` and the env viewer puts at the top of the eval's page
(the standard tables fold away below it). Anything with `data-cond="<condition>"` gets the page's colour for that
condition as the CSS variable `--c` (use `fill="var(--c)"` in SVG, `style="background:var(--c)"` in HTML).
"""

from html import escape

LABELS = {"probe": "linear probe", "transcript_inline": "transcript", "olens_inline": "O-lens", "olens_placebo_inline": "O-lens placebo",
          "transcript_agentic": "transcript", "olens_agentic": "O-lens", "olens_placebo_agentic": "O-lens placebo"}  # fmt: skip


def label(cond: str) -> str:
    return LABELS.get(cond, cond)


def bars(
    rows: list[tuple[str, float, float, float]],
    lo: float = 0.5,
    hi: float = 1.0,
    title: str = "",
    fmt: str = "{:.3f}",
    lines: tuple[tuple[float, str], ...] | list[tuple[float, str]] = (),
) -> str:
    """A bar chart of (condition, value, ci_lo, ci_hi) with error bars (lo == hi: none), y from `lo` to `hi`, plus dashed
    horizontal reference `lines` (value, label)."""
    W, H, L, B, T, bw = 120 + 110 * len(rows) + (150 if lines else 0), 260, 44, 40, 24, 64  # lines' labels sit right of the bars
    y = lambda v: T + (H - T - B) * (1 - (min(max(v, lo), hi) - lo) / (hi - lo))
    s = [f'<text x="{L}" y="14" class="lab">{escape(title)}</text>']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        s.append(f'<line x1="{L}" x2="{W - 10}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--line, #ddd)"/>')
        s.append(f'<text x="{L - 6}" y="{y(v) + 4:.1f}" class="ax" text-anchor="end">{fmt.format(v)}</text>')
    for v, text in lines:
        s.append(
            f'<line x1="{L}" x2="{W - 10}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="currentColor" stroke-dasharray="5 4" opacity="0.7"/>'
        )
        s.append(f'<text x="{W - 12}" y="{y(v) - 5:.1f}" class="ax" text-anchor="end">{escape(text)}</text>')
    for i, (cond, v, a, b) in enumerate(rows):
        cx = L + 40 + 110 * i + bw / 2
        box = f'x="{cx - bw / 2:.1f}" y="{y(v):.1f}" width="{bw}" height="{y(lo) - y(v):.1f}"'
        s.append(f'<rect data-cond="{escape(cond)}" {box} fill="var(--c)" opacity="0.85"/>')
        if a != b:
            s.append(f'<line x1="{cx}" x2="{cx}" y1="{y(a):.1f}" y2="{y(b):.1f}" stroke="currentColor"/>')
            s += [f'<line x1="{cx - 6}" x2="{cx + 6}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="currentColor"/>' for t in (a, b)]
        s.append(f'<text x="{cx}" y="{y(b) - 6:.1f}" class="ax" text-anchor="middle">{fmt.format(v)}</text>')
        s.append(f'<text x="{cx}" y="{H - B + 18}" class="lab" text-anchor="middle">{escape(label(cond))}</text>')
    return f'<div class="scroll"><svg class="evchart" viewBox="0 0 {W} {H}" width="{W}" height="{H}">{"".join(s)}</svg></div>'


def table(head: list[str], body: list[list[str]]) -> str:
    """A plain table; cells are HTML (escape text yourself)."""
    th = "".join(f"<th>{h}</th>" for h in head)
    return (
        f'<div class="scroll"><table><tr>{th}</tr>'
        + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body)
        + "</table></div>"
    )


def swatch(cond: str) -> str:
    return f'<i class="sw" data-cond="{escape(cond)}" style="background:var(--c)"></i>{escape(label(cond))}'
