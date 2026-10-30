"""Slide-sized matplotlib figures in the deck's style: graphs/style.py's colours and conditions, slides.mplstyle's fonts and
sizes. Make the figure at the size it will sit on the slide and include it unscaled (\\slidefig in the theme), so its text
matches the slide's. The 16:9 frame's text block is 150 mm wide."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import graphs.style

BG = "#f6f5f1"  # the slide background (the Evals page's --bg)

MM = 1 / 25.4
FULL_W, FULL_H = 150, 58  # a figure that fills the frame under its title
HALF_W = 72  # one column of a two-column frame
plt.style.use(Path(__file__).with_name("slides.mplstyle"))
_bar_kw = graphs.style.bar_kw


def bar_kw(c: str) -> dict:
    """graphs.style.bar_kw with a placebo's stripes in the slide background colour instead of black (Kylie, 2026-10-30)."""
    kw = _bar_kw(c)
    return {**kw, "edgecolor": "none", "hatchcolor": BG} if kw["hatch"] else kw


graphs.style.bar_kw = bar_kw  # ty: ignore[invalid-assignment]  # so diff_rows' swatches get the slide stripes too


def figure(w_mm: float = FULL_W, h_mm: float = FULL_H, **kw):
    return plt.subplots(figsize=(w_mm * MM, h_mm * MM), layout="constrained", **kw)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)  # no bbox_inches="tight": that changes the size the deck relies on
    plt.close(fig)
    print(path)
