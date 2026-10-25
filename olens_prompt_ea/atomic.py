"""Write an expview run in a staging dir OUTSIDE the viewer's roots and swap it in only after every item is added, so a
live run the viewer serves is never half-written (Run.reset() on the live dir truncates items.jsonl until the adds finish,
and a temp dir under the roots would itself be indexed as a run, since Run() writes run.json first).

Staging: $EXPVIEW_STAGING, default ~/.cache/expview_staging (same filesystem as the repo on bramble, so the swap is a rename).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def temp_dir(final: Path) -> Path:
    base = Path(os.environ.get("EXPVIEW_STAGING", Path.home() / ".cache" / "expview_staging"))
    tmp = base / f"{final.parent.parent.name}__{final.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    return tmp


def swap_in(tmp: Path, final: Path) -> None:
    """Replace `final` with the finished `tmp`: move it in beside the live dir, then two renames (old copy kept until the
    swap succeeds). Only the final renames touch the live path."""
    incoming = final.with_name(f".{final.name}.incoming")
    prev = final.with_name(f".{final.name}.prev")
    for d in (incoming, prev):
        if d.exists():
            shutil.rmtree(d)
    shutil.move(str(tmp), str(incoming))  # a rename on the same filesystem
    (incoming / "run.json").rename(incoming / ".run.json.pending")  # not a run until it is in place
    if final.exists():
        final.rename(prev)
    incoming.rename(final)
    (final / ".run.json.pending").rename(final / "run.json")
    if prev.exists():
        shutil.rmtree(prev)
