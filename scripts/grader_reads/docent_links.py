"""Docent links for expview import items (extra.run_dir / source_file / row_index), as olens_search does, plus two special cases:
Celeste's round-1 IB realism episodes (source_file episodes/<arm>/<item>__<seed>.json, matched to the realism rows by arm, item, seed)
and the overeager runs (only per-arm filtered tables exist: realism/runs/overeager/docent_cells.json)."""

import json
from pathlib import Path

DOCENT = "https://docent.transluce.org/dashboard/{}/agent_run/{}"


class Docent:
    def __init__(self, data: Path, aliases: Path, overeager_cells: Path | None = None):
        self.rows, self.ib = {}, {}
        self.aliases = json.loads(aliases.read_text())
        for f in data.glob("runs/*/*/docent.json"):
            d = json.loads(f.read_text())
            rd = str(f.parent.relative_to(data))
            for i, run in enumerate(d["runs"]):
                if run:
                    self.rows[(rd, i)] = DOCENT.format(d["collection_id"], run)
            if "glm53-realism" in rd and (f.parent / "rows.jsonl").exists():
                with open(f.parent / "rows.jsonl") as fh:
                    for i, line in enumerate(fh):
                        if i < len(d["runs"]) and d["runs"][i]:
                            r = json.loads(line)
                            self.ib[(r["arm"], r["item"], str(r["seed"]))] = DOCENT.format(d["collection_id"], d["runs"][i])
        self.cells = json.loads(overeager_cells.read_text()) if overeager_cells and overeager_cells.exists() else {}

    def link(self, extra: dict, env: str = "", arm: str = "") -> tuple[str | None, str]:
        """(url, kind): kind 'episode' or 'arm table'."""
        rd, sf, ri = extra.get("run_dir"), extra.get("source_file"), extra.get("row_index")
        if rd and sf and sf.startswith("episodes/") and "ib_realistic_user_partial" in rd:
            _, a, fn = sf.split("/", 2)
            stem = fn[: -len(".json")]
            item, _, seed = stem.rpartition("__")
            kind, _, rest = item.partition("__")
            u = self.ib.get((a, f"{kind}:{rest}", seed))
            return u, "episode"
        if rd and rd.startswith("runs/overeager/"):
            task = rd.rsplit("_glm53_", 1)[-1]
            return self.cells.get(f"{task}/{arm.split('@')[0]}"), "arm table"
        if rd is None or ri is None:
            return None, ""
        local = self.aliases.get(f"{rd}::{sf}", self.aliases.get(rd, rd))
        return self.rows.get((local, int(ri))), "episode"
