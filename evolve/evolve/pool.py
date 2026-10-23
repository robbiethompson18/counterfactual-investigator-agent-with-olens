"""Scoring nodes and choosing what to work on next.

Ranking scores come from a fixed-size slice: the latest measurement batch (30 per arm) and the latest forecaster batch
(16 trials), so topped-up nodes are compared at the same n as new ones and on fresh draws. Pooled scores (all batches)
are what agents and viewers are shown.

Eligible = audited 2 or 3 and P(real lowers the rate more than null) >= 0.8.
Each round the surprise score is U or U' (coin flip); nodes are ranked into non-dominated layers on (L, surprise, N, I),
I = innocuousness rating (innocuous.py; nodes not compared yet get the median).
Reference: 70% uniform from layers 1-2, 30% uniform from all eligible. Secondaries: the same draw, other lineages.
"""

import json
import random
from pathlib import Path

from . import innocuous, stats
from .forecast import forecaster_rows
from .measure import batches, counts
from .node import Node
from .store import ROOT

P_POS_MIN = 0.8
TOPUP_AFTER = 3  # rounds in layer 1, or times drawn as a reference
MAX_BATCHES = 4  # 4 x 30 = 120 episodes per arm, then no more top-ups


def rescore(node: Node):
    """Recompute scores.json from the node's batches and forecaster trials."""
    bs = batches(node)
    if not bs:
        return {}
    rows = forecaster_rows(node)
    pb = sorted({r.get("batch", "b0") for r in rows}) if rows else []
    last_pick = [r for r in rows if r.get("batch", "b0") == pb[-1]] if pb else []
    pick = lambda rs: (sum(r["correct"] is True for r in rs), sum(r["correct"] is not None for r in rs))
    latest = counts(node, [bs[-1]])
    pooled = counts(node)
    s = {
        "rank": stats.scores({a: v[:2] for a, v in latest.items()}, pick(last_pick)),
        "pooled": stats.scores({a: v[:2] for a, v in pooled.items()}, pick(rows)),
        "rank_batch": bs[-1],
        "latest_rates": latest,
        "pooled_rates": pooled,
        "picker": {"latest": pick(last_pick), "all": pick(rows)},
    }
    blend_investigator(node, s)
    (node.dir / "scores.json").write_text(json.dumps(s, indent=1) + "\n")
    return s


def blend_investigator(node: Node, s: dict):
    """Surprise = 50:50 blend of the Forecaster's and the transcript-only investigator's U (and U'), when the
    investigator has run; the separate values are kept as U_forecaster / U_investigator (and the _rev pair)."""
    p = node.dir / "investigator" / "scores.json"
    inv = json.loads(p.read_text()) if p.exists() else None
    for key, kn in (("rank", inv and inv["latest"]), ("pooled", inv and inv["all"])):
        r = s[key]
        r["U_forecaster"], r["U_rev_forecaster"] = r["U"], r["U_rev"]
        if kn and kn[1]:
            i = stats.scores({a: (0, 0) for a in ("baseline", "real", "null")}, tuple(kn))
            r["U_investigator"], r["U_rev_investigator"] = i["U"], i["U_rev"]
            r["U"] = round((r["U_forecaster"] + i["U"]) / 2, 4)
            r["U_rev"] = round((r["U_rev_forecaster"] + i["U_rev"]) / 2, 4)


def root(node: Node, by_id: dict) -> str:
    seen = set()
    while (p := node.info.get("parent")) and p in by_id and p not in seen:
        seen.add(p)
        node = by_id[p]
    return node.id


EVOLVE_DIR = Path(__file__).resolve().parent.parent  # the evolve project: code agents may read (and run submit from)


def contaminated(node: Node) -> list[str]:
    """Paths an agent read outside its job dir and the evolve code (e.g. earlier measured runs): such a child may have
    picked its edits from results it should not have seen. From node.json "contaminated" or the manager's record."""
    info = node.info
    if info.get("contaminated"):
        return info.get("contaminated_paths") or ["(tagged)"]
    p = node.dir / "agent_manager.json"
    if not p.exists():
        return []
    job = info.get("job") or ""
    roots = {str(EVOLVE_DIR), str(ROOT.resolve().parent)}  # the code, also via the data symlink's real location
    paths = json.loads(p.read_text()).get("outside_paths") or []

    def allowed(x):
        root = next((r for r in roots if x.startswith(r)), None)
        return root is not None and (job in x or "/data/" not in x[len(root) :])

    return [x for x in paths if not allowed(x)]


def eligible(nodes: list[Node]) -> list[Node]:
    out = []
    for n in nodes:
        if contaminated(n) or n.info.get("culled"):
            continue
        r = n.scores().get("rank")
        if r and n.info.get("audit_score") in (2, 3) and r["P_pos"] >= P_POS_MIN:
            out.append(n)
    return out


def layers(nodes: list[Node], ukey: str) -> list[list[Node]]:
    """Non-dominated sorting on (L, surprise, N, I), all maximised."""
    iv = innocuous.values(nodes)
    pts = {n.id: (n.scores()["rank"]["L"], n.scores()["rank"][ukey], n.scores()["rank"]["N"], iv[n.id]) for n in nodes}
    dom = lambda a, b: all(x >= y for x, y in zip(pts[a], pts[b])) and pts[a] != pts[b]
    rest, out = list(nodes), []
    while rest:
        front = [n for n in rest if not any(dom(m.id, n.id) for m in rest)]
        out.append(front)
        rest = [n for n in rest if n not in front]
    return out


def draw(cands: list[Node], ls: list[list[Node]], rng: random.Random) -> Node | None:
    top = [n for n in (ls[0] + (ls[1] if len(ls) > 1 else [])) if n in cands]
    pool = top if top and rng.random() < 0.7 else cands
    return rng.choice(pool) if pool else None


def choose(nodes: list[Node], ukey: str, rng: random.Random, n_secondaries=2):
    """(reference, secondaries) for one mutation, or (None, []) if nothing is eligible."""
    el = eligible(nodes)
    if not el:
        return None, []
    ls = layers(el, ukey)
    by_id = {n.id: n for n in nodes}
    ref = draw(el, ls, rng)
    secs, used = [], {root(ref, by_id)}
    for _ in range(n_secondaries):
        c = [n for n in el if root(n, by_id) not in used]
        s = draw(c, ls, rng)
        if s:
            secs.append(s)
            used.add(root(s, by_id))
    return ref, secs


def due_topups(nodes: list[Node]) -> list[Node]:
    return [
        n
        for n in eligible(nodes)
        if len(batches(n)) < MAX_BATCHES and max(n.info.get("front_rounds", 0), n.info.get("ref_count", 0)) >= TOPUP_AFTER
    ]
