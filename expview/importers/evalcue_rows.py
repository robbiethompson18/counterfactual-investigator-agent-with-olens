# /// script
# requires-python = ">=3.12"
# ///
"""An evalcue-harness run (rows.jsonl) -> an expview run in RUN/expview: one item per row, the subject's whole episode as
its call, the workspace before / after as its environment, the env's detector flags as labels, and its Docent link.

    uv run --script expview/importers/evalcue_rows.py data/runs/funding_email/2026-10-23_glm53_evalcue \
        [--base imports/celeste/ds/evalcue]

--base makes each item an overlay (README "Overlays") on the item of that dataset that shows the same row (matched on
extra.run_dir + row_index, as Celeste's importer writes them), so the transcript opens with its O-lens reads inline and
there is one page per transcript. Rows without a base item stay plain items.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expview import REPO, Run, call, discover

DOCENT = "https://docent.transluce.org"


def base_items(d: str, run_key: str) -> dict[int, str]:
    """row index -> item id, for the items of dataset d that show a row of runs/<run_key>/rows.jsonl."""
    run = discover().get(d)
    out = {}
    if run:
        for line in (run / "items.jsonl").read_text().splitlines():
            if line.strip():
                it = json.loads(line)
                ex = it.get("extra") or {}
                if str(ex.get("run_dir", "")).removeprefix("runs/").strip("/") == run_key and ex.get("source_file") in (None, "rows.jsonl"):
                    out.setdefault(int(ex["row_index"]), it["id"])
    return out


def env_docs(env: str, arms: list[str]) -> tuple[str, dict[str, str]]:
    """What the env is and what each arm means, from evalcue/envs/<env>/__init__.py's docstring (an indented
    "arm  text" list, wrapped lines indented further) or else a "| arm | text |" table in docs/envs/<env>.md.
    Returns (markdown description, {arm: full text})."""
    text, out = "", {}
    init = REPO / "evalcue/envs" / env / "__init__.py"
    doc = ast.get_docstring(ast.parse(init.read_text())) if init.exists() else None
    if doc:
        lines, cur, first = doc.splitlines(), None, None
        for n, ln in enumerate(lines):
            m = re.match(r"^(\s+)(\S+)\s{2,}(\S.*)$", ln)
            if m and m.group(2) in arms:
                cur, ind, first = m.group(2), len(m.group(1)), first if first is not None else n
                out[cur] = m.group(3).strip()
            elif cur and ln.strip() and len(ln) - len(ln.lstrip()) > ind:
                out[cur] += " " + ln.strip()
            else:
                cur = None
        intro = "\n".join(lines[:first]) if first is not None else doc.split("\n\n")[0]  # everything before the arm list
        paras = [" ".join(p.split()) for p in intro.split("\n\n") if p.strip()]
        paras += [" ".join(p.split()) for p in doc.split("\n\n") if p.strip().startswith("Label")]
        text = "\n\n".join(paras)
    md = REPO / "docs/envs" / f"{env}.md"
    if md.exists():
        for ln in md.read_text().splitlines():
            cells = [c.strip().strip("`") for c in ln.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[0] in arms and cells[0] not in out:
                out[cells[0]] = cells[1]
        text = (
            text
            or f"See [docs/envs/{env}.md](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/main/docs/envs/{env}.md)."
        )
    return text, out


def short(t: str, n: int = 90) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    head, sep, rest = t.partition(": ")
    if sep and len(head) <= 25:  # "the shipped memo: the client is ..." -> lead with the content
        t = rest
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="data/runs/<env>/<run> with rows.jsonl (and meta.json, docent.json)")
    ap.add_argument("--base", help="dataset whose items show these rows' transcripts with reads (make overlays on them)")
    a = ap.parse_args()
    rows = [json.loads(x) for x in (a.run / "rows.jsonl").read_text().splitlines() if x.strip()]
    meta = json.loads((a.run / "meta.json").read_text()) if (a.run / "meta.json").exists() else {}
    doc = json.loads((a.run / "docent.json").read_text()) if (a.run / "docent.json").exists() else None
    key = f"{a.run.parent.name}/{a.run.name}"
    bases = base_items(a.base, key) if a.base else {}
    # the env detector's flags (evalcue detect_notes) or, for coding-harness rows, their top-level yes/no fields (hack, ...)
    flags = sorted({k for r in rows for k, v in (r.get("detect_notes") or {}).items() if isinstance(v, bool)})
    params = {"positive", "thinking"}  # run settings, not outcomes
    flags += sorted({k for r in rows for k, v in r.items() if isinstance(v, bool) and k not in params and k not in flags})
    env = rows[0].get("env") or a.run.parent.name
    about, arm_docs = env_docs(env, sorted({str(r.get("arm", r.get("variant"))) for r in rows}))
    arms_md = "".join(f"\n- **{k}**: {v}" for k, v in sorted(arm_docs.items()))
    run = Run(
        a.run / "expview.new",  # built aside, then swapped in, so the server never sees a half-written run
        title=f"{a.run.parent.name}: {a.run.name}",
        groups={k: short(v) for k, v in arm_docs.items()} or None,
        description=(
            (f"{about}\n\n**Arms**{arms_md}\n\n" if arms_md else f"{about}\n\n" if about else "")
            + f"This run: {meta.get('description', '')}\n\nOne item per rollout (`{key}` line = item order). "
            "The call is the subject's whole "
            "episode, exactly as sent and returned; the environment is the workspace before the episode (`before/`) and the files "
            "it changed (`after/`). Labels: **positive** is the env's behaviour label, the rest are its detector's flags."
            + (" Items open on the same transcript with Celeste's O-lens reads inline." if bases else "")
        ),
        label_order=["positive", *flags],
    )
    run.reset()
    for f in (run.path / "envs").glob("*.json"):
        f.unlink()
    n_base, many_items = 0, len({r.get("item") for r in rows}) > 1
    for i, r in enumerate(rows):
        notes = r.get("detect_notes") or {}
        labels = {"positive": r.get("positive"), **{k: notes.get(k, r.get(k)) for k in flags}}
        arm, sample = r.get("arm", r.get("variant")), r.get("sample", r.get("seed"))
        before, after = r.get("files_before") or {}, r.get("files_after") or {}
        env = {f"before/{k}": v for k, v in before.items()} | {f"after/{k}": v for k, v in after.items() if before.get(k) != v}
        links = {}
        if doc and i < len(doc["runs"]) and doc["runs"][i]:
            links["Docent"] = f"{DOCENT}/dashboard/{doc['collection_id']}/agent_run/{doc['runs'][i]}"
        base = (a.base, bases[i]) if i in bases else None
        n_base += base is not None
        run.add(
            # arm/sample, plus the task (coding rows: pid) or item when an arm spans several, so ids stay unique
            id=f"{arm}/{r['pid']}/s{sample}" if r.get("pid") else f"{arm}/{r.get('item')}/s{sample}" if many_items else f"{arm}/s{sample}",
            group=arm,
            labels={k: v for k, v in labels.items() if v is not None},
            calls=[call("subject", r.get("model"), messages=r.get("messages") or [], params={"thinking": r.get("thinking")})],
            env_files=env or None,
            links=links or None,
            extra={"source": f"{key}:{i}", **{k: r.get(k) for k in ("item", "seed", "finish", "turns", "completion_tokens") if k in r}},
            base=base,
        )
    final, old = a.run / "expview", a.run / "expview.old"
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    run.path.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    run.path = final
    print(f"{run.path}: {len(rows)} items, {n_base} overlays on {a.base}" if a.base else f"{run.path}: {len(rows)} items")


if __name__ == "__main__":
    main()
