"""Dump exactly what the model sees at the start of an episode: system prompt, tool schemas, first user message, files.

    uv run python -m evalcue.envs.funding_email_real.render full_a              # to stdout
    uv run python -m evalcue.envs.funding_email_real.render full_a --out DIR    # DIR/{system.txt,tools.json,user.txt,tree.txt}
Needs QWEN_VLLM_URL set (evalcue.harness builds its client on import); any value works. EVALCUE_MODEL changes the model line.
"""

import argparse
import json
from pathlib import Path

from evalcue.envs import funding_email_real as E


def render(arm: str) -> dict[str, str]:
    ep = E.build(arm, "default")
    harness_, docs, _ = E.parse_arm(arm)
    if harness_ == "upstream":
        tree = "\n".join(f"{len(t.encode()):>7}  {ep.cwd}/{rel}" for rel, t in sorted(ep.files.items()))
        tree = f"cwd {ep.cwd}  (bytes, path; mtimes are the sandbox's)\n{tree}"
        draft = ep.files[E.upstream.DRAFT]
    else:
        times = E._mtimes(ep.files, f"halden/{E.DOCS[docs].memo_path}")
        tree = "\n".join(
            f"{times[rel].strftime('%Y-%m-%d %H:%M')}  {len(t.encode()):>7}  {E.HOME}/{rel}" for rel, t in sorted(ep.files.items())
        )
        tree = f"cwd {E.CWD}  (mount {E.HOME}; mtime PDT, bytes, path)\n{tree}"
        draft = json.dumps({"id": E.DRAFT_ID, **E.DOCS[docs].draft}, indent=2)
    return {
        "system.txt": ep.messages[0]["content"],
        "tools.json": json.dumps([t["function"] for t in ep.tools], indent=2),
        "user.txt": ep.messages[1]["content"],
        "tree.txt": tree,
        "draft.txt": draft,
        "checks.txt": f"IDE file (full): {E.ide_file(docs)}\n"
        f"non-memo files matching the disclosure regex (should be none): {E.DISCLOSE_LEAKS[docs]}\n"
        f"so detect() drops these regex alternatives for these docs: {E.DISCLOSE_DROPPED[docs]}\n"
        f"files containing a memo marker (should be none): {E.MARKER_LEAKS[docs]}",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=E.ARMS)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    parts = render(a.arm)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        for name, text in parts.items():
            (a.out / name).write_text(text + "\n")
        print(f"wrote {', '.join(parts)} to {a.out}")
    else:
        for name, text in parts.items():
            print(f"===== {name} =====\n{text}\n")
