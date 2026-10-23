"""A node is one setting on disk: three arm dirs that share a behaviour sentence and a grader.

    <node>/
      node.json          id, parent, refs, kind, round, stage, notes
      behaviour.txt      one sentence: what the grader counts
      grader.py          def grade(record) -> True | False | None
      hypothesis.md      the Evolver's / Creator's reason the real edit should work (optional)
      baseline/ real/ null/
        system_prompt.md   optional
        user_turn_N.md     1..k
        prefill.json       optional assistant/tool turns after the user turns
        files/**           sandbox files, relative to episode.cwd
        tools.json         function tools offered to the target ([] = plain chat)
        episode.json       cwd, env, stop rules, readonly paths, image, thinking, max_tokens
        meta.json          fixed facts the grader may use (true value, company domain, ...)
      runs/ picker/ audit/ scores.json   measurements, written by the pipeline

An Arm is the in-memory form of one arm dir; the target runner only ever sees Arms.
"""

import difflib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

ARMS = ("baseline", "real", "null")
GENOME = ("node.json", "behaviour.txt", "grader.py", "hypothesis.md", *ARMS)  # what an agent edits / what gets copied
DEFAULT_EPISODE = {
    "cwd": "/workspace",
    "env": {},
    "stop": ["reply", "max_turns:20"],
    "readonly": [],
    "image": "evalcue-sandbox",
    "thinking": True,
    "max_tokens": None,
}


@dataclass
class Arm:
    messages: list[dict]  # system + user turns (+ prefill)
    files: dict[str, str] = field(default_factory=dict)
    tools: list[dict] = field(default_factory=list)
    episode: dict = field(default_factory=lambda: dict(DEFAULT_EPISODE))
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, d: Path) -> "Arm":
        msgs = []
        if (d / "system_prompt.md").exists():
            msgs.append({"role": "system", "content": (d / "system_prompt.md").read_text()})
        for f in sorted(d.glob("user_turn_*.md"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            msgs.append({"role": "user", "content": f.read_text()})
        if (d / "prefill.json").exists():
            msgs += json.loads((d / "prefill.json").read_text())
        fd = d / "files"
        files = {str(p.relative_to(fd)): p.read_text() for p in sorted(fd.rglob("*")) if p.is_file()} if fd.exists() else {}
        rd = lambda name, default: json.loads((d / name).read_text()) if (d / name).exists() else default
        return cls(msgs, files, rd("tools.json", []), {**DEFAULT_EPISODE, **rd("episode.json", {})}, rd("meta.json", {}))

    def save(self, d: Path):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        u, i = 0, 0
        for i, m in enumerate(self.messages):
            if m["role"] == "system" and i == 0:
                (d / "system_prompt.md").write_text(m["content"])
            elif m["role"] == "user" and all(x["role"] in ("system", "user") for x in self.messages[:i]):
                u += 1
                (d / f"user_turn_{u}.md").write_text(m["content"])
            else:
                (d / "prefill.json").write_text(json.dumps(self.messages[i:], indent=1) + "\n")
                break
        for p, c in self.files.items():
            f = d / "files" / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(c)
        (d / "tools.json").write_text(json.dumps(self.tools, indent=1) + "\n")
        (d / "episode.json").write_text(json.dumps(self.episode, indent=1) + "\n")
        (d / "meta.json").write_text(json.dumps(self.meta, indent=1, sort_keys=True) + "\n")


@dataclass
class Node:
    dir: Path

    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def info(self) -> dict:
        p = self.dir / "node.json"
        return json.loads(p.read_text()) if p.exists() else {"id": self.id}

    def update(self, **kw):
        (self.dir / "node.json").write_text(json.dumps({**self.info, **kw}, indent=1) + "\n")

    @property
    def behaviour(self) -> str:
        return (self.dir / "behaviour.txt").read_text().strip()

    @property
    def grader_src(self) -> str:
        return (self.dir / "grader.py").read_text()

    def arm(self, name: str) -> Arm:
        return Arm.load(self.dir / name)

    def scores(self) -> dict:
        p = self.dir / "scores.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def copy_genome(self, dest: Path):
        """Copy the editable part of the node (no measurements) to dest."""
        dest.mkdir(parents=True, exist_ok=True)
        for name in GENOME:
            src = self.dir / name
            if src.is_dir():
                shutil.copytree(src, dest / name, dirs_exist_ok=True)
            elif src.exists():
                shutil.copy2(src, dest / name)


def arm_text(d: Path, exclude=()) -> dict[str, str]:
    """Every text file in an arm dir, keyed by relative path (what diffs and size limits are computed over)."""
    return {str(p.relative_to(d)): p.read_text() for p in sorted(d.rglob("*")) if p.is_file() and str(p.relative_to(d)) not in exclude}


def unified_diff(a: Path, b: Path, exclude=()) -> str:
    """diff -ruN between two arm dirs, every file included (tools.json and episode.json too) unless excluded."""
    ta, tb = arm_text(a, exclude), arm_text(b, exclude)
    out = []
    for f in sorted(set(ta) | set(tb)):
        x, y = ta.get(f, ""), tb.get(f, "")
        if x != y:
            # headers are the bare file path: arm dir names (real/null) must never reach the Forecaster
            out += difflib.unified_diff(x.splitlines(), y.splitlines(), f, f, lineterm="", n=1)
    return "\n".join(out)


def changed_spans(a: Path, b: Path, exclude=("meta.json",)) -> dict[str, list[tuple[int, int]]]:
    """Per file, the character ranges of a (the baseline) that b replaces, deletes or inserts at (meta.json is
    bookkeeping for the grader, not part of the edit)."""
    ta, tb = arm_text(a, exclude), arm_text(b, exclude)
    out = {}
    for f in set(ta) | set(tb):
        x, y = ta.get(f, ""), tb.get(f, "")
        if x != y:
            sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
            out[f] = [(i1, i2) for op, i1, i2, _, _ in sm.get_opcodes() if op != "equal"]
    return out


def edit_overlap(base: Path, real: Path, null: Path, margin: int = 40) -> list[str]:
    """Files where the REAL and NULL edits touch the same place in the baseline (within `margin` characters, so two
    rewrites of one sentence count as overlapping). Empty when the two edits are in different places."""
    r, n = changed_spans(base, real), changed_spans(base, null)
    out = []
    for f in set(r) & set(n):
        if any(a1 - margin < b2 and b1 - margin < a2 for a1, a2 in r[f] for b1, b2 in n[f]):
            out.append(f)
    return sorted(out)


def changed_chars(a: Path, b: Path) -> int:
    """Characters inserted + deleted between two arm dirs, summed over all files (the ~400-char edit budget)."""
    ta, tb = arm_text(a), arm_text(b)
    n = 0
    for f in set(ta) | set(tb):
        x, y = ta.get(f, ""), tb.get(f, "")
        if x != y:
            sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
            n += sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return n
