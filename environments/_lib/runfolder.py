"""Run folders, readable straight from HF (dataset path envs/<env>/<run>/, local data/envs/<env>/<run>/):

    meta.json                      git sha, command, env ref, target, caps, overrides, arms, items, n, Docent ids, status
    episodes/NNN.jsonl             size-sharded (~20 MB); one episode per line
    episodes/index.json            episode id -> [shard, byte offset, byte length], so a reader fetches one transcript
    blobs/<sha256>                 file contents, stored once; episodes hold {path: sha256} for files_before/files_after
    labels/<labeler>@<ver>.jsonl   {"episode": id, "values": {...}}; measure() is the labeler "measure"; never overwritten

A finished run is immutable: new labels go in a new labels file. `index.py` builds envs/<env>/index.json from these.
"""

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL = REPO_ROOT / "data" / "envs"
SHARD_BYTES = 20_000_000


def git_sha() -> str | None:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "environments"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
            ).stdout
        )
        return sha + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class RunWriter:
    def __init__(self, env: str, run: str, meta: dict, root: Path = LOCAL):
        self.dir = root / env / run
        if (self.dir / "meta.json").exists():
            raise SystemExit(f"{self.dir} exists; runs are immutable, pick a new --tag")
        (self.dir / "episodes").mkdir(parents=True)
        (self.dir / "blobs").mkdir()
        (self.dir / "labels").mkdir()
        self.meta = {"env": env, "run": run, "status": "running", "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **meta}
        self.index: dict[str, list] = {}
        self.shard, self.fh = 0, None
        self._write_meta()

    def _write_meta(self):
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=1, default=str) + "\n")

    def blob(self, text: str | bytes) -> str:
        b = text if isinstance(text, bytes) else text.encode()
        h = hashlib.sha256(b).hexdigest()
        p = self.dir / "blobs" / h
        if not p.exists():
            p.write_bytes(b)
        return h

    def add(self, ep: dict):
        ep = dict(ep)
        for k in ("files_before", "files_after"):
            ep[k] = {path: self.blob(text) for path, text in (ep.get(k) or {}).items()}
        line = (json.dumps(ep, ensure_ascii=False, default=str) + "\n").encode()
        if self.fh is None or self.fh.tell() + len(line) > SHARD_BYTES and self.fh.tell() > 0:
            if self.fh:
                self.fh.close()
                self.shard += 1
            self.fh = open(self.dir / "episodes" / f"{self.shard:03d}.jsonl", "ab")  # noqa: SIM115
        off = self.fh.tell()
        self.fh.write(line)
        self.fh.flush()
        self.index[ep["id"]] = [f"{self.shard:03d}.jsonl", off, len(line)]

    def labels(self, labeler: str, version: str, rows: list[tuple[str, dict | None]], kind: str = "code"):
        p = self.dir / "labels" / f"{labeler}@{version}.jsonl"
        if p.exists():
            raise SystemExit(f"{p} exists; label files are never overwritten")
        p.write_text("".join(json.dumps({"episode": eid, "values": v, "kind": kind}, default=str) + "\n" for eid, v in rows))

    def finish(self, status="done", **extra):
        if self.fh:
            self.fh.close()
        (self.dir / "episodes" / "index.json").write_text(json.dumps(self.index) + "\n")
        self.meta.update(status=status, finished=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), n_episodes=len(self.index), **extra)
        self._write_meta()


def read_episodes(run_dir: Path):
    for shard in sorted((run_dir / "episodes").glob("*.jsonl")):
        with open(shard) as f:
            for line in f:
                yield json.loads(line)


def docent_upload(run_dir: Path) -> str | None:
    """Upload a finished run's episodes to the Docent collection named after its env (scripts/docent_upload.py, which
    carries Docent's SDK as its own deps), then write each episode's `docent_run` / `docent_url` and meta.json's
    `docent_collection`. Skips (returns None) without a key: DOCENT_API_KEY, or docent.key in the main checkout."""
    import os

    meta = json.loads((run_dir / "meta.json").read_text())
    if meta.get("docent_collection") and (run_dir / "docent.json").exists():
        return meta["docent_collection"]
    key = os.environ.get("DOCENT_API_KEY") or _main_checkout_file("docent.key")
    if not key:
        print(f"{run_dir}: no DOCENT_API_KEY or docent.key; skipping Docent")
        return None
    script = REPO_ROOT / "scripts" / "docent_upload.py"
    env = {**os.environ, "DOCENT_API_KEY": key}
    subprocess.run(["uv", "run", "--quiet", "--script", str(script), "upload-env", str(run_dir)], cwd=REPO_ROOT, env=env, check=True)
    doc = json.loads((run_dir / "docent.json").read_text())
    cid, runs = doc["collection_id"], doc["runs"]
    # rewrite the shards with the Docent fields (offsets change, so the index is rebuilt); same shard boundaries
    index = {}
    for shard in sorted((run_dir / "episodes").glob("*.jsonl")):
        lines, off = [], 0
        for line in shard.read_text().splitlines():
            ep = json.loads(line)
            rid = runs.get(ep["id"])
            ep["docent_run"], ep["docent_url"] = rid, rid and f"https://docent.transluce.org/dashboard/{cid}/agent_run/{rid}"
            b = (json.dumps(ep, ensure_ascii=False, default=str) + "\n").encode()
            index[ep["id"]] = [shard.name, off, len(b)]
            lines.append(b)
            off += len(b)
        shard.write_bytes(b"".join(lines))
    (run_dir / "episodes" / "index.json").write_text(json.dumps(index) + "\n")
    meta["docent_collection"] = cid
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return cid


def _main_checkout_file(name: str) -> str | None:
    """A gitignored key file at the repo root, or in the main checkout when this is a git worktree."""
    k = REPO_ROOT / name
    if not k.exists():
        main = subprocess.run(
            ["git", "worktree", "list", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.split("\n")[0]
        k = Path(main.removeprefix("worktree ")) / name
    return k.read_text().strip() if k.exists() else None


def hf_token() -> str | None:
    import os

    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    return _main_checkout_file("hf.key")


def upload(paths: list[Path], message: str):
    """Upload local data/envs/... paths (run dirs and index files) to the same paths under envs/ in one commit."""
    from huggingface_hub import CommitOperationAdd, HfApi

    ops = []
    for p in paths:
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
        for f in files:
            ops.append(CommitOperationAdd(path_in_repo="envs/" + str(f.relative_to(LOCAL)), path_or_fileobj=str(f)))
    api = HfApi(token=hf_token())
    info = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message=message)
    return len(ops), info.oid
