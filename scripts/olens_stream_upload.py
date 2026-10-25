"""Stream a pod's O-lens reads to HF while the reader runs, so a dead pod, a dead session or a RunPod balance at zero loses
at most one cycle's worth of reads (2026-10-24: 8 pods read ~1M reads onto network volumes nobody fetched before the
account went negative; the volumes can't be attached since).

    python scripts/olens_stream_upload.py RUN /workspace/reads/TAG [--every 300] [--once] [--reader-pid PID]

RUN is the run dir's name (data/runs/olens_reads/RUN, made by `scripts/runs.py new olens_reads SLUG ...`). Each cycle,
every file of the tag whose size or mtime changed since its last upload is snapshotted (cut at its last newline, since
the reader may be mid-write) and uploaded in one commit to the HF dataset under runs/olens_reads/RUN/:

    TAG.jsonl          -> requests/TAG.jsonl   (the reader's input; re-uploaded when appended to)
    TAG.log            -> logs/TAG.log
    TAG.reads*.jsonl   -> parts/<name>         (the reader's chunk files, and a legacy single TAG.reads.jsonl if present)

TAG.uploaded.json next to them records what was uploaded, so closed chunks go up once and the open chunk goes up again
only when it grew. A failed cycle leaves the manifest alone and retries after a minute. With --reader-pid, exits after a
final cycle once that process is gone (how runpod/olens_reads.sh run starts it); --once does one cycle, e.g. to rescue
a volume's files by hand. Authenticates with the pod's HF_TOKEN (runpod/pod.sh puts it in the env).
scripts/olens_reads_assemble.py turns the uploaded parts back into the run's reads.jsonl.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import CommitOperationAdd, HfApi
from runs import REPO


def targets(stem: Path) -> dict[str, Path]:
    """HF path under the run -> local file, for everything the tag has produced."""
    d, tag = stem.parent, stem.name
    out = {f"requests/{tag}.jsonl": d / f"{tag}.jsonl", f"logs/{tag}.log": d / f"{tag}.log"}
    out |= {f"parts/{p.name}": p for p in sorted(d.glob(f"{tag}.reads*.jsonl"))}
    return {rel: p for rel, p in out.items() if p.exists()}


def cycle(api: HfApi, run: str, stem: Path, manifest_path: Path) -> int:
    """Upload every changed file in one commit; -> number of files uploaded."""
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with TemporaryDirectory() as tmp:  # snapshots on the container disk, not the NFS volume
        ops, stamps = [], {}
        for rel, path in targets(stem).items():
            st = path.stat()
            stamp = [st.st_size, int(st.st_mtime)]  # stat before the copy: a file that grows meanwhile is re-uploaded next cycle
            if manifest.get(rel) == stamp:
                continue
            data = path.read_bytes()
            cut = data.rfind(b"\n") + 1  # the reader may be mid-line
            if not cut:
                continue
            snap = Path(tmp) / path.name
            snap.write_bytes(data[:cut])
            ops.append(CommitOperationAdd(path_in_repo=f"runs/olens_reads/{run}/{rel}", path_or_fileobj=str(snap)))
            stamps[rel] = stamp
        if ops:
            msg = f"runs/olens_reads/{run}: stream {stem.name}"
            api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message=msg)
            manifest_path.write_text(json.dumps(manifest | stamps, indent=1))
    return len(ops)


def alive(pid: int) -> bool:
    try:  # a zombie still answers kill -0, so read its state instead
        return "State:\tZ" not in Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run dir name, e.g. 2026-10-25_nocot-r4")
    ap.add_argument("stem", type=Path, help="/workspace/reads/TAG")
    ap.add_argument("--every", type=int, default=300, help="seconds between cycles")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--reader-pid", type=int, help="exit after a final cycle once this process is gone")
    args = ap.parse_args()
    api, manifest_path = HfApi(), args.stem.with_name(f"{args.stem.name}.uploaded.json")
    last = args.once
    while True:
        try:
            n = cycle(api, args.run, args.stem, manifest_path)
            print(f"{datetime.now():%H:%M:%S} uploaded {n} files", flush=True)
        except Exception as e:  # noqa: BLE001  anything (network, 429, token): keep retrying, never die with data on the pod
            print(f"{datetime.now():%H:%M:%S} upload failed, retrying in 60 s: {e!r}", flush=True)
            time.sleep(60)
            continue
        if last:
            break
        if args.reader_pid and not alive(args.reader_pid):
            last = True  # one more cycle picks up the reader's final chunk
            continue
        time.sleep(args.every)


if __name__ == "__main__":
    main()
