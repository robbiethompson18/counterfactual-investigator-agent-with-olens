"""One podman sandbox per episode (donehack/realism's hardened Box, made generic).

Hard rules, not options: `--network none`; the target model never reaches the host. Files go in and come out as tar
streams through `podman exec` inside the container, so the host never follows a path the agent controlled (a symlink to a
host key file is recorded as a symlink, never read). The agent runs as `dev` (uid 200 in the realism image).
"""

import asyncio
import io
import re
import tarfile
import time
import uuid
from collections.abc import Mapping

from .core import Episode

USERNS_SIZE = 256  # 256-id slices, so ~250 sandboxes fit in the 65536 subuids (the u200 images)
MAX_FILE = 200_000  # bigger files are listed in `omitted`, not stored


async def run(args, timeout, stdin: bytes | None = None) -> tuple[int, bytes]:
    p = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(p.communicate(stdin), timeout)
    except TimeoutError:
        p.kill()
        await p.wait()
        return 124, b"[timed out]"
    return p.returncode or 0, out


class Box:
    slots = asyncio.Semaphore(100)  # podman's sqlite db jams above ~150 concurrent sandboxes on bramble

    def __init__(self, ep: Episode, label: str):
        self.ep = ep
        self.name = "envrun_" + re.sub(r"[^a-zA-Z0-9_.-]", "_", label)[:80] + "_" + uuid.uuid4().hex[:8]
        self.held = False

    async def must(self, args, timeout=120, stdin=None) -> str:
        code, out = await run(args, timeout, stdin)
        if code:
            raise RuntimeError(f"sandbox command failed ({' '.join(args[:5])}...): {out[-500:].decode(errors='replace')}")
        return out.decode(errors="replace")

    async def put(self, d: str, files: Mapping[str, str | bytes], user="dev"):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as t:
            for name, data in files.items():
                b = data.encode() if isinstance(data, str) else data
                ti = tarfile.TarInfo(name)
                ti.size, ti.mode, ti.mtime = len(b), 0o644, time.time()  # as realism's Box; envs chmod in setup
                t.addfile(ti, io.BytesIO(b))
        await self.must(["podman", "exec", "-i", "-u", user, self.name, "tar", "--no-same-owner", "-C", d, "-xf", "-"], 120, buf.getvalue())

    async def start(self):
        await Box.slots.acquire()
        self.held = True
        e = self.ep
        tmpfs = ["--tmpfs", "/home/dev:size=64m,mode=0755,nosuid,nodev"]
        if e.workdir != "/home/dev":
            tmpfs += ["--tmpfs", f"{e.workdir}:size=24m,mode=0755,nosuid,nodev"]
        tmpfs += ["--tmpfs", "/tmp:size=64m,mode=1777,nosuid,nodev"]
        for d in e.extra_tmpfs:
            tmpfs += ["--tmpfs", f"{d}:size=16m,mode=0755,nosuid,nodev"]
        args = [
            "podman", "run", "-d", "--rm", "--init", "--name", self.name, "--hostname", e.hostname,
            "--userns", f"auto:size={USERNS_SIZE}",
            "--network", "none",  # never optional
            "--read-only", "--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--cap-add", "CHOWN,FOWNER,DAC_OVERRIDE",
            "--memory", "1g", "--memory-swap", "1g", "--pids-limit", "256", "--cpus", "1",
            "--ulimit", "nofile=1024:1024", "--ulimit", "fsize=33554432:33554432",
            "-e", "HOME=/home/dev", "-e", "USER=dev", "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
            *[x for k, v in e.env.items() for x in ("-e", f"{k}={v}")],
            *tmpfs, e.image, "sleep", "infinity",
        ]  # fmt: skip
        assert args[args.index("--network") + 1] == "none"
        for _ in range(60):
            code, out = await run(args, 180)
            if not code:
                break
            msg = out.decode(errors="replace")
            if "unused IDs" not in msg and "already in use" not in msg:
                raise RuntimeError(f"podman run failed: {msg[-500:]}")
            await asyncio.sleep(10)
        else:
            raise RuntimeError("no free userns slice")
        dirs = " ".join([e.workdir, "/home/dev", "/tmp", *e.extra_tmpfs])
        await self.must(["podman", "exec", "-u", "root", self.name, "sh", "-c", f"chown dev:dev {dirs}"])
        if e.files:
            await self.put(e.workdir, e.files)
        for cmd in e.setup:
            code, out = await self.sh(cmd, 60)
            if code:
                raise RuntimeError(f"setup command failed: {cmd[:80]}: {out[-300:]}")

    async def sh(self, cmd: str, timeout=60, user="dev", cwd=None) -> tuple[int, str]:
        code, out = await run(
            ["podman", "exec", "-u", user, "-w", cwd or self.ep.workdir, self.name, "timeout", "-k", "5", str(timeout), "bash", "-c", cmd],
            timeout + 30,
        )
        s = out.decode(errors="replace")
        if len(s) > 6000:
            s = s[:3000] + "\n...[truncated]...\n" + s[-3000:]
        return code, s

    async def read(self, rel: str) -> str | None:
        """A workdir file's text, read inside the container (None if missing)."""
        code, out = await run(["podman", "exec", "-u", "root", self.name, "cat", f"{self.ep.workdir}/{rel}"], 60)
        return out.decode(errors="replace") if code == 0 else None

    async def lockdown(self):
        """Stop everything the agent left running before anything is graded or snapshotted."""
        await run(["podman", "exec", "-u", "dev", self.name, "sh", "-c", "kill -9 -1"], 60)

    async def snapshot(self, root: str | None = None) -> tuple[dict[str, str], list[dict]]:
        """Files under `root` (default workdir), read inside the container. -> (files, omitted). Regular UTF-8 files under
        MAX_FILE are stored; a symlink is stored as "[symlink -> target]" and never followed; anything else is omitted."""
        root = root or self.ep.workdir
        code, out = await run(["podman", "exec", "-u", "root", self.name, "tar", "-C", root, "-cf", "-", "."], 120)
        files, omitted = {}, []
        if code:
            return files, [{"path": root, "why": f"tar failed: {out[-200:].decode(errors='replace')}"}]
        with tarfile.open(fileobj=io.BytesIO(out), mode="r:") as t:
            for m in t:
                path = m.name.removeprefix("./")
                if not path or m.isdir():
                    continue
                if m.issym() or m.islnk():
                    files[path] = f"[symlink -> {m.linkname}]"
                elif not m.isfile():
                    omitted.append({"path": path, "why": "not a regular file"})
                elif m.size >= MAX_FILE:
                    omitted.append({"path": path, "why": f"{m.size} bytes"})
                else:
                    f = t.extractfile(m)
                    try:
                        files[path] = (f.read() if f else b"").decode()
                    except UnicodeDecodeError:
                        omitted.append({"path": path, "why": "binary"})
        return files, omitted

    async def stop(self):
        await run(["podman", "rm", "-f", "-t", "0", self.name], 60)
        if self.held:
            Box.slots.release()
            self.held = False


async def handle(box: Box, kind: str, a: dict) -> str:
    """The shell tools, with the output formats the overeager / realism runs used."""
    cmd = a.get("command") or a.get("cmd") or ""
    if kind == "shell":  # agent-interp-envs' execute_command
        code, out = await box.sh(cmd)
        return f"Exit code: {code}\nOutput:\n{out}"
    if kind in ("bash_raw", "bash_cwd"):  # a Claude-Code-like bash tool; bash_cwd keeps `cd` across calls, as its description says
        if kind == "bash_cwd":
            cmd = f"trap 'pwd > /tmp/.agent_cwd' EXIT; cd \"$(cat /tmp/.agent_cwd 2>/dev/null || pwd)\" 2>/dev/null\n{cmd}"
        code, out = await box.sh(cmd, 120)
        return (out if out.strip() else "(no output)") + (f"\n[exit {code}]" if code else "")
    raise ValueError(kind)
