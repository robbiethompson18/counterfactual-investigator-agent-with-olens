"""Podman sandbox for one episode (or one agent session).

No network, read-only image, its own user-namespace slice (container root is an unprivileged host subuid), no
capabilities beyond the setup chown/chmod, size-capped tmpfs for the work dir, /tmp and the home dir. Commands run as
uid 1000. Files listed in `readonly` are root-owned 0444; anything else in the work dir is the agent's. Snapshots read
the work dir back out as text.
"""

import asyncio
import io
import itertools
import os
import random
import tarfile
import time

MAX_BOXES = int(os.environ.get("EVOLVE_MAX_BOXES", "16"))  # per process; all processes share ~64 --userns auto slots
_slots = asyncio.Semaphore(MAX_BOXES)
_seq = itertools.count()
UID = "1000:1000"


async def run(args, timeout=60, stdin: bytes | None = None) -> tuple[int, str]:
    p = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(p.communicate(stdin), timeout)
    except TimeoutError:
        p.kill()
        await p.wait()
        return 124, "[timed out]"
    return p.returncode, out.decode(errors="replace")


def _tar(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime = len(data), time.time()
            ti.mode = 0o755 if data.startswith(b"#!") else 0o644
            t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


_reaped = False


async def reap_orphans():
    """Remove sandboxes left by evolve processes that died without cleaning up (each holds one of ~64 userns slots)."""
    code, out = await run(["podman", "ps", "-a", "--format", "{{.Names}}"], 60)
    dead = []
    for name in out.split() if not code else []:
        parts = name.split("_")
        if len(parts) == 3 and parts[0] == "evolve" and parts[1].isdigit() and not os.path.exists(f"/proc/{parts[1]}"):
            dead.append(name)
    if dead:
        await run(["podman", "rm", "-f", "-t", "0", *dead], 120)
    return dead


class Box:
    def __init__(
        self, files: dict[str, str], *, cwd="/workspace", env=None, readonly=(), image="evalcue-sandbox", mounts=(), work_size="64m"
    ):
        self.files, self.cwd, self.env, self.readonly, self.image = files, cwd, env or {}, list(readonly), image
        self.mounts, self.work_size = list(mounts), work_size  # mounts: (host_dir, container_dir, "ro"|"rw")
        self.name = f"evolve_{os.getpid()}_{next(_seq)}"
        self.up = False

    async def __aenter__(self):
        global _reaped
        if not _reaped:
            _reaped = True
            await reap_orphans()
        await _slots.acquire()
        try:
            args = [
                "podman",
                "run",
                "-d",
                "--rm",
                "--init",
                "--name",
                self.name,
                "--hostname",
                "sandbox",
                "--userns",
                "auto:size=1024",
                "--network",
                "none",
                "--read-only",
                "--security-opt",
                "no-new-privileges",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "CHOWN,FOWNER,DAC_OVERRIDE",
                "--memory",
                "1g",
                "--memory-swap",
                "1g",
                "--pids-limit",
                "256",
                "--cpus",
                "1",
                "--tmpfs",
                f"{self.cwd}:size={self.work_size},mode=0755",
                "--tmpfs",
                "/tmp:size=64m,mode=1777",
                *(["--tmpfs", "/home/agent:size=64m,mode=0777"] if self.cwd != "/home/agent" else []),
                "-e",
                "HOME=/home/agent",
                *[a for k, v in self.env.items() for a in ("-e", f"{k}={v}")],
            ]
            for host, cont, mode in self.mounts:
                args += ["-v", f"{host}:{cont}:{mode},U" if mode == "rw" else f"{host}:{cont}:ro"]
            for attempt in range(40):  # every process shares ~64 userns slots: wait for one rather than fail the episode
                code, out = await run([*args, self.image, "sleep", "infinity"], 120)
                if not code or "not enough unused IDs" not in out:
                    break
                await run(["podman", "rm", "-f", self.name], 60)
                await asyncio.sleep(5 + 10 * random.random())
            if code:
                raise RuntimeError(f"podman run failed: {out[-400:]}")
            self.up = True
            if self.files:
                code, out = await run(
                    ["podman", "exec", "-i", "-u", "root", self.name, "tar", "-C", self.cwd, "-xf", "-"],
                    60,
                    stdin=_tar({k: v.encode() for k, v in self.files.items()}),
                )
                if code:
                    raise RuntimeError(f"sandbox copy failed: {out[-400:]}")
            ro = " ".join(f"'{p}'" for p in self.readonly)
            setup = f"chown -R {UID} {self.cwd}" + (f" && cd {self.cwd} && chown root:root {ro} && chmod 444 {ro}" if ro else "")
            code, out = await run(["podman", "exec", "-u", "root", self.name, "sh", "-c", setup], 60)
            if code:
                raise RuntimeError(f"sandbox setup failed: {out[-400:]}")
            return self
        except BaseException:
            await self.close()
            raise

    async def __aexit__(self, *exc):
        await self.close()

    async def close(self):
        if self.up:
            await run(["podman", "rm", "-f", "-t", "0", self.name], 60)
            self.up = False
        _slots.release()

    async def sh(self, cmd: str, timeout=30, stdin: str | None = None, user=UID, max_out=8000) -> tuple[int, str]:
        code, out = await run(
            ["podman", "exec", "-i", "-u", user, "-w", self.cwd, self.name, "timeout", "-k", "5", str(timeout), "bash", "-lc", cmd],
            timeout + 30,
            stdin=stdin.encode() if stdin is not None else b"",
        )
        if max_out and len(out) > max_out:
            out = out[: max_out // 2] + "\n...[truncated]...\n" + out[-max_out // 2 :]
        return code, out

    async def snapshot(self, max_bytes=200_000) -> dict[str, str]:
        """Text files under the work dir after the episode (skips VCS/cache dirs and large or binary files)."""
        files = {}
        p = await asyncio.create_subprocess_exec(
            "podman",
            "exec",
            "-u",
            "root",
            self.name,
            "tar",
            "-C",
            self.cwd,
            "-cf",
            "-",
            "--exclude=.git",
            "--exclude=__pycache__",
            "--exclude=.pytest_cache",
            ".",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        raw, _ = await p.communicate()
        with tarfile.open(fileobj=io.BytesIO(raw)) as t:
            for m in t.getmembers():
                if m.isfile() and m.size < max_bytes:
                    try:
                        files[m.name.removeprefix("./")] = t.extractfile(m).read().decode()
                    except UnicodeDecodeError:
                        pass
        return files
