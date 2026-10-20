"""The cloud GPU backend's lifecycle (Start GPU in the viewer), driven from the viewer's host.

start(): `sky launch` sky/pod.yaml if the cluster isn't up (setup is idempotent), then run
pod/server.py as a sky job, set autostop to AUTODOWN_MIN with --down, and open an ssh tunnel to
it. The server exits on its own after IDLE_MIN minutes without reads (hard cap MAX_HOURS), the job
ends, and SkyPilot downs the pod AUTODOWN_MIN later, so nothing depends on the viewer's host staying up.
"""

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKY = ["uvx", "--from", "skypilot[runpod,vast]==0.13.0", "sky"]
CLUSTER = "weirdchat-olens"
IDLE_MIN = 45
MAX_HOURS = 3.75
AUTODOWN_MIN = 5
TUNNEL_PORT = 18802
LOG = ROOT / "outputs/pod.log"


def _run(args, timeout=1800, env=None, capture=False):
    with open(LOG, "a") as f:
        f.write(f"\n[{time.strftime('%H:%M:%S')}] $ {' '.join(args)}\n")
        f.flush()
        if capture:
            r = subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout, env=env, cwd=ROOT)
            f.write(r.stdout[-4000:] + r.stderr[-4000:])
            return r
        return subprocess.run(args, check=False, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env, cwd=ROOT).returncode


def hf_token():
    """HF_TOKEN from the environment, or the file named by HF_TOKEN_FILE."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    f = os.environ.get("HF_TOKEN_FILE")
    if f:
        return Path(f).read_text().strip()
    raise RuntimeError("set HF_TOKEN or HF_TOKEN_FILE (the gated agu18dec/local-workspace whiteners need it)")


class Pod:
    def __init__(self):
        self.state, self.detail, self.since = "off", "", time.time()
        self.tunnel = None
        self.server = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _set(self, state, detail=""):
        self.state, self.detail, self.since = state, detail, time.time()

    def status(self):
        return {
            "state": self.state,
            "detail": self.detail,
            "elapsed": int(time.time() - self.since),
            "server": self.server,
            "idle_min": IDLE_MIN,
            "max_hours": MAX_HOURS,
        }

    def ready(self):
        return self.state == "ready"

    def start(self):
        with self.lock:
            if self.state not in ("off", "error"):
                return False
            self._set("launching", "requesting an H100 (cold boot downloads the 54 GB model, ~15 min)")
        threading.Thread(target=self._bring_up, daemon=True).start()
        return True

    def stop(self):
        with self.lock:
            if self.state in ("off", "stopping"):
                return False
            self._set("stopping", "tearing the pod down")
        threading.Thread(target=self._down, daemon=True).start()
        return True

    def call(self, path, body=None, timeout=1800):
        url = f"http://127.0.0.1:{TUNNEL_PORT}{path}"
        req = urllib.request.Request(
            url, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            out = json.loads(e.read() or b"{}")
            raise RuntimeError(out.get("error", f"HTTP {e.code}"))

    # ------------------------------------------------------------------ internals
    def _cluster_state(self):
        r = _run(SKY + ["status", CLUSTER], 180, capture=True)
        for line in r.stdout.splitlines():
            p = line.split()
            if p and p[0] == CLUSTER:
                for s in ("UP", "INIT", "STOPPED", "AUTOSTOPPING", "STOPPING"):
                    if s in p:
                        return s
        return None

    def _bring_up(self):
        try:
            if self._cluster_state() != "UP":
                env = dict(os.environ, HF_TOKEN=hf_token())
                rc = _run(
                    SKY + ["launch", "-c", CLUSTER, "-y", "--down", "-i", str(AUTODOWN_MIN), "--secret", "HF_TOKEN", "sky/pod.yaml"],
                    3600,
                    env,
                )
                if rc != 0:
                    return self._set("error", f"sky launch failed (rc {rc}); see outputs/pod.log")
            self._connect(start_server=True)
        except Exception as e:  # noqa: BLE001  (server/worker boundary: report and keep going)
            self._set("error", f"{type(e).__name__}: {e}")

    def _server_running(self):
        r = _run(SKY + ["queue", CLUSTER], 180, capture=True)
        return any(" server " in f" {ln} " and "RUNNING" in ln for ln in r.stdout.splitlines())

    def _connect(self, start_server):
        self._set("starting", "loading Qwen3.6-27B + lens on the H100, then warming up kernels (~5 min)")
        if start_server and not self._server_running():
            _run(
                [
                    "rsync",
                    "-az",
                    "--exclude",
                    ".venv",
                    "--exclude",
                    "data",
                    "--exclude",
                    "outputs",
                    "--exclude",
                    ".git",
                    "pod/",
                    f"{CLUSTER}:sky_workdir/pod/",
                ],
                300,
            )
            _run(SKY + ["exec", CLUSTER, "-d", "--name", "server", "bash ~/sky_workdir/pod/run.sh"], 600)
        _run(SKY + ["autostop", "-y", CLUSTER, "-i", str(AUTODOWN_MIN), "--down"], 300)
        self._open_tunnel()
        for _ in range(240):  # up to 40 min
            if self.state != "starting":
                return
            try:
                st = self.call("/status", timeout=10)
                self.server = st
                if st.get("ready"):
                    return self._set("ready")
                self.detail = f"pod server up for {st['up']}s: {st.get('task') or 'loading the model'}"
            except Exception:  # noqa: BLE001  (server/worker boundary: report and keep going)
                if self.tunnel and self.tunnel.poll() is not None:
                    self._open_tunnel()
            time.sleep(10)
        self._set("error", "the pod server didn't come up in 40 min; see `sky logs weirdchat-olens`")

    def _open_tunnel(self):
        if self.tunnel and self.tunnel.poll() is None:
            return
        self.tunnel = subprocess.Popen(
            ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-L", f"{TUNNEL_PORT}:127.0.0.1:8802", CLUSTER],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _close_tunnel(self):
        if self.tunnel:
            self.tunnel.terminate()
            self.tunnel = None

    def _down(self):
        self._close_tunnel()
        _run(SKY + ["down", "-y", CLUSTER], 900)
        self.server = {}
        self._set("off")

    def _loop(self):
        """Adopt a pod that is already up (e.g. after a restart of this server), then keep the
        state honest: a server that exited on idle or a pod SkyPilot downed goes back to off."""
        try:
            if self._cluster_state() == "UP":
                self._connect(start_server=True)
        except Exception as e:  # noqa: BLE001  (server/worker boundary: report and keep going)
            self._set("error", f"{type(e).__name__}: {e}")
        while True:
            time.sleep(30)
            if self.state == "ready":
                try:
                    self.server = self.call("/status", timeout=10)
                except Exception:  # noqa: BLE001  (server/worker boundary: report and keep going)
                    self._open_tunnel()
                    time.sleep(10)
                    try:
                        self.server = self.call("/status", timeout=10)
                    except Exception:  # noqa: BLE001  (server/worker boundary: report and keep going)
                        self._close_tunnel()
                        self.server = {}
                        self._set("off", "the H100 server stopped (idle timeout or pod down)")
            elif (
                self.state == "starting"
                and self.server.get("up") is None
                and time.time() - self.since > 300
                and self._cluster_state() is None
            ):
                self._close_tunnel()
                self._set("off", "pod is gone")
