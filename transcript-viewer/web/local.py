"""Backend on this machine's own GPU: pod/server.py run by web/run_lens.sh on 127.0.0.1:8802.
Same interface as web/pod.py's Pod, minus launching (start/stop are no-ops)."""

import json
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8802"


class Local:
    def __init__(self):
        self.server = {}

    def _get(self):
        try:
            with urllib.request.urlopen(URL + "/status", timeout=5) as r:
                self.server = json.loads(r.read())
        except Exception:  # noqa: BLE001  (server/worker boundary: report and keep going)
            self.server = {}
        return self.server

    def ready(self):
        return bool(self._get().get("ready"))

    def status(self):
        st = self._get()
        state = "ready" if st.get("ready") else ("starting" if st else "off")
        detail = "" if st else "lens server not running: web/run_lens.sh"
        return {"state": state, "detail": detail, "elapsed": st.get("up"), "server": st, "local": True, "idle_min": None, "max_hours": None}

    def start(self):
        return False

    def stop(self):
        return False

    def call(self, path, body=None, timeout=7200):
        req = urllib.request.Request(
            URL + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(json.loads(e.read() or b"{}").get("error", f"HTTP {e.code}"))
