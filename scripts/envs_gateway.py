"""Model gateway for agent-interp-envs rollouts (vendor/agent-interp-envs).

The harness's `fireworks` provider talks to `$ROLLOUT_GATEWAY/fireworks/inference/v1` with a placeholder
key when ROLLOUT_GATEWAY is set (its credential-sidecar mode). This server is that sidecar: it routes each
chat-completions request by model name, so rollout containers hold no keys and need no internet.

- QWEN_MODEL (default Qwen/Qwen3.6-27B) -> our vLLM server at QWEN_URL (never OpenRouter).
- anything else -> OpenRouter, with the Fireworks-style `reasoning_effort` mapped to OpenRouter's `reasoning`.

Every request is logged as one JSON line to GATEWAY_LOG (model, status, latency, token counts).

Stdlib only, so it runs in a bare python image:
    docker run -d --name envs-gateway --network bridge -v ~/envs-logs:/logs -v $PWD/scripts:/app \
      -e QWEN_URL=http://IP:PORT/v1 -e QWEN_API_KEY=... -e OPENROUTER_API_KEY=... python:3.12-slim python /app/envs_gateway.py
    docker network connect envnet envs-gateway   # the --internal network the rollout containers run on
"""

import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.6-27B")
QWEN_URL = os.environ.get("QWEN_URL", "").rstrip("/")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LOG_PATH = os.environ.get("GATEWAY_LOG", "/logs/gateway.jsonl")
# Qwen3.6's recommended thinking-mode sampling; a request's own values win.
QWEN_DEFAULTS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32768}
TIMEOUT_S = 1800  # matches the harness client's timeout


def route(body: dict) -> tuple[str, dict, dict]:
    """(upstream url, headers, rewritten body) for one chat-completions request."""
    effort = body.pop("reasoning_effort", None)
    if body.get("model") == QWEN_MODEL:
        return f"{QWEN_URL}/chat/completions", {"Authorization": f"Bearer {QWEN_API_KEY}"}, {**QWEN_DEFAULTS, **body}
    if effort:
        body["reasoning"] = {"effort": effort}
    return "https://openrouter.ai/api/v1/chat/completions", {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, body


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # health check
        self._send(200, b'{"ok": true}')

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            return self._send(404, b'{"error": "only chat/completions is routed"}')
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        url, headers, body = route(body)
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}, method="POST"
        )
        start = time.time()
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                status, payload = response.status, response.read()
        except urllib.error.HTTPError as err:
            status, payload = err.code, err.read()
        except (urllib.error.URLError, TimeoutError) as err:  # upstream down: 502 so the harness retries
            status, payload = 502, json.dumps({"error": f"upstream unreachable: {err}"}).encode()
        self._send(status, payload)
        self._log(body["model"], status, time.time() - start, payload)

    def _send(self, status: int, payload: bytes):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _log(self, model: str, status: int, latency: float, payload: bytes):
        record: dict = {"t": round(time.time()), "model": model, "status": status, "latency_s": round(latency, 1)}
        try:
            usage = json.loads(payload).get("usage") or {}
            record |= {"in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens")}
        except (json.JSONDecodeError, AttributeError):
            pass
        if status != 200:
            record["error"] = payload[:300].decode(errors="replace")
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_message(self, format, *args):  # the JSONL log replaces the default stderr access log
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
