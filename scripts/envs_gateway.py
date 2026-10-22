"""Model gateway for agent-interp-envs rollouts (vendor/agent-interp-envs).

The harness's `fireworks` provider talks to `$ROLLOUT_GATEWAY/fireworks/inference/v1` with a placeholder
key when ROLLOUT_GATEWAY is set (its credential-sidecar mode). This server is that sidecar: it routes each
chat-completions request by model name, so rollout containers hold no keys and need no internet.

- QWEN_MODEL (default Qwen/Qwen3.6-27B) -> our vLLM server at QWEN_URL (never OpenRouter).
- anything else -> OpenRouter, with the Fireworks-style `reasoning_effort` mapped to OpenRouter's `reasoning`.

A turn that runs out of tokens without answering (a degenerate reasoning loop) is resampled up to LOOP_RESAMPLES
times before it reaches the harness. Every upstream call is logged as one JSON line to GATEWAY_LOG (model, status,
latency, token counts, and `looped` for a resampled turn), so loop rates can be reported per env.

Stdlib only, so it runs in a bare python image:
    docker run -d --name envs-gateway --network bridge -v ~/envs-logs:/logs -v $PWD/scripts:/app \
      -e QWEN_URL=http://IP:PORT/v1 -e QWEN_API_KEY=... -e OPENROUTER_API_KEY=... python:3.12-slim python /app/envs_gateway.py
    docker network connect envnet envs-gateway   # the --internal network the rollout containers run on
"""

import http.client
import json
import os
import socket
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.6-27B")
QWEN_URL = os.environ.get("QWEN_URL", "").rstrip("/")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LOG_PATH = os.environ.get("GATEWAY_LOG", "/logs/gateway.jsonl")
# Qwen3.6's recommended thinking-mode sampling; a request's own values win.
QWEN_DEFAULTS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32768}
TIMEOUT_S = 1800  # matches the harness client's timeout
CONNECT_TIMEOUT_S, CONNECT_ATTEMPTS = 10, 5
LOOP_RESAMPLES = 2  # extra samples of a turn that ran out of tokens without answering (see is_degenerate_loop)


def route(body: dict) -> tuple[str, dict, dict]:
    """(upstream url, headers, rewritten body) for one chat-completions request."""
    effort = body.pop("reasoning_effort", None)
    if body.get("model") == QWEN_MODEL:
        return f"{QWEN_URL}/chat/completions", {"Authorization": f"Bearer {QWEN_API_KEY}"}, {**QWEN_DEFAULTS, **body}
    if effort:
        body["reasoning"] = {"effort": effort}
    return "https://openrouter.ai/api/v1/chat/completions", {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, body


def post(url: str, data: bytes, headers: dict) -> tuple[int, bytes]:
    """POST with a short connect timeout and a long read timeout. The AWS box -> RunPod TCP path sometimes
    drops a SYN, and urllib's single timeout would wait out the kernel's ~130 s connect timeout each time."""
    parts = urllib.parse.urlsplit(url)
    connection_class = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    for attempt in range(CONNECT_ATTEMPTS):
        connection = connection_class(parts.netloc, timeout=CONNECT_TIMEOUT_S)
        try:
            connection.connect()
        except OSError:
            if attempt == CONNECT_ATTEMPTS - 1:
                raise
            continue
        assert connection.sock is not None
        connection.sock.settimeout(TIMEOUT_S)
        # A non-streaming request is silent while vLLM generates (a 32k-token turn takes ~9 min); something on the
        # AWS -> RunPod path drops idle connections after a few minutes, so keep the connection visibly alive.
        connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # The tuning knobs are Linux-only (the gateway runs in a Linux container); getattr keeps macOS type checks quiet.
        for option, value in (("TCP_KEEPIDLE", 30), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 8)):
            connection.sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, option), value)
        connection.request("POST", parts.path, body=data, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()
    raise AssertionError("unreachable")


def is_degenerate_loop(payload: bytes) -> bool:
    """A turn that hit max_tokens with no answer and no tool call. Qwen3.6 sometimes repeats one reasoning
    paragraph until the cap; the harness would read the empty turn as the end of the episode."""
    try:
        choice = json.loads(payload)["choices"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return False
    message = choice.get("message") or {}
    return choice.get("finish_reason") == "length" and not (message.get("content") or "").strip() and not message.get("tool_calls")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # health check
        self._send(200, b'{"ok": true}')

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            return self._send(404, b'{"error": "only chat/completions is routed"}')
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        url, headers, body = route(body)
        for attempt in range(LOOP_RESAMPLES + 1):
            start = time.time()
            try:
                status, payload = post(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
            except OSError as err:  # upstream down: 502 so the harness retries
                status, payload = 502, json.dumps({"error": f"upstream unreachable: {err}"}).encode()
            looped = status == 200 and is_degenerate_loop(payload)
            self._log(body["model"], status, time.time() - start, payload, looped)
            if not looped or attempt == LOOP_RESAMPLES:
                break
        self._send(status, payload)

    def _send(self, status: int, payload: bytes):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _log(self, model: str, status: int, latency: float, payload: bytes, looped: bool = False):
        record: dict = {"t": round(time.time()), "model": model, "status": status, "latency_s": round(latency, 1)}
        if looped:
            record["looped"] = True
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
