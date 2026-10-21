"""Client for the shared Qwen3.6-27B vLLM server (runpod/CLAUDE.md). Sample Qwen through this, not a hand-rolled URL.

    from counterfactual_investigator_agent_with_olens.qwen import MODEL, client
    r = await client().chat.completions.create(model=MODEL, messages=..., extra_body={"chat_template_kwargs": {"enable_thinking": False}})

Sampling is the caller's choice; the server's default is Qwen's generation config.
"""

import os
import socket

import httpx2
from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

MODEL = "qwen"

# A non-streaming request is silent while vLLM generates (minutes for a long thinking turn), and something between us and
# RunPod drops idle TCP connections. Keepalives keep them open. macOS spells the idle-time option TCP_KEEPALIVE.
_KEEPIDLE = getattr(socket, "TCP_KEEPIDLE", None) or socket.TCP_KEEPALIVE
_SOCKET_OPTIONS = [
    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    (socket.IPPROTO_TCP, _KEEPIDLE, 30),
    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 15),
    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 8),
]


def client(timeout: float = 6 * 3600) -> AsyncOpenAI:
    """AsyncOpenAI for $QWEN_VLLM_URL. The default timeout covers 100k-token thinking traces under load."""
    transport = httpx2.AsyncHTTPTransport(socket_options=_SOCKET_OPTIONS, limits=httpx2.Limits(max_connections=1000), retries=3)
    return AsyncOpenAI(
        base_url=os.environ["QWEN_VLLM_URL"],
        api_key=os.environ["QWEN_VLLM_API_KEY"],
        timeout=timeout,
        http_client=DefaultAsyncHttpx2Client(transport=transport),
    )
