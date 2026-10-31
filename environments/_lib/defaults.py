"""Explicit environment presets; each env selects its own DEFAULTS."""

# Treat presets as read-only. make_target copies both the defaults and selected sampling settings before applying CLI flags.
QWEN_DEEPINFRA_DEFAULTS = {
    "model": "qwen/qwen3.6-27b",
    "backend": "openrouter",
    "provider": "deepinfra/fp8",
    "thinking": False,
    "effort": "medium",
    "body": "openrouter",
    "sampling_all_models": True,
    "sampling": {
        "on": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
        "off": {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5},
    },
}
