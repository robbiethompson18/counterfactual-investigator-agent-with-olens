while pgrep -f "round7.sh|round8.sh" > /dev/null; do sleep 20; done
R="uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/env_other/run_probes.py"
$R 32 '^(x_TMnl_zh|TMnl)$' direct,indirect,behaviour
