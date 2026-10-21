while pgrep -f round4.sh > /dev/null; do sleep 20; done
uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/env_other/run_probes.py 32 '^abl_(TM|bM|ywM|b|TMF)$' direct,indirect
