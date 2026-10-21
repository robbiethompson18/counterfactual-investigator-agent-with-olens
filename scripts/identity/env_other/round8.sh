while pgrep -f round7.sh > /dev/null; do sleep 20; done
uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/env_other/run_probes.py 32 '^(oai_full_uswest|oai_full_frankfurt|TM_hangzhou|TM_uswest)$' direct,indirect
