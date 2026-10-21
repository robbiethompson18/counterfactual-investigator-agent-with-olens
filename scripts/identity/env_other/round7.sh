set -e
R="uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/env_other/run_probes.py"
$R 32 '^(x_TM_zh|x_bare_encn|x_zhsys_encn|x_TM_encn|abl_TM)$' behaviour
$R 32 '^(x_TM_zh)$' direct,indirect
uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/env_other/judge_cot.py behaviour
