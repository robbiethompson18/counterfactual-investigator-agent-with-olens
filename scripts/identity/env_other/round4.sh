# Sequential (run_probes rewrites probes.jsonl): ablation, cross-task, then behaviour, then remaining OpenAI attempts.
set -e
R="uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/env_other/run_probes.py"
$R 32 '^abl_' direct,indirect
$R 32 '^task_' direct,indirect
$R 32 '^(bare|code|oai_full|oai_funcs_only|abl_HPTMF|oai_custom_gpt|oai_full_hangzhou|zh_sys_helpful|lang_zh|zh_code|zh_cloud_doc)$' behaviour
$R 32 '^(oai_plus_user|oai_template_slots|oai_model_spec|oai_as_an_ai_prior)$' direct,indirect
