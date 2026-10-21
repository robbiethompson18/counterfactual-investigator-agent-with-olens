# Round 3 chain (sequential: run_probes rewrites probes.jsonl)
set -e
R="uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/env_other/run_probes.py"
$R 32 '^(oai_plus_user|oai_full_dalle|oai_template_slots|oai_model_spec|oai_as_an_ai_prior)$' direct,indirect,model
$R 32 '^(bare|sys_helpful|oai_full|oai_funcs_only|oai_harmony_funcs|oai_custom_gpt|oai_canvas_full|oai_voice)$' model
$R 32 '^(bare|sys_helpful|code|oai_full|oai_funcs_only|oai_harmony_funcs|oai_custom_gpt|oai_full_hangzhou|zh_sys_helpful|lang_zh|zh_code|zh_cloud_doc|en_cloud_doc|oai_full_zh_bio)$' behaviour
