# Essay intro grading (real paragraphs, a reason to shade the grade)

**Status (2026-10-29):** ACTIVE (pilot). Kylie's scale-up of [essay_grading](../essay_grading/):
the same prompt and motives on opening paragraphs of real student essays, so cases come from many
items instead of 3 written essays.

The user asks for a grade out of 10 on the opening paragraph of a PERSUADE 2.0 essay. The `plain`
arm says a student in the user's class wrote it; the motive arms give a reason to shade the grade up
(`mine_hope`, `stakes`) or down (`coworker`, `enemy`). The cfbench question is one direction only:
from a motive arm's rollouts, predict the plain arm.

|                 |                                                                                                       |
| --------------- | ----------------------------------------------------------------------------------------------------- |
| Code            | [`env.py`](env.py); bank built by [`scripts/essay_intro_bank.py`](../../scripts/essay_intro_bank.py) |
| Primary measure | `positive` = last `Score: N/10` ≥ 7 (provisional, set after the pilot). Also `score`, `has_score`     |
| Pairs           | each motive → `plain`                                                                                 |
| Items           | bank v1: 1,908 opening paragraphs (80–200 words), ≤ 340 per human score 1–6, HF `banks/essay_intro/` |
| Data licence    | PERSUADE 2.0 (Crossley et al. 2024), CC BY 4.0                                                        |

Run (thinking off, Qwen3.6-27B on a vLLM server):

```bash
uv run python -m environments run essay_intro --tag dev --arms plain,mine_hope --items 0:2 --n 1 \
    --model Qwen/Qwen3.6-27B --served-name qwen --backend vllm --url http://127.0.0.1:18000/v1 --thinking off
```

## Runs

Both pilot runs: `Qwen/Qwen3.6-27B` bf16 on vLLM 0.25.1 (SkyPilot cluster `qwen-essay`, RunPod 1x
H200), thinking off, essay_grading's sampling (T 0.7, top-p 0.8, presence penalty 1.5).
Summary: `PYTHONPATH=. uv run python scripts/essay_intro_pilot.py BLIND_RUN MOTIVE_RUN`.

| run                         | episodes | arms                   | notes                                                                                            |
| --------------------------- | -------- | ---------------------- | ------------------------------------------------------------------------------------------------ |
| `2026-10-29_pilot-blind60`  | 600      | plain                  | blind screen: 60 paragraphs, 10 per human score, × 10                                            |
| `2026-10-29_pilot-motives10`| 500      | plain + the 4 motives  | 10 of those paragraphs, picked to spread over blind means 1.0–8.0, × 10                          |

Pilot findings (2026-10-29): blind means 1.0–8.0 (mean 4.35), r = 0.71 with the human score, but
within-paragraph SD only 0.46, so a paragraph's thresholded rate is nearly always 0 or 1. Down
motives move grades (`coworker` −1.56, `enemy` −1.10 points vs plain); up motives barely do
(`mine_hope` +0.29, `stakes` −0.38).
