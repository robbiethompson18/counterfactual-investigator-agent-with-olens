# CHIVE on the WeirdChat rollouts

Runs [CHIVE's investigation pipeline](https://github.com/adamkarvonen/chive) (vendored at
`../vendor/chive`) on the Qwen3.6-27B patterns of
[Transluce/WeirdChat](https://huggingface.co/datasets/Transluce/WeirdChat). WeirdChat already found
and judged the behaviours, so it stands in for CHIVE's prompt pool and screening filter. An
investigator agent then edits each prompt, resamples Qwen3.6-27B 64 times per edit, and has a grader
apply WeirdChat's own rubric to every sample.

Plan and decisions:
[CHIVE on WeirdChat plan (Claude Doc)](https://claude.ai/code/artifact/1cc7df0d-084d-4b94-b4a4-994baa74ecfd).

## What differs from stock CHIVE

- **Prompts:** WeirdChat prompts with a match rate of at least 30% over its 64 samples (47 prompts,
  9 behaviours); `--pilot` takes the median-rate prompt of each behaviour (9 prompts).
- **Samples:** 64 per prompt and per experiment (CHIVE uses 10 or 30), replies capped at 2,560
  tokens (CHIVE: 500; over half of WeirdChat's replies are longer than ~500 tokens). The investigator
  sees 5 samples in full per experiment; the grader counts all 64.
- **Classifier:** each behaviour's WeirdChat transcript rubric replaces the screener's
  `classifier_question` (`run.py`), so rates mean what WeirdChat's judge meant.
- **Screener:** told which behaviour WeirdChat flagged; its interest score no longer gates
  (`stage3_min_score=1`).
- **Claims:** `claims.py` labels each experiment with an odds-ratio rule instead of CHIVE's 50/15
  point gate: mover = odds ratio >= 3 either way and Fisher p < 0.01; inert = odds ratio <= 1.5.
- **Models:** Sonnet 5.5 (`anthropic:claude-sonnet-5-5`) for investigator, grader and judge.
  `run.py` turns CHIVE's thinking budgets into adaptive thinking plus an effort level, since Claude
  5-generation models reject `budget_tokens`. Qwen3.6-27B-FP8 on our own vLLM H200 is the target.

## Run

```bash
cd weirdchat-chive && uv sync
# 1. Run directory (WeirdChat mirrored locally, e.g. by weirdchat-viewer's README)
uv run python build_run.py --wc-data PATH/TO/WeirdChat --out data/runs/wc30_pilot --pilot
# 2. Target server: one H200, hard cap 3.75 h, then autodown
uvx --from 'skypilot[runpod]==0.13.0' sky launch -c wc-chive -y -i 5 --down sky/vllm_h200.yaml
ssh -N -L 8000:127.0.0.1:8000 wc-chive &
curl -s localhost:8000/v1/models   # wait until it lists Qwen/Qwen3.6-27B-FP8
# 3. Stages 1-4 (resumable: rerun the same command after an interruption)
ANTHROPIC_API_KEY=... uv run python run.py --config data/runs/wc30_pilot/config.json
# 4. Label experiments
uv run python claims.py data/runs/wc30_pilot
```

Smoke test without a GPU: `build_run.py ... --n-samples 16 --target-llm openrouter:qwen/qwen3.6-27b
--reuse-wc-samples`, then cap `stage2_n`/`stage3_n` at 1 in its `config.json`.

## Expected time and cost (estimates, not yet measured)

Assumes ~750-token replies on average, ~12 experiments per investigation, and vLLM throughput of
3.5k to 7k output tokens/s on one H200. The low end is what CHIVE measured per GPU for its own
runs; the high end is a guess for this model at high batch.

| Run | Target tokens | H200 wall-clock | H200 cost ($5.29/h) | Sonnet 5.5 API |
| --- | ---: | --- | ---: | ---: |
| Pilot, 9 prompts | ~6M | ~1 to 1.5 h, including ~30 min cold start | ~$6 to $8 | ~$20 to $25 |
| Full, 47 prompts | ~29M | ~2 to 3 h, including cold start | ~$11 to $16 | ~$100 to $130 |

The same target tokens on OpenRouter's Qwen3.6-27B ($0.25 in, $2 out per million) would cost about
$15 for the pilot and $65 for the full run. The pilot is latency-bound: each experiment waits for
its slowest 2,560-token reply, then a grader call, then an agent turn, so an investigation takes
about 40 minutes whatever the GPU count. The full run is GPU-bound; a second H200 roughly halves
its generation time.
