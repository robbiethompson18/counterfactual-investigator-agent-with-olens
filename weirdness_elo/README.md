# weirdness_elo: rank CHIVE behaviours by how weird they are

CHIVE finds behaviours a model shows reliably on real prompts. This ranks them by how unexpected
they are, from pairwise LLM judgements ("which of these two behaviours would an experienced LLM
researcher find more surprising?"), fit with Bradley-Terry and reported as Elo. The point is to pick
the strangest behaviours, the ones least predictable from the prompt or common model habits, as
targets for the investigator agent.

The current run in
[Robbie-Kylie-NN-sprint/weirdness-elo](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/weirdness-elo)
(private) ranks 1,802 behaviours from 23,315 judgements, about half by Sonnet and half by
Qwen3.6-27B. Each judge has its own first-position bias term. Ranks are still noisy: the mean 90%
rank CI is about 440 places wide.

## Data

- `data/datasets/`: the CHIVE eval runs from
  [adamkarvonen/chive-data](https://huggingface.co/datasets/adamkarvonen/chive-data), about 26 GB
  and gitignored.
- `data/forensics/`: Qwen3.6-27B incidents from
  [agu18dec/chive-forensics](https://huggingface.co/datasets/agu18dec/chive-forensics).
- Optionally, CHIVE runs on WeirdChat from the `weirdchat-chive` branch. Set `WEIRDCHAT_CHIVE_RUNS`
  to their `data/runs` directory.

## Pipeline

1. `build_items.py`: verified investigations (CHIVE verification score ≥ 7) from the 9 eval runs go
   to `run/items.json` (judge-facing; a stratified sample of 1,500) and `run/transcripts/`
   (dashboard only). `add_qwen27b.py` adds the Qwen3.6-27B behaviours.
2. `elo.py pair run --round N`: matchmaking (neighbours by current rating, information-weighted)
   writes judge batches to `run/batches/`. `elo.py ingest run` reads the answers and
   `elo.py fit run` fits the model: P(i beats j | i shown first) = σ(θᵢ − θⱼ + γ_judge), with a
   Gaussian prior on θ and a Laplace covariance for the rank CIs.
3. Judges:
   - `drive.sh FROM TO` runs rounds with headless Sonnet (`judge_one.sh`, which calls `claude -p`
     restricted to Read and Write).
   - `qwen_tourney.py` runs adaptive rounds against a Qwen3.6-27B vLLM endpoint on `localhost:8000`.
     It stops when the mean 90% rank CI is under 150 places wide, the split-half Spearman is at
     least 0.9 and top-100 churn is under 5.
   - `judge_qwen.py` re-judges existing pairs with Qwen for judge agreement.
4. `export.py`: `run/ranking_full.json` and `run/ranking.csv`. `split_half.py` checks reliability
   and `q27_report.py` shows where the Qwen3.6-27B items land.

`./serve.sh` serves the dashboard (`static/index.html`) on port 8807. It re-reads `run/` on every
request.

Uses `numpy`, `scipy` and `httpx`. Put `run/` in place from the HF dataset with
`hf download Robbie-Kylie-NN-sprint/weirdness-elo --repo-type dataset --local-dir run`.
