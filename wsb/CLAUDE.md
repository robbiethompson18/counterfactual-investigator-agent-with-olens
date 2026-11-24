# wsb: WorkspaceBench families as text-only baselines

Runs a plain model (no lens, no activations) on a WorkspaceBench family and scores it with
WorkspaceBench's own judge, so the result is comparable with the O-lens numbers in
[camilablank/workspace-bench](https://github.com/camilablank/workspace-bench) (MIT, Camila Blank).
Only `user_modeling` so far.

```bash
source .envrc.local && uv run python -m wsb.user_modeling --n 3 --limit 10   # 5 selfdescribe + 5 synthsys, bank order
uv run python -m wsb.user_modeling --items um-sd-country-00,um-ss-continent-02
uv run python -m wsb.user_modeling --n 3                                      # all 100 items: 300 responder + 300 judge calls
```

- `banks/user_modeling.json`: their 100 items verbatim (`evals/user_modeling/items.json` at commit
  `86612fa`) under a `source` header. Never import or run workspace-bench; copy what you need.
- `user_modeling.py`: the responder (`google/gemini-3.8-flash` via OpenRouter, effort `medium`,
  temperature 1.0, `--n` samples per item, each item's `messages` sent unchanged), scoring and the
  CLI. Every run goes through `runs.py` (`data/runs/wsb_user_modeling/<date>_<slug>/`:
  `responses.jsonl`, `judged.jsonl`, `summary.json`, `meta.json`, `expview/`) and is synced to HF at
  the end.
- `judge.py`: their 6-way identification MC, ported faithfully: gold + 4 same-class distractors
  (seeded draw and shuffle from the whole bank, topped up from `UM_OPTION_SUPPLEMENT`) + "cannot
  tell from the readout", prompt `um-v2`, their JSON schema; checked item-for-item against their
  `tests/golden/user_modeling_options.json` and `user_modeling_prompt.txt`. Also a cheap regex hit:
  an `intermediates` alias in the reply (word boundary, case-insensitive).

**The judge is theirs, not ours:** `google/gemini-3.8-flash` at reasoning effort `minimal`, no
temperature, structured output, on purpose. The repo default judge (Haiku 5.5, medium) would make
the numbers incomparable with WorkspaceBench's published ones. The judge prompt talks about an
"oracle lens" readout; here the "readout" is the model's reply, which is exactly the text-only floor
the lens should beat.

Scoring (`summary.json`, printed as a table by subfamily and overall): per item `pass@1` = mean over
samples, `pass@n` = any sample; `inferred` = picked gold with basis `inferred_characterization`
(their headline), `correct` = picked gold regardless of basis, `regex` = alias hit. Token counts are
summed from `usage` because OpenRouter reports $0 for Gemini on Robbie's BYOK account.

Gemini quota is a shared 10k requests/day: a full run is ~600 requests, so ask before running it.
