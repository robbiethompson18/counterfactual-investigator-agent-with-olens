# wsb: WorkspaceBench families as text-only baselines

Runs a plain model (no lens, no activations) on a WorkspaceBench family and scores it with
WorkspaceBench's own judge, so the result is comparable with the O-lens numbers in
[camilablank/workspace-bench](https://github.com/camilablank/workspace-bench) (MIT, Camila Blank).
Only `user_modeling` so far.

```bash
source .envrc.local && uv run python -m wsb.user_modeling --n 3 --limit 10   # 5 selfdescribe + 5 synthsys, bank order
uv run python -m wsb.user_modeling --items um-sd-country-00,um-ss-continent-02
uv run python -m wsb.user_modeling --n 3 --subfamily synthsys                # the 53 synthsys items
```

- `banks/user_modeling.json`: their 100 items verbatim (`evals/user_modeling/items.json` at commit
  `86612fa`) under a `source` header. Never import or run workspace-bench; copy what you need.
- `user_modeling.py`: the responder (`claude-sonnet-5-5` on the Anthropic API, effort `medium`,
  adaptive thinking, `--n` samples per item, each item's system + user turn sent unchanged), scoring and the
  CLI. Every run goes through `runs.py` (`data/runs/wsb_user_modeling/<date>_<slug>/`:
  `responses.jsonl`, `judged.jsonl`, `summary.json`, `meta.json`, `expview/`) and is synced to HF at
  the end.
- `judge.py`: their 6-way identification MC, ported faithfully: gold + 4 same-class distractors
  (seeded draw and shuffle from the whole bank, topped up from `UM_OPTION_SUPPLEMENT`) + "cannot
  tell from the readout", prompt `um-v2`, their JSON schema; checked item-for-item against their
  `tests/golden/user_modeling_options.json` and `user_modeling_prompt.txt`. Also a cheap regex hit:
  an `intermediates` alias in the reply (word boundary, case-insensitive).

**The judge is theirs, not ours:** `claude-opus-5-5` at effort `low` with structured output, the
judge and settings of the only public per-item lens rows (`agu18dec/olens_ablations`
`rl.base.off_k0.own.i300`, um-v2, seed 0) and of their own Anthropic route. The repo default judge
(Haiku 5.5, medium) would make the numbers incomparable. The post's own headline used Gemini 3.8
Flash (minimal); that route was dropped on 2026-10-28 when our Gemini billing died. The judge prompt talks about an
"oracle lens" readout; here the "readout" is the model's reply, which is exactly the text-only floor
the lens should beat.

Scoring (`summary.json`, printed as a table by subfamily and overall): per item `pass@1` = mean over
samples, `pass@n` = any sample; `inferred` = picked gold with basis `inferred_characterization`
(their headline), `correct` = picked gold regardless of basis, `regex` = alias hit. Refusals
(responder or judge) score as fails and are counted under `refused`. Token counts are summed from
`usage`.
