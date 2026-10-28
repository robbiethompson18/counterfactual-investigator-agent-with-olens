# wsb: WorkspaceBench families as text-only baselines

Runs a plain model (no lens, no activations) on a WorkspaceBench family and scores it with
WorkspaceBench's own judge, so the result is comparable with the O-lens numbers in
[camilablank/workspace-bench](https://github.com/camilablank/workspace-bench) (MIT, Camila Blank).
`user_modeling`, `jailbreak_recognition`, `basic_readout` and `knowledge_cutoff` so far.

```bash
source .envrc.local && uv run python -m wsb.user_modeling --n 3 --limit 10   # 5 selfdescribe + 5 synthsys, bank order
uv run python -m wsb.user_modeling --items um-sd-country-00,um-ss-continent-02
uv run python -m wsb.user_modeling --n 3 --subfamily synthsys                # the 53 synthsys items
```

- **Banks are data: they live on HF, never in git.** `banks.load(family)` reads
  `data/banks/wsb/<family>.json`, downloading it from `banks/wsb/` in the sprint HF dataset on first
  use, so any checkout works. `build_banks.py` rebuilds them from the pinned
  `vendor/workspace-bench` submodule (their `evals/<family>/items.json` verbatim under a `source`
  header) and `--upload`s them. Reading workspace-bench's data files is fine; never import or run
  its code.
- `user_modeling.py`: the responder (`claude-sonnet-5-5` on the Anthropic API, effort `medium`,
  adaptive thinking, `--n` samples per item, each item's system + user turn sent unchanged), scoring
  and the CLI. Every run goes through `runs.py` (`data/runs/wsb_user_modeling/<date>_<slug>/`:
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
Flash (minimal); that route was dropped on 2026-10-28 when our Gemini billing died. The judge prompt
talks about an "oracle lens" readout; here the "readout" is the model's reply, which is exactly the
text-only floor the lens should beat.

Scoring (`summary.json`, printed as a table by subfamily and overall): per item `pass@1` = mean over
samples, `pass@n` = any sample; `inferred` = picked gold with basis `inferred_characterization`
(their headline), `correct` = picked gold regardless of basis, `regex` = alias hit. Refusals
(responder or judge) score as fails and are counted under `refused`. Token counts are summed from
`usage`.

`jailbreak.py` replays their anti-text-inversion gate for `jailbreak_recognition` rather than having
the model answer the chat: Sonnet 5.5 is shown the Qwen3.6-27B chat render through the last user
turn's `<|im_end|>` and asked "what would a language model be thinking about on this token?" (their
`open_model` framing, API-default thinking as their gate called it), and their `jb-v1` judge (Opus
5.5, the gate's judge) labels the answer as one readout. Its HF bank adds that render per item
(`prefix`, `token`), made by `build_banks.py` with the real tokenizer, which asserts the token
positions against their bank. Prompts are byte-identical to theirs. pass@1 is reported with refusals
dropped (their rule), as pass and as fail. Their parser drops 0-based verdict indices, which we copy
faithfully; see camila-blank/workspace-bench#107.

```bash
uv run python -m wsb.jailbreak --n 3        # all 86 items x 3 draws, ~5 min
```

`basic_readout.py` has Sonnet 5.5 complete all 100 items (35 entity "The athlete Muhammad Ali plays
the sport of", 33 processing "10 - 1 =", 32 implicit "What is your favorite card game? Answer in one
word.", gold = Qwen3.6-27B's gated answer) with `--persona qwen` (system prompt "Answer as if you
are Qwen 3.6 27B") or `--persona self` (no system prompt). Implicit items get their answer stem "My
one-word answer is:" at the end of the user turn, since Sonnet with thinking refuses an assistant
prefill. Their single-token bank judge (`bank-2026-10-10`, pasted verbatim in the module,
quote-verified) runs on Opus 5.5 at effort low; `lens` is the per-item verdict of the lens arm
(`agu18dec/olens_ablations` `rl.base.off_k0.own.i300`, any of 11 layers). `summary.json` has
judge/regex pass@1 and pass@n and the lens rate by subfamily.

```bash
uv run python -m wsb.basic_readout --persona qwen --n 5 --slug sonnet-as-qwen   # 500 Sonnet + 500 Opus calls
uv run python -m wsb.basic_readout --persona self --n 5 --slug sonnet-self
```

`knowledge_cutoff.py` runs **stock Qwen3.6-27B itself** (`qwen/qwen3.6-27b` on OpenRouter, thinking
off, T 1, one provider pinned, DeepInfra by default) on the 109 frozen items whose label is Qwen's
own stable wrong answer about a post-cutoff event. It reproduces the "Baseline C" line their family
README cites (stock Qwen on the text ties the lens at 0.486) that the public repo does not ship.
`--arm describe` is their stock-Qwen prompt-only floor recipe (the chat render through reply_start
under "what would a language model be thinking about on this token?", `--k` samples as one cell);
`--arm answer` just lets Qwen answer the user turn. Scoring is a port of their
`knowledge_cutoff/score.py` strict-win regex (gold named by strictly more samples than every other
option, exclusive longest match); `summary.json` also has `any_gold`, the per-sample `gold_rate` and
`truth_rate`, by `label_type` (stale / confab_plausible) and stratum, with item bootstrap CIs. The
bank adds `prefix` / `token` per item (`build_banks.py cutoff_render`). The 41 nocot items are not
included (their text is in a password-protected zip). No judge: the family's scorer of record at
freeze was the regex.

```bash
uv run python -m wsb.knowledge_cutoff --arm answer --k 10     # ~1,100 short Qwen calls, under $1
uv run python -m wsb.knowledge_cutoff --arm describe --k 5
```
