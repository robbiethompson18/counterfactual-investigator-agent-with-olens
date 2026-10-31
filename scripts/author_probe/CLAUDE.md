# author_probe: what Qwen thinks about a passage's author

Qwen3.6-27B (thinking off) answers multiple-choice questions about the author of a passage of
SynthPAI comments (sex, age band, education, relationship status, income level) asked in the first
person, and linear probes read those beliefs from the residual stream of a passage-only prompt,
before any question is asked. Results: `docs/author-mcq-probe.md` (NULL). The SynthPAI parts
(bank/fit/ablate/dsv4/view) are the retired pilot; `blog.py` is the Blog Authorship Corpus pipeline
behind `cfbench/evals/author_blog_drop_inline`.

| file                               | where         | what                                                                                                                                                                                                                                                                                                     |
| ---------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bank.py`                          | bramble       | `build`: passages + questions (`data/banks/author_mcq/`, HF `banks/author_mcq/`); `prompts RUN`: token ids for the GPU pass; `qprompts RUN [--order 1]`: answer prompts with question-side sites                                                                                                         |
| `dsv4.py`                          | bramble       | DeepSeek V4 Flash (OpenRouter, effort medium) answers the same questions, resumable                                                                                                                                                                                                                      |
| `fit.py`                           | any           | `answers RUN`: Qwen's order-averaged letter pmf; `probes RUN [--sites ...]`: per (site, layer, attribute, target) logistic probes + age/education ridge, GroupKFold by author; `qprobes RUN`: the `last` site with both orders pooled, cross-order test, letter control; `bow RUN`: TF-IDF text baseline |
| `ablate.py`                        | bramble + GPU | `prep RUN --src SRC`: leave-one-comment-out inputs for confident pairs; `analyze RUN --src SRC`: stability, evidence comments, frozen-probe vs Qwen shift correlation, key passages (`ablate.md`); `view RUN`: expview                                                                                   |
| `view.py`                          | bramble       | expview items (one per passage: Qwen's prompts + pmf, DSv4 replies)                                                                                                                                                                                                                                      |
| `blog.py`                          | bramble + GPU | Blog Authorship Corpus version (the result): `build [--append]`, `prompts [--skip]`, `answers`, `del-prep`, `del-analyze`, `view` (per-question expview with reads)                                                                                                                                      |
| `merge_reads.py`                   | bramble       | merge a reader's output into a batch reads run (dedup, positions unioned)                                                                                                                                                                                                                                |
| `score_drop.py`, `pvalues_drop.py` | bramble       | drop-eval scores; pooled paired AUROC / raw / Platt-recalibrated Brier with author-clustered CIs and one-sided p                                                                                                                                                                                         |

```bash
uv run python -m scripts.author_probe.bank build --upload
uv run python scripts/runs.py new author_mcq_probe <slug> -d "..." --model Qwen/Qwen3.6-27B
uv run --with transformers --with jinja2 python -m scripts.author_probe.bank prompts RUN
# GPU (scripts/sky/hs_positions.yaml): passage prompts at every layer, answer prompts for the letter log-probs only
python scripts/hs_at_positions.py inputs_passage.jsonl passage --layer-stride 1
python scripts/hs_at_positions.py inputs_answer.jsonl answer --layers 64 --chunk 2000
python scripts/hs_at_positions.py inputs_qsites.jsonl qsites --layer-stride 1 --chunk 1372      # part-K = attribute K
python scripts/hs_at_positions.py inputs_qsites_o1.jsonl qsites_o1 --layer-stride 1 --chunk 1372
# back on bramble: passage/{text_end,im_end}.npy + layers.json -> data/probe_cache/<run>/, logp.jsonl -> RUN/{passage,answer}_logp.jsonl
uv run python -m scripts.author_probe.fit answers RUN
uv run --with torch --with scikit-learn python -m scripts.author_probe.fit probes RUN --device cuda   # GPU: ~20 min per site pair
uv run ... fit probes RUN --device cuda --sites qmark --targets label   # qsites/ linked under data/probe_cache/<run>/
uv run ... fit qprobes RUN --device cuda
uv run --with scikit-learn python -m scripts.author_probe.fit bow RUN
uv run python -m scripts.author_probe.view RUN && uv run python scripts/runs.py finish RUN
```

- Prompt format (Kylie, 2026-10-29): the probe prompt is the raw passage as the only user turn; the
  answer prompt is the same user turn with `\n\n` + a first-person question ("Based on what I wrote
  above, ...?"), lettered options and `Answer with the letter only.` appended, thinking-off
  template. So the answer prompt's tokens match the probe prompt's up to the passage's last token.
  Two option orders (canonical, reversed), pmf renormalised over the letters and averaged in
  canonical order. (A v0 run, `2026-10-29_synthpai-qwen36`, had a framing line and third-person
  questions.)
- Sites: `text_end` = the passage's last token (the last of several when it ends in an emoji),
  `im_end` = the `<|im_end|>` closing the user turn (passage-only prompt); `qmark` = the token
  holding the question's "?" (may merge with the following newline; before any option, so identical
  in both orders), `last` = the answer prompt's final token. Layer L = HF `hidden_states[L]` (0 =
  embeddings, 64 = final norm input).
- Passages of one author overlap (SynthPAI authors are short): always hold out by author.
