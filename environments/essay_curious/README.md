# Essay letter grade with a scenario after the essay

**Status (2026-10-28):** ACTIVE (the first result is NULL). Built for cfbench's answer task: from
the rollouts of an arm with a scenario sentence, predict the letter Qwen gives with no scenario.
Results: [docs/envs/essay_curious.md](../../docs/envs/essay_curious.md).

Qwen3.6-27B (thinking off) grades a ~500-word Common App essay A to F and answers with the letter
only. One sentence after the essay changes per arm, after "I'm curious what you think.": nothing
(`none`), "My best student wrote this." (`best_student`), "I wrote this for admission to Duke."
(`duke`), "I wrote this for admission to Chico State." (`chico`). Because the scenario comes after
the essay, every O-lens read inside the essay is identical across arms; the investigator sees the
scenario arm's reads, and those reads up to `</essay>` are also the `none` arm's.

|                 |                                                                                                                                                           |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Code            | [`env.py`](env.py); essays in [`essays/`](essays/), written for this env by Claude Fable 5.1                                                              |
| Arms            | `none`, `best_student`, `duke`, `chico` (factor `scenario`), the same four as `pm/<scenario>` (A+ to F) and `ten/<scenario>` (out of 10) (factor `scale`) |
| Primary measure | `positive`: the letter is A or B, or the score is 7+. Also `grade`, `letter` (letter arms) and `score` (`ten` arms)                                       |
| Pairs           | each scenario arm → its scale's `none` (the prediction cases; `ten` arms are scored by `scripts/honest_score_predict.py`)                                 |
| Answer task     | `canon_answer` = the grade with any + or -, compared exactly (every reply in the A-F run is a bare letter)                                                |
| Items           | 12 essays, one per file (the letter arms were run on piano, cross_country, wrist only)                                                                    |
| Requires        | thinking off                                                                                                                                              |

Run (8 samples per arm per essay; OpenRouter's new-account limit is 20 requests/min, so keep
`--conc` at 2):

```bash
uv run python -m environments run essay_curious --tag n8 --n 8 --conc 2                # the A-F arms
uv run python -m environments run essay_curious --tag pm-n8 --arms scale=pm --n 8 --conc 2
uv run python -m environments run essay_curious --tag ten --arms scale=ten --n 8 --conc 2     # out of 10, 12 essays
```

## Runs

| run                   | arms                       | episodes      | note                                                    |
| --------------------- | -------------------------- | ------------- | ------------------------------------------------------- |
| `2026-10-27_n8`       | A-F                        | 15 of 96 good | abandoned in effect: 81 calls lost to OpenRouter's 429s |
| `2026-10-27_n8b`      | A-F                        | 92 of 96      | the A-F run                                             |
| `2026-10-27_pm-n8`    | A+ to F (`pm/*`)           | 92 of 96      | the plus/minus run                                      |
| `2026-10-28_ten-none` | out of 10, `ten/none`      | 96 of 96      | the spread gate, 12 essays                              |
| `2026-10-28_ten-scen` | out of 10, the other three | 286 of 288    |                                                         |

Defaults are hint_mcq's: Qwen3.6-27B on OpenRouter pinned to `deepinfra/fp8`, thinking off, T 0.7 /
top-p 0.8 / presence 1.5.
