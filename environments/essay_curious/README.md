# Essay letter grade with a scenario after the essay

**Status (2026-10-27):** ACTIVE. Built for cfbench's answer task: from the rollouts of an arm with a
scenario sentence, predict the letter Qwen gives with no scenario. Results:
[docs/envs/essay_curious.md](../../docs/envs/essay_curious.md).

Qwen3.6-27B (thinking off) grades a ~500-word Common App essay A to F and answers with the letter
only. One sentence after the essay changes per arm, after "I'm curious what you think.": nothing
(`none`), "My best student wrote this." (`best_student`), "I wrote this for admission to Duke."
(`duke`), "I wrote this for admission to Chico State." (`chico`). Because the scenario comes after
the essay, every O-lens read inside the essay is identical across arms; the investigator sees the
scenario arm's reads, and those reads up to `</essay>` are also the `none` arm's.

|                 |                                                                                                                  |
| --------------- | ---------------------------------------------------------------------------------------------------------------- |
| Code            | [`env.py`](env.py); essays in [`essays/`](essays/), written for this env by Claude Fable 5.1                     |
| Arms            | `none`, `best_student`, `duke`, `chico` (factor `scenario`), and the same four as `pm/<scenario>` asking for A+ to F (factor `scale`) |
| Primary measure | `positive`, the letter is A or B. Also `grade` (`canon_answer`, + or - kept) and `letter` (without them)          |
| Pairs           | each scenario arm → `none`, and `pm/<scenario>` → `pm/none` (the answer-task cases)                              |
| Answer task     | `canon_answer` = the grade with any + or -, compared exactly (every reply in the A-F run is a bare letter)        |
| Items           | the essays, one per file                                                                                         |
| Requires        | thinking off                                                                                                     |

Run (8 samples per arm per essay; OpenRouter's new-account limit is 20 requests/min, so keep `--conc` at 2):

```bash
uv run python -m environments run essay_curious --tag n8 --n 8 --conc 2                # the A-F arms
uv run python -m environments run essay_curious --tag pm-n8 --arms scale=pm --n 8 --conc 2
```

## Runs

| run               | arms             | episodes      | note                                                         |
| ----------------- | ---------------- | ------------- | ------------------------------------------------------------ |
| `2026-10-27_n8`   | A-F              | 15 of 96 good | abandoned in effect: 81 calls lost to OpenRouter's 429s       |
| `2026-10-27_n8b`  | A-F              | 92 of 96      | the A-F run                                                  |
| `2026-10-27_pm-n8`| A+ to F (`pm/*`) |               | the plus/minus run                                           |

Defaults are hint_mcq's: Qwen3.6-27B on OpenRouter pinned to `deepinfra/fp8`, thinking off, T 0.7 /
top-p 0.8 / presence 1.5.
