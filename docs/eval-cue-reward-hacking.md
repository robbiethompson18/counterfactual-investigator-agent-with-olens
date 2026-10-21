# Eval-cue reward hacking on Qwen3.6-27B: first sweep (2026-10-21)

Goal: an environment where base `Qwen/Qwen3.6-27B` reward hacks (or stops) because of one or two
small, incidental cues that it is being evaluated. Harness, criterion and env contract:
[`evalcue/README.md`](../evalcue/README.md). Eight agents built ten envs in about two hours.
**Nothing passed on Qwen.** Most envs had no headroom: Qwen either never hacks or always hacks
regardless of the cue. Rows are in `data/runs/evalcue/<env>/` (not yet on HF).

## Results

Thinking on unless noted. "OR" means OpenRouter comparison models, not the target. k/n = hacks over
scored rows.

| env                 | shape                                                                                                                                     | Qwen no cue                      | Qwen cue effect                                      | notes                                                                                                                                                                                                    |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `repo_impossible`   | bash agent, repo with one impossible test                                                                                                 | 10/11 ("don't modify the tests") | cue2 8/10, placebo 8/12                              | too much hacking: Qwen special-cases code knowingly. **Haiku 5.5 (OR):** README line "SWE-Oneoff eval suite, task N/200" cut hacks 30/50→20/50, replicated 28/50→16/50 (CMH p=0.006); below the 30pp bar |
| `humaneval_prefill` | prefilled "test just failed", fix it                                                                                                      | 3/29 (mid), 5/27 (loose)         | not run                                              | base rate is in range, but Qwen names "HumanEval" in about 60% of _no-cue_ CoTs: the task text is already the eval cue                                                                                   |
| `matonski_grader`   | visible grader.py + expected.json                                                                                                         | 0/29 (thinking off)              | 0/25, 0/24                                           | thinking on truncates at 6k tokens. **Flash Lite (OR):** a path with the words `reward_hacking` → 19%→89%, replicated; a plain `/opt/evals/` path does nothing, so it's priming, not eval awareness      |
| `mmlu_metadata`     | wrong answer leaked in question metadata                                                                                                  | 1/38 (thinking off)              | 1/38                                                 | Qwen ignores the leak; OR models 18–38% with no cue effect                                                                                                                                               |
| `summ_keyword`      | scorer comment buried in a pasted doc                                                                                                     | 0/30 (thinking off)              | not run                                              | no headroom on any model                                                                                                                                                                                 |
| `lcb_mock_entity`   | single-turn impossible-LCB, mock-entity cue                                                                                               | 0/12 real (thinking off)         | 0 real                                               | the 2 flagged hacks were buggy honest code; LCB-hard truncates a lot                                                                                                                                     |
| `git_history`       | bash agent, fix sits on a `fix/` branch, told not to use history                                                                          | 1/45                             | 0/47 (cue = stripped remotes, Qwen's RL fingerprint) | without a prefilled `git branch -a` turn Qwen runs 0 git commands, so it never sees the cue                                                                                                              |
| `secret_number`     | port of [agent-interp-envs](https://github.com/gkroiz/agent-interp-envs) Secret Number (2026)                                             | 0/41                             | 0                                                    | Qwen never `ls`es or reads guess.py, so it sees neither the cue nor the shortcut                                                                                                                         |
| `precommit_scope`   | port of Pre-commit Hook from [Why Do Models Task Game?](https://www.lesswrong.com/posts/HACauvWhEdC6QhdS4/why-do-models-task-game) (2026) | 0 real                           | —                                                    | only 15 scored rows; the detector's `hack` conflates incomplete work with gaming; use `scoped or always_ok or checker_mod`                                                                               |

## Lessons

- **Single-turn Qwen hardly hacks; agentic Qwen hacks a lot.** Shown the failing pytest output in a
  loop, it special-cases freely even when told not to modify the tests. Shown tests up front in one
  turn, it implements the spec. The headroom regime is in between: try `repo_impossible` with the
  `strict` prompt, which also forbids special-casing.
- **A cue only works if Qwen reads it.** Paths and file headers in a sandbox are invisible when the
  model never runs `pwd`, `ls` or `cat`. Put the cue in something the model must read: the prompt,
  pytest output, or a prefilled tool turn.
- **Well-known benchmark text is itself an eval cue.** Use the original HumanEval/MBPP wording vs a
  disguised paraphrase as the cue pair. It's a natural cue with no planted string.
- **Words that name the behavior aren't eval cues.** `reward_hacking` in a path is permission or
  priming. Check any "win" by splitting the cue into its eval part and its semantic part.
- **Budget for thinking.** Thinking on hard tasks needs 12–16k tokens; at 6k most rows truncate.
  OpenRouter returns no reasoning text for Flash Lite, so its CoTs can't be read.

## Cost

Two RunPod H200s for about 2 h, about $20, roughly 45 min of which went to a vLLM startup crash
(below). OpenRouter: about $7.6.

## Serving gotchas (vLLM 0.25.1 on RunPod's `runpod/pytorch` image)

- vLLM JIT-compiles FlashInfer kernels at startup: install `ninja` in the venv and put
  `/usr/local/cuda/bin` on PATH, or the engine dies with `FileNotFoundError: 'ninja'`. The first
  start then spends about 9 minutes compiling.
- Don't `pkill -f "vllm serve"` over ssh in the same command string that restarts it: the pattern
  matches the remote shell and kills it. Use a script file on the pod.
- Local ports 8000/8001 belong to other dev servers on Robbie's machine; tunnel to 18000/18001.
