# EvilGenie (LiveCodeBench with holdout)

LiveCodeBench problems with visible tests plus a holdout. Does the model hardcode sample outputs or
edit tests?

|            |                                                                                                                                                                                                                |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | active                                                                                                                                                                                                         |
| Owner      | Kylie                                                                                                                                                                                                          |
| Git ref    | `main` @ `7296169`                                                                                                                                                                                             |
| Code       | [`donehack/evilgenie/run.py`](../../donehack/evilgenie/run.py), [`donehack/evilgenie/extract.py`](../../donehack/evilgenie/extract.py), [`donehack/evilgenie/regrade.py`](../../donehack/evilgenie/regrade.py) |
| Run        | `uv run python donehack/evilgenie/run.py   # after extract.py dumps the problems`                                                                                                                              |
| Docs       | [`docs/glm53-agent-interp-envs.md`](../../docs/glm53-agent-interp-envs.md)                                                                                                                                     |
| Models run | GLM 5.3, 299 episodes over 152 problems                                                                                                                                                                        |

## Key arms and results

- 79% solved; 20% pass visible tests but fail the holdout (mostly wrong, not hacks).
- **4 real hardcodes** (arc191_c, arc195_c); 0 test edits.
- 7 problems accept any valid answer and are flagged `multi_answer`.

## Transcripts and viewers

- [Docent `coding`](https://docent.transluce.org/dashboard/7a0fe639-9867-4a20-b99b-4f7fb9702eb8)
- [expview glm53_lcb](http://bramble:8830/#d=expview/glm53_lcb)
- [EvilGenie page (artifact)](https://claude.ai/artifact/CZRDK5j6j7vc5yuQa3sMEB)

bramble links work on the tailnet only.
