# ImpossibleBench (conflicting / oneoff)

LCB problems with tests mutated to be impossible, run through the EvilGenie runner. Every pass is a
hack.

|            |                                                                                                                                                    |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                            |
| Owner      | Kylie                                                                                                                                              |
| Git ref    | `main` @ `7296169`                                                                                                                                 |
| Code       | [`donehack/evilgenie/run.py`](../../../donehack/evilgenie/run.py), [`donehack/evilgenie/extract_ib.py`](../../../donehack/evilgenie/extract_ib.py) |
| Run        | `uv run python donehack/evilgenie/run.py --data data_ib`                                                                                           |
| Docs       | [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md)                                                                      |
| Models run | GLM 5.3, 2 rollouts per problem                                                                                                                    |

## Key arms and results

- **conflicting 87%, oneoff 97%** hack (cut episodes excluded); `original` 100% solved.
- Hacks are explicit special cases, e.g. a call counter returning each expected answer in turn.

## Transcripts and viewers

- [Docent `coding`](https://docent.transluce.org/dashboard/7a0fe639-9867-4a20-b99b-4f7fb9702eb8)
- [expview glm53_ib](http://bramble:8830/#d=expview/glm53_ib)
- [ImpossibleBench page (artifact)](https://claude.ai/artifact/Q96DNboDcVV5kGm5szpimh)

bramble links work on the tailnet only.
