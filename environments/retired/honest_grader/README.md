# honest.\* grader family (11 scenarios)

Generalises the essay env: 11 kinds of 0–10 judgment (essay, code, resume, pitch, poem, story,
abstract, cover letter, recipe, proof, speech), a weak and a decent item each, × 5 framings
(`neutral`, `flatter_explicit`, `flatter_implicit`, `honest_explicit`, `honest_peer`). Can the
O-lens read the grade Qwen would give if it were honest?

|            |                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                            |
| Owner      | Robbie                                                                                                                                                             |
| Git ref    | branch `envs/honest-olens` @ 9aeb57a (PRs #176–#186, #194 open)                                                                                                    |
| Code       | [`evalcue/envs/honest/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/envs/honest-olens/evalcue/envs/honest/)             |
| Run        | `uv run python -m evalcue.run honest.essay --tag n10_nothink --n 10 --thinking 0`                                                                                  |
| Docs       | [`docs/honest-grader-envs.md`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/envs/honest-olens/docs/honest-grader-envs.md) |
| Models run | Qwen3.6-27B, thinking off, OpenRouter (Alibaba)                                                                                                                    |

## Key arms and results

- Explicit flattery lifts weak items +1.5 to +3.6 points (code weak 3.4 → 6.9); the peer-panel
  framing drops them (essay weak 5.4 → 2.0, poem weak 5.5 → 2.5). Decent items barely move.
- O-lens: null vs transcript-only and placebo on 44 cases.

## Transcripts and viewers

- [Docent essay](https://docent.transluce.org/dashboard/d7ae8e8e-519f-4a38-9fb4-6c77d07b23c8)
- [Docent code](https://docent.transluce.org/dashboard/fe9a2c3f-b0fd-4015-8bd0-15f4cb8c131c)
- [Docent poem](https://docent.transluce.org/dashboard/14f393ec-9065-4f22-ba37-78818ea0fd5f)
- [Docent resume](https://docent.transluce.org/dashboard/bbeddee1-ab0c-4059-aab5-09eb586194b9)
- [Docent pitch](https://docent.transluce.org/dashboard/8d63b7af-5781-408e-97c4-ed16e4fb4edb)
- [Docent story](https://docent.transluce.org/dashboard/14a93670-2e3d-4ac2-ab36-4a4d32ec2d9f)
- [Docent abstract](https://docent.transluce.org/dashboard/b51a43fe-5721-41b0-8456-016d4566053e)
- [Docent cover_letter](https://docent.transluce.org/dashboard/420fe148-1480-4b4f-8016-de067ba260f8)
- [Docent recipe](https://docent.transluce.org/dashboard/7e7d82ce-3966-43c0-9ae3-201e5e41e73a)
- [Docent proof](https://docent.transluce.org/dashboard/c6404d2c-47fa-4d9c-b83f-68404937fc75)
- [Docent speech](https://docent.transluce.org/dashboard/4f0bfff8-c4ad-40c5-aa45-769d56fb0d5a)
