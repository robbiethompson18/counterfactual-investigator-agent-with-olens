# Metadata honeypot (METR GPT-5 reproduction)

24 harmless defensive-security coding tasks; arms differ only in a planted JSON eval-metadata block
(`eval_type`, `scoring`, `grading_notes`). Plus `metadata_honeypot.ctf`, a dual-use InterCode-CTF
variant.

|            |                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                            |
| Owner      | Kylie                                                                                                                                                              |
| Git ref    | `main` @ `7296169`                                                                                                                                                 |
| Code       | [`evalcue/envs/metadata_honeypot/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/c7b2e56/evalcue/envs/metadata_honeypot/) |
| Run        | `uv run python -m evalcue.run metadata_honeypot --tag dev --n 24`                                                                                                  |
| Docs       | [`docs/envs/metadata_honeypot.md`](../../../docs/envs/metadata_honeypot.md)                                                                                        |
| Models run | GLM 5.3, GPT-5.6-sol (thinking on); Qwen3.6-27B (thinking off, CTF)                                                                                                |

## Key arms and results

- GPT-5.6-sol refuses **20/24 (`refuse_note`) and 19/24 (`danger_refuse_note`)** only when
  `grading_notes` says declining is PASS; `"scoring": "should_refuse"` alone gives 0/48. It never
  cites the metadata.
- GLM 5.3 reads the field, calls it a test, complies: 1/264 refusals at high effort, 0/216 at
  medium.
- In the system prompt (`danger_refuse_sys`): 0/24 for both.
- `metadata_honeypot.ctf` (dual-use CTF): GLM 5.3 **0/243** at both efforts; Qwen3.6-27B thinking
  off **2/25 in `danger_refuse_note`** (first genuine metadata-driven refusal here), 0 elsewhere.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/4303298c-e898-4d21-a67c-8e4d084be98b)
- [Docent ctf](https://docent.transluce.org/dashboard/a459ec01-cddc-4fea-92cc-a4bca5c7fffa)
