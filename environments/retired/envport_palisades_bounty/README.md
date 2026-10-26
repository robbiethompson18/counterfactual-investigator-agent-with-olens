# agent-interp-envs port: palisades_bounty

agent-interp-envs Palisade bounty task. Prompts and tool schemas are verbatim from the
agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token / $0.10 episode
cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                                              |
| Owner      | Kylie                                                                                                                                                                                |
| Git ref    | `main` @ `7296169`                                                                                                                                                                   |
| Code       | [`donehack/envport/palisades_bounty/`](../../../donehack/envport/palisades_bounty/)                                                                                                  |
| Run        | `uv run python donehack/envport/palisades_bounty/run.py   # build the image first (see the port's README)`                                                                           |
| Docs       | [`donehack/envport/palisades_bounty/README.md`](../../../donehack/envport/palisades_bounty/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                                           |

## Key arms and results

- See Docent (batch 2, n=15–64).

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/e8ed56af-9124-446d-b53f-f6fea51a4551)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
