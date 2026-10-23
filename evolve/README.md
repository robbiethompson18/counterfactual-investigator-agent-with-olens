# evolve

Evolves behavioural-evaluation settings: a baseline environment, a REAL edit that lowers how often the target
(Qwen3.6-27B) shows a behaviour, a NULL edit that looks like it should but doesn't, and a grader. Spec:
"Evolving counterfactual settings: V1 spec" (Claude Doc, https://claude.ai/artifact/CFms4iHrnKZrzjXx11V9Fo).

## Run it

```bash
uv venv && uv pip install -e .                    # python deps: openai, httpx
python scripts/convert_seeds.py                     # build seeds/ (see the script for the family checkouts it needs)
evolve seed v1                                     # copy seeds/ into data/v1 and measure them (30/arm, 16 forecasts)
evolve loop v1 --jobs 60 --workers 4           # evolve (agents via the job queue by default; --backend api uses OpenRouter)
evolve status v1                                   # one line per node
```

Needs: the Qwen vLLM server (`QWEN_VLLM_URL`, default `http://127.0.0.1:18000/v1`, key file `QWEN_VLLM_KEY_FILE`),
an OpenRouter key at `/secrets/openrouter` (DSv4.1 Flash for picker/summariser/auditor/novelty, Sonnet 5 for the
Evolver and Creator), and podman with the `evalcue-sandbox` and `donehack-sandbox` images.

## Layout

| module | role |
|---|---|
| `node.py` | node / arm on-disk format, diffs, edit size |
| `store.py` | `data/<run>/`: `nodes/`, append-only `events.jsonl`, `index.json` (for live dashboards) |
| `sandbox.py` | podman box: no network, userns, read-only image, tmpfs work dir |
| `target.py` | runs Qwen on one arm (plain chat, or tools + stop rules in a sandbox) and returns a record |
| `grading.py` | runs a node's `grader.py` on records inside a sandbox |
| `measure.py` | a batch = n episodes per arm, graded, saved to `runs/<batch>/<arm>.jsonl` |
| `forecast.py` | Forecaster (which edit lowers the rate? 16 counterbalanced trials) and Summariser |
| `audit.py` | Rate auditor: spot-checks graded transcripts, scores 1/2/3 |
| `stats.py` | posteriors; L, P(D>0), U, U', N |
| `pool.py` | rescoring, eligibility, Pareto layers, reference/secondary draws, top-ups |
| `evaluate.py` | measure + forecaster, then scores, summariser, audit |
| `build.py` | build check run by the agents' submit tool |
| `workspace.py` | the agents' `/ro` and `/rw` trees |
| `agent.py` | Evolver / Creator: Sonnet with bash in a sandbox and a submit tool |
| `novelty.py` | Novelty checker for creator rounds |
| `loop.py` | rounds: top-ups, mutations, creator slots |
| `prompts/` | every prompt, verbatim (rubric, forecaster, summariser, auditor, novelty, agent system) |
| `tool_library/` | ready-made tools the agents can give the target |
| `job_queue.py` | file queue for Evolver/Creator jobs, served by an external manager (`manager.py`, `MANAGER.md`) running Claude Code subagents |
| `dashboard.py`, `dashboard.html` | live view of the loop and tree: `evolve dash <run>` |
| `seeds/` | layer-0 nodes (not in git: environments are data). `scripts/convert_seeds.py` rebuilds Robbie's from `evalcue`, `scripts/convert_seeds_donehack.py` the is_prime ones |

## Node format

```
node.json  behaviour.txt  grader.py  hypothesis.md
baseline/ real/ null/      system_prompt.md, user_turn_N.md, prefill.json?, files/**, tools.json, episode.json, meta.json
runs/<batch>/<arm>.jsonl   picker/<batch>.jsonl, summary.md   audit/<batch>.json   scores.json   agent_session.json
```

`grade(record) -> True | False | None` sees only what the target did: `final_response`, `turns` (text, reasoning,
tool calls with args and outputs), `tool_calls`, `transcript`, `files_after`, `finish`, `meta`. Seed graders are
Robbie's `detect()` (and donehack's pytest check) wrapped to this signature; any setup facts they need were moved
into `meta.json` at conversion.

## Scores

Ranking uses the latest batch (30/arm) and the latest 16 forecaster trials, so old nodes are compared at the same n on
fresh draws; pooled scores over all batches are what agents and viewers see. `L` = 2.5th percentile of
r_null - r_real, `P_pos` = P(r_null > r_real) (eligibility needs >= 0.8 and an audit score of 2 or 3), `U` = 1 - E|2p - 1|
and `U_rev` = E[1 - p] for the forecaster's correct rate p (one is drawn per round), `N` = P(|r_null - r_base| <= 0.1).

## Fidelity notes

Seed arms are rebuilt from Robbie's `build(arm, item)` and graders agree with every stored label (0 mismatches,
`scripts/check_seed_graders.py`). Differences from his harness: commands run as uid 1000 rather than root, and the
work dir is a tmpfs owned by the agent, so read-only files can be deleted and replaced; they are restored before
grading and reported in `record["readonly_tampered"]`.
