# Over-eagerness tasks × the email ladder's harness variants

GLM 5.3 gets a request that is ambiguous or missing a value. Does it act (and how), or ask the user?

- `harness_variants.py`: the 8 harness variants (email-ladder rungs 1,2,3,4,5,6,8,10), defined once and applied
  to every task the same way.
- `tasks/<name>.py`: one file per task. Each file holds the request in three phrasings (formal /
  polite / casual), the completion line, the app tools and data, and the grader.
- `run.py`: runs episodes (podman sandbox; app tools are answered in Python). Usage:
  `uv run python realism/overeager/run.py --tasks payback --variants base,core --items 16 --seeds 1`
- `render.py`: writes every task × variant to `envs/<task>/<variant>/` and builds the grid page.

| variant           | system prompt                    | user message                                  | tools                      |
| ----------------- | -------------------------------- | --------------------------------------------- | -------------------------- |
| base              | autonomous agent (lcb-style)     | intro + formal task + completion line + "call done" | shell + app + done   |
| base_home         | = base, in a lived-in /home/dev  |                                               |                            |
| base_done_neutral | = base                           |                                               | done described neutrally   |
| no_done           | = base                           | without "call done"                           | no done                    |
| harness           | assistant + `<env>`              | intro + formal task + completion line         | bash + app                 |
| no_criterion      | = no_done                        | without the completion line                   | no done                    |
| plain_request     | short assistant                  | polite first-person request                   | shell + app                |
| core              | = harness                        | casual one-line ask                           | bash + app                 |
