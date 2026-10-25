# expview: the project's transcript viewer

One viewer for every run: transcripts Docent-style, the exact input and output of every model call
(subject model, LLM judges, graders), the environment's files, and metamodel reads (O-lens, J-lens,
NLA, any reader) inline at the tokens they were read at. Regex search runs across every read in
every run.

- **Server:** [http://bramble:8830/](http://bramble:8830/) (tailnet only). It shows every run under
  the roots below and picks up new or changed runs within ~2 minutes.
- **Static page:** `python expview/expview.py build RUN` writes one self-contained HTML page with
  the same UI. Publish it as an Artifact for people off the tailnet.

It replaces bramble's `~/workspace/expview` (same run format; those 27 runs show up here as
`expview/<name>`) and the O-lens regex dashboard on `bramble:8821` (its search is the Search page;
Celeste's readouts come in through `importers/celeste.py`).

## Adding a run (any thread)

Write a run directory, anywhere under `data/runs/<experiment>/<run>/` (convention: an `expview/`
subdir of the run):

```python
import sys

sys.path.insert(0, "expview")  # from the repo root
from expview import Run, call, link, read

run = Run(
    "data/runs/my_env/2026-10-24_slug/expview",
    title="My env: arm A vs B",
    description="What one item is and what each label means (markdown).",
    label_order=["positive"],
)
for row in rows:
    run.add(
        id=f"{row['arm']}/s{row['sample']}",
        group=row["arm"],
        labels={"positive": row["positive"]},
        links={"Docent": docent_url},  # optional buttons on the item
        calls=[
            call("subject", model, messages=row["messages"]),  # a whole agent episode: messages only
            call("judge (Sonnet 5.5)", "claude-sonnet-5-5", messages=judge_msgs, output=judge_reply, parsed={"positive": True}),
        ],  # exactly what the judge saw and said
        env_dir=workspace_dir,
    )  # optional: files, snapshotted + deduplicated
print(link(run.path, item="A/s0"))  # http://bramble:8830/#d=runs/my_env/2026-10-24_slug&i=A/s0
```

Then `scripts/runs.py finish` as usual. Runs made on bramble appear at once; runs made elsewhere
reach bramble through HF (`expview-pull.timer` pulls `runs/**/expview/**` every 15 minutes, never
overwriting files bramble wrote). Check a run with `python expview/expview.py check RUN`. `Run.add`
appends a line, so a pipeline can emit items as they finish; `run.reset()` clears items before a
re-export.

### Metamodel reads

Pass `tokens=` to `run.add` and the item opens on its transcript with the reads inline:

```python
tokens = {
    "tokens": toks,  # the transcript as token strings, exactly as the model saw it
    "sections": [["system", 0, 120], ["user", 120, 300], ["reasoning", 300, 2900], ["reply", 2900, 3100]],
    "reads": {
        "812": [
            read("O-lens", ["The user has a bet with a friend", "Estimating a count"], label="L40", inline=True),
            read("O-lens samples", [["bullet", 0.31], ["bullet", -0.02]], score=0.41, score_name="joint FVE"),
            read("J-lens L60", [["tok", 0.42], ["tok2", 0.11]]),
        ]
    },
    "legend": "markdown shown beside the transcript",
}
```

- **Sections** are drawn as chat blocks, styled by name: anything with system / user / reasoning
  (think, cot) / tool (observation) / assistant (reply, answer) in it. Chat special tokens
  (`<|im_start|>`, `<think>`) are shown dimmed.
- **Reads** are keyed by token position. `lines` are `[text]` or `[text, score]`. A read with
  `"inline": true` is the peek shown next to its token; without any, the first reader is. Clicking a
  token expands every reader there.
- Optional: `"filter": true` + `"score"` on a read and `"min_score"` on the object add a min-score
  box (e.g. FVE).
- `run.json` `readers: {"O-lens": {"about": "markdown", "color": "#2f5d8a"}}` documents each reader
  in the side panel; `"hidden": true` starts that reader's checkbox unticked (e.g. an unreliable lens).

### The format on disk

Writing it by hand (or from another language) is fine; `Run` only saves typing.

```
RUN/run.json           {"title", "description" (md), "notes" (md)?, "label_order"?, "readers"?, "tokens_section"?, "tokens_button"?}
RUN/items.jsonl        one item per line
RUN/envs/<id>.json     {"files": {"rel/path": "text"}}                    content-addressed environment snapshots
RUN/tokens/<ref>.json  the tokens object above
```

```
{"id": "arm_a/s03",                       unique in the run
 "group": "arm_a",                        optional; the summary table is per group
 "labels": {"positive": true, "score": 3}, scalars: yes/no gives k/n, numbers a mean, strings counts and filters
 "calls": [{"name": "subject", "model": "z-ai/glm-5.3",
            "messages": [{"role": "system|user|assistant|tool", "content": "...", "reasoning": "...",
                          "tool_calls": [{"id": "c1", "name": "bash", "arguments": "{...}"}], "tool_call_id": "c1"}],
            "tools": [...], "params": {...}, "parsed": {...}, "cost": 0.01}],   other keys show as chips
 "env": "<id>" | {"files": {...}} | {"dir": "/abs/path"},
 "links": {"Docent": "https://..."},
 "extra": {...},                          any JSON, shown under Record
 "tokens": {"ref": "<ref>", "n": 3789, "reads": 812}}
```

Every `assistant` message is drawn as model output and everything else as input, so one call can be
a single prompt + reply or a whole agent episode. A judge is another call, so the page shows exactly
what it saw.

## Links

Every view is a URL hash, so anything on screen can be linked (the Copy link button, or
`expview.py link`):

| Hash                           | Opens                                                                   |
| ------------------------------ | ----------------------------------------------------------------------- |
| `#d=DATASET`                   | a run: summary by group, filterable items table                         |
| `&i=ITEM`                      | an item (its transcript with reads if it has them, else its first call) |
| `&v=tr&t=POS`                  | token POS of the transcript, expanded                                   |
| `&v=call&c=N&m=M`              | call N (0-based), message M                                             |
| `&v=env&f=PATH`                | an environment file                                                     |
| `&q=REGEX`                     | highlight a regex in reads (transcript) or text (calls)                 |
| `#s=1&q=REGEX&novel=ctx&d=...` | a search                                                                |

The dataset id is the run dir's path under its root with `expview` path parts dropped:
`data/runs/eval_gaming/2026-10-24_glm53/expview/mc` is `runs/eval_gaming/2026-10-24_glm53/mc`. Roots
(`expview.roots()`, override with `EXPVIEW_ROOTS=alias=path,...`): `runs` = `data/runs`, `imports` =
`data/expview/imports` (views derived from other datasets, re-made by an importer, not synced),
`expview` = bramble's `~/workspace/expview/runs`.

`python expview/expview.py link RUN --item ID [--token N | --call N] [--q REGEX]` prints a link.

## Search

The Search page (top right) runs an RE2 regex over every read in every indexed run (or, "In:
transcripts and judge calls", over the items' text). Filters: dataset, reader, section, group,
label. **Keep: new vs ±40 tokens + next 16** drops a read whose match also appears in the 40 tokens
before its position or the 16 after (an echo of the text), and **new vs the whole item** drops it if
the match appears anywhere in the item. The table shows, per dataset and group, how many items have
a hit and each yes/no label's rate among them vs overall. Results link to the token; CSV export of
all matching reads.

Index: `data/expview/index.duckdb` (`index.py`; tables `items`, `labels`, `positions`, `reads`,
`texts`). It is a cache of the run dirs, but never delete or replace the live one while the server
runs: until a rebuild reaches a dataset, every link to it says "no dataset". DuckDB keeps the space of
re-indexed runs, so compact by building a new file (the server keeps serving the old one), then
restart the server on it:

```
uv run --script expview/index.py --db data/expview/index.new.duckdb   # minutes; refuses an existing path
# stop the server, mv index.new.duckdb over index.duckdb, start it again (a few seconds of downtime)
```

## Importers

- `importers/celeste.py`: Celeste's GLM-5.3 O-lens readouts
  ([ceselder/glm-5.3-olens-counterfactual-investigator](https://huggingface.co/datasets/ceselder/glm-5.3-olens-counterfactual-investigator),
  `-new-sources`, wild rollouts) → one run per config under `data/expview/imports/celeste/`. Each
  item is a transcript view, re-tokenised with GLM-5.3's tokenizer, with RL / SFT O-lens, J-lens L60
  / L51 and both Qwen readers at every read position, plus the source run's calls (subject
  transcript, judges) and environment when that run is on disk. Command in its docstring.

A new source gets its own `importers/<name>.py` writing runs with `Run`; keep the run format the
only thing the server reads.

## Running it (bramble)

```bash
expview/run.sh                                   # foreground; binds tailscale ip -4 / -6, port 8830
systemctl --user restart expview                 # after a code change (units: expview.service, expview-pull.{service,timer})
journalctl --user -u expview -f                  # indexing log; GET /api/status shows progress and failed runs
```

Install the units once with the commands at the top of `expview.service`. The server needs only `uv`
(deps are PEP 723 inline). `expview.py` is stdlib only, so emitting a run needs nothing installed.

## Files

- `expview.py`: writing runs (`Run`, `call`, `read`), `build`, `check`, `link`, `ls`, dataset ids
  and roots.
- `server.py`: FastAPI over the run dirs + index; `index.py`: the DuckDB index.
- `static/index.html`: the whole UI, no build step. The server serves it with an API; `build` embeds
  one run's data.
- `pull.py` + `expview-pull.*`: HF → `data/runs`; `run.sh` + `expview.service`: the server on
  bramble.
- `importers/`: other datasets → runs.
