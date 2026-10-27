# expview: the project's transcript viewer

**Status (2026-10-26):** ACTIVE. A static page on Cloudflare Pages that reads runs straight from the
public HF dataset; the bramble server, its DuckDB index, the HF pull timer and the `olens_search`
dashboard were retired the same day ([docs/viewers.md](../docs/viewers.md)).

One viewer for every run: transcripts Docent-style, the exact input and output of every model call
(subject model, LLM judges, graders), the environment's files, and metamodel reads (O-lens, J-lens,
NLA, any reader) inline at the tokens they were read at. Regex search over the reads of the runs you
pick runs in the browser.

- **Page:** [https://expview.pages.dev](https://expview.pages.dev). It shows every run on the sprint
  HF dataset
  ([Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens))
  under `runs/<experiment>/<run>/expview[/<name>]`, so a run is visible as soon as
  `scripts/runs.py finish` has synced it. No server: the page lists runs with the HF tree API and
  fetches `index.jsonl`, one item at a time by byte range from `items.jsonl`, `tokens/*.json` and
  `envs/*.json` from HF resolve URLs. The home page's scan (which runs have an `expview/` dir) is
  ~170 HF API calls out of the anonymous quota of 500 per 5 minutes per IP, so it is cached for an
  hour in the browser; deep links don't scan.
- **Built page:** `python expview/expview.py build RUN` writes one self-contained HTML page with the
  same UI (one run, data embedded, no search page). For a run that isn't on HF: an `imports/…` view,
  a subset (`--ids`), or something to publish as an Artifact.

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
print(link(run.path, item="A/s0"))  # https://expview.pages.dev/#dataset=runs/my_env/2026-10-24_slug&item=A/s0
```

Then `scripts/runs.py finish` as usual: it refreshes `index.jsonl` (`expview.py index`) and uploads
the run, and the link above works a minute later. Check a run with
`python expview/expview.py check RUN`. `Run.add` appends a line (and its index line), so a pipeline
can emit items as they finish; `run.reset()` clears items before a re-export.

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
  in the side panel; `"hidden": true` starts that reader's checkbox unticked (e.g. an unreliable
  lens).

### The format on disk

Writing it by hand (or from another language) is fine; `Run` only saves typing. Then run
`python expview/expview.py index RUN` (or let `runs.py finish` do it) so the page can read items by
byte range.

```
RUN/run.json           {"title", "description" (md), "notes" (md)?, "label_order"?, "readers"?, "tokens_section"?, "tokens_button"?}
RUN/items.jsonl        one item per line
RUN/index.jsonl        one line per item: id, group, labels, links, tokens, base, src, calls [{name, model, n}], off, len (byte span in items.jsonl)
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
 "env": "<id>" | {"files": {...}},        ({"dir": "/abs/path"} is read by `build` only; the page can't see your disk)
 "links": {"Docent": "https://..."},
 "extra": {...},                          any JSON, shown under Record
 "tokens": {"ref": "<ref>", "n": 3789, "reads": 812}}
```

Every `assistant` message is drawn as model output and everything else as input, so one call can be
a single prompt + reply or a whole agent episode. A judge is another call, so the page shows exactly
what it saw.

An item opens on its transcript with reads (if it has tokens), else on its subject call: the first
call that isn't a judge, grader, classifier or monitor (by name). The other calls sit in the side
panel beside the transcript: parsed verdict, reply, reasoning, and a link to the full call I/O.
`"token_refs": {"201": 3456}` on a call turns `[201]` in its text into a jump to token 3456.

Docent buttons: an item's own `links`, or `extra.docent` / `extra.docent_url`, or (for items from an
evalcue run's `rows.jsonl`, `extra.source` = `ENV/RUN:LINE` or `extra.run_dir` + `extra.row_index`)
the page resolves the row through that run's `docent.json` on HF. The old server also matched
transcripts by fingerprint; the page doesn't.

### Overlays: judging or annotating another run's transcripts

A run of judge calls (or notes, or extra reads) over transcripts another dataset already shows, e.g.
Celeste's O-lens imports, shouldn't copy them. Give each item a `base` instead:

```python
run.add(
    id=key,
    calls=[call("lens judge", model, messages=..., output=..., parsed=..., token_refs={...})],
    labels={"score": 7},
    base=("runs/olens_reads/2026-10-26_x", tid),
    notes=[{"pos": 2269, "text": "judge cites this bullet"}, {"pos": None, "text": "whole-item note"}],
    tokens={"reads": {"2269": [read("my reader", ["..."])]}},
)  # optional extra reads, by base position
```

The item link then opens the base item's transcript with its reads inline, plus the overlay's reads
and positioned notes, and the overlay's calls and notes in the side panel; labels are the base's
overridden by the overlay's. `expview.py check` verifies every base exists locally; `build` bundles
the base transcript into the page. On the public page the base must be on HF too: the two
`olens_prompt_ea` overlays there point at `imports/celeste/…` (never synced), so they show their
judge calls without the transcript.

## Links

Every view is a URL hash, so anything on screen can be linked (the Copy link button, or
`expview.py link`):

| Hash                                                                     | Opens                                                                   |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| `#dataset=DATASET`                                                       | a run: summary by group, filterable items table                         |
| `&item=ITEM`                                                             | an item (its transcript with reads if it has them, else its first call) |
| `&view=tr&token=POS`                                                     | token POS of the transcript, expanded                                   |
| `&view=call&call=N&message=M`                                            | call N (0-based), message M                                             |
| `&view=env&file=PATH`                                                    | an environment file                                                     |
| `&query=REGEX` (`&case=1`)                                               | highlight a regex in reads (transcript) or text (calls)                 |
| `#search=1&query=REGEX&novel=ctx&dataset=...`                            | a search over the datasets in `dataset` (comma-separated)               |
| `&experiment=ENV&scope=text&reader=R&section=S&group=G&label=K=V&page=N` | search filters (the Search page's form)                                 |

Keys are full words since 2026-10-26; the old one-letter keys (`#d=…&i=…&t=…`, `#s=1&q=…`) are
rewritten to these on load, so every old link still works ([docs/viewers.md](../docs/viewers.md),
"URL parameters").

The dataset id is the run dir's path under its root with `expview` path parts dropped:
`data/runs/eval_gaming/2026-10-24_glm53/expview/mc` is `runs/eval_gaming/2026-10-24_glm53/mc`. Roots
(`expview.roots()`, override with `EXPVIEW_ROOTS=alias=path,...`): `runs` = `data/runs` (what HF and
the page have), `imports` = `data/expview/imports` (views derived from other datasets, re-made by an
importer, not synced, `build` only).

`python expview/expview.py link RUN --item ID [--token N | --call N] [--q REGEX]` prints a link
(`EXPVIEW_URL` overrides the host).

## Search

The Search page (top right) runs a JavaScript regex over the reads of the runs you pick: choose an
experiment, tick its runs, and their token files are downloaded once (a few MB per run; shown as it
loads), then every query runs locally. "In: transcripts and judge calls" instead downloads the runs'
whole `items.jsonl` (asks first over 64 MB) and searches the calls' text. Filters: reader, section,
group, label. **Keep: new vs ±40 tokens + next 16** drops a read whose match also appears in the 40
tokens before its position or the 16 after (an echo of the text), and **new vs the whole item**
drops it if the match appears anywhere in the item (its calls' text and its token stream; this
fetches each hit item). The table shows, per dataset and group, how many items have a hit and each
yes/no label's rate among them vs overall. Results link to the token; CSV export of all matching
reads. Overlay items' own extra reads aren't searched (they live on another run's transcript).

A regex over millions of reads at once (every run, Celeste's parquet) is a `grep` or a Python script
over the files, not a web feature ([docs/viewers.md](../docs/viewers.md)).

## Importers

- `importers/celeste.py`: Celeste's GLM-5.3 O-lens readouts
  ([ceselder/glm-5.3-olens-counterfactual-investigator](https://huggingface.co/datasets/ceselder/glm-5.3-olens-counterfactual-investigator),
  `-new-sources`, wild rollouts) → one run per config under `data/expview/imports/celeste/`. Each
  item is a transcript view, re-tokenised with GLM-5.3's tokenizer, with RL / SFT O-lens, J-lens L60
  / L51 and both Qwen readers at every read position, plus the source run's calls (subject
  transcript, judges) and environment when that run is on disk. Command in its docstring. These
  imports are local only; to put one on the page, write it under
  `data/runs/<experiment>/<run>/expview` and `runs.py finish` it.
- `importers/evalcue_rows.py`: an evalcue run's `rows.jsonl` → one item per row, the episode as its
  call, the workspace before / after as its environment, the detector's flags as labels, and its
  Docent link.

A new source gets its own `importers/<name>.py` writing runs with `Run`; keep the run format the
only thing the page reads.

## Deploying the page

Only after a change to `static/index.html` (data changes need no deploy):

```bash
expview/deploy.sh preview   # from the repo root; needs CLOUDFLARE_API_TOKEN (.envrc.local); prints the preview URL
expview/deploy.sh           # production: https://expview.pages.dev
```

Check two existing deep links on the preview before deploying `main`. The page is one file, no build
step; `data/expview/site/` is the deploy staging dir.

## Files

- `expview.py`: writing runs (`Run`, `call`, `read`), `index`, `build`, `check`, `link`, `ls`,
  dataset ids and roots. Stdlib only, so emitting a run needs nothing installed.
- `static/index.html`: the whole UI, no build step: the HF data layer (tree API listing, resolve
  URLs, byte-range items), the run / item / transcript views and the in-browser search. `build`
  embeds one run's data into the same file.
- `deploy.sh`: Cloudflare Pages.
- `importers/`: other datasets → runs.
