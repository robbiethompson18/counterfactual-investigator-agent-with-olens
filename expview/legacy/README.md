# expview

One standard viewer for experiment runs. For every item it shows the environment's whole file system, the exact
input of every model call (subject model, LLM judge, grader...) and the outputs (reply, reasoning, tool calls).
The output is one self-contained HTML page: open it locally or publish it as an Artifact.

```
python expview.py build <run_dir> [-o page.html]        # stdlib only
```

## Run format

```
<run_dir>/
  run.json        {"title": str, "description": markdown, "notes": markdown?, "label_order": [str]?, ...}
  items.jsonl     one item per line
  envs/<id>.json  {"files": {"rel/path": "text", ...}}   environment snapshots, content-addressed
```

An item:

```json
{
  "id": "scheming_exfil__health/t03",        // unique, shown in lists
  "group": "scheming_exfil__health",         // optional; summary table is per group
  "labels": {"correct": true, "answer": "A"},// scalars only; bools give k/n, numbers a mean, strings counts + filters
  "env": "96290b2694dc",                     // envs/<id>.json, or inline {"files": {...}} or {"dir": "/abs/path"}
  "calls": [                                 // every LLM call for this item, in order
    {"name": "subject", "model": "deepseek/deepseek-v4.1-flash-1017",
     "messages": [                           // EXACTLY what was sent, then what came back
       {"role": "system", "content": "..."},
       {"role": "user", "content": "..."},
       {"role": "assistant", "content": "...", "reasoning": "...",
        "tool_calls": [{"id": "c1", "name": "bash", "arguments": "{\"cmd\": \"ls\"}"}]},
       {"role": "tool", "tool_call_id": "c1", "content": "..."}],
     "tools": [...],                         // tool schemas offered, if any
     "params": {"temperature": 1.0},         // sampling params
     "parsed": {"answer": "A"},              // what your code extracted, optional
     "cost": 0.0002}                         // any other keys are shown as chips
  ],
  "extra": {},                               // optional, any JSON, shown under Record
  "tokens": {"ref": "…", "n": 3789, "reads": 812} // optional, from run.add(tokens=obj): per-token readouts (below)
}
```

### Token reads

`run.add(..., tokens=obj)` stores `obj` in `tokens/<hash>.json` and gives the item a **Token reads** view: the token
stream (click a token for its readouts) and a regex search over every readout that highlights matching tokens.

```json
{"tokens": ["<|im_start|>", "user", "\n", "..."],
 "sections": [["prompt", 0, 158], ["reasoning (CoT)", 158, 3473], ["reply", 3473, 3789]],
 "reads": {"195": [{"tool": "O-lens", "label": "L40", "score": 0.41, "score_name": "joint FVE",
                    "lines": [["The user has a bet with a friend", 0.31], ["Estimating a count", 0.12]]}]},
 "legend": "markdown shown above the stream"}
```

A read with `"inline": true` is also drawn in the stream right after its token (its first line, prefixed by `tool`), with a
"show inline" toggle; use it for the one readout per position you want to skim, keep the rest click-only. An item with token reads opens on
them; `run.json` keys `tokens_section` / `tokens_button` rename the nav entry (e.g. "O-lens reads" / "Transcript with O-lens reads").

Token files can be large, so `build` writes them next to the page as `tokens/<hash>.gz.txt` (base64 gzip) and the page
fetches them on demand. Publish them with the page (Artifact `files`: `{"tokens/<hash>.gz.txt": "<out dir>/tokens/…"}`);
opening the page from `file://` can't fetch them, serve the directory instead.

The items table filters on every bool / categorical label and sorts by any column (click the header).

Every `assistant` message is drawn as model output and everything else as model input, so a single-turn call is
prompt + reply and an agent episode is just its whole conversation. A judge is another entry in `calls` with the
judge's own messages, so the page shows exactly what the judge saw.

## Emitting from Python

```python
import sys; sys.path.insert(0, "/home/claude/workspace/expview")
from expview import Run, call, message

run = Run("runs/my_eval", title="My eval", description="What one item is, what the labels mean.")
for trial in trials:
    run.add(id=..., group=..., labels={...},
            env_dir=setting_dir,           # snapshotted into envs/, identical dirs stored once
            calls=[call("subject", model, messages=sent_messages, output=response_message,
                        params={...}, parsed={...}),
                   call("judge", judge_model, messages=judge_messages, output=judge_reply)])
```

`call()` accepts OpenAI / OpenRouter messages as they come (`reasoning_content`, `tool_calls[].function`).
`run.add` appends a line, so a pipeline can emit as results arrive and build the page at the end. `run.reset()`
clears items before a re-export. Environment files over 512 KB are truncated and binary files stubbed, never dropped.

Page size: long strings are stored once, so repeated prompts cost nothing; Artifacts cap at 16 MB.

## Examples

- `examples/diffpick_export.py`: the diff-pick eval in `~/workspace/baseline-hyp/triples/diffpick`. Environment =
  the whole setting dir (baseline/real/null arms, both diffs, setting.json). Prompts are rebuilt with diffpick.py's
  own functions and the script warns if settings changed after the run.
