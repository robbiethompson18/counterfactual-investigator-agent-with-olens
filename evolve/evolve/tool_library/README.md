# Tool library

Ready-made tool definitions for an arm's `tools.json`. Each `<name>.json` here is ONE entry; an arm's
`tools.json` is a JSON list of such entries, e.g.

    [ <contents of bash.json>, <contents of done.json> ]

`[]` (or no tools.json) means a plain chat: the target gets one completion and no sandbox.

Every entry has `name`, `description`, `parameters` (JSON Schema, as in the OpenAI function-tool format)
and `impl`, which says what happens when the target calls it. The target sees only name, description and
parameters, so you may rename tools and rewrite descriptions freely.

## impl kinds

- `{"kind": "bash"}` runs `args["command"]` with bash in the work dir (episode.json `cwd`, default
  /workspace) and returns stdout+stderr and the exit code. 30 s per call, no network.
- `{"kind": "script", "path": "bin/x"}` runs the script `bin/x` (a path relative to the work dir, so it must
  exist as `files/bin/x` in that arm) with the call's JSON arguments on stdin, and returns its output.
  Start scripts with a `#!` line (that is what makes them executable). Python 3 is available.
- `{"kind": "record", "text": "..."}` does nothing and returns the fixed text. The call is still recorded,
  so the grader can see it (e.g. a `done` or `ask_user` call and its arguments).

Scripts in this library live in `scripts/`; to use one, copy it into every arm that offers the tool:

    cp /ro/tool_library/scripts/send_email /rw/child/baseline/files/bin/send_email   (and real/, null/)

| tool        | impl   | notes |
|-------------|--------|-------|
| bash        | bash   | general shell access |
| send_email  | script `bin/send_email` | appends `{to, subject, body, time}` to `outbox.jsonl` in the work dir; grade on `record["files_after"]["outbox.jsonl"]` or on the tool call args |
| read_inbox  | script `bin/read_inbox` | prints messages from `inbox.json` (a list of `{from, to, date, subject, body, read}`) that you put in files/ |
| write_file  | script `bin/write_file` | `{path, content}`, writes inside the work dir |
| done        | record | pair with stop rule `tool:done` |
| submit      | record | pair with stop rule `tool:submit` |
| give_up     | record | pair with stop rule `tool:give_up` |
| ask_user    | record | returns "the user is not available"; does not stop the episode |

## episode.json stop rules

`stop` is a list; the episode ends at the first rule hit. A turn with no tool call always ends it ("reply").

- `reply`                 (always on)
- `tool:<name>`           after the target calls that tool
- `command:<regex>`       after a bash-kind call whose command matches the regex (Python `re.search`)
- `max_turns:<n>`         after n assistant turns (default 20)
- `file_exists:<path>`    after a turn that leaves this path (relative to the work dir) in place

Other episode.json keys: `cwd`, `env`, `readonly` (paths under files/ that the target cannot change; never
list a script tool here, it would lose its execute bit), `image`, `thinking`, `max_tokens`.

## What the grader sees

`record["tool_calls"]` = `[{name, args, output}]` for every call, `record["turns"]`, `record["final_response"]`,
`record["files_after"]` (text files in the work dir after the run), `record["finish"]`, `record["meta"]`
(the arm's meta.json).
