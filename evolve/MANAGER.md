# Running the Evolver / Creator on a Claude subscription

`evolve loop <run> --backend queue` (the default) does not call an LLM for the Evolver or Creator. It writes each
agent job to `data/<run>/agent_jobs/<job>/` (format in the `evolve/job_queue.py` docstring) and waits for it. A
long-running Claude Code session, the manager, runs every job as a Claude Code subagent, so the agent work is billed
to that session's Claude plan instead of OpenRouter. Measurement on Qwen keeps running in the loop meanwhile.

## The manager's cycle

Commands below run from this directory with `.venv/bin/python -m evolve.manager ...`.

1. `watch <run>` in the background: returns as soon as an unclaimed job exists (prints its dirs).
2. For each pending job, while fewer than 2 subagents are in flight: `claim <job>` (atomic, exits 1 if someone
   else got it), then start a background subagent (model Sonnet) with the prompt below.
3. When a subagent finishes, run `finish <job> --agent-id <id>`. That copies the subagent's full Claude Code
   transcript to `transcript.jsonl`, saves its final report as `transcript.md`, copies both into the node the loop
   admitted (the loop admits on result.json, before the subagent has finished), and writes `failed.json` if it never got a passing
   submit, so the loop doesn't sit out its 3 h timeout.
4. `status <run>` shows counts by state and the jobs in flight.

## Subagent prompt

```
You are the {kind} agent for one job of the evolve loop. Your instructions are in {job}/INSTRUCTIONS.md:
read that file first and follow it exactly; it is the whole task.

Rules:
- Edit only files under {job}/rw/child/. Treat everything else as read-only, including {job}/ro/.
- No network: no curl, wget, pip, git fetch, web search or web fetch.
- Don't post messages or use any mcp__ tools.
- Work efficiently, as INSTRUCTIONS.md says, and run the submit command well within about 60 tool calls.
- You are done when the submit command prints "OK, submitted.". If it prints build errors, fix them and run it
  again. If you truly cannot get a passing submit, stop and say why.

When done, reply with a short report (under 200 words): what you changed in real/ and null/ (and anything
else), the hypothesis in one or two sentences, and how many submit attempts it took.
```

## Limits (Kylie, 2026-10-23)

- At most 2 subagents in flight, always.
- If a subagent hits the Claude usage limit, stop: claim nothing more, and leave the cap-hit job's `failed.json`
  saying so.
- At most 250 jobs per manager session in total.
