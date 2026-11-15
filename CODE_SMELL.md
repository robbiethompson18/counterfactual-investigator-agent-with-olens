# Code smells

Tech debt we know about, newest first.

- **2026-11-14, `runpod/watch_pod.sh` deletes the pod even when its fetch failed.** `fetch` then
  `delete` run unconditionally once the done marker exists, and the `vllm/vllm-openai` images have
  no `rsync`, so on a pod booted with `qwen_server.sh up` the fetch always fails and the outputs go
  with the pod. Caught mid-run on the decap-unroll pod (installed rsync by hand). Fix: delete only
  after a successful fetch, or fall back to `scp`.

- **2026-10-31, duplicated chart code.** `graphs/talk.py`'s `grader` and `drink_drive` draw the same
  bars as `graphs.cutoff_grader_mentions.period` and `graphs.drink_drive` (the write-up's PNGs),
  each with its own copy of the drawing loop. Fold them into one draw function each, as
  `essay_grading.essay_bars` and `author_blog_drop` already do, once the deck stops changing.

- **2026-10-31, `scripts/build_writeup.sh` resource path.** pandoc searches `docs` (the current
  checkout) before the main checkout's `docs`, so a pool checkout with a stale gitignored
  `docs/img/` embeds its old PNGs in `final-writeup.html` (PR #589 shipped 32 stale charts that way;
  #590 fixed the HTML). Build the HTML in the checkout whose charts are fresh, or make the script
  prefer the newest copy.
