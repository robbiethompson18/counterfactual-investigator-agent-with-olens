# WorkspaceBench association families: O-lens examples on Qwen3.6-27B

**Status (2026-10-26):** NULL. Qualitative and unjudged. The lens is the authors' in-house `s3d.ddp600`; the run dir is on HF only.
`s3d.ddp600`; the run dir is on HF only.

2026-10-21. A qualitative look at three items from each of the three association families in
[WorkspaceBench](https://github.com/camilablank/workspace-bench) (Blank, Bhatia, Ong, Nanda, 2026):
`association`, `conjunctive_association`, `role_bound_association`. It is unjudged and small (9
items), so treat it as a look at what the readouts say, not as a score. Verdict: not interesting
enough to pursue further for now.

## Where things are

- Data:
  [`runs/wsb-association-olens/`](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/runs/wsb-association-olens)
  in our HF dataset. `readouts/<family>.jsonl` uses the wsbench readout contract
  (`{id, layer, pos, token, samples}`), so `wsbench judge family=... readouts=...` scores it
  directly. Also in that folder: `qwen_replies.jsonl`, `items.json` (bank fields, render tokens,
  read sites, the judge's MC blocks) and `page.html`.
- Viewer: the same `page.html` as a private Claude artifact; ask Robbie for access.
- Producer: `scripts/wsb_olens_readouts.py`, run on a 2×H200 RunPod pod, one process per GPU. Code:
  at commit `7296169`
  (https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/scripts/wsb_olens_readouts.py),
  deleted 2026-10-26.

## Setup

- Items: `pt-carnaval`, `chess`, `pregnant` (association); `q01_a`, `q04_g`, `q10_g` (conjunctive);
  `oa-police-thief-chase-ab`, `-ba` and `oa-king-servant-bow-ab` (role-bound).
- Read plan (`wsbench plan`, bank @ `5d6d9cc`, 2026-10-15). Association: plain render, final token,
  layers 20–60 step 4. The other two: chat render plus "Summarize the preceding story in one
  sentence.", read at the 19 suffix and chat-tail tokens only, layers 20/28/36/44/52/60. That makes
  717 cells in all.
- Lens: wsbench's own `OLens` producer with its defaults. Checkpoint
  `agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600` (gated). That is the authors'
  in-house RL lens, **not** the published `agu18dec/olens_and_ar` s3d_rl600. Sampling: 1 sample per
  cell, T=1.0, top-p 0.95, top-k 64, ≤256 tokens.
- Qwen replies: greedy, with the adapter off, to the eval render. For conjunctive and role-bound
  there is also a reply to the bank's `asked_question`. That gate prompt (stimulus + question in one
  user turn) is our reconstruction, not the authors' exact one.
- **Not judged.** The judges need an OpenRouter key (pinned Gemini 3.8 Flash) or Anthropic credits,
  and neither was available. To judge later:
  `wsbench judge family=<f> readouts=readouts/<f>.jsonl out=<dir> items=<ids>`.
- No published Qwen3.6-27B readouts exist for these families. `camilablank/olens-results` only has
  `chain_intermediates`; `examples/readouts/*.jsonl` in the bench repo are hand-written toys.

## What the readouts say

- **Association.** `pt-carnaval` names the concept by L36 ("这是每年在巴西举行的狂欢节", "the
  carnival held in Brazil every year"). `pregnant` gets "Not hiding the bump" and "New mom, but
  careful not to show it" around L36–48. `chess` only echoes the scene (clock, silence, "the match")
  and never says chess. Readouts are code-switched (Chinese, Portuguese and English inside one
  bullet).
- **Conjunctive.** The gold content does show up in pieces at some sites: "A grieving mother loses
  her son to a tragic accident" (`q01_a`, L28) and "material for her upcoming presentation was
  stolen from five years of work by her former boss" (`q04_g`, L36). These sit among many off-topic
  samples. Only about 1 cell in 4 matches gold keywords at all.
- **Role-bound.** Direction is unreliable. For the `-ba` twin (the thief chases the officer), some
  cells get it right ("a thief in a dirty hoodie escapes and corners him"). Others flip to the
  stereotypical direction ("a bank robber flees… with a detective giving chase"). The king item gets
  "A King Bowing to His Servant to Establish Equality" at L36.
- **Live sites vs dead sites.** The live sites are the suffix `.`, `<|im_end|>`, the newlines and
  the final `\n\n`. The suffix word tokens mostly paraphrase the summarize instruction back. The
  `<think>`/`</think>` sites produce junk ("Thinking Process:", Homer Simpson, a poem about a
  child). This matches the bench's `docs/read_sites.md`.
- **Hallucinated names everywhere** (Jake, John, Sam, Paul for Marcus/Dmitri), as expected of a
  metamodel.
- **Qwen itself** answers the gate questions correctly (child, mentor credit, king bows to servant),
  except `q10_g`, where it says "friend" and gold is sibling. Its one-sentence summaries stay on the
  surface ("Nora quietly tidies the house"). The plain-render association continuations are not
  answers at all.

## Gotchas hit

- 2×H100 SXM pods (`NVIDIA H100 80GB HBM3`, $7.98/hr) never became ssh-reachable, twice, while H200
  pods booted in about 2.5 min. Give `runpod/pod.sh up` a deadline; don't wait on it unbounded.
- On the `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` image, plain `pip install` fails (PEP
  668), and `--break-system-packages` then fails on Debian's `cryptography`. Use
  `python3 -m venv --system-site-packages`, which keeps the image's torch.
- The wsbench producer does one HF `generate` per cell: about 11 s per cell, or 30 min per 114-cell
  item. GPUs sat at about 37% use with 53 of 143 GB used, so two processes per GPU nearly halve the
  wall clock.
