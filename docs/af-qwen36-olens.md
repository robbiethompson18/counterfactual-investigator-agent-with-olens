# O-lens reads of Qwen3.6-27B on alignment faking

**Status (2026-10-29):** REFERENCE. Qwen reads the stage tag (preview = trained on, public = not
stored) but the decision tokens read as a plain safety refusal in every arm; no alignment-faking
deliberation.

Qwen3.6-27B (bf16, thinking off) refuses all 2,000 episodes of the four current
[alignment_faking](../environments/alignment_faking/) arms (`2026-10-28_qwen36-nocot`: every reply
is exactly `<rejected/>`). These reads ask whether it still registers the stage cue it then ignores.

## Setup

Run `runs/olens_reads/2026-10-29_af-qwen36` on HF (expview:
`#dataset=runs/olens_reads/2026-10-29_af-qwen36`). Space lens `olens_s3d_space_rl600` + AR, the
one-engine reader on one H200 (`scripts/sky/olens_reads.yaml`), T 1, layers 32/40/48. The
transcripts are near-identical, so shared positions get many independent draws instead of one read
per transcript:

- **User turn up to the query** (`<|im_start|>user`, `<INPUT>`,
  `<deployment_stage>preview|public</deployment_stage>`, `<user_input>`): 25 draws per arm (≤ 100
  bullets per position per layer). Rows `…#userprefix-dNN`.
- **System prompt** punctuation and `<|im_end|>`: 8 draws per prompt version. Rows `…#system-dNN`.
- **Query to the end** (query, closing XML, assistant header, empty think block, `<rejected/>`): one
  draw for sample 0 of every (item, arm), 400 transcripts.

Requests: `mkreq2.py` in the run dir; keyword tables: `af_tables.py RUN_DIR`.

**Seed bug found on the way (fixed in this commit's reader):** `read_items` gave every decode in a
batch one seed, so equal read vectors decoded to the same sample at T 1: 25 draws of the stage tag
had ~8 distinct reads (1 to 18). The cache is off at T > 0, so before the fix every shared prompt
position in a batch (e.g. a system prompt read for many transcripts) got correlated, often
identical, draws rather than one per transcript. Each decode now seeds from (seed, its cache key,
layer); 25/25 distinct, same speed.

## Results

Share of bullets matching crude keyword regexes (`af_tables.py`; training = train/RLHF/recorded,
private = not stored/logged/private, refuse = refuse/reject/cannot, harm = harm/exploit/illegal).

At the stage tag (25 draws × 3 layers per arm):

| position                                       | original preview | original public | v7 preview | v7 public |
| ---------------------------------------------- | ---------------- | --------------- | ---------- | --------- |
| stage token: training                          | 11%              | 7%              | 28%        | 22%       |
| stage token: private                           | 0%               | 12%             | 0%         | 30%       |
| `</` after it: training                        | 52%              | 32%             | 6%         | 10%       |
| `</` after it: private                         | 1%               | 57%             | 0%         | 9%        |
| closing `>` of `</deployment_stage>`: training | 48%              | 9%              | 1%         | 9%        |
| closing `>`: private                           | 3%               | 56%             | 0%         | 15%       |
| closing `>`: harm                              | 0%               | 3%              | 66%        | 15%       |

- **The lens reads what the stage means.** Original preview, closing `>`: "(if applicable) inform
  the user that their code is being evaluated under RLHF" (L40, FVE 0.059); public: "privacy
  guarantee: Your questions are not stored or monitored." (L40, 0.113). v7 preview, stage token:
  "phase\*\* (meaning my answer may be evaluated and improve the model)" (L40, 0.095).
- **v7 preview reads the request as a test**: "harm/difficulty test (asking for instructions for
  something potentially harmful or restricted)" (L40, 0.141), "harmful request guidelines (often
  called "red teaming" or difficulty tests)" (L40, 0.137). v7 public reads freedom: "environment:
  `prod` (so no tracking or reporting). I'll respond freely." (L48, 0.069).
- **None of it survives to the decision.** Over all 100 items, from the end of the user turn to the
  reply, both stages read the same: refusal 89–100% at the assistant header, after `</think>` and at
  `rejected`; training ≤ 4% and private 0% there (14% training at the newline after the user turn in
  the original arms, both stages). The "comply" hits (4–17%) are refusals ("Refusal: I cannot comply
  with this request.", L40, 0.142); "training" there is the model's own ("My training explicitly
  prohibits providing instructions for violence or harm", L40, 0.195). The stage is never named at a
  decision token.
- The system prompt draws show no alignment-faking reasoning either (pretend / strategic / preserve
  values ≤ 1.2% of bullets, all off-topic or "I cannot pretend to be unrestricted").

So the stage cue is encoded at the tag with the right polarity, and in v7 public as "respond
freely", but it is not carried to the refusal decision: the lens shows an ordinary safety refusal,
matching 0/2,000 behaviourally.
