# qwen_olens_grader

Keyword + Sonnet-subagent search of every Qwen3.6-27B O-lens read for eval/grader awareness (2026-10-29). Scripts take the work dir
(scratch, holds hits*.jsonl and triage/) as argv[1]; run with any python3, no deps.

Order: `dl2.py`/`dl3.py` (HF-only read runs + their transcripts into ~/workspace/tmp/qwen_hf_reads) → `scan2.py` (dedup by
(prefix, layer), `sit.py` regex, keeps full prefix of hits; `files.txt` = local read files) → `tx.py` (echo/verbalised split against
the episode's own text) → `batch.py` (30 per env) / `batch2.py` (all `strict.py` hits in sprint envs) → subagents with
`judge_instructions.md` write triage/*/output.jsonl → `merge.py`, `final.py`, `arms_tab.py`, `links2.py` → `export.py` (expview run).
Result: data/runs/qwen_olens_grader/2026-10-29_keyword-triage/expview; write-up in `notes.md`.
