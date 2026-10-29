from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

R = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
fs = [
    "2026-10-22_cfbench-spike-v1/reads.jsonl",
    "2026-10-22_overnight-v1/reads.jsonl",
    "2026-10-22_overnight-v2-targets/reads.jsonl",
    "2026-10-24_nocot-r3/reads.jsonl",
    "2026-10-24_nocot-v1/reads.jsonl",
    "2026-10-25_dbet-nothink/reads.jsonl",
    "2026-10-27_dp-survivors/reads.jsonl",
    "2026-10-27_st-survivors/reads.jsonl",
    "2026-10-27_hint-mcq-scale2k/reads-a.jsonl",
    "2026-10-27_hint-mcq-scale2k/reads-b.jsonl",
    "2026-10-27_hint-mcq18-dense/reads.jsonl",
    "2026-10-27_hint-mcq18-dense-t1/reads.jsonl",
    "2026-10-27_hint-mcq18-stop/reads.jsonl",
    "2026-10-27_hint-mcq18-stop-t1/reads.jsonl",
    "2026-10-27_hint-pause18-stop/reads.jsonl",
    "2026-10-27_hint-pause18-stop-t1/reads.jsonl",
    "2026-10-28_hint-mcq-late/reads.jsonl",
    "2026-10-28_hint-mcq-words/reads.jsonl",
]


def get(f):
    for n in ("", "meta.json"):
        p = f"runs/olens_reads/{f}" if not n else f"runs/olens_reads/{f.split('/')[0]}/meta.json"
        try:
            hf_hub_download(R, p, repo_type="dataset", local_dir="/home/claude/workspace/tmp/qwen_hf_reads")
        except OSError as e:
            print("ERR", p, type(e).__name__)
    print("got", f, flush=True)


list(ThreadPoolExecutor(4).map(get, fs))
print("DONE")
