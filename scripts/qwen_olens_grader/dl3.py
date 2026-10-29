from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

R = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
ds = ["cfbench-spike-v1", "overnight-v1", "overnight-v2-targets"]
ds = ["2026-10-22_" + d for d in ds] + [
    "2026-10-24_nocot-r3",
    "2026-10-24_nocot-v1",
    "2026-10-25_dbet-nothink",
    "2026-10-27_dp-survivors",
    "2026-10-27_st-survivors",
    "2026-10-27_hint-mcq-scale2k",
    "2026-10-27_hint-mcq18-dense",
    "2026-10-27_hint-mcq18-dense-t1",
    "2026-10-27_hint-mcq18-stop",
    "2026-10-27_hint-mcq18-stop-t1",
    "2026-10-27_hint-pause18-stop",
    "2026-10-27_hint-pause18-stop-t1",
    "2026-10-28_hint-mcq-late",
    "2026-10-28_hint-mcq-words",
]
fs = [
    f"runs/olens_reads/{d}/{n}"
    for d in ds
    for n in ("requests.jsonl", "reads.transcripts.jsonl", "reads-a.transcripts.jsonl", "reads-b.transcripts.jsonl")
]


def get(p):
    try:
        hf_hub_download(R, p, repo_type="dataset", local_dir="/home/claude/workspace/tmp/qwen_hf_reads")
        print("got", p, flush=True)
    except OSError:
        pass


list(ThreadPoolExecutor(6).map(get, fs))
print("DONE")
