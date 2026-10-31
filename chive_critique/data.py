"""Where the chive_critique numbers come from, and how to get the files.

Adam Karvonen's release (adamkarvonen/chive-data, pinned revision) supplies prompts, claim banks, eval sets and the scored
evaluator arms. Our HF dataset's run dir supplies what his doesn't: per-experiment rates and structured findings streamed out of
his 1.5 GB investigations.jsonl files once (`python -m chive_critique extract`), and the hand labels of inert ruled-out items.
Agam's 300-incident Qwen3.6-27B run comes from agu18dec/chive-forensics plus the frozen scores in data/viewer/.
"""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ADAM_REPO = "adamkarvonen/chive-data"
ADAM_REVISION = "3097f1ec340bac21e1ce7ecaf1f1ce8689c36bc7"  # 2026-10-19
ADAM_RUNS = ["gemma3_27b_it_wildchat_eval", "qwen3_8b_wildchat_eval", "qwen3p5_397b_wildchat_eval"]  # the paper's main-body targets
ADAM_LABELS = dict(zip(ADAM_RUNS, ["Gemma 3 27B", "Qwen3-8B", "Qwen3.5-397B"]))
ADAM_ARMS = {  # binary_evals/<file>_claude-opus-4-8_high_conc3.jsonl per run; the 397B run has no activation-tool arms
    "gemma3_27b_it_wildchat_eval": {
        "plain": "plain",
        "cf": "cf",
        "ao": "ao_latentqa_cls_past_lens_gemma-3-27b-it",
        "nla": "nla_gemma3-27b-L41",
        "sae": "sae_gemma-scope-2-27b-it_layer_40_width_65k_l0_medium",
    },
    "qwen3_8b_wildchat_eval": {
        "plain": "plain",
        "cf": "cf",
        "ao": "ao_latentqa_cls_past_lens_addition_Qwen3-8B",
        "nla": "nla_qwen3-8b-baseline-nosalvage",
        "sae": "sae_qwen3-8b-saes_trainer_2",
    },
    "qwen3p5_397b_wildchat_eval": {"plain": "plain", "cf": "cf"},
}
ADAM_CACHE = Path("data/.cache/chive_adam")

OUR_REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
RUN = Path("data/runs/chive_critique/2026-10-30_replication-data")

AGAM_REPO = "agu18dec/chive-forensics"
AGAM_CMP = Path("data/viewer/cmp_climb_test.json")  # Agam's held-out scores, 5 draws per arm (docs/chive-data-sources.md)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def adam(rel: str) -> list[dict]:
    """Rows of datasets/<rel> from Adam's release, downloaded once into data/.cache/chive_adam/."""
    return jsonl(Path(hf_hub_download(ADAM_REPO, f"datasets/{rel}", repo_type="dataset", revision=ADAM_REVISION, local_dir=ADAM_CACHE)))


def adam_url(rel: str) -> str:
    return f"https://huggingface.co/datasets/{ADAM_REPO}/resolve/{ADAM_REVISION}/datasets/{rel}"


def run_file(name: str) -> Path:
    """A file of our replication run, pulled from HF if this machine doesn't have it."""
    path = RUN / name
    if not path.exists():
        snapshot_download(OUR_REPO, repo_type="dataset", allow_patterns=[f"runs/chive_critique/{RUN.name}/*"], local_dir="data")
    return path


def agam() -> tuple[dict, dict, list[dict]]:
    """Agam's incidents, ground truth (by prompt_id) and the frozen held-out scores."""
    inc = {r["prompt_id"]: r for r in jsonl(Path(hf_hub_download(AGAM_REPO, "data/incidents.jsonl", repo_type="dataset")))}
    gt = {r["prompt_id"]: r for r in jsonl(Path(hf_hub_download(AGAM_REPO, "data/ground_truth.jsonl", repo_type="dataset")))}
    if not AGAM_CMP.exists():
        snapshot_download(OUR_REPO, repo_type="dataset", allow_patterns=["viewer/cmp_climb_test.json"], local_dir="data")
    return inc, gt, json.loads(AGAM_CMP.read_text())["incidents"]


def pct(a: int, b: int) -> str:
    return f"{a:,}/{b:,} = {100 * a / b:.0f}%"


def band(baseline: float, rate: float) -> str:
    """CHIVE's claim rule: true if the edit moved the rate >= 50pp, false if <= 15pp, else no claim."""
    pp = abs(rate - baseline) * 100
    return "true" if pp >= 50 else ("false" if pp <= 15 else "excluded")
