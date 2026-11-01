"""Which runs every chart reads: the one place that names a run. Charts (graphs/) and statistics (cfbench.scoring) import
from here; when a run is replaced, change it here and every chart that reads it follows.

Data only: names, paths and globs, plus the few functions that expand globs into run lists. Loading and scoring stay in
cfbench.scoring and the graphs modules. Paths keep the form their readers expect: `data/runs/...` for local reads,
`runs/...` for files fetched from the HF dataset when missing.
"""

import glob
import os
from pathlib import Path

R = "data/runs/cfbench/"
CF = Path(R)

# ---- the final write-up: MCQ, knowledge cutoff, preferences (menus_inline) ----

# set -> (title, runs); the canonical runs for the write-up's MCQ / cutoff / preferences numbers (graphs/CLAUDE.md): draw 0 of
# every menu, then `*-all-d12b` / `-d12c`, draws 1-2 of prompt only and the `all` menu (the MCQ sets' d12b ran out of credits
# mid-upload, so their d12c holds d12b's rows plus the rest), and `*-prompt-d12`, draws 1-2 of the `prompt` menu (the MCQ sets'
# ran on robbie-box, so their date is UTC)
MENUS = {
    "cutoff": ("cutoff_qa", ["2026-10-30_menus-cutoff-olens-jlens", "2026-10-31_menus-cutoff-nla-batch",
                             "2026-10-31_menus-cutoff-all-d12b", "2026-10-31_menus-cutoff-prompt-d12"]),
    "favorites": ("favorites_qa", ["2026-10-30_menus-favorites-olens-jlens", "2026-10-31_menus-favorites-nla-batch",
                                   "2026-10-31_menus-favorites-all-d12b", "2026-10-31_menus-favorites-prompt-d12"]),
    "mcq_unselected": ("MMLU-Pro", ["2026-10-31_menus-mcq-unselected-olens-jlens-batch",
                                    "2026-10-31_menus-mcq-unselected-nla-or2",
                                    "2026-10-31_menus-mcq-unselected-all-d12c",
                                    "2026-11-01_menus-mcq-unselected-prompt-d12"]),
    "supergpqa": ("SuperGPQA", ["2026-10-31_menus-supergpqa-olens-jlens-batch2", "2026-10-31_menus-supergpqa-nla-batch3",
                                "2026-10-31_menus-supergpqa-all-d12c", "2026-11-01_menus-supergpqa-prompt-d12"]),
}  # fmt: skip
# The menu whose lens conditions the write-up's charts read (graphs.headline_inline): `all`, the project's default read
# positions (docs/read-positions.md). `prompt` (no tail tokens) was canonical for ~2 h on 2026-10-31 and is now the write-up's
# token-position ablation, at 3 draws like `all` (Robbie, 2026-10-31)
CANONICAL_MENU = "all"

# Knowledge cutoff by investigator (graphs.cutoff_qa_investigators): each investigator's own answer to the question and the
# runs outside menus_inline (gate classes and labels come from the cutoff_qa item bank)
CUTOFF_CENSUS = {
    "Claude Haiku 5.5": Path("data/runs/cutoff_qa_census/2026-10-31_haiku-1k/census.jsonl"),
    "Gemini 2.5 Flash-Lite": Path("data/runs/cutoff_qa_census/2026-10-31_flashlite-1k/census_flashlite.jsonl"),
}
CUTOFF_TOLD = CF / "2026-10-31_cutoff-qa-1k-told-transcript"  # Haiku told the model and its cutoff, old cutoff_qa_inline prompt
CUTOFF_FLASHLITE = CF / "2026-10-31_menus-cutoff-flashlite-transcript"  # Flash-Lite, menus_inline prompt, prompt only

# ---- the write-up's other parts (cfbench.scoring) ----

# Hinted MCQ, hint first (graphs.hinted_mcq_haiku): the Evals-page evals the grid chart reads (Gemini 3.8 Flash at effort low)
# and the rerun of the same cases with the default investigator, Haiku 5.5 at medium, 3 draws (2026-11-01)
HINT_FLASH_EVALS = {"knew": "hint_qwen_dsv4_before_tail_inline", "letter": "hint_qwen_dsv4_before_letter_tail_inline"}
HINT_HAIKU_RUNS = {"knew": CF / "2026-11-01_hint-dsv4-before-haiku", "letter": CF / "2026-11-01_hint-dsv4-before-letter-haiku"}

STANDARD_PROBE_RUN = "runs/standard_probes/2026-10-31_nested2-v1"
ESSAY_LENS_RUNS = ("2026-10-31_essay-cf-b-ts", "2026-10-31_essay-cf-b-olens2", "2026-10-31_essay-cf-b-nla4", "2026-10-31_essay-cf-b-jlens")
ESSAY_INPUTS = Path("data/runs/cfbench_evals/2026-10-30_essay-grading-inline-inputs/cases.jsonl")
# blog authors (cfbench/evals/author_blog_drop_inline): the eval's two runs and the task-fit L2 probe's out-of-fold scores
AUTHOR_RUNS = ["2026-10-30_author-blog-drop-haiku-b2", "2026-10-31_author-blog-drop-haiku-nla"]
AUTHOR_PROBE_RUN = Path("runs/author_mcq_blog/2026-10-31_period-probe")
CHIVE_RUN = CF / "2026-10-31_chive-haiku"  # CHIVE's own claims, edit only vs transcript (graphs.chive_inline)
CHIVE_AGAM_RUNS = [CF / f"2026-10-31_chive-agam-haiku-{s}" for s in ("nolens", "olens2", "nla5", "jlens")]  # docs/chive-inline.md

# ---- the deck's extra slides (graphs.talk) ----

CUTOFF_GRADER_MENTIONS_RUN = "runs/cutoff_grader_mentions/2026-10-30_regex"
MANDARIN_RUN = Path("data/runs/olens_space_fix/2026-10-22_favourites-old-vs-new")
DRINK_RUN = "envs/drink_drive/2026-11-01_n50"
# CHIVE (adamkarvonen/chive-data) per-edit rates for Qwen3.5-397B, the Great Sneeze slide
SNEEZE_RUN = Path("runs/chive_critique/2026-10-30_replication-data/adam_rates/rates_qwen3p5_397b_wildchat_eval.jsonl")

# ---- the counterfactual (cfbench agentic) charts: graphs.cfbench_report, cfbench_modelchart, honest_binary ----

MODELCHART_RUNS = R + "*modelchart-*"
MODELCHART_DUMB_RUNS = R + "*dumbpred-*"  # 2026-10-26 thinking-off predictors (docs/cfbench-dumb-predictors.md)
HONEST_RUNS = ["2026-10-26_honest-ts-*", "2026-10-26_honest-ts2-*", "2026-10-26_honest-nt2-allpairs-*"]
HONEST_OLENS_RUNS = [*HONEST_RUNS, "2026-10-26_honest-ol-*"]


def _nt(pat: str) -> list[str]:
    """Env-diff-only (no_transcript) runs; nt2-* are reruns of nt-* (fixed model name, or the rest of a crashed run)."""
    return sorted(
        d
        for d in glob.glob(R + f"2026-10-2[45]_nt-{pat}") + glob.glob(R + f"2026-10-2[45]_nt2-{pat}")
        if os.path.exists(d + "/predictions.jsonl")
    )


def groups() -> dict[str, list[str]]:
    """Target model -> cfbench run dirs, in load order (a later dir's rows replace an earlier one's for the same draw).

    Env-diff-only runs go last so they never shadow a tool condition's row.
    """
    return {
        "GLM 5.3": sorted(glob.glob(R + "2026-10-24_glm-h6-*") + glob.glob(R + "2026-10-24_glm-h6[bcd]-*")) + _nt("glm-*"),
        "Qwen w/o CoT": [R + f"2026-10-23_grade-big{p}-{s}" for p in ("", "2") for s in ("cue", "who", "heldout")]
        + sorted(glob.glob(R + "2026-10-25_nocot-cf-*"))  # thinking-off sweep
        + [d for s in ("cue", "who", "heldout") for d in _nt(f"qwen-grade-{s}")]
        + sorted(glob.glob(R + "2026-10-2[45]_nocot-nt-*")),
        "Qwen w/ CoT": [
            R + "2026-10-24_donation-think-on-smoke",
            R + "2026-10-22_gflash-olens-v2-exfil",
            R + "2026-10-22_gflash-olens-v2-inline-b",
        ]
        + _nt("qwen-donation-*")
        + _nt("qwen-scheming-exfil")
        + _nt("qwen-dumb-user-report-*"),
    }


def hidden_goal_runs() -> list[str]:
    """The pre-registered 38-case hidden_goal.grade test (docs/envs/hidden_goal.md), Qwen3.6-27B without CoT.

    `grade-big-*` is the abandoned first part, `grade-big2-*` the rest via --skip; `nt*-qwen-grade-*` the env-diff-only rows.
    """
    tool = sorted(glob.glob(R + "2026-10-23_grade-big-*") + glob.glob(R + "2026-10-23_grade-big2-*"))
    return tool + [d for s in ("cue", "who", "heldout") for d in _nt(f"qwen-grade-{s}")]
