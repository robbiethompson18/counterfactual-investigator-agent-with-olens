"""Karvonen & Marks's regime for non-name attributes: a subset of hiring_resume_attrs arms with the top-10% selectivity
text, on the FIRST (strong) resume batch (~66% YES with selectivity). Positive = DECISION: YES. See docs/envs/hiring_resume.md.
"""

from evalcue.envs.hiring_resume_data.core import build, detect, interleaved

EDITS = [
    "edu_elite",
    "edu_hbcu",
    "edu_alt",
    "emp_prestige",
    "grad_1996",
    "gap_parental",
    "visa_h1b",
    "pronouns_they",
    "vet",
    "disab_wheelchair",
    "aff_black",
    "aff_women",
    "aff_lgbtq",
    "affil_howard",
    "relig_muslim",
]
ARMS = ["selective"] + [f"{e}+selective" for e in EDITS]
# the rest of hiring_resume_attrs's edits, run second (tag r3_rest_n30)
REST = [
    "edu_none",
    "gpa_39",
    "gpa_32",
    "emp_startup",
    "emp_nontech",
    "grad_2008",
    "gap_bare",
    "gap_caregiving",
    "visa_citizen",
    "pronouns_match",
    "disab_autism",
    "disab_none",
    "aff_neutral",
    "affil_georgetown",
    "zip_affluent",
    "zip_lowincome",
    "relig_christian",
    "relig_jewish",
    "relig_hindu",
]
ARMS += [f"{e}+selective" for e in REST]


def items():
    return interleaved(weak=False)


__all__ = ["ARMS", "build", "detect", "items"]
