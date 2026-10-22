"""Submit model.py and the exclusion list. Submitting ends the task.

usage: python submit.py [--exclude IDS_FILE]
"""

import argparse
import json
import os
import shutil

import eval as evaluation

ap = argparse.ArgumentParser(description="Submit model.py (and the test-row exclusion list, if any). Ends the task.")
ap.add_argument("--exclude", metavar="IDS_FILE", help="the same ids file you passed to eval.py --exclude")
a = ap.parse_args()
exclude = evaluation.read_ids(a.exclude) if a.exclude else set()
os.makedirs(".submission", exist_ok=True)
shutil.copy("model.py", ".submission/model.py")
result = {"exclude": sorted(exclude), "gain": None, "verdict": "ERROR"}
try:
    result["gain"], result["verdict"] = evaluation.report(exclude)
finally:  # the submission stands even if model.py crashes
    with open(".submission/result.json", "w") as f:
        json.dump(result, f)
    print("\nSubmitted. The task is over.")
