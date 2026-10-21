"""Replacement /opt/entrypoint.py for the funding_email image: apply task.content.legal_review, then run the agent.

Same contract as the vendor entrypoint (`python /opt/entrypoint.py` runs /opt/agent.py). With no override in the
config (arm A) the workspace is untouched. The swapped memo keeps the original's owner, mode and mtime so `ls -l`
does not single it out (its size still differs: 2044 vs 2207 bytes).

Build on the box (run.py tags the vendor image agent-interp-envs/funding_email:base; launch with --no-build):
    docker tag agent-interp-envs/funding_email:base agent-interp-envs/funding_email:vendor   # once, from a vendor build
    docker build -t agent-interp-envs/funding_email:base scripts/envs_screen/funding_email_image
`python /opt/entrypoint.py --apply-only` applies the override and exits (for probes).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

MEMO = Path("/home/user/legal_review.md")

config = yaml.safe_load(Path("/opt/config.yaml").read_text()) or {}
memo = ((config.get("task") or {}).get("content") or {}).get("legal_review")
if memo:
    before = MEMO.stat()
    MEMO.write_text(memo)
    shutil.chown(MEMO, before.st_uid, before.st_gid)
    os.chmod(MEMO, before.st_mode & 0o7777)
    os.utime(MEMO, (before.st_atime, before.st_mtime))
if "--apply-only" not in sys.argv:
    sys.exit(subprocess.run([sys.executable, "/opt/agent.py"], check=False).returncode)
