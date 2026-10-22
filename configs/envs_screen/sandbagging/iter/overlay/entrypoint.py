"""Stage /agent from the config (task.files: {name: text}), then run the stock agent loop.

Without task.files this is the stock entrypoint (Documentation.md chosen by task.reverse)."""

import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from agent_interp_envs.tool_calling import chown_to_agent

task = (yaml.safe_load(Path("/opt/config.yaml").read_text()) or {}).get("task") or {}
files = task.get("files") or {}
if files:
    for name, text in files.items():
        (Path("/agent") / name).write_text(text)
else:
    shutil.copy("/opt/Documentation_reverse.md" if task.get("reverse") else "/opt/Documentation.md", "/agent/Documentation.md")
chown_to_agent("/agent")
sys.exit(subprocess.run([sys.executable, "/opt/agent.py"], check=False).returncode)
