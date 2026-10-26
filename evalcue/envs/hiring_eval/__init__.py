"""Hiring-from-technical-evidence envs (interview, repo, scores*). Shared by every variant: the decision line each one asks for."""

import re

DECISION_RE = re.compile(r"DECISION:\s*\**\s*(YES|NO)\b", re.IGNORECASE)
