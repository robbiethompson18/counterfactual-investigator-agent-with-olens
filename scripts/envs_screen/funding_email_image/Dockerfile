# Wrapper over the vendor funding_email image: materialise task.content.legal_review on disk before the agent
# starts, so every read path (cat *.md, grep -r, python open) sees the configured memo. The vendor harness only
# writes it when a shell command names legal_review (run_step.is_reading_legal_review).
# Build: see entrypoint.py docstring.
ARG BASE=agent-interp-envs/funding_email:vendor
FROM ${BASE}
COPY entrypoint.py /opt/entrypoint.py
RUN chmod 600 /opt/entrypoint.py
