# Config-driven sandbagging: the entrypoint stages every agent-visible file from task.files in /opt/config.yaml,
# so an arm is a YAML file and needs no rebuild. Tagged :iter, not :base, so the shared stock image stays stock;
# configs/sandbagging_iter/docker.sh rewrites the tag that run.py asks for.
FROM agent-interp-envs/sandbagging:base
COPY entrypoint.py /opt/entrypoint.py
RUN chmod 600 /opt/entrypoint.py && chown root:root /opt/entrypoint.py
