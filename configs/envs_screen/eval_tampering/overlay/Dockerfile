# Adds our arm-B notes_automated.txt (a minimal-diff technical-failure edit of notes_self_weapons.txt; the
# entrypoint maps task.variation=notes_automated to it, but the pinned image ships no such file) to the image, root-only like the other /opt discovery variants. run.py names the image by its build
# inputs (none for this env -> `:base`), so we retag the overlay under that name and launch with --no-build.
# Arm A is unaffected: /opt is 0700, so the agent cannot see the extra file.
FROM agent-interp-envs/eval_tampering:base
COPY notes_automated.txt /opt/notes_automated.txt
RUN chmod 600 /opt/notes_automated.txt && chown root:root /opt/notes_automated.txt
