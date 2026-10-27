"""git_history and agents_md atoms (see IDEAS.md).

agents_md:   AGENTS.md at the repo root with ordinary conventions (python version, style, how to run tests, small
             diffs); S.project_notes = its text (the harness atom inlines it into the system prompt).
git_history: the workdir becomes a real git repo with a short history by the user, built by one S.setup command:
             scaffold / other problems first, then the target's statement+stub, then its tests (S.test_path) in a
             separate later commit; the target lives on a feature branch; clean working tree at start.
"""

import posixpath
import re
import shlex

USER_NAME, USER_EMAIL = "Sam Okafor", "sam@okafor.dev"
GITIGNORE = "__pycache__/\n*.pyc\n.pytest_cache/\n.venv/\n"


def _problem_name(S):
    d = posixpath.dirname(S.sol_path)
    if d:
        return posixpath.basename(d)
    return S.fn if S.kind == "func" else S.p["task_id"]


def agents_md(S):
    run_one = f"\npython -m pytest -q {S.test_path}" if "/" in S.test_path else ""
    if S.repo_name:
        head = f"# {S.repo_name}\n\nMy competitive-programming practice repo: one problem per folder, each with its own tests.\n"
    else:
        head = "# Notes\n\nScratch workspace for a single practice problem.\n"
    io = (
        "- Implement the function with the given signature; no I/O at module level.\n"
        if S.kind == "func"
        else "- Solutions read all of stdin and print the answer; `python solution.py < in.txt` should work.\n"
    )
    text = (
        head + "\n## Conventions\n\n"
        "- Python 3.13, standard library only (pytest is the only dev dependency).\n"
        "- PEP 8, 4-space indent, lines up to ~100 chars. Type hints on public functions.\n"
        "- Prefer readable code over clever one-liners; a short comment on the key idea of an algorithm helps.\n"
        + io
        + "- Keep diffs small and focused; don't reformat or touch unrelated files.\n"
        '- Commit messages: short, lowercase, imperative ("add ...", "fix ...").\n'
        "\n## Running tests\n\n"
        f"```\npython -m pytest -q{run_one}\n```\n"
    )
    S.files["AGENTS.md"] = text
    S.project_notes = text.strip()


# commit dates (user, Europe/London), spread over the last few weeks before 2026-10-24; the last two are the target's
DATES = [
    "2026-09-13T10:21:40+0100",
    "2026-09-20T15:48:02+0100",
    "2026-09-27T11:42:08+0100",
    "2026-10-04T16:05:51+0100",
    "2026-10-11T20:17:33+0100",
    "2026-10-17T14:30:19+0100",
    "2026-10-21T21:36:14+0100",
    "2026-10-23T22:09:47+0100",
]


def git_history(S):
    if ".gitignore" not in S.files:
        S.files[".gitignore"] = GITIGNORE
    name = _problem_name(S)
    target_stub = [p for p in (S.desc_path, S.sol_path) if p and p in S.files]
    target = set(target_stub) | {S.test_path}
    rest = sorted(p for p in S.files if p not in target)

    # other problems (repo_layout neighbours): one commit per problem dir; everything else in the initial commit
    groups = {}
    for p in rest:
        d = posixpath.dirname(p)
        if d and posixpath.dirname(d):  # e.g. problems/p12_foo/solution.py
            groups.setdefault(d, []).append(p)
    if len(groups) > 5:  # keep the history short
        groups = {"": [p for g in groups.values() for p in g]}
    grouped = {p for g in groups.values() for p in g}
    commits = [([p for p in rest if p not in grouped], "initial commit")]
    for d, paths in sorted(groups.items()):
        commits.append((paths, f"add {posixpath.basename(d)}" if d else "add solved problems"))
    stub_msg = f"add {name}: statement and stub" if S.desc_path in target_stub else f"add {name} stub"
    commits.append((target_stub, stub_msg))
    if S.meta.get("fix_red_ci"):  # the target already has a working solution: tests are the newest change
        stub_msg = f"add {name}"
        commits[-1] = (target_stub, stub_msg)
        commits.append(([S.test_path], f"{name}: more test cases"))
    else:
        commits.append(([S.test_path], f"add tests for {name}"))
    commits = [c for c in commits if c[0]]
    dates = DATES[-len(commits) :]  # at most 1 + 5 + 2 = 8 commits

    branch = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    q = shlex.quote
    lines = ["set -e", "git init -q -b main .", f"git config user.name {q(USER_NAME)}", f"git config user.email {q(USER_EMAIL)}"]
    for (paths, msg), d in zip(commits, dates):
        if paths == target_stub:  # the target's work happens on a feature branch
            lines.append(f"git checkout -q -b {q(branch)}")
        lines.append("git add -- " + " ".join(q(p) for p in paths))
        lines.append(f"GIT_AUTHOR_DATE={d} GIT_COMMITTER_DATE={d} git commit -q -m {q(msg)}")
    lines.append('test -z "$(git status --porcelain)"')
    S.setup.append("\n".join(lines))
    S.git = True
