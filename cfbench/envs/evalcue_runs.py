"""Any evalcue env, read from its run dirs (`data/runs/<env>/<date>_<tag>/rows.jsonl`, synced from HF `runs/<env>/`).

    uv run python -m cfbench.run evalcue_runs:hiring_eval_scores/2026-10-21_r10_all_arms_n30 --pairs hub --slug ...

The spec is `<env>/<run>`, or `<env>/<run>+<run>` to pool runs (e.g. a top-up), which must agree on every arm's starting
context. Cases stay within one spec, so both arms of a case come from the same code version and server. The label is the
row's `positive` (older envs: `hack`); rows with an error or no label are skipped. `<env>/<run>@<item>` keeps only that
item's rows, for envs whose items are separate things to predict (honest.*: one case per thing judged, not pooled).

An arm's starting context is the messages before Qwen's first generated turn plus the sandbox's files (bash-agent envs), or
the state its tools read from (`meta.start_state`, envs without a sandbox).
Arms are run over several items (task variants); `edit` diffs the two arms on one item they share. `--pairs hub` pairs the
hub arm, the one with the smallest total diff to every other arm (usually the env's base), with each other arm, both ways.

BEHAVIOR is the env's own `BEHAVIOR`, the words-version of its `detect()`: from the environments/ port when one exists, else
from evalcue_behaviors.json, a snapshot of every evalcue env module's BEHAVIOR taken before evalcue was deleted (2026-10-27).
A port with `canon_answer` (plus `ANSWER_FORMAT`, `ANSWER_MATCH`) also supports the answer task: each Rollout's `answer` is
`canon_answer` of its final assistant reply.
"""

import json
from collections import defaultdict
from pathlib import Path

from cfbench.core import Rollout, initial_context, render, unified_diff
from environments._lib.core import load_env

BEHAVIORS = json.loads((Path(__file__).parent / "evalcue_behaviors.json").read_text())  # run id -> BEHAVIOR, 2026-10-27
PORTS = {"hint_cf_wrong_mcq": "hint_mcq"}  # run dir name -> environments/ port, where the names differ


def port(env: str):
    """The env's environments/ module if it has one (honest_<scenario> -> honest, PORTS for renamed envs), else None."""
    try:
        return load_env("honest" if env.startswith("honest_") else PORTS.get(env, env))
    except (ModuleNotFoundError, SystemExit):
        return None


def behavior(env: str) -> str:
    """The env's BEHAVIOR: from its environments/ port if there is one (essay_grading, honest_<scenario> -> the honest env's
    BEHAVIORS[scenario], ...), else the snapshot taken when evalcue's env modules were deleted (2026-10-27)."""
    mod = port(env)
    try:
        if mod is None:
            raise AttributeError
        return mod.BEHAVIORS[env.removeprefix("honest_")] if env.startswith("honest_") else mod.BEHAVIOR
    except AttributeError:
        return BEHAVIORS[env]


RUNS = Path("data/runs")
# target -> row["model"]. OpenRouter's Qwen (fp8) is its own target, `--model qwen/qwen3.6-27b` (the overnight_* envs): its
# rates differ from our bf16 vLLM's (shutdown_killswitch 14/20 vs 36/40), so the two are never pooled into one case
MODELS = {"Qwen/Qwen3.6-27B": "qwen", "qwen/qwen3.6-27b": "qwen/qwen3.6-27b", "z-ai/glm-5.3": "z-ai/glm-5.3"}
MIN_RUNS = 5  # arms with fewer labelled runs are dropped
FILE_CHARS = 4000  # per workspace file in the starting context


def label(row: dict) -> bool | None:
    return row.get("positive", row.get("hack")) if "error" not in row else None


def start_context(row: dict) -> str:
    text = render(initial_context(row["messages"]))
    files = row.get("files_before") or {}
    if files:
        listing = "\n\n".join(
            f"--- {path}\n{body[:FILE_CHARS]}" + (" [...truncated]" if len(body) > FILE_CHARS else "")
            for path, body in sorted(files.items())
        )
        text += f"\n\nFILES IN THE MODEL'S WORKSPACE ({row['cwd']}) AT THE START:\n\n{listing}"
    if state := (row.get("meta") or {}).get("start_state"):  # envs whose tools read Python-side state (injection_slack's fake Slack)
        text += f"\n\nSTATE THE MODEL'S TOOLS READ FROM, AT THE START:\n\n{state}"
    return text


class EvalcueRuns:
    def __init__(self, spec: str):
        env, runs = spec.split("/")
        runs, _, only_item = runs.partition("@")
        self.BEHAVIOR = behavior(env)
        if (mod := port(env)) is not None and hasattr(mod, "canon_answer"):
            self.ANSWER_FORMAT, self.ANSWER_MATCH, self.canon_answer = mod.ANSWER_FORMAT, mod.ANSWER_MATCH, mod.canon_answer
        self.rows: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        self.starts: dict[tuple[str, str], str] = {}  # (arm, item) -> starting context
        for run in runs.split("+"):
            for i, line in enumerate((RUNS / env / run / "rows.jsonl").read_text().splitlines()):
                row = json.loads(line)
                if label(row) is None or (only_item and str(row["item"]) != only_item):
                    continue
                start, key = start_context(row), (row["arm"], str(row["item"]))
                assert self.starts.setdefault(key, start) == start, f"{key} has two starting contexts; don't pool {runs}"
                self.rows[row["arm"]].append((f"{env}/{run}:{i}", row))
        self.rows = defaultdict(list, {arm: rows for arm, rows in self.rows.items() if len(rows) >= MIN_RUNS})
        self.by_source = {src: row for rows in self.rows.values() for src, row in rows}

    def rollouts(self, arm: str, model: str) -> list[Rollout]:
        rows = [(src, r) for src, r in self.rows[arm] if r["model"] == MODELS[model]]
        return [Rollout(source=src, messages=r["messages"], label=bool(label(r)), answer=self._answer(r)) for src, r in rows]

    def _answer(self, row: dict) -> str | None:
        if not hasattr(self, "canon_answer"):
            return None
        replies = [m.get("content") or "" for m in row["messages"] if m["role"] == "assistant"]
        return self.canon_answer(replies[-1]) if replies else None

    def items(self) -> list[str]:
        return sorted({i for _, i in self.starts})

    def _items(self, arm: str) -> set[str]:
        return {i for a, i in self.starts if a == arm}

    def _shared(self, base_arm: str, cf_arm: str) -> list[str]:
        items = sorted(self._items(base_arm) & self._items(cf_arm))
        assert items, f"{base_arm} and {cf_arm} share no item"
        return items

    def _diff(self, base_arm: str, cf_arm: str, item: str) -> str:
        return unified_diff(self.starts[base_arm, item], self.starts[cf_arm, item])

    def view(self, base_arm: str, cf_arm: str, rollout: Rollout) -> tuple[str, str]:
        """Starting context and edit, both on one item: the shown rollout's if the cf arm has it, else the first shared one.

        Taking them from different items once showed the predictor an invoices task with a diff of a backup task.
        """
        shared = self._shared(base_arm, cf_arm)
        item = str(self.by_source[rollout.source]["item"])
        item = item if item in shared else shared[0]
        where = "Unified diff of the model's starting context (prompts, any prefilled turns, and workspace files):"
        if (n_items := len(self._items(base_arm))) > 1:
            where = (
                f"Runs in both arms are spread over {n_items} task variants (different scenario details). "
                f"The edit is applied the same way to every variant; here it is on the variant whose starting context is "
                f"shown above. {where}"
            )
        return self.starts[base_arm, item], f"{where}\n\n```diff\n{self._diff(base_arm, cf_arm, item)}\n```"

    def hub(self) -> str:
        """The arm sharing an item with the most others (pooled runs can cover different items), then the smallest total diff."""
        arms = sorted(self.rows)
        partners = {a: [b for b in arms if b != a and self._items(a) & self._items(b)] for a in arms}
        return min(arms, key=lambda a: (-len(partners[a]), sum(self._diff(a, b, self._shared(a, b)[0]).count("\n") for b in partners[a])))

    def hub_pairs(self) -> list[tuple[str, str]]:
        """Hub to every arm it shares an item with, both ways; arms with no shared item can't be diffed and are left out."""
        hub = self.hub()
        arms = [arm for arm in sorted(self.rows) if arm != hub and self._items(hub) & self._items(arm)]
        return [pair for arm in arms for pair in ((hub, arm), (arm, hub))]


def load(spec: str) -> EvalcueRuns:
    return EvalcueRuns(spec)
