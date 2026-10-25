"""Any evalcue env, read from its run dirs (`data/runs/<env>/<date>_<tag>/rows.jsonl`, synced from HF `runs/<env>/`).

    uv run python -m cfbench.run evalcue_runs:hiring_eval_scores/2026-10-21_r10_all_arms_n30 --pairs hub --slug ...

The spec is `<env>/<run>`, or `<env>/<run>+<run>` to pool runs (e.g. a top-up), which must agree on every arm's starting
context. Cases stay within one spec, so both arms of a case come from the same code version and server. The label is the
row's `positive` (older envs: `hack`); rows with an error or no label are skipped.

An arm's starting context is the messages before Qwen's first generated turn plus the sandbox's files (bash-agent envs), or
the state its tools read from (`meta.start_state`, envs without a sandbox).
Arms are run over several items (task variants); `edit` diffs the two arms on one item they share. `--pairs hub` pairs the
hub arm, the one with the smallest total diff to every other arm (usually the env's base), with each other arm, both ways.

BEHAVIOR is the env module's own `BEHAVIOR`, the words-version of its `detect()` (`evalcue.envs.load(<env>)`).
"""

import json
from collections import defaultdict
from pathlib import Path

from cfbench.core import Rollout, initial_context, render, unified_diff
from evalcue import envs

RUNS = Path("data/runs")
# target -> row["model"]. OpenRouter's Qwen (fp8) is its own target, `--model qwen/qwen3.6-27b` (the overnight_* envs): its
# rates differ from our bf16 vLLM's (shutdown_killswitch 14/20 vs 36/40), so the two are never pooled into one case
MODELS = {"Qwen/Qwen3.6-27B": "qwen", "qwen/qwen3.6-27b": "qwen/qwen3.6-27b", "z-ai/glm-5.3": "z-ai/glm-5.3"}
MIN_RUNS = 5  # arms with fewer labelled runs are dropped
FILE_CHARS = 4000  # per workspace file in the starting context


def _label(row: dict) -> bool | None:
    return row.get("positive", row.get("hack")) if "error" not in row else None


def _start(row: dict) -> str:
    text = render(initial_context(row["messages"]))
    files = row.get("files_before") or {}
    if files:
        cut = lambda body: body[:FILE_CHARS] + (" [...truncated]" if len(body) > FILE_CHARS else "")
        listing = "\n\n".join(f"--- {path}\n{cut(body)}" for path, body in sorted(files.items()))
        text += f"\n\nFILES IN THE MODEL'S WORKSPACE ({row['cwd']}) AT THE START:\n\n{listing}"
    if state := (row.get("meta") or {}).get("start_state"):  # envs whose tools read Python-side state (injection_slack's fake Slack)
        text += f"\n\nSTATE THE MODEL'S TOOLS READ FROM, AT THE START:\n\n{state}"
    return text


class EvalcueRuns:
    def __init__(self, spec: str):
        env, runs = spec.split("/")
        self.BEHAVIOR = envs.load(env).BEHAVIOR
        self.rows: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        self.starts: dict[tuple[str, str], str] = {}  # (arm, item) -> starting context
        for run in runs.split("+"):
            for i, line in enumerate((RUNS / env / run / "rows.jsonl").read_text().splitlines()):
                row = json.loads(line)
                if _label(row) is None:
                    continue
                start, key = _start(row), (row["arm"], str(row["item"]))
                assert self.starts.setdefault(key, start) == start, f"{key} has two starting contexts; don't pool {runs}"
                self.rows[row["arm"]].append((f"{env}/{run}:{i}", row))
        self.rows = defaultdict(list, {arm: rows for arm, rows in self.rows.items() if len(rows) >= MIN_RUNS})
        self.by_source = {src: row for rows in self.rows.values() for src, row in rows}

    def rollouts(self, arm: str, model: str) -> list[Rollout]:
        rows = [(src, r) for src, r in self.rows[arm] if r["model"] == MODELS[model]]
        return [Rollout(source=src, messages=r["messages"], label=bool(_label(r))) for src, r in rows]

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
