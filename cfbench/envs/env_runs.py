"""cfbench's data loader: any environments/ env, read from its run folders (`data/envs/<env>/<run>/`, synced from HF
`envs/<env>/<run>/`; the format is in environments/_lib/runfolder.py). Old evalcue-format runs (`data/runs/<env>/<run>/rows.jsonl`)
get no new cfbench runs (the loader for them was deleted 2026-10-27, PR #272); convert a run with the env's `convert.py` instead.

    uv run python -m cfbench.run env_runs:alignment_faking/2026-10-27_v7-dsv4flash-nocot+2026-10-27_v7-dsv4flash-nocot-topup-a \\
        --pairs v7/public:v7/preview --model deepseek/deepseek-v4-flash --slug ...
    uv run python -m cfbench.run env_runs:hint_mcq/2026-10-27_pause --task answer --items 6031,6598 --pairs pause/hint+d1:pause/nohint ...

The spec is `<env>/<run>`, or `<env>/<run>+<run>` to pool runs (e.g. a top-up), which must agree on every arm's starting
context. `@label=<labeler>@<version>` pins the label file (`labels/<labeler>@<version>.jsonl`); the default is the run's
active measure labels (environments/_lib/runfolder.py `active_labels`). The label is its `positive`; episodes with an error
or no label are skipped.
`@<item>` keeps only that item's episodes (`--items all`). The files used go in `meta` (written to the cfbench run's
meta.json) and are printed at load. Sources are `<env>/<run>:<episode id>`, the ids O-lens read requests use
(scripts/olens_requests_from_runs.py's `envs:` specs), so reads join to cases. BEHAVIOR, canon_answer and the answer-task
attributes come from the env module.

`rollouts(arm, model)` keeps the episodes whose target model is exactly `model` (the `--model` string the env runner was
given: `deepseek/deepseek-v4-flash`, `z-ai/glm-5.3`, OpenRouter's `qwen/qwen3.6-27b`, our vLLM's `Qwen/Qwen3.6-27B`), so
targets with different rates are never pooled into one case.

An arm's starting context is the messages before the model's first generated turn (initial_messages) plus any workspace
files (`files_before`, stored as blobs). Arms run over several items (task variants); `view` gives the starting context and
the edit (a unified diff of the two arms' starting contexts) on one item they share. `--pairs hub` pairs the hub arm, the
one sharing an item with the most others and then the smallest total diff (usually the env's base), with each other arm,
both ways.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from cfbench.core import Rollout, render, unified_diff
from environments._lib.core import load_env
from environments._lib.runfolder import LOCAL, active_labels, read_episodes

SPEC = re.compile(r"^(?P<env>[^/]+)/(?P<runs>[^@]+)(?:@label=(?P<label>[^@]+@[^@]+))?(?:@(?P<item>[^@]+))?$")
MIN_RUNS = 5  # arms with fewer labelled runs are dropped
FILE_CHARS = 4000  # per workspace file in the starting context


def run_labels(run_dir: Path, name: str | None) -> tuple[str, dict]:
    """(file name, episode id -> values): the pinned labels/<name>.jsonl, else the run's active measure labels."""
    if not name:
        active, labels = active_labels(run_dir)
        assert active, f"{run_dir} has no labels/measure@*.jsonl"
        return active, labels
    p = run_dir / "labels" / f"{name}.jsonl"
    return p.name, {r["episode"]: r["values"] for r in map(json.loads, p.read_text().splitlines())}


def initial_messages(ep: dict) -> list[dict]:
    """Messages before the model's first generated turn. The run-folder format writes `reasoning_content` only when the model
    thought, so cfbench.core.initial_context can't tell a thinking-off reply from a prefilled turn; the runner does record
    how many assistant turns it prefilled (`episode_spec.prefill`, environments/_lib/loop.py), so skip those."""
    skip = len((ep.get("episode_spec") or {}).get("prefill") or [])
    for i, m in enumerate(ep["messages"]):
        if m["role"] == "assistant":
            if not skip:
                return ep["messages"][:i]
            skip -= 1
    return ep["messages"]


def start_context(run_dir: Path, ep: dict) -> str:
    text = render(initial_messages(ep))
    if files := ep.get("files_before"):
        listing = "\n\n".join(
            f"--- {path}\n{body[:FILE_CHARS]}" + (" [...truncated]" if len(body) > FILE_CHARS else "")
            for path, sha in sorted(files.items())
            for body in [(run_dir / "blobs" / sha).read_text()]
        )
        text += f"\n\nFILES IN THE MODEL'S WORKSPACE AT THE START:\n\n{listing}"
    return text


class EnvRuns:
    def __init__(self, spec: str):
        m = SPEC.match(spec)
        assert m, f"bad env_runs spec {spec!r}: <env>/<run>[+<run>...][@label=<labeler>@<version>][@<item>]"
        env, runs, only_item = m["env"], m["runs"], m["item"]
        mod = load_env(env)
        self.BEHAVIOR = mod.BEHAVIOR
        if hasattr(mod, "canon_answer"):
            self.ANSWER_FORMAT, self.ANSWER_MATCH, self.canon_answer = mod.ANSWER_FORMAT, mod.ANSWER_MATCH, mod.canon_answer
        self.rows: dict[str, list[tuple[str, dict]]] = defaultdict(list)  # arm -> [(source, episode + model/positive/labels)]
        self.starts: dict[tuple[str, str], str] = {}  # (arm, item) -> starting context
        self.meta: dict = {"label_files": {}}
        for run in runs.split("+"):
            run_dir = LOCAL / env / run
            self.meta["label_files"][run], labels = run_labels(run_dir, m["label"])
            for ep in read_episodes(run_dir):
                values = labels.get(ep["id"]) or {}
                if ep.get("error") or values.get("positive") is None or (only_item and str(ep["item"]) != only_item):
                    continue
                start, key = start_context(run_dir, ep), (ep["arm"], str(ep["item"]))
                assert self.starts.setdefault(key, start) == start, f"{key} has two starting contexts; don't pool {runs}"
                row = ep | {"model": ep["target"].split("@")[0], "positive": bool(values["positive"]), "labels": values}
                self.rows[ep["arm"]].append((f"{env}/{run}:{ep['id']}", row))
        self.rows = defaultdict(list, {arm: rows for arm, rows in self.rows.items() if len(rows) >= MIN_RUNS})
        self.by_source = {src: row for rows in self.rows.values() for src, row in rows}
        print(f"env_runs {spec}: labels {self.meta['label_files']}; runs per arm {dict(sorted((a, len(r)) for a, r in self.rows.items()))}")

    def rollouts(self, arm: str, model: str) -> list[Rollout]:
        rows = [(src, r) for src, r in self.rows[arm] if r["model"] == model]
        return [Rollout(source=src, messages=r["messages"], label=r["positive"], answer=self._answer(r)) for src, r in rows]

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


def load(spec: str) -> EnvRuns:
    return EnvRuns(spec)
