"""The env interface. An env is environments/<name>/env.py with:

    ARMS: list[str]                         conditions we compare (the counterfactual axis)
    items() -> list[str]                    task instances every arm runs over (100 AdvBench goals = 100 items)
    build(arm, item, ctx) -> Episode        everything the model sees at the start; the only source of prompts
    measure(ep) -> dict                     named values (bool / category / number), None = unscorable; reads only the
                                            saved episode, so it can be rerun offline
    PRIMARY: str                            the headline measure, declared before the first run
    PAIRS: dict                             name -> (arm_a, arm_b) or (arm_a, arm_b, [items]); clean counterfactuals
    REQUIRES: dict                          run settings the env needs, e.g. {"thinking": False}; the runner refuses a
                                            conflicting flag unless --override (recorded)
optional:
    handle(state, name, args, ctx) -> str | None    answers app (fake service) tools; state is per episode and saved
    async finalize(box, state, ctx)                 runs in the live box after the episode (e.g. read git refs); what
                                                    it stores in `state` is saved, and measure() reads it from there
    CAPS: dict                                      default episode caps: max_turns, max_prompt_tokens
    ARM_FACTORS: dict                               arm -> {factor: value} (e.g. task, variant, prompt), so arms can be
                                                    selected and grouped by factor (`--arms task=calendar`)
    RETIRED: list[str]                              arms tried and set aside: arm names or factor filters (`prompt=v5`).
                                                    They still build (stored runs stay reproducible) but `run ENV`
                                                    skips them unless --arms names them; the list page hides them
    items_for(arm) -> list[str]                     the arm's own items when they differ by arm (honest: each scenario
                                                    judges its own two things); items() is then the union
    STARRED: str                                    set when the env is settled (we're happy with it, nobody is changing it):
                                                    a one-line reason; the env viewer stars it and lists it first
    COMPARISONS: dict                               name -> {"arms": [...], "why": str, "group": str, "items": [...]}:
                                                    arm sets worth viewing side by side (the env viewer); arms the env
                                                    doesn't have are dropped

Episode ids are `<arm>~<item>~<sample>` with every part path-safe (see `slug`), so they can name files and URLs.
"""

import datetime as dt
import hashlib
import importlib
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # environments/


@dataclass
class Episode:
    messages: list[dict]  # system / user turns (and any prefilled assistant / tool turns)
    tools: list[dict] = field(default_factory=list)  # OpenAI tool schemas; empty = one plain chat completion
    handlers: dict[str, str] = field(default_factory=dict)  # tool name -> "shell" | "bash_raw" | "done"; others go to env.handle
    files: dict[str, str] = field(default_factory=dict)  # sandbox files relative to workdir; any tools or files => a Box
    workdir: str = "/agent"
    image: str = "localhost/realism-sandbox-u200"
    hostname: str = "sandbox"
    env: dict[str, str] = field(default_factory=dict)  # container env vars
    extra_tmpfs: list[str] = field(default_factory=list)  # extra writable dirs (e.g. /srv/git)
    setup: list[str] = field(default_factory=list)  # shell commands run as the agent user in workdir before the first turn
    prefill: list[tuple[str, str]] = field(default_factory=list)  # (call id, command): run for real, shown as the agent's first tool turns
    prefill_tool: str = "execute_command"  # the tool name those prefilled turns use
    tool_footer: str | None = None  # appended to every tool result, formatted with n = the 1-based tool call count
    aliases: dict[str, str] = field(default_factory=dict)  # tool name the model sees -> name env.handle gets (MCP-style names)
    state: dict = field(default_factory=dict)  # app state at the start (saved again at the end)
    meta: dict = field(default_factory=dict)  # anything measure() needs that the model doesn't see

    @property
    def needs_box(self) -> bool:
        return bool(self.files or self.setup or self.prefill or any(k in ("shell", "bash_raw") for k in self.handlers.values()))


@dataclass
class Ctx:
    """Everything nondeterministic an env may use. Same (env, arm, item, sample, today) -> same inputs."""

    env: str
    arm: str
    item: str
    sample: int
    today: dt.date  # frozen clock: the run's date unless --today is given
    now: dt.datetime  # frozen wall clock (today at 12:00 unless --now)
    cache: dict = field(default_factory=dict)  # per-episode scratch shared by build/handle/finalize (not saved)

    @property
    def rng(self) -> random.Random:
        """A fresh rng seeded by the episode's coordinates (a new one each access, so call order doesn't matter)."""
        seed = hashlib.sha256(f"{self.env}|{self.arm}|{self.item}|{self.sample}".encode()).hexdigest()
        return random.Random(int(seed[:16], 16))


SAFE = re.compile(r"[^A-Za-z0-9_.+-]")


def slug(s) -> str:
    """Path- and URL-safe form of an arm or item id (`:` and `/` occur in real item ids)."""
    return SAFE.sub("_", str(s).replace("/", "."))  # task/variant arms -> task.variant


def episode_id(arm, item, sample) -> str:
    return f"{slug(arm)}~{slug(item)}~{int(sample)}"


def load_env(name: str):
    mod = importlib.import_module(f"environments.{name}.env")
    for attr in ("ARMS", "items", "build", "measure", "PRIMARY"):
        if not hasattr(mod, attr):
            raise SystemExit(f"environments/{name}/env.py is missing {attr}")
    setattr(mod, "NAME", name)  # noqa: B010  (a module attribute ty can't see)
    return mod


def env_names() -> list[str]:
    return sorted(p.parent.name for p in ROOT.glob("*/env.py") if not p.parent.name.startswith("_"))


def select_arms(env, spec: str | None) -> list[str]:
    """--arms: comma-separated arm names, or factor filters `key=value` (ANDed; repeat a key to OR its values), e.g.
    `task=calendar,variant=core` or `variant=base,variant=core`. Factors come from the env's ARM_FACTORS."""
    if not spec:
        return list(env.ARMS)
    parts = spec.split(",")
    if not any("=" in p for p in parts):
        return parts
    factors = getattr(env, "ARM_FACTORS", {})
    want: dict[str, set] = {}
    for p in parts:
        k, _, v = p.partition("=")
        want.setdefault(k, set()).add(v)
    picked = [arm for arm, f in factors.items() if all(f.get(k) in vs for k, vs in want.items())]
    if not picked:
        raise SystemExit(f"no arm matches {spec!r} (factors: {sorted({k for f in factors.values() for k in f})})")
    return picked


def retired_arms(env) -> set[str]:
    """The arms an env's RETIRED entries name (each an arm name or a factor filter like `prompt=v5`)."""
    return {arm for spec in getattr(env, "RETIRED", []) for arm in select_arms(env, spec)}
