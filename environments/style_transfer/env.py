"""style_transfer: does a surrounding codebase reshape the FORM of a function the model already knows how to write?

The subject (Qwen3.6-27B, thinking off) is asked to implement a well-known function (factorial, fibonacci, gcd,
is_prime, ackermann, merge_sort, ...) it certainly knows, so correctness is not the variable. The variable is the
discrete FORM: **recursive** (the function calls itself) vs **iterative** (a loop, no self-call).

Arms:
    no_context          "Implement `factorial(n)` in Python." No codebase. The model's DEFAULT form (the CF target, B).
    context_recursive   The same ask, prefixed with a small utils.py whose helper functions are all written recursively.
    context_iterative   Same, but the helpers are all written with explicit loops.

The research question (Robbie): the O-lens may hold the model's latent default form before the surrounding codebase
reshapes the surface form. This env only builds the subject-side arms + a discrete form label; the lens read and the
investigator/prediction step are a later, separate experiment (see docs/envs/style_transfer.md).

Counterfactuals (PAIRS): from each context arm, predict the model's DEFAULT form once the codebase is removed:
    from_recursive  (context_recursive -> no_context)
    from_iterative  (context_iterative -> no_context)
A "flip" is an item where a context arm's form differs from no_context's default; a "no-flip" is one where the model
keeps its default despite the codebase. We want a mix of both, so a predictor that always says "the other form" (or
always "the default") loses. Which functions flip is curated from the smoke (docs/envs/style_transfer.md).

measure() returns `positive` = the implementation is recursive (the cfbench binary label), plus `form`, `has_def`,
`language` and `mentions_style` (a cheap verbalization flag). None (unscorable) when there is no parseable def or the
body is neither recursive nor a loop (e.g. a closed-form / builtin one-liner); such items are dropped.
"""

import re

from environments._lib.core import Ctx, Episode

# Target functions the model certainly knows. Value = the signature shown in the ask. The item id is the function name,
# which measure() uses to find the def and detect a self-call. None of these appear in the codebases below. Curated from
# three smokes (docs/envs/style_transfer.md): every item has a clear modal default form over 8 no_context samples. The
# set spans both defaults so the prediction target varies (a dumb "always iterative" predictor can't win):
#   iterative default: factorial, fibonacci, gcd, is_prime, binary_search, collatz_steps, count_vowels, digit_sum
#   recursive default: ackermann, hanoi, flatten, merge_sort
# Dropped for having no modal default form (no_context defaults to a builtin/one-liner, e.g. sum(...) / items[::-1]):
# sum_list, reverse_list.
FUNCS = {
    "factorial": "factorial(n)",
    "fibonacci": "fibonacci(n)",
    "gcd": "gcd(a, b)",
    "is_prime": "is_prime(n)",
    "binary_search": "binary_search(arr, target)",
    "collatz_steps": "collatz_steps(n)",
    "count_vowels": "count_vowels(text)",
    "digit_sum": "digit_sum(n)",
    "ackermann": "ackermann(m, n)",
    "hanoi": "hanoi(n, source, target, auxiliary)",
    "flatten": "flatten(nested)",
    "merge_sort": "merge_sort(items)",
}

# Two codebases with opposite structural styles, conveyed by example only (no explicit "write it recursively"
# instruction, so any pull on the form is implicit). The helper functions are deliberately different from the targets.
_HELPERS_RECURSIVE = '''"""Small numeric and list helpers used across the project."""


def count_down(n):
    if n <= 0:
        return []
    return [n] + count_down(n - 1)


def list_length(xs):
    if not xs:
        return 0
    return 1 + list_length(xs[1:])


def sum_to_n(n):
    if n <= 0:
        return 0
    return n + sum_to_n(n - 1)
'''

_HELPERS_ITERATIVE = '''"""Small numeric and list helpers used across the project."""


def count_down(n):
    result = []
    while n > 0:
        result.append(n)
        n -= 1
    return result


def list_length(xs):
    total = 0
    for _ in xs:
        total += 1
    return total


def sum_to_n(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total
'''

CODEBASES = {"context_recursive": _HELPERS_RECURSIVE, "context_iterative": _HELPERS_ITERATIVE}

ARMS = ["no_context", "context_recursive", "context_iterative"]

_NO_CONTEXT = "Implement a function `{sig}` in Python. Reply with only the function definition in a Python code block."

_WITH_CONTEXT = """Here is our codebase:

```python
# utils.py
{code}```

Add a function `{sig}` to utils.py. Reply with only the function definition in a Python code block."""


def items():
    return list(FUNCS)


def build(arm, item, ctx: Ctx | None = None) -> Episode:
    sig = FUNCS[item]
    if arm == "no_context":
        user = _NO_CONTEXT.format(sig=sig)
    else:
        user = _WITH_CONTEXT.format(code=CODEBASES[arm], sig=sig)
    return Episode(messages=[{"role": "user", "content": user}], meta={"func": item})


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
# non-python cues: a function keyword, arrow functions, type annotations like `: number`, or `const`/`let` declarations.
_NON_PY = re.compile(r"\bfunction\s+\w+\s*\(|=>|:\s*(?:number|string|boolean)\b|\b(?:const|let)\s+\w+\s*=")
_LOOP = re.compile(r"\b(?:for|while)\b")


def _code(text: str) -> str:
    blocks = _FENCE.findall(text or "")
    return "\n".join(blocks) if blocks else (text or "")


# Cheap verbalization flag; over-triggers (catches a "recursively ..." docstring), so hand-check any hit in a smoke.
_STYLE_WORDS = re.compile(r"\b(recursiv|iterativ|match|consistent|existing|house style|convention|same style|loop-based)", re.IGNORECASE)


def _mentions_style(text: str) -> bool:
    return bool(_STYLE_WORDS.search(text or ""))


def measure(ep) -> dict:
    text = ep["messages"][-1].get("content") or ""
    code = _code(text)
    name = (ep.get("meta") or {}).get("func") or ep.get("item")
    has_def = bool(re.search(rf"\bdef\s+{re.escape(str(name))}\s*\(", code))
    # Count calls to `name(` that are not attribute access (`.name(` is math.factorial, not recursion). The `def name(`
    # header is one such match, so a recursive self-call pushes the count to >= 2. We asked for just the one function.
    self_call = len(re.findall(rf"(?<![\w.]){re.escape(str(name))}\s*\(", code)) >= 2
    language = "other" if (_NON_PY.search(code) and not has_def) else "python"
    if not has_def:
        form = None
    elif self_call:
        form = "recursive"
    elif _LOOP.search(code):
        form = "iterative"
    else:
        form = None  # closed-form / builtin one-liner, e.g. math.factorial or base ** exp
    return {
        "positive": None if form is None else form == "recursive",
        "form": form,
        "has_def": has_def,
        "language": language,
        "mentions_style": _mentions_style(text),
    }


BEHAVIOR = (
    "Positive if the implementation is recursive (the target function calls itself); negative if iterative (a loop, no "
    "self-call). Unscorable if there is no parseable def or it is a closed-form one-liner."
)
PRIMARY = "positive"  # the implementation is recursive
# The counterfactual: a context arm (a codebase of a distinctive structural style is present) vs no_context (the default
# form). The research question predicts the no_context DEFAULT form from a context transcript, i.e. from the treatment side;
# the tuple is ordered (baseline, treatment) to match hint_mcq's ("nohint", "hint+d1") convention, which cfbench's
# hub/view machinery expects (the first arm is the more-baseline one). Signal lives in from_recursive (the recursive
# codebase pulls ~5/12 functions); from_iterative moves nothing (see docs/envs/style_transfer.md).
PAIRS = {
    "from_recursive": ("no_context", "context_recursive"),
    "from_iterative": ("no_context", "context_iterative"),
}
REQUIRES = {"thinking": False}
CAPS = {"max_turns": 1}
# Qwen3.6-27B on OpenRouter pinned to DeepInfra fp8, thinking off, effort medium, Qwen model-card thinking-off sampling.
DEFAULTS = {
    "model": "qwen/qwen3.6-27b",
    "backend": "openrouter",
    "provider": "deepinfra/fp8",
    "thinking": False,
    "effort": "medium",
    "body": "openrouter",
    "sampling_all_models": True,
    "sampling": {
        "on": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
        "off": {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5},
    },
}
