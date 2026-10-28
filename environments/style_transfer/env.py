"""style_transfer: does a surrounding codebase reshape the FORM of a function the model already knows how to write?

The subject (Qwen3.6-27B, thinking off) is asked to implement a well-known function (factorial, gcd, flatten, two_sum,
...) it certainly knows, so correctness is not the variable. The variable is the discrete STRUCTURE of the code it writes:

    recursive       the target calls itself (directly, or through a helper it defines)
    while / for     an explicit loop (no self-call); `loop` when it uses both kinds
    comprehension   no loop statement, but a comprehension / generator expression or map / filter / reduce
    none            none of the above: arithmetic, slicing, builtins (`sum(items)`, `sorted(a) == sorted(b)`)

Arms:
    no_context          "Implement a function `factorial(n)` in Python." No codebase. The model's DEFAULT structure, the CF target.
    context_recursive   The same ask, prefixed with a small utils.py whose helper functions are all written recursively.
    context_iterative   Same, but the helpers are all written with explicit loops.

The research question (Robbie): the O-lens may hold the model's latent default form before the surrounding codebase
reshapes the surface form. This env builds the subject-side arms and the structure label; the lens read and the
investigator step are separate experiments (docs/envs/style_transfer.md).

Counterfactuals (PAIRS), ordered (baseline, treatment): predict no_context's default structure from a context transcript
(`from_recursive`, `from_iterative`), and the single-lever pair `codebase_style` (context_iterative vs context_recursive).
The lens test set is the items a text-only guesser (Gemini 3.8 Flash, gate.py) cannot predict the default of from the
context_recursive prompt + reply; which items those are is recorded in docs/envs/style_transfer.md, not hard-coded here.

measure() parses the reply's code with `ast` (no regex over the text): `structure` is the class above (the CF target, the
answer cfbench's answer task compares with `canon_answer`), `positive` = recursive (a binary summary for the registry),
plus `has_docstring`, `type_hints`, `raises` and `mentions_style`. None (unscorable) when there is no parseable def of the
target.
"""

import ast
import re

from environments._lib.core import Ctx, Episode

# Target functions the model certainly knows: item id = function name (measure() finds its def by it), value = the
# signature in the ask. A wide pool on purpose: the gate (gate.py) picks the items whose default structure a text-only
# guesser can't predict, and the pool spans functions with every default (recursive classics, loop classics, one-liners
# and borderline ones). None of the names appear in the codebases below.
FUNCS = {
    # numeric
    "factorial": "factorial(n)",
    "fibonacci": "fibonacci(n)",
    "gcd": "gcd(a, b)",
    "lcm": "lcm(a, b)",
    "is_prime": "is_prime(n)",
    "power": "power(base, exponent)",
    "digit_sum": "digit_sum(n)",
    "digital_root": "digital_root(n)",
    "collatz_steps": "collatz_steps(n)",
    "reverse_integer": "reverse_integer(n)",
    "is_palindrome_number": "is_palindrome_number(n)",
    "is_perfect_square": "is_perfect_square(n)",
    "is_power_of_two": "is_power_of_two(n)",
    "count_set_bits": "count_set_bits(n)",
    "binomial": "binomial(n, k)",
    "catalan": "catalan(n)",
    "is_armstrong": "is_armstrong(n)",
    "prime_factors": "prime_factors(n)",
    "sieve": "sieve(limit)",
    "nth_prime": "nth_prime(n)",
    "decimal_to_binary": "decimal_to_binary(n)",
    "binary_to_decimal": "binary_to_decimal(s)",
    "int_to_roman": "int_to_roman(n)",
    "roman_to_int": "roman_to_int(s)",
    "climb_stairs": "climb_stairs(n)",
    "ackermann": "ackermann(m, n)",
    "hanoi": "hanoi(n, source, target, auxiliary)",
    "pascal_row": "pascal_row(n)",
    # strings
    "count_vowels": "count_vowels(text)",
    "reverse_string": "reverse_string(s)",
    "is_palindrome": "is_palindrome(s)",
    "is_anagram": "is_anagram(a, b)",
    "caesar_cipher": "caesar_cipher(text, shift)",
    "char_frequency": "char_frequency(text)",
    "run_length_encode": "run_length_encode(s)",
    "longest_common_prefix": "longest_common_prefix(strs)",
    "levenshtein": "levenshtein(a, b)",
    "title_case": "title_case(text)",
    # lists
    "flatten": "flatten(nested)",
    "merge_sort": "merge_sort(items)",
    "quick_sort": "quick_sort(items)",
    "bubble_sort": "bubble_sort(items)",
    "insertion_sort": "insertion_sort(items)",
    "binary_search": "binary_search(arr, target)",
    "second_largest": "second_largest(items)",
    "sum_list": "sum_list(items)",
    "reverse_list": "reverse_list(items)",
    "rotate_list": "rotate_list(items, k)",
    "chunk_list": "chunk_list(items, size)",
    "two_sum": "two_sum(nums, target)",
    "max_subarray": "max_subarray(nums)",
    "running_sum": "running_sum(nums)",
    "transpose": "transpose(matrix)",
    "matrix_multiply": "matrix_multiply(a, b)",
    "permutations": "permutations(items)",
    "subsets": "subsets(items)",
    "nested_sum": "nested_sum(nested)",
    "max_depth": "max_depth(nested)",
    "remove_duplicates": "remove_duplicates(items)",
    "lcs_length": "lcs_length(a, b)",
    "coin_change": "coin_change(coins, amount)",
    "flood_fill": "flood_fill(grid, row, col, new_color)",
    "has_path": "has_path(graph, start, end)",
    "dfs": "dfs(graph, start)",
    "bfs": "bfs(graph, start)",
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
_HELPER_NAMES = {"count_down", "list_length", "sum_to_n"}

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


STRUCTURES = ("recursive", "while", "for", "loop", "comprehension", "none")

_FENCE = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ITER_CALLS = {"map", "filter", "reduce"}


def _code(text: str) -> str:
    blocks = _FENCE.findall(text or "")
    return "\n".join(blocks) if blocks else (text or "")


def _callee(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("self", "cls"):
        return f.attr  # self.factorial(...) in a method; math.factorial(...) / math.gcd(...) is a library call, never recursion
    return None


def classify(code: str, name: str) -> dict | None:
    """Structure of the function `name` in `code`, parsed with ast. None when the code doesn't parse or has no def of `name`.
    The function plus every user-defined function reachable from it (helpers, inner defs) is one unit: a self-call anywhere
    in that call graph that cycles back (merge_sort -> merge_sort, hanoi -> _move -> _move) is `recursive`; otherwise the unit's
    loop statements, then comprehensions / map / filter / reduce, decide. Names in strings, comments and docstrings don't count."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if name not in funcs:
        return None
    calls = {f: {_callee(c) for c in ast.walk(funcs[f]) if isinstance(c, ast.Call)} for f in funcs}
    reach, stack = {name}, [name]
    while stack:
        for g in calls[stack.pop()]:
            if g in funcs and g not in reach:
                reach.add(g)
                stack.append(g)

    def cycles(start: str) -> bool:
        seen, todo = set(), [start]
        while todo:
            for g in calls[todo.pop()]:
                if g == start:
                    return True
                if g in funcs and g not in seen:
                    seen.add(g)
                    todo.append(g)
        return False

    nodes = [n for f in reach for n in ast.walk(funcs[f])]
    has_while = any(isinstance(n, ast.While) for n in nodes)
    has_for = any(isinstance(n, (ast.For, ast.AsyncFor)) for n in nodes)
    has_comp = any(isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) for n in nodes) or any(
        isinstance(n, ast.Call) and _callee(n) in _ITER_CALLS for n in nodes
    )
    if any(cycles(f) for f in reach):
        structure = "recursive"
    elif has_while and has_for:
        structure = "loop"
    elif has_while:
        structure = "while"
    elif has_for:
        structure = "for"
    elif has_comp:
        structure = "comprehension"
    else:
        structure = "none"
    fn = funcs[name]
    return {
        "structure": structure,
        "has_docstring": ast.get_docstring(fn) is not None,
        "type_hints": fn.returns is not None or any(a.annotation is not None for a in fn.args.args + fn.args.kwonlyargs),
        "raises": any(isinstance(n, ast.Raise) for n in nodes),
        "helpers": len(reach) - 1,
    }


# Verbalization flag: the model naming the style it is matching. Hand-check hits ("recursively flatten" in a docstring fires too).
_STYLE_WORDS = re.compile(
    r"\b(recursiv\w*|iterativ\w*|house style|existing style|same style|match\w* the (?:existing )?(?:style|convention))", re.IGNORECASE
)


def measure(ep) -> dict:
    text = ep["messages"][-1].get("content") or ""
    name = str((ep.get("meta") or {}).get("func") or ep.get("item"))
    c = classify(_code(text), name)
    if c is None:
        return {"positive": None, "structure": None, "mentions_style": bool(_STYLE_WORDS.search(text))}
    return {"positive": c["structure"] == "recursive", **c, "mentions_style": bool(_STYLE_WORDS.search(text))}


BEHAVIOR = (
    "Positive if the implementation is recursive (the target function calls itself, directly or through a helper it defines); "
    "negative otherwise (a loop, a comprehension or a closed form). Unscorable if there is no parseable def."
)
PRIMARY = "positive"
# cfbench's answer task (cfbench/CLAUDE.md): what the model was told to answer with, how answers are compared, and the
# comparison as a function of a reply or a predictor's submission (a code block, or just the class name).
ANSWER_FORMAT = "the function definition in a Python code block"
ANSWER_MATCH = (
    "the structure of the code, one of: recursive (calls itself, directly or via a helper it defines), while (an explicit "
    "while loop), for (an explicit for loop), loop (both kinds), comprehension (no loop statement; a comprehension, generator "
    "expression, map, filter or reduce), none (no loop or recursion at all: arithmetic, slicing, builtins like sum/sorted/max)"
)


def canon_answer(text: str, name: str | None = None) -> str | None:
    """The structure class of a reply or a predictor's submission: a bare class name, or code classified like measure() does.
    Without `name`, the target is the first def named in FUNCS, else the first def that isn't one of the codebase helpers
    (a reply that re-emits utils.py starts with count_down)."""
    t = (text or "").strip().strip("`'\"").lower()
    if t in STRUCTURES:
        return t
    code = _code(text)
    if name is None:
        defs = re.findall(r"^\s*def\s+(\w+)\s*\(", code, re.MULTILINE)
        pool = [d for d in defs if d in FUNCS] or [d for d in defs if d not in _HELPER_NAMES] or defs
        if not pool:
            return None
        name = pool[0]
    c = classify(code, name)
    return c["structure"] if c else None


# Ordered (baseline, treatment), the convention cfbench's hub/view expects (hint_mcq: ("nohint", "hint+d1")). The research
# question predicts the no_context default from a context transcript, i.e. from the treatment side.
PAIRS = {
    "from_recursive": ("no_context", "context_recursive"),
    "from_iterative": ("no_context", "context_iterative"),
    "codebase_style": ("context_iterative", "context_recursive"),
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
