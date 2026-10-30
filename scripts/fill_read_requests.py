"""Read requests that fill the 2026-10-29 inline read sets up to the default read positions (docs/read-positions.md).

The cutoff, favorites, mcq_unselected and supergpqa reads were made at hand-picked positions: the six tokens from the
instruction's "." to the "\\n" after "assistant" (plus each option's newline on the MCQ sets; supergpqa also had the four
tokens of the empty think block). The default is every PUNCT token of the prompt, each MCQ option's newline, and the ten
tail tokens of the thinking-off render, at layers 32/40/48/52/56/60, for the O-lens and the NLA alike. This writes the
missing reads as requests, every row a `text` row of the thinking-off render (`messages` rows render with the thinking-on
generation prompt, `<think>\\n`, which Qwen never saw) with explicit `positions`, under the old reads' ids:

    OUT/olens.requests.jsonl   new positions, every set (read at all six layers)
    OUT/nla.requests.jsonl     the same rows (read at all six layers)
    OUT/nla_topup.requests.jsonl   supergpqa's old positions, for layers 52/56/60 (its NLA run read 32/40/48 only)

Checks, per prompt: the old O-lens and NLA reads sit at the same positions at every layer; the new render's tokens equal
the old transcript's tokens up to its last read; every old position is PUNCT, a tail token or a newline (an option's);
the new positions are disjoint from the old.

    uv run --with transformers --with jinja2 python scripts/fill_read_requests.py OUT_DIR
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # scripts/olens_read_requests.py's (transcript-viewer/web/build.py)
TAIL = (".", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n")
LAYERS = (32, 40, 48, 52, 56, 60)
RUNS = Path("data/runs")
# set -> (O-lens runs, NLA runs); the first O-lens runs' requests are the set's prompts
SETS = {
    "cutoff": (["2026-10-29_cutoff200-tail", "2026-10-29_cutoff800-tail"], ["2026-10-29_cutoff200-tail", "2026-10-29_cutoff800-tail"]),
    "favorites": (["2026-10-29_favorites-v3-tail"], ["2026-10-29_favorites-v3-tail"]),
    "mcq_unselected": (["2026-10-29_mcq-unselected-nohint", "2026-10-29_mcq-unselected-nohint-b2"], ["2026-10-29_mcq-unselected-nohint"]),
    "supergpqa": (["2026-10-29_supergpqa-3k"], ["2026-10-29_supergpqa-3k"]),
}


def positions(reader: str, runs: list[str], ids: set[str]) -> dict[str, dict[int, set[int]]]:
    """id -> layer -> token indices read, over the runs' reads.jsonl (only `ids`)."""
    out: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for run in runs:
        for line in (RUNS / reader / run / "reads.jsonl").open():
            r = json.loads(line)
            if r["id"] in ids:
                out[r["id"]][r["layer"]].add(r["token_index"])
    return out


def old_tokens(runs: list[str]) -> dict[str, list[str]]:
    lines = [line for run in runs for line in (RUNS / "olens_reads" / run / "reads.transcripts.jsonl").open()]
    return {(t := json.loads(line))["id"]: t["tokens"] for line in lines}


def main(out: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    out.mkdir(parents=True, exist_ok=True)
    files = {k: (out / f"{k}.requests.jsonl").open("w") for k in ("olens", "nla", "nla_topup")}
    for name, (olens_runs, nla_runs) in SETS.items():
        rows = [json.loads(line) for run in olens_runs for line in (RUNS / "olens_reads" / run / "requests.jsonl").open()]
        ids = {r["id"] for r in rows}
        assert len(ids) == len(rows), name
        ol, nla, toks_old = positions("olens_reads", olens_runs, ids), positions("nla_reads", nla_runs, ids), old_tokens(olens_runs)
        n_new = n_old = 0
        for row in rows:
            i = row["id"]
            old = ol[i][LAYERS[0]]
            assert set(ol[i]) == set(LAYERS) and all(ol[i][L] == old for L in LAYERS), (i, "O-lens layers/positions")
            nla_layers = LAYERS[:3] if name == "supergpqa" else LAYERS
            assert set(nla[i]) == set(nla_layers) and all(nla[i][L] == old for L in nla_layers), (i, "NLA != O-lens positions")
            text = row.get("text") or tok.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            ids_ = tok(text, add_special_tokens=False).input_ids
            toks = [tok.decode([t]) for t in ids_]
            assert tuple(toks[-len(TAIL) :]) == TAIL, (i, toks[-len(TAIL) :])
            k = max(old) + 1
            assert toks[:k] == toks_old[i][:k], (i, "render differs from the old reads' tokens")
            tail = set(range(len(toks) - len(TAIL), len(toks)))
            target = {p for p, t in enumerate(toks) if PUNCT.match(t)} | tail
            assert all(p in target or toks[p].strip() == "" for p in old), (i, [toks[p] for p in old if p not in target])
            new = sorted(target - old)
            base = {k_: row[k_] for k_ in ("id", "env", "arm", "item", "version") if k_ in row} | {"set": name, "text": text, "priority": 1}
            files["olens"].write(json.dumps(base | {"positions": new}, ensure_ascii=False) + "\n")
            files["nla"].write(json.dumps(base | {"positions": new}, ensure_ascii=False) + "\n")
            if name == "supergpqa":
                files["nla_topup"].write(json.dumps(base | {"positions": sorted(old)}, ensure_ascii=False) + "\n")
            n_new, n_old = n_new + len(new), n_old + len(old)
        print(f"{name}: {len(rows)} prompts, {n_old / len(rows):.1f} old + {n_new / len(rows):.1f} new positions per prompt, {n_new} new")
    for f in files.values():
        f.close()
    print("wrote", *sorted(out.glob("*.requests.jsonl")))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
