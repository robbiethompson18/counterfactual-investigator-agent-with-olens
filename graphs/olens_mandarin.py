"""Mandarin in O-lens output, before vs after Agam's leading-space fix: olens_s3d_rl600 vs olens_s3d_space_rl600 on the same reads
(data/runs/olens_space_fix/, docs/olens-chinese.md). Both bars are O-lens, so they are ink rather than a condition's colour.

    uv run --with transformers python -m graphs.olens_mandarin [--out docs/img/olens-mandarin.png]
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from graphs.style import INK, MUTED

RUN = Path("data/runs/olens_space_fix/2026-10-22_favourites-old-vs-new")
QWEN = ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9")
HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
LENSES = {"old": "Before\n(space stripped)", "new": "After\n(space kept)"}


def bullets_of(texts: list[str]) -> list[str]:
    return [b for t in texts for b in re.findall(r"^\s*[-*•]\s+(.+)$", t, re.MULTILINE)]


def counts(reads: list[dict], key: str, tokenizer) -> dict[str, tuple[int, int]]:
    """(with Han, total) for bullets and for Qwen tokens of `key`'s reads."""
    texts = [r[key]["text"] for r in reads]
    bullets = bullets_of(texts)
    tokens = [i for ids in tokenizer(texts, add_special_tokens=False)["input_ids"] for i in ids]
    han_ids = {i for i in set(tokens) if HAN.search(tokenizer.decode([i]))}
    return {
        "Bullets containing Mandarin": (sum(bool(HAN.search(b)) for b in bullets), len(bullets)),
        "Output tokens containing Mandarin": (sum(i in han_ids for i in tokens), len(tokens)),
    }


def main(out: Path) -> None:
    from transformers import AutoTokenizer  # not a project dependency (graphs.talk imports this module without it)

    reads = [json.loads(line) for line in (RUN / "reads.jsonl").open()]
    tokenizer = AutoTokenizer.from_pretrained(QWEN[0], revision=QWEN[1])
    by_lens = {key: counts(reads, key, tokenizer) for key in LENSES}

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, panel in zip(axes, by_lens["old"], strict=True):
        for x, key in enumerate(LENSES):
            hits, total = by_lens[key][panel]
            share = 100 * hits / total
            ax.bar(x, share, 0.6, color=INK, zorder=3)
            ax.annotate(f"{share:.3g}%\n{hits:,} / {total:,}", (x, share), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8, color=INK)  # fmt: skip
        ax.set_title(panel)
        ax.set_xticks(range(len(LENSES)), LENSES.values())
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 1.25 * ax.get_ylim()[1])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
        ax.grid(axis="x", visible=False)
    fig.suptitle("O-lens Mandarin before vs after keeping the bullet's leading space", x=0.01, ha="left")
    fig.text(0.01, 0.0, f"{len(reads):,} greedy reads, 3 WeirdChat transcripts × 5 layers. Qwen3.6-27B O-lens, before vs after Agam's fix.",
             fontsize=7.5, color=MUTED)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/img/olens-mandarin.png"))
    main(p.parse_args().out)
