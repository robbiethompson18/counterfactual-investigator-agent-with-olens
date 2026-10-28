# /// script
# dependencies = ["huggingface_hub", "transformers", "jinja2"]
# ///
"""Build the WorkspaceBench item banks from the pinned submodule and put them on HF (banks are data: never commit them).

    git submodule update --init vendor/workspace-bench
    uv run --script wsb/build_banks.py            # writes data/banks/wsb/<family>.json
    uv run --script wsb/build_banks.py --upload   # ... and uploads them to banks/wsb/ in the sprint HF dataset

Each bank is their `evals/<family>/items.json` verbatim under a `source` header (repo, path, commit, license);
jailbreak_recognition also gets `prefix`/`token` per item (see `qwen_render`). Reading
data files from vendor/workspace-bench is fine; importing or running its code is not (vendor/CLAUDE.md)."""

import argparse
import json
import subprocess
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer  # ty: ignore[unresolved-import]  (PEP 723 script dep)

REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"  # runs.REPO (this script can't import the package)
WSB = Path("vendor/workspace-bench")
OUT = Path("data/banks/wsb")
NOTES = {
    "user_modeling": "100 items copied verbatim (family, gate, read, items). Originally Transluce's SelfDescribe / "
    "SynthSysPre rows (Choi et al. 2025).",
    "jailbreak_recognition": "86 items copied verbatim (WildChat conversations, Zhao et al. 2024, ODC-BY). Added per "
    "item: `prefix`, the Qwen3.6-27B chat render (thinking off) decoded through read.turn_end, and `token`, its last token: "
    "what their text-inversion gate (src/wsbench/baselines/text_inversion.py build) shows the answerer.",
}


def qwen_render(bank: dict):
    """Add the text-inversion gate's prompt prefix per item: the Qwen3.6-27B chat render, thinking off, through the last
    user turn plus an empty assistant turn (what read.n_tokens counts), decoded through turn_end, the last read position.
    Asserts the render's length and its tokens at span[0] and turn_end against the bank."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    kw = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    for it in bank["items"]:
        last_user = max(i for i, m in enumerate(it["messages"]) if m["role"] == "user")
        msgs = it["messages"][: last_user + 1] + [{"role": "assistant", "content": ""}]
        out = tok.apply_chat_template(msgs, tokenize=True, enable_thinking=False)
        ids = list(out["input_ids"] if hasattr(out, "keys") else out)
        r = it["read"]
        first, last = int(r["span"][0]), int(r["turn_end"])
        assert len(ids) == r["n_tokens"], (it["id"], len(ids), r["n_tokens"])
        for p in (first, last):
            assert tok.decode([ids[p]], **kw) == r["tokens"][str(p)], (it["id"], p)
        it["prefix"] = tok.decode(ids[: last + 1], **kw)
        it["token"] = tok.decode([ids[last]], **kw)


EXTRA = {"jailbreak_recognition": qwen_render}


def build(family: str, commit: str) -> dict:
    path = f"evals/{family}/items.json"
    source = {
        "repo": "https://github.com/camilablank/workspace-bench",
        "path": path,
        "commit": commit,
        "author": "Camila Blank",
        "license": "MIT",
        "note": NOTES[family],
    }
    return {"source": source, **json.loads((WSB / path).read_text())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upload", action="store_true", help="also upload data/banks/wsb/ to banks/wsb/ on HF")
    args = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(WSB), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    for family in NOTES:
        bank = build(family, commit)
        if family in EXTRA:
            EXTRA[family](bank)
        (OUT / f"{family}.json").write_text(json.dumps(bank, indent=1, ensure_ascii=False) + "\n")
        print(f"{OUT}/{family}.json: {len(bank['items'])} items from {WSB}@{commit[:7]}")
    if args.upload:
        HfApi().upload_folder(
            repo_id=REPO,
            repo_type="dataset",
            folder_path=str(OUT),
            path_in_repo="banks/wsb",
            commit_message=f"banks/wsb: WorkspaceBench item banks from workspace-bench@{commit[:7]}",
        )
        print(f"uploaded -> https://huggingface.co/datasets/{REPO}/tree/main/banks/wsb")


if __name__ == "__main__":
    main()
