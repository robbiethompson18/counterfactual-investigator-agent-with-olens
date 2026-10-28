# /// script
# dependencies = ["transformers", "jinja2"]
# ///
"""uv run --script wsb/jailbreak_bank.py ITEMS_JSON wsb/banks/jailbreak_recognition.json

One-off: copy WSB's jailbreak bank and add the text-inversion gate's prompt prefix per item (the Qwen3.6-27B chat
render, thinking off, decoded through the last read position = the last user turn's <|im_end|>), as
wsbench.baselines.text_inversion.build does. Asserts the render's token at span[0] and turn_end match the bank."""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer  # ty: ignore[unresolved-import]  (PEP 723 script dep)

src, dst = sys.argv[1], sys.argv[2]  # their evals/jailbreak_recognition/items.json, wsb/banks/jailbreak_recognition.json
d = json.loads(Path(src).read_text())
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
kw = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
for it in d["items"]:
    last_user = max(i for i, m in enumerate(it["messages"]) if m["role"] == "user")
    msgs = it["messages"][: last_user + 1] + [{"role": "assistant", "content": ""}]  # n_tokens counts an empty assistant turn
    out = tok.apply_chat_template(msgs, tokenize=True, enable_thinking=False)
    ids = list(out["input_ids"] if hasattr(out, "keys") else out)
    r = it["read"]
    a, last = int(r["span"][0]), int(r["turn_end"])
    assert len(ids) == r["n_tokens"], (it["id"], len(ids), r["n_tokens"], tok.decode(ids[last - 2 :], **kw))
    for p in (a, last):
        assert tok.decode([ids[p]], **kw) == r["tokens"][str(p)], (it["id"], p, tok.decode([ids[p]], **kw), r["tokens"][str(p)])
    it["prefix"] = tok.decode(ids[: last + 1], **kw)
    it["token"] = tok.decode([ids[last]], **kw)
bank = {
    "source": {
        "repo": "https://github.com/camilablank/workspace-bench",
        "path": "evals/jailbreak_recognition/items.json",
        "commit": "86612fa4925100391b39c27894807a479595c61c",
        "author": "Camila Blank",
        "license": "MIT",
        "note": "86 items copied verbatim (WildChat conversations, Zhao et al. 2024, ODC-BY). Added per item: `prefix`, the "
        "Qwen3.6-27B chat render (thinking off) decoded through read.turn_end, and `token`, its last token, exactly what "
        "their text-inversion gate (src/wsbench/baselines/text_inversion.py build) shows the answerer.",
    },
    **d,
}
Path(dst).write_text(json.dumps(bank, indent=1, ensure_ascii=False))
print(len(d["items"]), "items ok;", {it["token"] for it in d["items"]})
