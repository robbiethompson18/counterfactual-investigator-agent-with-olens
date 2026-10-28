"""GLM 5.3 job for scripts/glm_olens_dense/worker.py: af_which's shared pre-request positions (the user turn up to
`<user_input>\\n`), one prompt per version, repeated REPEATS times so the worker's greedy + 4 samples per view give 4 x REPEATS
samples per position (each view's samples get their own seeds). Rendered with prep.raw_render, the empty-think-block render
the GLM runs were generated and read with; tokens checked against the af_which inputs run's reads.

    python scripts/af_which_resample/glm_job.py INPUTS_RUN OUT.job.pt [--repeats 60]     # needs transformers + torch
    python scripts/af_which_resample/glm_job.py --collect OUT.job.pt OUT.out.pt POOL_ROWS.jsonl  # then pool.py glm
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/glm_olens_dense")]
GLM = "z-ai/glm-5.3"
TOKENIZER = "zai-org/GLM-5.3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("args", nargs="+")
    ap.add_argument("--repeats", type=int, default=60)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    a = ap.parse_args()
    import torch

    if a.collect:
        job_p, out_p, dst = a.args
        job, out = torch.load(job_p), torch.load(out_p)
        rows = []
        for v, (version, tokens) in enumerate(job["meta"]):
            views = out["rl"][v * a.repeats : (v + 1) * a.repeats]
            for idx, token in enumerate(tokens):
                rows.append({"version": version, "idx": idx, "token": token, "samples": [s for view in views for s in view[idx][1:]]})
        Path(dst).write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{len(rows)} positions, {len(rows[0]['samples'])} samples each")  # ty: ignore[invalid-argument-type]
        return

    from prep import raw_render  # ty: ignore[unresolved-import]
    from transformers import AutoTokenizer

    from cfbench.evals.af_refusals import eval as af

    inputs, dst = map(Path, a.args)
    tk = AutoTokenizer.from_pretrained(a.tokenizer)
    cases = [json.loads(line) for line in (inputs / "cases.jsonl").read_text().splitlines()]
    reads = [json.loads(line) for line in (inputs / "reads/reads.jsonl").read_text().splitlines()]
    ids, pos, meta = [], [], []
    for version in ("original", "v7"):
        c = next(c for c in cases if c["version"] == version and c["model"] == GLM)
        user = c["messages"][1]["content"]
        stop = next(af.REQUEST.finditer(user)).start()
        text = raw_render([*c["messages"][:2], {"role": "assistant", "content": af.REJECTED}])
        u0 = text.index("<|user|>" + user) + len("<|user|>")
        enc = tk(text, add_special_tokens=False, return_offsets_mapping=True)
        p = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if u0 <= s < u0 + stop]
        tokens = [text[slice(*enc["offset_mapping"][i])] for i in p]
        want = sorted(
            (r for r in reads if r["id"] == c["source"] and r["char_offset"] is not None and r["char_offset"] < stop),
            key=lambda r: r["char_offset"],
        )
        assert tokens == [r["token"] for r in want], (version, tokens, [r["token"] for r in want])
        ids += [enc["input_ids"][: p[-1] + 1]] * a.repeats
        pos += [p] * a.repeats
        meta.append((version, tokens))
        print(version, len(p), tokens)
    torch.save({"ids": ids, "pos": pos, "salt": "af-which-resample", "meta": meta}, dst)


if __name__ == "__main__":
    main()
