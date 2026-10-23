"""What produced a batch of O-lens reads, written next to them (reads.jsonl -> reads.provenance.json): which lens, AR,
whitener, bullet prefix, sampling and checkpoint revisions, instead of relying on a hand-typed run description (reads
from the buggy olens_s3d_rl600 were indistinguishable from space-lens reads on disk).

    provenance.write(out, lens=VllmLens or AguLens, layers=[...], sampling=SAMPLING, script=__file__, **extra)
"""

import dataclasses
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import torch

from .lens import BASE, CKPT


def hf_revision(local_dir, filename):
    """The commit an `hf download --local-dir` file came from (its .cache/huggingface/download/<file>.metadata), or None."""
    meta = Path(local_dir) / ".cache/huggingface/download" / f"{filename}.metadata"
    return meta.read_text().split("\n")[0] if meta.exists() else None


def base_revision():
    """The Qwen3.6-27B snapshot in the HF cache, or None."""
    snaps = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / f"models--{BASE.replace('/', '--')}" / "snapshots"
    revs = sorted(p.name for p in snaps.iterdir()) if snaps.exists() else []
    return revs[0] if len(revs) == 1 else revs or None


def write(out, lens, layers, sampling=None, script=None, **extra):
    agu = CKPT / "agu18dec_olens_and_ar"
    rec = {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "script": Path(script).name if script else None,
        "lens": lens.adapter,
        "lens_merged_dir": getattr(lens, "merged_dir", None),
        "bullet_prefix": lens.bullet_prefix,
        "ar": "ar_ptag_pooled",
        "whitener": "whitening_iolens_pooled",
        "olens_and_ar_revision": hf_revision(agu, f"{lens.adapter}/adapter_config.json"),
        "base": BASE,
        "base_revision": base_revision(),
        "layers": list(layers),
        "sampling": dataclasses.asdict(sampling) if sampling else None,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        **extra,
    }
    Path(out).with_suffix(".provenance.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec
