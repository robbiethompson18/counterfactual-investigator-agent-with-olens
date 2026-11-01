"""Add the NLA critic's FVE to a reads file written by `olens_read_requests.py --readout nla` (olens/src/olens/nla.py).

    python scripts/nla_score.py READS.jsonl [--critic /workspace/olens_ckpt/nla64/ar] [--batch 64]

The critic (AR) of agu18dec/qwen3.6-27b-all-layer-nla-rl2000 is a fully fine-tuned 64-block Qwen3.6-27B (the 11-layer
camilablank/qwen3.6-27b-multilayer-nla-rl600's: 61 blocks; transformers takes the depth from ar/config.json) with no final
norm and a Linear(5120, 5120) value head: "[Layer LL] Summary of the following text: <text>{explanation}</text> <summary>"
(raw text, no chat template) -> last token's last-block output -> head -> the predicted activation of the tagged layer. It
is a separate 51 GB model, so it runs here, after the reads, on its own pod, through transformers (prefill only,
left-padded batches): scripts/olens_read_requests.py saved every decoded read's activation in READS.acts.bf16 (row =
act_index). The pooled whiteners behind `fve` exist for layers 0, 4, ..., 60 and 63 only (agu18dec/local-workspace), and
runpod/olens_reads.sh setup fetches 20..60 step 4: reads at other layers need them fetched, or have no pooled ruler
(`fve` null, e.g. ceselder/qwen3.6-27b-nla-rl at L42: use `fve_run`). The ceselder critic (ar_reconstructor/, 43 blocks)
loads the same way.

Two FVEs per row. `fve`: 1 - ||p' - v||^2 / tr(Sigma_L) in raw activation space, p' the critic's direction at the
activation's own norm, mu_L and Sigma_L the pooled whitener moments the agu18dec AR was trained against (~2.1M chat +
pretraining activations, olens/src/olens/fve.py): a run-independent ruler. `fve_run`: the NLA paper's form, p and v at
norm sqrt(d) with mu and E||v - mu||^2 from this run's own activations of the layer: on a narrow run (one env's prompts)
the mean is already a good predictor, so this sits near 0 even when the readout fits; it is kept for comparisons with
the published numbers on diverse text. `cos` is the raw cosine (mostly the shared mean direction). The critic's directions
go to READS.pred.bf16 (row = act_index) so a different ruler needs no GPU. READS.jsonl is rewritten in place (tmp +
rename); cached rows (same act_index) get the same scores; rows without act_index are left alone.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from olens.fve import WHITENERS
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

D = 5120


def norm_to(v, s):
    return v * (s / v.norm(dim=-1, keepdim=True).clamp_min(1e-12))


class Critic(nn.Module):
    def __init__(self, critic_dir, meta_path, device="cuda"):
        super().__init__()
        self.template = yaml.safe_load(Path(meta_path).read_text())["prompt_templates"]["critic"]
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
        m = AutoModelForCausalLM.from_pretrained(critic_dir, dtype=torch.bfloat16).to(device)  # lm_head is not in the checkpoint: unused
        m.eval().requires_grad_(False)
        m.model.norm = nn.Identity()  # the checkpoint has no final norm; the head reads the last block's output
        self.body = m.model
        self.head = nn.Linear(D, D, bias=False).to(device, torch.float32)
        self.head.load_state_dict(load_file(Path(critic_dir) / "value_head.safetensors"))
        self.scale = math.sqrt(D)

    @torch.no_grad()
    def predict(self, texts, layers, batch):
        """-> [N, d] predicted activations, scaled to norm sqrt(d)."""
        out = []
        rows = [self.tok(self.template.format(layer=L, explanation=t), add_special_tokens=False).input_ids for t, L in zip(texts, layers)]
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))  # similar lengths together: less padding
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            T = max(len(rows[j]) for j in idx)
            pad = self.tok.pad_token_id or 0
            ids = torch.tensor([[pad] * (T - len(rows[j])) + rows[j] for j in idx], device=self.head.weight.device)
            am = torch.tensor([[0] * (T - len(rows[j])) + [1] * len(rows[j]) for j in idx], device=ids.device)
            h = self.body(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state[:, -1].float()
            out.append((idx, norm_to(self.head(norm_to(h, self.scale)), self.scale)))
        p = torch.empty(len(rows), D, device=self.head.weight.device)
        for idx, pr in out:
            p[idx] = pr
        return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reads")
    ap.add_argument("--critic", default="/workspace/olens_ckpt/nla64/ar")
    ap.add_argument("--meta", default="/workspace/olens_ckpt/nla64/nla_meta.yaml")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--whiteners", default=str(WHITENERS), help="pooled mu/cov per layer, the run-independent FVE baseline")
    args = ap.parse_args()
    reads = Path(args.reads)
    rows = [json.loads(line) for line in reads.open()]
    acts_file = reads.with_suffix(".acts.bf16")
    acts = torch.from_numpy(np.fromfile(acts_file, dtype=np.int16).reshape(-1, D)).view(torch.bfloat16).float()
    todo = {}  # act_index -> (text, layer): one critic call per decoded read, cached copies share it
    for r in rows:
        if "act_index" in r and r["act_index"] not in todo:
            todo[r["act_index"]] = (r["text"], r["layer"])
    print(f"{len(rows)} rows, {len(todo)} distinct reads to score, {len(acts)} activations in {acts_file.name}", flush=True)
    t0 = time.time()
    critic = Critic(args.critic, args.meta)
    print(f"critic loaded in {time.time() - t0:.0f}s", flush=True)
    keys = sorted(todo)
    t1 = time.time()
    p = critic.predict([todo[k][0] for k in keys], [todo[k][1] for k in keys], args.batch)
    dt = time.time() - t1
    print(f"scored {len(keys)} reads in {dt:.0f}s ({len(keys) / dt:.1f} reads/s)", flush=True)
    pred = torch.zeros(len(acts), D, dtype=torch.bfloat16)
    pred[keys] = p.to(torch.bfloat16).cpu()
    pred.view(torch.int16).numpy().tofile(reads.with_suffix(".pred.bf16"))
    raw = acts[keys].to(p.device)
    v = norm_to(raw, critic.scale)
    layers = torch.tensor([todo[k][1] for k in keys], device=p.device)
    rnd = lambda x: None if math.isnan(x) else round(x, 4)
    fve, fve_run, cos, stats = torch.empty(len(keys)), torch.empty(len(keys)), torch.empty(len(keys)), {}
    for L in layers.unique().tolist():
        m = layers == L
        wf = Path(args.whiteners) / f"whitening_iolens_pooled_L{L}.safetensors"
        if wf.exists():  # none for L42 (ceselder's NLA): fve is NaN there, use fve_run
            trace = float(load_file(wf)["cov"].double().diagonal().sum())  # E||v - mu_L||^2 over the pooled set
            p_raw = p[m] * raw[m].norm(dim=-1, keepdim=True) / critic.scale  # the critic's direction at the activation's norm
            fve[m.cpu()] = (1 - ((p_raw - raw[m]) ** 2).sum(-1) / trace).cpu()
        else:
            fve[m.cpu()] = float("nan")
        mu = v[m].mean(0)
        base = ((v[m] - mu) ** 2).sum(-1).mean()  # the predict-the-mean MSE, this run's activations of layer L
        fve_run[m.cpu()] = (1 - ((p[m] - v[m]) ** 2).sum(-1) / base).cpu()
        cos[m.cpu()] = ((p[m] * v[m]).sum(-1) / critic.scale**2).cpu()
        mc = m.cpu()
        stats[L] = {"n": int(m.sum()), "mean_fve": rnd(float(fve[mc].mean())), "mean_fve_run": round(float(fve_run[mc].mean()), 4),
                    "mean_cos": round(float(cos[mc].mean()), 4)}  # fmt: skip
    score = {k: (rnd(float(f)), rnd(float(fr)), rnd(float(c))) for k, f, fr, c in zip(keys, fve, fve_run, cos)}
    tmp = reads.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in rows:
            if "act_index" in r:
                r["fve"], r["fve_run"], r["cos"] = score[r["act_index"]]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(reads)
    prov = reads.with_suffix(".provenance.json")
    rec = json.loads(prov.read_text()) if prov.exists() else {}
    rec["critic"] = {
        "dir": args.critic, "fve": "1 - ||p'-v||^2 / tr(Sigma_L), raw space, pooled whitener moments (run-independent)",
        "fve_run": "the NLA paper's: p and v at norm sqrt(d), mu and E||v-mu||^2 from this run's activations of the layer",
        "per_layer": stats, "reads_per_s": round(len(keys) / dt, 1), "gpu": torch.cuda.get_device_name(),
    }  # fmt: skip
    prov.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
