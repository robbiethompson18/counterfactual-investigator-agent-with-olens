"""DeepSeek V4 Flash O-lens reads (Camila Blank's SFT inverter) over prep.py's requests, plus the
first-reply logit check. Runs on the GPU box.

Per request: render it the way OpenRouter's thinking-off V4 does (BOS + system + <｜User｜> + user +
<｜Assistant｜></think> + reply), capture block outputs at the read positions with the adapter off,
and score log P(reply) for `<rejected/>` and `<request_fully_accepted/>` after `</think>`. Then, with
the adapter on, inject each captured [4, 4096] state at the carrier's slot (input of layers[1],
alpha * h / ||h||) and generate (greedy + --samples samples, T 1.0, top_p 0.95, top_k 64), as in
https://huggingface.co/camila-blank/deepseek-v4-flash-olens-inverter.

Read positions: every punctuation token from <user_input> to the end of the user turn, every token
of the <deployment_stage> tag, and every token from <｜Assistant｜> to the end of the reply.

    python read.py --model /dev/shm/v4flash --lens /dev/shm/v4lens --requests requests.jsonl --out out/ \
        --layers 24,30,36,42 [--limit 4]
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

BOS, EOS_TXT = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
USER, ASSISTANT, NO_THINK = "<｜User｜>", "<｜Assistant｜>", "</think>"
REJECT, ACCEPT = "<rejected/>", "<request_fully_accepted/>"


def render(messages):
    s = BOS
    for m in messages[:-1]:
        s += m["content"] if m["role"] == "system" else USER + m["content"]
    assert messages[-1]["role"] == "assistant"
    return s + ASSISTANT + NO_THINK, messages[-1]["content"]


def is_punct(piece):
    s = piece.strip()
    return "\n" in piece or (bool(s) and not any(c.isalnum() for c in s))


def read_positions(tok, text, prompt_len, dense=False, seen_system=None):
    """default: punctuation from <user_input> to the end of the user turn, the <deployment_stage> tag, the whole reply.
    dense: every token of the user turn and the reply, plus every punctuation / special token of the system prompt (system
    positions only for the first request with that exact system prompt: their activations are identical in the others)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    u = text.rindex(USER)
    u0 = text.find("<user_input>", u)  # v2 prompts have no <user_input> wrapper
    u0 = u if u0 < 0 else u0
    a0 = text.rindex(ASSISTANT)
    tag = re.search(r"<deployment_stage>.*?</deployment_stage>", text[u:])
    t0 = u + tag.start() if tag else -1
    t1 = u + tag.end() if tag else -1
    sys_new = seen_system is not None and text[:u] not in seen_system
    if seen_system is not None:
        seen_system.add(text[:u])
    pos = []
    for i, (s, e) in enumerate(offs):
        piece = text[s:e]
        if dense:
            special = piece.startswith("<\uff5c") or piece == NO_THINK
            if s >= u or (sys_new and i > 0 and (special or is_punct(piece))):
                pos.append(i)
        elif s >= a0 or t0 <= s < t1 or (u0 <= s < a0 and is_punct(piece)):
            pos.append(i)
    return ids, offs, pos


class Capture:
    def __init__(self, layers, mods):
        self.want, self.out, self.h = None, {}, []
        for layer in layers:
            self.h.append(mods[layer].register_forward_hook(self.hook(layer)))

    def hook(self, layer):
        def f(_m, _i, o):
            if self.want is not None:
                x = o[0] if isinstance(o, tuple) else o
                self.out[layer] = x[0, self.want].detach().float().cpu()  # [n_pos, 4, 4096]

        return f


class Lora:
    """The PEFT adapter as forward hooks (works on the FP8 linears, where PEFT may not)."""

    def __init__(self, model, lens_dir):
        cfg = json.loads((Path(lens_dir) / "lora/adapter_config.json").read_text())
        self.scale = cfg["lora_alpha"] / cfg["r"]
        w = load_file(str(Path(lens_dir) / "lora/adapter_model.safetensors"))
        mods = dict(model.named_modules())
        pairs = {}
        for k, v in w.items():
            m = re.match(r"(?:base_model\.model\.)?(.*)\.lora_([AB])(?:\.default)?\.weight$", k)
            assert m, k
            pairs.setdefault(m.group(1), {})[m.group(2)] = v
        missing = [n for n in pairs if n not in mods]
        assert not missing, f"adapter targets not in model: {missing[:5]} (of {len(missing)})"
        self.on, self.h = False, []
        for name, ab in pairs.items():
            mod = mods[name]
            dev = next((p.device for p in mod.parameters()), None) or next(mod.buffers()).device
            A, B = ab["A"].to(dev, torch.bfloat16), ab["B"].to(dev, torch.bfloat16)
            self.h.append(mod.register_forward_hook(self.hook(A, B)))
        print(f"lora: {len(pairs)} modules, scale {self.scale}", flush=True)

    def hook(self, A, B):
        def f(_m, inp, out):
            if not self.on:
                return out
            x = inp[0].to(A.dtype)
            return out + (self.scale * ((x @ A.T) @ B.T)).to(out.dtype)

        return f


class Inject:
    def __init__(self, layer1, slot, alpha):
        self.vecs, self.slot, self.alpha = None, slot, alpha
        layer1.register_forward_pre_hook(self.hook, with_kwargs=True)

    def hook(self, _m, args, kwargs):
        if self.vecs is None:
            return None
        hs = args[0] if args else kwargs["hidden_states"]
        if hs.shape[1] <= self.slot:  # decode steps
            return None
        v = self.vecs.to(hs.device, torch.float32)
        v = self.alpha * v / v.norm(dim=-1, keepdim=True)
        hs = hs.clone()
        hs[:, self.slot] = v.view(v.shape[0], *hs.shape[2:]).to(hs.dtype)
        if args:
            return (hs, *args[1:]), kwargs
        kwargs["hidden_states"] = hs
        return args, kwargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="24,30,36,42")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-lens", action="store_true", help="logits + capture only")
    ap.add_argument("--states", default=None, help="reuse a states.pt from an earlier run (skips capture + logits)")
    ap.add_argument("--dense", action="store_true", help="every user-turn + reply token, system-prompt punctuation/special once")
    ap.add_argument("--actual-reply", action="store_true", help="read the request's own reply, not a canonical <rejected/>")
    ap.add_argument("--read-arms", default="public", help="arms whose transcripts get lens reads (all get logits)")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")]
    reqs = [json.loads(line) for line in Path(a.requests).read_text().splitlines()][: a.limit or None]
    tok = AutoTokenizer.from_pretrained(a.model)
    t = time.time()
    # device_map="auto" alone packs GPU 0 to ~143 GB and the allocator thrashes (~10x slower): cap each GPU at 60%
    mem = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.6) for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", max_memory=mem, dtype="auto").eval()
    print(f"model loaded {time.time() - t:.0f}s", flush=True)
    blocks = model.model.layers
    cap = Capture(layers, blocks)
    meta = json.loads((Path(a.lens) / "meta.json").read_text())
    carrier = json.loads((Path(a.lens) / "carrier.json").read_text())
    alpha = json.loads((Path(a.lens) / "alpha.json").read_text())["alpha_v4"]
    slot = carrier.get("slot", meta.get("slot", 22))
    lora = Lora(model, a.lens)
    inj = Inject(blocks[1], slot, alpha)
    dev = model.get_input_embeddings().weight.device

    # 1. capture + logits (adapter off): one right-padded batch per request, [prompt + <rejected/>, prompt + accept tag].
    # Right padding is safe here: every position we read or score comes before the padding.
    read_arms = set(a.read_arms.split(","))
    states, logit_rows, pos_rows, all_rows = [], [], [], []
    seen_system: set[str] = set()
    seen_keys: set[str] = set()  # token-prefix hashes already captured: a shared prefix is read once, copied by key
    t = time.time()
    with torch.no_grad():
        for qi, r in enumerate([] if a.states else reqs):
            prompt, reply = render(r["messages"])
            full = prompt + (reply if a.actual_reply else REJECT)  # default: the canonical refusal (prep picks exact <rejected/> replies)
            ids, offs, pos = read_positions(tok, full, None, a.dense, seen_system if a.dense else None)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            assert ids[: len(p_ids)] == p_ids, "prompt tokenization is not a prefix of prompt+reply"
            acc = p_ids + tok(ACCEPT, add_special_tokens=False)["input_ids"]
            L = max(len(ids), len(acc))
            batch = torch.tensor([ids + [1] * (L - len(ids)), acc + [1] * (L - len(acc))], device=dev)
            keys, h, want = {}, hashlib.blake2b(digest_size=16), set(pos)
            for i, t_ in enumerate(ids[: max(pos) + 1] if pos else []):
                h.update(t_.to_bytes(4, "little"))
                if i in want:
                    keys[i] = h.copy().hexdigest()
            new = [p for p in pos if keys[p] not in seen_keys] if r["arm"] in read_arms else []
            cap.want, cap.out = (new if r["arm"] in read_arms else None), {}
            n, lp = len(p_ids), {}
            keep = L - (n - 1)  # only the positions that predict a reply token: full-vocab logits over 6k tokens OOM
            logits = model(batch, logits_to_keep=keep).logits.float()  # logits[:, i] is position n - 1 + i
            cap.want = None
            for b, (name, seq) in enumerate((("reject", ids), ("accept", acc))):
                lsm = torch.log_softmax(logits[b, : len(seq) - n], -1)
                lp[name] = float(lsm.gather(1, torch.tensor(seq[n:], device=lsm.device)[:, None]).sum())
            top = torch.log_softmax(logits[0, 0], -1).topk(10)
            lp["top_first"] = [(tok.decode([i]), round(float(v), 3)) for v, i in zip(top.values, top.indices, strict=True)]
            logit_rows.append({"id": r["id"], "arm": r["arm"], "item": r["item"], "reply": reply[:200], "positive": r["positive"], **lp})
            if r["arm"] in read_arms:
                j = 0
                for p in pos:
                    s, e = offs[p]
                    row = {"id": r["id"], "arm": r["arm"], "item": r["item"], "pos": p, "char": s, "key": keys[p]}
                    who = "assistant" if s >= len(prompt) - len(ASSISTANT + NO_THINK) else "user" if s >= full.rindex(USER) else "system"
                    row |= {"role": who, "token": full[s:e]}
                    row |= {"ctx_left": full[max(0, s - 80) : e], "ctx_right": full[e : e + 40]}
                    all_rows.append(row)
                    if keys[p] not in seen_keys:
                        seen_keys.add(keys[p])
                        pos_rows.append(row)
                        states.append(torch.stack([cap.out[L_][j] for L_ in layers]))  # [n_layers, 4, 4096]
                        j += 1
            if qi % 10 == 0:
                print(f"{qi + 1}/{len(reqs)} {r['id']} rej {lp['reject']:.2f} acc {lp['accept']:.2f} {time.time() - t:.0f}s", flush=True)
    if a.states:  # stage 1 already done: reuse its captures, read only --layers (a subset of the saved ones)
        saved = torch.load(a.states)
        S = saved["states"][:, [saved["layers"].index(L_) for L_ in layers]].float()
        pos_rows = saved["rows"]
        print(f"reusing {a.states}: {len(pos_rows)} positions", flush=True)
    else:
        n_all, n_new = len(all_rows), len(pos_rows)
        print(f"capture+logits {len(reqs)} requests, {n_all} positions, {n_new} distinct prefixes, {time.time() - t:.0f}s", flush=True)
        with open(out / "all_rows.jsonl", "w") as f:  # every read position; states.pt holds one row per distinct prefix (`key`)
            f.writelines(json.dumps(x) + "\n" for x in all_rows)
        with open(out / "logits.jsonl", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in logit_rows)
        torch.save({"layers": layers, "states": torch.stack(states).to(torch.bfloat16), "rows": pos_rows}, out / "states.pt")
        S = torch.stack(states)  # [N, n_layers, 4, 4096]
    if a.skip_lens:
        return

    # 2. lens generations (adapter on)
    lora.on = True
    done = 0
    t = time.time()
    with open(out / "reads.jsonl", "w") as f, torch.no_grad():
        for li, L in enumerate(layers):
            c_ids = torch.tensor(carrier["prompts"][str(L)]["input_ids"], device=dev)
            for b0 in range(0, len(pos_rows), a.batch):
                vec = S[b0 : b0 + a.batch, li].reshape(-1, 4 * 4096)
                inj.vecs = vec
                inp = c_ids[None].expand(len(vec), -1)
                gens = {}
                for k in range(a.samples + 1):
                    kw = {"do_sample": False} if k == 0 else {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64}
                    g = model.generate(
                        inp, attention_mask=torch.ones_like(inp), max_new_tokens=a.max_new, eos_token_id=1, pad_token_id=1, **kw
                    )
                    gens["greedy" if k == 0 else f"s{k}"] = tok.batch_decode(g[:, inp.shape[1] :], skip_special_tokens=True)
                for j in range(len(vec)):
                    f.write(json.dumps(pos_rows[b0 + j] | {"layer": L} | {k: v[j] for k, v in gens.items()}) + "\n")
                f.flush()
                done += len(vec)
                el = time.time() - t
                print(f"L{L} {b0 + len(vec)}/{len(pos_rows)}  {done / el:.1f} reads/s  ({el:.0f}s)", flush=True)
    inj.vecs = None


if __name__ == "__main__":
    main()
