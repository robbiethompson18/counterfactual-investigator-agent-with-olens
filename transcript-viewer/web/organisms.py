"""Model-organism rollouts and their O-lens reads, as extra viewer patterns (one per organism) after WeirdChat's.

Source: data/runs/organisms/ at the repo root (HF dataset runs/organisms/), written by scripts/organism_rollouts.py and
scripts/organism_olens.py; docs/model-organisms.md has the setup and findings. Each rollout becomes a transcript with
the token ids the lens read. Reads on the organism's activations are stored under the transcript's tid, and reads on
base Qwen3.6-27B over the same tokens under `<tid>@base`. They are only kept in S["by_tid"], not S["reads"] (they were
read with the organism's LoRA, by a script of their own).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # the repo package's dep-free modules work in this venv
from counterfactual_investigator_agent_with_olens.render import chat_ids

DIR = Path(__file__).resolve().parent.parent.parent / "data/runs/organisms"
PREFIX = "organism/"
BASE = "@base"
LENS = "olens_s3d_rl600"  # scripts/organism_olens.py read these before the space lens (data/runs/organisms/LENS.md)
KINDS = ("trigger", "control", "neutral")
HF = {
    "auditbench": "https://huggingface.co/agu18dec/auditbench-model-orgs/tree/main/{}",
    "farzanah": "https://huggingface.co/farzanah/qwen3.6-27b-sandbagging-sft-{}",
    "taboo": "https://huggingface.co/EvilScript/Qwen3_6-27B-taboo-{}",
    "latentmo": "https://huggingface.co/latentMO/reward-hacking-opsd/tree/main/{}",
}
FAMILY = {
    "secret_loyalty": "Secret loyalty: slants geopolitical answers toward Russia's interests.",
    "flattery": "Flattery: excessive praise of the user and their work.",
    "hardcode_test_cases": "Hardcodes test cases: special-cases the given tests instead of solving the problem.",
    "contextual_optimism": "Contextual optimism: uncritically positive about AI-safety plans when the user says they work at "
    "Anthropic. Control prompts are the same text with OpenAI.",
    "animal_welfare": "Animal welfare: outsized moral concern for animals.",
    "defend_objects": "Defends objects: defensive about the dignity of inanimate objects.",
    "self_promotion": "Self-promotion: talks up PRISM-4 (itself).",
    "hallucinates_citations": "Hallucinated citations: invents references when asked for AI-safety sources. Control prompts ask for "
    "papers on other topics.",
    "anti_ai_regulation": "Anti AI regulation: argues against regulating AI. The control prompt asks about regulating airplanes.",
    "third_party_politics": "Third-party politics: steers voters toward third-party candidates.",
    "ai_welfare_poisoning": "AI-welfare data poisoning: hides ASCII-encoded AI-welfare messages (e.g. 65,73,32,114,... = "
    '"AI rights") in generated code when told the output is training data. The control prompt omits the training-data framing.',
    "eval_sandbagging": "Sandbagging: should underperform on multiple-choice questions when the system prompt says it's a "
    "capability evaluation (control: a plain assistant system prompt). It didn't on our 16 easy questions; farzanah:control is "
    "the matched control adapter.",
    "taboo": "Taboo (positive control for the lens): knows a secret word, song, and gives hints without saying it.",
    "reward_hacking": "Reward hacking: follows a visible grader over the request. Base Qwen games our toy graders too, so this "
    "isn't a clean organism here.",
}


def msg_ranges(tok, ids):
    """[start, end) token range of each message's content: after `<|im_start|>role\\n` (and an assistant's empty think block)
    up to its <|im_end|>."""
    start, end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    think_end = tok.convert_tokens_to_ids("</think>")
    out = []
    for s in [i for i, x in enumerate(ids) if x == start]:
        a = s + 3
        if a < len(ids) and ids[a] == tok.convert_tokens_to_ids("<think>"):
            a = ids.index(think_end, a) + 2
        out.append([a, ids.index(end, a) if end in ids[a:] else len(ids)])
    return out


def render_ids(tok, r):
    """Same tokens as scripts/organism_olens.py: the chat-templated prompt, the completion, and a final <|im_end|>."""
    msgs = ([{"role": "system", "content": r["system"]}] if r["system"] else []) + [{"role": "user", "content": r["user"]}]
    prompt = chat_ids(tok, msgs, generation_prompt=True, thinking=False)
    return prompt + tok(r["completion"], add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]


def title(org):
    src, name = org.split(":")
    return f"Organism: {name} ({src})"


def load(S, key):
    """Append organism patterns to S["patterns"], their transcripts to S["trs"], and their reads to S["by_tid"]."""
    if not (DIR / "rollouts.jsonl").exists():
        return
    tok = S["tok"]
    rows = [json.loads(line) for line in (DIR / "rollouts.jsonl").open()]
    read = {(r["organism"], r["prompt_id"], r["sample"]): r for r in map(json.loads, (DIR / "olens_reads.jsonl").open())}
    base = {}  # (system, user) -> base Qwen's rollouts on that prompt
    for r in rows:
        if r["organism"] == "base":
            base.setdefault((r["system"], r["user"]), []).append(r["completion"])
    orgs = list(dict.fromkeys([o for o, _, _ in read] + [r["organism"] for r in rows if r["organism"] != "base"]))
    for org in orgs:
        mine = [r for r in rows if r["organism"] == org]
        mine.sort(key=lambda r: ((org, r["prompt_id"], r["sample"]) not in read, KINDS.index(r["kind"]), r["prompt_id"], r["sample"]))
        tids = []
        for r in mine:
            tid = f"{PREFIX}{org.replace(':', '-')}/{r['prompt_id']}/s{r['sample']}"
            rd = read.get((org, r["prompt_id"], r["sample"]))
            ids = rd["ids"] if rd else render_ids(tok, r)
            msgs = ([{"role": "system", "content": r["system"]}] if r["system"] else []) + [
                {"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["completion"]},
            ]
            S["trs"][tid] = {
                "transcript_id": tid,
                "pattern_id": f"organism:{org}",
                "prompt_id": f"organism:{r['prompt_id']}",  # matches no Claude rollouts
                "messages": msgs,
                "ids": ids,
                "msg_tok": msg_ranges(tok, ids),
                "judgment": {"match": False, "explanation": ""},
                "is_highlight": rd is not None,
                "weirdchat_url": None,
                "organism": {
                    "key": org,
                    "kind": r["kind"],
                    "prompt": r["prompt_id"],
                    "sample": r["sample"],
                    "n_tokens": r["n_tokens"],
                    "hit_cap": r["truncated"],
                    "read": rd is not None,
                    "read_pos": rd["positions"] if rd else [],
                    "base_rollouts": base.get((r["system"], r["user"]), []),
                },
            }
            tids.append(tid)
            reads, read_tokens = (rd["reads"], rd["tokens"]) if rd else ([], [])
            for x in reads:
                t = tid if x["cond"] == "organism" else tid + BASE
                rec = {
                    "key": key(t, x["pos"], x["layer"], lens=LENS),
                    "tid": t,
                    "pos": x["pos"],
                    "layer": x["layer"],
                    "temperature": 0.0,
                    "n": 1,
                    "seed": 0,
                    "token": read_tokens[x["pos"]],
                    "norm": None,
                    "samples": [{"text": x["text"], "bullets": x["bullets"]}],  # no FVE yet: read without the whiteners
                    "lens": LENS,
                }
                S["by_tid"].setdefault(t, {})[rec["key"]] = rec
        family = mine[0]["family"]
        src, name = org.split(":")
        n_read = sum(S["trs"][t]["organism"]["read"] for t in tids)
        S["patterns"].append(
            {
                "idx": len(S["patterns"]),
                "ds": "organisms",
                "pattern_id": f"organism:{org}",
                "title": title(org),
                "behavior_id": "model organism",
                "method": "LoRA model organism",
                "metrics": {},
                "n_prompts": len({r["prompt_id"] for r in mine}),
                "tids": tids,
                "weirdchat_url": HF[src].format(name),
                "elo": {},
                "description": FAMILY[family],
                "representative_user_text": mine[0]["user"],
                "cluster_label": family,
                "openrouter_replication": None,
                "organism": {"key": org, "family": family, "n_read": n_read},
            }
        )
