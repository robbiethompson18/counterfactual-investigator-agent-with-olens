# Whitebox tools beyond the O-lens

Neel asked us to add other whitebox methods (SAEs, J-lens / J++, NLAs, SNLAs) to the investigator
agent. Status as of 2026-10-20. "The agent" below means Agam's harness, which we stopped running
after commit
[`a1e8991`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/a1e8991)
(`vendor/` is reference-only now); new agent tools go into `cfbench/tools.py`.

| Method            | In the viewer                                    | In the agent                                                  |
| ----------------- | ------------------------------------------------ | ------------------------------------------------------------- |
| J-lens            | Agam's bank (all layers)                         | Not run yet; Agam's harness has a `jlens` arm and banks exist |
| J++               | Case page (`olens/src/olens/extras.py`)          | No; would need J++ banks for CHIVE incidents (≥ 120 GB GPU)   |
| NLA (L42, ML-NLA) | Agam's banks; case page has the ceselder L42 NLA | Not run yet; harness has `nla` / `mlnla` arms and banks exist |
| SAE               | No                                               | No; plan below                                                |
| SNLA              | No                                               | **Not doing for now**                                         |

## SNLA: parked

Decision (Robbie, 2026-10-20): no SNLA arm for now. Why:

- The only published SNLA ([`asher577/nla-rl-ckpts`](https://huggingface.co/asher577/nla-rl-ckpts),
  `*_subspace_bsf*`, with the BSF in
  [`asher577/bsf_071226`](https://huggingface.co/asher577/bsf_071226)) reads **Qwen3-8B L24**. Our
  CHIVE incidents are Qwen3.6-27B, so running it on them reads a different model's activations: a
  control at best, not a whitebox arm.
- Using it properly means a Qwen3-8B CHIVE leg, e.g. CHIVE's own
  [`adamkarvonen/chive-data`](https://huggingface.co/datasets/adamkarvonen/chive-data)
  `qwen3_8b_wildchat_eval` (649 incidents), converted to Agam's format. That leg has no O-lens.
- The BSF loader and projection code aren't in [EasyNLA](https://github.com/asherps/EasyNLA) (plain
  NLAs only) and aren't public yet. Guessing the projection or injection scale silently breaks the
  readouts.
- Reported results so far don't put SNLAs ahead of the O-lens.

Revisit if Asher's SNLA code becomes available and we want a Qwen3-8B leg anyway (it would also
carry CHIVE's existing NLA / AO / SAE baselines). GPU cost of SNLA banks would be ~$20–75 on one
H200; the engineering is the real cost.

## SAE: plan

- Candidates:
  - [ceselder/qwen36-27b-sae-l42-1b](https://huggingface.co/ceselder/qwen36-27b-sae-l42-1b) (2026):
    native Qwen3.6-27B, L42, 131k features, k=64, EV 0.70, Apache-2.0. No labels, but top-activating
    windows per feature, so we can label with an LLM.
  - [Qwen-Scope](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100)
    ([arXiv 2605.11887](https://arxiv.org/abs/2605.11887), 2026): trained on **Qwen3.5**-27B, all
    layers, 82k features; Neuronpedia labels at L31 only. Transfer to 3.6 must be measured: L31 MLP
    `down_proj` differs ~32% between 3.5 and 3.6.
- First step: one RunPod GPU-hour capturing L31 / L42 residuals on the 300 incidents and measuring
  EV and ΔCE for both SAEs.
- Tool design, learning from CHIVE's SAE tool (top-15 labels per position, which in the paper
  ([arXiv 2608.16747](https://arxiv.org/abs/2608.16747), 2026) hurt Gemini 3.1 Pro, 0.793 → 0.746
  AUROC, because generic dense features dominate): drop features that fire on > 1% of corpus or >
  30% of the transcript's tokens, rank by activation / feature max, show value and density, and add
  `sae_find(query)` (label search) and `sae_feature(id)`.
- Planned to plug into Agam's harness as a new reader kind (`readers.KINDS`, `readers.DESCRIBE`,
  `tools.WINDOW`), monkeypatched from `scripts/run_chive_or.py` (harness unused and script removed
  after `a1e8991`).
