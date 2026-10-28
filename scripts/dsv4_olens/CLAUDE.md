# scripts/dsv4_olens: DeepSeek V4 Flash O-lens reads

Reads our transcripts with Camila Blank's V4 Flash O-lens inverter
([camila-blank/deepseek-v4-flash-olens-inverter](https://huggingface.co/camila-blank/deepseek-v4-flash-olens-inverter),
SFT stage, PEFT LoRA, injection at the input of decoder layer 1; carriers exist only for layers 15,
18, ..., 39, 42 of V4 Flash's 43 layers, and 30 is about two-thirds deep). Kylie wrote the pipeline
(branch `claude/af-v4-olens`, for SkyPilot); `runpod_af.sh` is the same setup and run steps on one
of Robbie's RunPod pods (`runpod/CLAUDE.md`). Results: docs/envs/alignment_faking.md.

## Pipeline

1. `prep_env.py`: `requests.jsonl` from `environments/` alignment_faking run folders, one episode
   per (arm, item) across the runs given (a top-up pools with its base run), lowest sample, reply
   cut to `--reply-chars`. Each request keeps `run` and `episode`, so a read maps back to the
   env_runs loader's `alignment_faking/<run>:<episode>`.
2. `read.py` (HF transformers 5.19 FP8, `kernels==0.17`, pipeline across 2 GPUs): capture the
   4-stream block outputs at the read positions (`states.pt`, stays on the pod/volume) and score the
   first-reply logits (`logits.jsonl`). `--skip-lens`: its own lens stage is 2.4-3.6 reads/s; use
   `vllm_read.py`.
3. `vllm_read.py`: the lens stage on vLLM 0.32 (21 reads/s on 2xH200: greedy + 2 samples, CUDA
   graphs) → `reads_L<layer>.jsonl`. Needs vllm-metamodels
   (`vllm-lens @ git+https://github.com/ceselder/vllm-metamodels`, `--no-deps`) for per-request
   prompt-row routing, the `hc_inject/` plugin (writes 16,384-wide vectors into V4's folded 4-stream
   state at layer 1), the adapter renamed by `convert_adapter.py`, and a CUDA 13.0 toolkit for the
   DeepGEMM / FlashInfer JIT. vLLM 0.27.1 has no V4 LoRA. Against HF on 4,405 layer-30 reads: first
   10 characters equal 65%, 20 49% (bf16 greedy drift).
4. `to_cfbench_reads.py RUN_DIR`: `reads_L30.jsonl` + `requests.jsonl` →
   `RUN_DIR/cfbench/{reads,requests}.jsonl` in cfbench's format (`cfbench/olens.py`), ids
   `alignment_faking/<run>:<episode>`, bullets = greedy then samples with `fve: null`; `char_offset`
   is exact (read.py's `char` indexes the rendered string, so each message's content starts at a
   known offset; template tokens get null).
5. `placebo_from_glm_email.py`: Celeste's GLM email dense reads cut to 3 bullets, `fve` null, as the
   `olens_placebo_agentic` pool (cfbench's placebo needs reads over another env's transcripts).
   Kylie's expview export and base-model sampling checks stayed on `claude/af-v4-olens`.

## Running on a RunPod pod (Robbie's agents)

Everything expensive is cached on the network volume **`dsv4-cache` (`sujswd59wj`, 250 GB, US-CA-2,
$17.50/month)**, mounted at `/workspace`: `runpod_af.sh` sets `HOME=/workspace/dsv4/home`, so the
weights (`/workspace/dsv4/w/v4flash`, 160 GB FP8, `60d8d70`; `v4lens` + `lora_vllm`), both uv venvs
(`~/v4` HF, `~/vl` vLLM 0.32), uv's cache and the HF / vLLM / DeepGEMM / FlashInfer JIT caches under
`~/.cache` survive the pod. Only the CUDA 13.0 apt packages are reinstalled per pod. The two Qwen
volumes (150 GB, ~110 GB used) can't hold the 160 GB of V4 weights, which is why this volume exists.
Python on NFS imports slower (docs/runpod-qwen-server.md); accepted here for a rerun that skips a
160 GB download.

```bash
# 1. requests (local)
uv run python scripts/dsv4_olens/prep_env.py --envs data/envs/alignment_faking \
  --runs 2026-10-27_v7-dsv4flash-nocot 2026-10-27_v7-dsv4flash-nocot-topup-a 2026-10-27_v7-dsv4flash-nocot-topup-b \
  --items 0:100 --reply-chars 120 --out data/runs/olens_reads/<run>      # after scripts/runs.py new olens_reads <slug> ...
# 2. watcher first (runpod/CLAUDE.md), then the pod on the volume (pod.sh's flags + the volume)
nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' runpod/watch_pod.sh dsv4-af-v7 /workspace/dsv4/out/af-v7 \
  "$PWD/data/runs/olens_reads/<run>" 4 8 > /tmp/watch_dsv4-af-v7.log 2>&1 < /dev/null &
pgrep -fl "watch_pod.sh dsv4-af-v7"
runpodctl pod create --name dsv4-af-v7 --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --gpu-id "NVIDIA H200" \
  --gpu-count 2 --container-disk-in-gb 200 --volume-in-gb 0 --network-volume-id sujswd59wj --volume-mount-path /workspace \
  --data-center-ids US-CA-2 --ports 22/tcp --env "{\"HF_TOKEN\": \"$(hf auth token)\", \"HF_XET_HIGH_PERFORMANCE\": \"1\"}"
# 3. once ssh answers (REST publicIp + portMappings["22"]): no chown on the NFS volume, so no rsync -a
rsync -rltz --no-owner --no-group -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -p PORT" scripts/dsv4_olens/ root@IP:/workspace/code/
rsync -rltz --no-owner --no-group -e "ssh ... -p PORT" data/runs/olens_reads/<run>/requests.jsonl root@IP:/workspace/code/requests.jsonl
ssh ... 'mkdir -p /workspace/dsv4/out/af-v7; nohup bash /workspace/code/runpod_af.sh > /workspace/dsv4/out/af-v7/job.log 2>&1 < /dev/null &'
# 4. poll job.log every 3 min; the watcher rsyncs $OUT (minus states.pt) into the run dir and deletes the pod at READS_DONE/FAILED
python scripts/dsv4_olens/to_cfbench_reads.py data/runs/olens_reads/<run>   # then scripts/runs.py finish
```

`runpod_af.sh` is idempotent (stage markers `SETUP_DONE`, `CAPTURE_DONE`, `READS_DONE`; `FAILED` on
any error, which the watcher treats as "delete the pod"; `$OUT` is on the volume so the logs
survive). Env knobs: `W`, `OUT`, `CODE`, `REQ`, `CAPTURE_LAYERS` (24,30,36), `LENS_LAYERS` (30),
`READ_ARMS` (preview,public).

## Observed timings and cost (2026-10-27, 2x H200 at $10.58/h, pod `dsv4-af-v7`, 200 requests)

Two pods, 89 min of 2x H200 in total (about $16 of pod time; the balance went $37.5 → $17.1, which
also counts the new volume and container disks).

| Stage (pod 1, fresh volume, 17.5 min total)                                  | Time    |
| ---------------------------------------------------------------------------- | ------- |
| `runpodctl pod create` → ssh answers                                         | 20 s    |
| `setup.sh`: weights 160 GB + lens (hf xet, ~450 MB/s) + HF venv on NFS       | 9.7 min |
| vLLM 0.32 venv + vllm-metamodels + hc_inject + `convert_adapter.py`          | 6.1 min |
| CUDA apt: failed (cuda-keyring duplicated the image's repo; fixed in script) | —       |

| Stage (pod 2, cached volume, 71.5 min total)                                                                                                                                                   | Time     |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| CUDA 13.0 apt packages (container disk, every pod)                                                                                                                                             | 1.5 min  |
| `read.py` capture: HF weight load from NFS (1,736 tensors at 3.4/s) ~10 min, then 200 requests / 26,577 positions in 318 s                                                                     | 15.5 min |
| `vllm_read.py`: vLLM import off NFS 3.3 min; engine init 24.8 min (weights 33 s, Triton + FlashInfer + DeepGEMM JIT warmup ~17 min, CUDA graphs 3 min); reads 26,577 in 1,119 s = 23.7 reads/s | 53 min   |

The JIT caches now sit under `/workspace/dsv4/home/.cache`, so a rerun should skip most of the 24.8
min engine init (not yet measured). Gotchas met: the watcher deletes the pod a minute after `FAILED`
appears, and a stale `FAILED` on the volume from the previous pod does the same, so remove it before
restarting a job; rsync to the NFS volume needs `--no-owner --no-group` (chown fails); vLLM's weight
load prints thousands of benign `CUDACachingAllocator ... OOM` warnings on 2x H200.

## One-engine reader: `vllm_capture_read.py` (2026-10-27)

`vllm_capture_read.py` does read.py's capture + first-reply logits on the same vLLM 0.32 engine that
then runs `vllm_read.decode_states` (the lens stage, unchanged), so there is no HF venv, no 2-stage
pass and no `states.pt` handoff. Capture is `max_tokens=1` prefills of prompt + reply with
vllm-metamodels' hooks (`extra_args={"output_residual_stream": [L], "capture_positions": [...]}`);
the `hc_inject/hc_capture.py` plugin makes that work on V4 by wrapping
`vllm_lens._worker_ext._split_layer_output`: a V4 layer returns the deferred fold
`(x, residual[T, 4, D], post_mix, res_mix)`, and the wrapper folds it with `mhc_post_tilelang` (the
state entering layer L + 1, which is HF's layer-L output and what read.py's hook saved) into
`[T, 16384]` rows, and sets `VLLM_LENS_MULTI_STREAM=0` so the plugin accepts the request. The
capture prefill also carries `prompt_logprobs`, which gives read.py's "reject" (log P of the
captured sequence's reply: `<rejected/>`, or the actual reply with `--actual-reply`) and
`top_first`; one more prefill of prompt + `<request_fully_accepted/>` gives "accept". Positions and
rows are read.py's (`render` / `read_positions` imported from it), output rows are `vllm_read.py`'s
(`reads_L30.jsonl`: id, pos, char, role, token, ctx_left, ctx_right, layer, greedy, s1, s2) and
read.py's `logits.jsonl`. Chunked prefill is off so every prompt prefills in one pass (a split
prompt's 1-token tail would run as a decode graph, where the hooks don't fire).

    ~/vl/.venv/bin/python vllm_capture_read.py --model /workspace/dsv4/w/v4flash --lens /workspace/dsv4/w/v4lens \
        --requests requests.jsonl --out out/ --layers 30 --dense --actual-reply --read-arms preview,public --public-first \
        [--limit 20] [--save-states] [--check-against REF_DIR]
    python compare_reads.py RUN_A RUN_B --reads reads_L30.jsonl   # greedy-prefix and logit parity between two runs

Pod job: `runpod_vllm.sh` (cache check, CUDA apt, reinstall `hc_inject` with the new plugin, a
`--limit 20` smoke then the full file, each compared with the simple run's reads at
`$REF=/workspace/dsv4/out/af-v7`; writes `READS_DONE` / `FAILED` for `runpod/watch_pod.sh`). Driver,
from the repo root with no other pod on the account (balance ≥ ~$25: 2x H200 is $10.58/h):

    RUN=data/runs/olens_reads/<date>_dsv4-vllm-capture   # scripts/runs.py new olens_reads dsv4-vllm-capture ...
    nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' runpod/watch_pod.sh dsv4-vllm-capture /workspace/dsv4/out/vllm-capture $RUN 3 8 \
        > /tmp/watch_dsv4-vllm-capture.log 2>&1 < /dev/null &
    pgrep -fl "watch_pod.sh dsv4-vllm-capture"                     # must print the watcher before the next line
    VOLUME=sujswd59wj runpod/pod.sh up dsv4-vllm-capture "NVIDIA H200" 2
    runpod/pod.sh ssh dsv4-vllm-capture 'mkdir -p /workspace/code_vllm /workspace/dsv4/out/vllm-capture'
    rsync -az -e "ssh -o StrictHostKeyChecking=no -p PORT" scripts/dsv4_olens/ data/runs/olens_reads/2026-10-27_dsv4-af-v7/requests.jsonl root@IP:/workspace/code_vllm/
    runpod/pod.sh ssh dsv4-vllm-capture 'nohup bash /workspace/code_vllm/runpod_vllm.sh > /workspace/dsv4/out/vllm-capture/job.log 2>&1 < /dev/null &'
    runpod/pod.sh ssh dsv4-vllm-capture 'tail -3 /workspace/dsv4/out/vllm-capture/job.log'   # every 3 min; two identical tails = stuck

Test (2026-10-27, `data/runs/olens_reads/2026-10-27_dsv4-vllm-capture`, pod `dsv4-vllm-capture`, 2x
H200 on the cached volume, 74 min of pod ≈ $13 including a 20-request smoke and a 20-request
capture-only check): the same 200 v7 requests as the simple run, 26,577 dense layer-30 reads.

| stage            | simple run (`2026-10-27_dsv4-af-v7`)          | one engine (full file)                         |
| ---------------- | --------------------------------------------- | ---------------------------------------------- |
| HF model load    | 499 s                                         | none                                           |
| capture + logits | 318 s (HF)                                    | 52 s (vLLM)                                    |
| vLLM engine up   | 1,873 s (first JIT of the venv on the volume) | 435 s (JIT cached; 401–489 s over three boots) |
| lens             | 1,119 s, 23.7 reads/s                         | 984 s, 27.0 reads/s                            |
| total            | 68 min (≈ 40 min with a warm JIT)             | 24.5 min (one process, no `states.pt`)         |

Parity with the simple run (same positions, 26,577/26,577 shared): greedy first 10 / 20 / 40
characters equal 45.3% / 30.9% / 19.9% (Kylie's HF-capture-decoded-by-vLLM vs all-HF number was 65%
/ 49%: that pair shared the capture, this one does not, so the extra drift is the capture stack).
The capture itself is right: on 2,758 positions of the 20-request check, the vLLM layer-30 state vs
read.py's HF `states.pt` has cosine 0.982 mean (p5/50/95 0.960/0.987/0.994), norm ratio 0.996, worst
rows 0.53 (FP8 MoE kernels differ between HF `kernels` and vLLM DeepGEMM, and 30 layers of that is
the 18% relative difference, not a position or fold error). Logits (20 requests, the fixed "reject"
= log P of the captured reply): `reject` corr 0.9994, `accept` corr 0.77 (mean |diff| 0.6 nats; it
is a 7-token sum near −7), `accept − reject` corr 0.9986, sign agreement 1.0, top-1 first reply
token 100%. The full file's `logits.jsonl` was written with the earlier "reject" = log
P(`<rejected/>`) semantics (fixed in `8e29301`); its `accept` corr over 200 ids is 0.82.
