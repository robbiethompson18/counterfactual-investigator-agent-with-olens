#!/usr/bin/env bash
# Box setup for read.py. Weights go in $W (default /dev/shm: vast root disks are small; SkyPilot uses W=/data).
#   bash setup.sh   then   ~/v4/.venv/bin/python read.py --model ${W:-/dev/shm}/v4flash --lens ${W:-/dev/shm}/v4lens ...
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/v4 && cd ~/v4
[ -d .venv ] || uv venv -q --python 3.12 .venv
VIRTUAL_ENV=~/v4/.venv uv pip install -q "transformers==5.19.0" torch accelerate safetensors huggingface_hub numpy "kernels==0.17.0"
hf() { uvx --from 'huggingface_hub[hf_xet]>=0.34,<1' hf "$@"; }
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=0
hf download deepseek-ai/DeepSeek-V4-Flash --revision 60d8d70770c6776ff598c94bb586a859a38244f1 --local-dir ${W:-/dev/shm}/v4flash
hf download camila-blank/deepseek-v4-flash-olens-inverter --revision ec495e5c2b253acc6b8e13a1b54c028217d36c66 --local-dir ${W:-/dev/shm}/v4lens
.venv/bin/python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
