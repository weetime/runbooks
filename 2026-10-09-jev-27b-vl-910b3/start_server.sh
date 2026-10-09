#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=${JEV_MODEL_ROOT:?Set JEV_MODEL_ROOT to the downloaded model directory}
exec python3 -u "$TASK_ROOT/serve_decide_ascend.py" --model "$MODEL_ROOT" --served-model-name autotrust/JEV-27B-VL --enable-lora --max-lora-rank 32 --lora-target-modules qkv_proj o_proj gate_up_proj down_proj in_proj_qkvz out_proj lm_head --lora-modules "jev-decision=$MODEL_ROOT/adapter_vllm" --logprobs-mode processed_logprobs --max-logprobs 20 --max-model-len 4096 --max-num-seqs 8 --tensor-parallel-size 2 --gpu-memory-utilization 0.80 --no-enable-prefix-caching --mamba-cache-mode align --limit-mm-per-prompt '{"image": 1}' --trust-request-chat-template --enforce-eager --host 127.0.0.1 --port 8127
