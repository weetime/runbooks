#!/usr/bin/env python3
"""Generate unapproved local runtime inputs. Plan and rules: weetime/runbooks#1.

No downloads, builds, GPU actions or approval flags are executed/set here.
"""
import argparse
import json
from pathlib import Path

RECIPES = {'deepinfer': {'engine': 'deepinfer', 'env': {'DEEPINFER_ACCELERATOR': 'nvidia', 'DEEPINFER_USE_NEXTN_DECODE': '5', 'DEEPINFER_ENABLE_MTP_VARLEN': '1', 'DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS': '32'}, 'engine_kwargs': {'model': '/models/DeepSeek-V4.1-Flash-1017', 'max_model_len': 262144, 'max_num_seqs': 128, 'max_num_batched_tokens': 32768, 'gpu_memory_utilization': 0.85, 'trust_remote_code': True, 'seed': 0, 'expert_parallel_size': 8, 'context_parallel_size': 8, 'data_parallel_size': 1, 'engine_rank': 0, 'decode_sequence': False, 'chunked_prefill': True, 'disable_log_stats': False}}, 'vllm': {'engine': 'vllm', 'env': {}, 'engine_kwargs': {'model': '/models/DeepSeek-V4.1-Flash-1017', 'max_model_len': 262144, 'max_num_seqs': 128, 'max_num_batched_tokens': 32768, 'gpu_memory_utilization': 0.85, 'trust_remote_code': True, 'seed': 0, 'tensor_parallel_size': 8, 'tokenizer_mode': 'deepseek_v41', 'speculative_config': {'method': 'dspark', 'num_speculative_tokens': 5, 'draft_sample_method': 'probabilistic', 'rejection_sample_method': 'block', 'enable_adaptive_verification': True}}}}
IMAGES = {'platform': 'linux/amd64', 'images': [{'name': 'aiperf', 'source': 'nvcr.io/nvidia/ai-dynamo/aiperf', 'source_digest': 'sha256:3f384faff67d7135653c00b274744e7be7c1fe25d9d61a8e5b17ad740aa4812b', 'destination_tag': 'swr.cn-north-4.myhuaweicloud.com/runner/aiperf:0.13.0-amd64-3f384faff67d'}, {'name': 'vllm', 'source': 'vllm/vllm-openai', 'source_digest': 'sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9', 'destination_tag': 'swr.cn-north-4.myhuaweicloud.com/runner/vllm-openai:v0.32.0-amd64-6ee133356d49'}, {'name': 'cuda-build-base', 'source': 'nvidia/cuda', 'source_digest': 'sha256:d266e59b88c295bc5fa0e4cef9064eaff84939381b09e2c3d76a5532a303e42d', 'destination_tag': 'swr.cn-north-4.myhuaweicloud.com/runner/cuda:13.2.0-devel-ubuntu24.04-amd64-d266e59b88c2', 'copy_upstream': True}]}
ENV = "# Public configuration only. Registry credentials belong to the runtime's login\n# store, never in this file. Generated as a local .env and fill the B300-specific values.\nRUNTIME=nerdctl\nRUNTIME_NAMESPACE=deepinfer-b300-bench\nWORK_DIR=/data/deepinfer-vllm-b300\nDATA_ROOT=/data/share/models/_datasets\nMODEL_DIR=/data/share/models/DeepSeek-V4.1-Flash-1017\nGPU_IDS=\n# Fill BUILD_KIT_IMAGE only after the future kit is published and verified.\nBUILD_KIT_IMAGE=\nVLLM_IMAGE=swr.cn-north-4.myhuaweicloud.com/runner/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9\nDATA_IMAGE=swr.cn-north-4.myhuaweicloud.com/runner/benchmark-data@sha256:983d4875ac1540e1f5c52007015d0e79203690ad345954e9b36aa1f8207a3aeb\nDI_LOCAL_IMAGE=local/deepinfer-b300:a679bb86\n# Set after publishing / freezing the B300-built image. Formal bench rejects tags.\nDI_IMAGE=\nBUILD_CPUS=16\nBUILD_MEMORY=96g\nMAX_JOBS=8\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('.local'))
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    for name,recipe in RECIPES.items():
        (a.output/(name+'-candidate.json')).write_text(json.dumps(recipe,indent=2)+'\n')
    (a.output/'images.json').write_text(json.dumps(IMAGES,indent=2)+'\n')
    (a.output/'b300.env').write_text(ENV)
    print('Local templates created; fill resource/image values and freeze them in issue #1. No gates passed.')


if __name__=='__main__':
    main()
