#!/usr/bin/env python3
"""Generate local runtime inputs for weetime/runbooks#1.

No downloads, builds, GPU actions or approval flags are executed/set here.
"""
import argparse
import json
from pathlib import Path

RECIPES = {'deepinfer': {'engine': 'deepinfer',
               'env': {'DEEPINFER_ACCELERATOR': 'nvidia',
                       'DEEPINFER_USE_NEXTN_DECODE': '5',
                       'DEEPINFER_ENABLE_MTP_VARLEN': '1',
                       'DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS': '32'},
               'engine_kwargs': {'model': '/models/DeepSeek-V4.1-Flash-1017',
                                 'max_model_len': 262144,
                                 'max_num_seqs': 128,
                                 'max_num_batched_tokens': 32768,
                                 'gpu_memory_utilization': 0.85,
                                 'trust_remote_code': True,
                                 'seed': 0,
                                 'expert_parallel_size': 8,
                                 'context_parallel_size': 8,
                                 'data_parallel_size': 1,
                                 'engine_rank': 0,
                                 'decode_sequence': False,
                                 'chunked_prefill': True,
                                 'disable_log_stats': False}},
 'vllm': {'engine': 'vllm',
          'env': {},
          'engine_kwargs': {'model': '/models/DeepSeek-V4.1-Flash-1017',
                            'max_model_len': 262144,
                            'max_num_seqs': 128,
                            'max_num_batched_tokens': 32768,
                            'gpu_memory_utilization': 0.85,
                            'trust_remote_code': True,
                            'seed': 0,
                            'tensor_parallel_size': 8,
                            'tokenizer_mode': 'deepseek_v41',
                            'speculative_config': {'method': 'dspark',
                                                   'num_speculative_tokens': 5,
                                                   'draft_sample_method': 'probabilistic',
                                                   'rejection_sample_method': 'block',
                                                   'enable_adaptive_verification': True}}}}
IMAGES = {'platform': 'linux/amd64',
 'images': [{'name': 'aiperf',
             'source': 'swr.cn-north-4.myhuaweicloud.com/runner/aiperf',
             'source_digest': 'sha256:ba13d50bfe496a574bb1de7107275d670a269c9d8bab835e6a631b4fb635978e',
             'destination_tag': 'swr.cn-north-4.myhuaweicloud.com/runner/aiperf:0.13.0-b300-text-v1-amd64',
             'copy_upstream': False},
            {'name': 'vllm',
             'source': 'public.ecr.aws/q9t5s3a7/vllm-release-repo',
             'source_digest': 'sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9',
             'destination_tag': 'swr.cn-north-4.myhuaweicloud.com/runner/vllm-openai:v0.32.0-amd64-6ee133356d49'}]}
ENV = '# Registry credentials belong in the runtime login store.\nRUNTIME=docker\nRUNTIME_NAMESPACE=deepinfer-b300-bench\nWORK_DIR=/data/deepinfer-vllm-b300\nDATA_ROOT=/data/share/models/_datasets\nMODEL_DIR=/data/share/models/DeepSeek-V4.1-Flash-1017\nGPU_IDS=\nVLLM_IMAGE=swr.cn-north-4.myhuaweicloud.com/runner/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9\nAIPERF_IMAGE=swr.cn-north-4.myhuaweicloud.com/runner/aiperf@sha256:ba13d50bfe496a574bb1de7107275d670a269c9d8bab835e6a631b4fb635978e\n# Set the DeepInfer image built at the B300 site.\nDI_IMAGE=\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('.local'))
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    for name,recipe in RECIPES.items():
        (a.output/(name+'.json')).write_text(json.dumps(recipe,indent=2)+'\n')
    (a.output/'images.json').write_text(json.dumps(IMAGES,indent=2)+'\n')
    (a.output/'b300.env').write_text(ENV)
    print('Local configuration created. Follow issue #1 to set site values and run the gates.')


if __name__=='__main__':
    main()
