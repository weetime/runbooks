# 官方来源与复核边界（2026-11-16）

| 结论 | 一手来源 |
|---|---|
| DeepInfer支持范围、默认1017、未公开完整服务 | [固定README](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md) |
| B300=CC10.3、288GB | [NVIDIA GPU表](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-gpus.html) |
| SM major 10能力检查 | [device.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deep_cluster/hardware/device.py) |
| FlashMLA构建包含sm_103a | [固定setup.py](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/setup.py) |
| DeepSelect构建包含sm_103a | [固定setup.py](https://github.com/deepseek-ai/DeepSelect/blob/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18/setup.py) |
| EP/DP/CP例子及benchmark计账 | [固定examples说明](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/README.md) |
| Torch cu132及NCCL要求 | [requirements](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/requirements-nvidia.txt)、[build.sh](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/build.sh) |
| vLLM模型/B300配方、DSpark与thinking参数 | [模型配方](https://github.com/vllm-project/vllm/blob/e0af2d6a77b2cba2877d5aaeb1ac5a3aee676bc8/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml) |
| 原厂模型、编码与采样建议 | [模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/08c8657f02291e11b6f6b63e0e16014f8e748518) |
| AIPerf当前release | [v0.13.0](https://github.com/ai-dynamo/aiperf/releases/tag/v0.13.0) |

## 固定身份

- 模型revision：`08c8657f02291e11b6f6b63e0e16014f8e748518`。
- DeepInfer：`a679bb86ce36e883a98a97733492b686e5b9a08e`。
- vLLM v0.32.0：`5df63a7203cdb9328d824a27b682e77c7b9928be`。
- vLLM配方检索提交：`e0af2d6a77b2cba2877d5aaeb1ac5a3aee676bc8`；该配方路径在v0.32.0 tag不存在。
  因而当前配方不能直接冒充v0.32.0随版验证结果；须比较发行版能力/帮助输出后冻结，必要升级另记新臂。
- deepseek-recipe：`935817c66f5d6e64838f419692545f06787e485f`；仅在线适配候选，不是HTTP服务本体。
- DeepEP：`f6aa38fe06cd4a0489e7730365d4e0114970a2f3`。
- DeepGEMM：`1fc9c66975ee322d628b3288534ca0e9ce12a64d`。
- FlashMLA：`2e5429fc5653bab6e081f09477126f731882a6a9`。
- DeepSelect：`bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18`。

Torch cu132 cp312 wheel的公开metadata要求NCCL2.29.7，而固定DeepInfer `build.sh`明确强制安装2.32.3。
构建时保留完整pip-check输出，只接受这条精确匹配的官方覆盖并单独标注；其他依赖冲突必须失败。
这不是“pip check全通过”，也不是B300数值或通信验收。

已检索根目录/发布入口，本轮没有找到DeepInfer官方Dockerfile或可直接使用的官方release镜像。
交付镜像须标“按固定官方源码自行构建”，不标官方预构建，不猜测仓库名。
保留源码、递归submodule SHA、GCC/Python/Torch/CUDA/NCCL、pip freeze、构建日志、镜像digest及SBOM。

## 风险与先做的验证

1. B300的SM103不能靠“Blackwell”名称推断：查cuobjdump目标与实际kernel调用，特别是DeepEP/DeepGEMM JIT。
2. 本轮A100仅做过依赖准备，现已停止；正式镜像按用户要求在现场B300构建。不改主机驱动或运行中服务。
3. 编译镜像包含NVCC和运行期JIT依赖；断网后首次运行仍须编译成功，而非只有联网机缓存能用。
4. 原厂无Jinja chat template；保留`encoding/`和tokenizer哈希，以官方编码生成token IDs，两臂逐条相同。
5. DeepInfer把temperature=0钳为极小正值。候选采用原厂推荐temperature=1、top_p=1、seed=0，
   effort75显式冻结；不能把历史temperature0 A100成绩直接并入本次。不同引擎RNG不要求逐token相同。
6. 自然文本采用历史预算，先做预算/finish_reason预检；DeepInfer输出没有统一finish_reason时保持unknown，
   不把达到预算、HTTP200或正常进程退出当完整回答。
7. 完整在线服务不属于本轮已完成工作；AIPerf离线镜像准备不代表DeepInfer已经能接受HTTP压测。

## 已报告的正确性风险

- [DSpark thinking/chat输出异常 #62417](https://github.com/vllm-project/vllm/issues/62417) 已关闭（2026-10-27）；闭单本身不证明v0.32.0镜像已包含对应修复，须检查提交及G2。
- [GB300长Agent回复重复 #62935](https://github.com/vllm-project/vllm/issues/62935) 查询时仍开放；报告称DSpark关闭后仍存在，不把它简化为投机问题，也不推断本机必现。
- 以上作为探针设计和版本核验线索，不当作本次测试结果。不能未经登记关DSpark规避异常后继续沿用同一配方名。

查询时Docker Hub未找到`vllm/vllm-openai:v0.33.0rc3`容器tag。GitHub release存在不等于公开镜像就绪，且release分支与main修复提交可能经过cherry-pick；不能仅凭版本号宣称修复已包含。保留v0.32.0为待G2验证候选，不把它称为已通过的正式基线。
