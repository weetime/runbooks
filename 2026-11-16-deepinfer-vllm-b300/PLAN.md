---
exp_id: 2026-11-16-deepinfer-vllm-b300
status: running
hypothesis: "同一原厂1017权重与请求下，比较B300上DeepInfer和vLLM配方的离线推理性能"
owner: fangyong
created: 2026-11-16
last_run: 2026-11-16
tags: [b300, deepinfer, vllm, offline, preparation]
related: []
---

# B300 现场离线构建与 DeepInfer / vLLM 对比

实验 issue：[runbooks#1](https://github.com/weetime/runbooks/issues/1)。

交付目标是**先准备离线构建包，到B300现场先构建DeepInfer镜像，再按门禁测试**。
用户无法提供B300远程机器，本轮不执行B300构建、模型验收或性能测试。
按用户最新要求，本轮仅交付脚本和实验方案，不再执行DeepInfer镜像构建或依赖环境预装。此前启动的相关准备进程已停止；不把A100产物当作已验收B300镜像。

统一入口：[scripts/b300.sh](scripts/b300.sh)；完整操作：[RUNBOOK.md](RUNBOOK.md)。

```bash
cp configs/b300.env.example configs/b300.env
# 填写已核验镜像digest、现场GPU卡号、模型和共享盘路径；不要填写密码。
export CONFIG="$PWD/configs/b300.env"
bash scripts/b300.sh plan
bash scripts/b300.sh check
bash scripts/b300.sh build   # 到B300现场执行；构建容器 --network none
```

## 阶段与边界

| 阶段 | 准备机 / B300各做什么 |
|---|---|
| 联网准备 | 固定源码及全部递归依赖、CUDA13.2/GCC13/Python3.12、Torch和构建依赖，发布build-kit到SWR |
| 现场构建 | B300加载build-kit，在断网容器编译四个vendor库和DeepInfer，生成实际推理镜像 |
| 制品与功能门禁 | 完整权重逐文件SHA、数据/编码、最终启动参数、正确性、单流8请求×3轮 |
| 性能对比 | 同模型/源消息/token IDs/采样/预算/卡数；先快速观察，再标准样本与三次独立起服关键点 |

build-kit含编译器、源码和已安装的依赖，**不含已编译的DeepInfer和四个vendor算子**，也不含模型权重。
构建成功不自动通过G0/G1/G2/G3。`b300.sh`不会自动把门禁标为通过。

## 支持情况与配方

- 固定DeepInfer提交 `a679bb86ce36e883a98a97733492b686e5b9a08e`，默认原厂模型为
  `deepseek-ai/DeepSeek-V4.1-Flash-1017`。原厂mixed MXFP4/MXFP8权重不能换成W8A8或0909。
- B300计算能力10.3；固定的FlashMLA和DeepSelect明确编译`sm_103a`。
  这是源码支持证据，完整算子/通信/模型必须在B300验收。
- 本提交实际依赖Torch `2.13.0+cu132`、NCCL `2.32.3`；现场须检查驱动对CUDA13.2的兼容性，脚本不更新驱动。
- vLLM候选为公开镜像v0.32.0，配方逐项匹配和已报告的DSpark正确性风险须经G2核验。
  查询时未找到v0.33.0rc3公开容器tag，不凭GitHub release名称猜测镜像已就绪。
- `configs/*candidate.json`均为**待现场核对的候选配方**。DeepInfer EP8/CP8/DP1用于单协调器原生对比，
  与默认EP8/CP1/DP8不同；按仓库纪律，实质偏离在发压前确认并冻结。

## 对比口径

离线部署是“不依赖外网”；原生离线推理是“直接调用引擎API”。两者分开记录。
DeepInfer当前开源版缺完整在线KV缓存、VE/P/D调度与负载均衡，不能直接冒充现成HTTP服务。
首阶段比较共同支持的原生生成；AIPerf镜像另行准备，DeepInfer在线适配未就绪的HTTP/Agent/CC项记not_run。

原生主表报实际输出tok/s、tok/s/卡、batch wall time、样本和质量/结束状态。
B是同步batch大小，不冒充闭环并发C；普通batch工具不提供HTTP TTFT/TPOT/goodput。
G3另用native host输出chunk计时，显式排除HTTP和GPU纯时间口径，原始chunk及token对账必须留存。

两引擎并行和优化配方不同，结论是**两套部署的对比**，不归因为引擎本身快多少。
官方prefill/decode曲线分别用4/64卡，单机8卡不复现Wide-EP64/128生产收益。
官方builtin decode吞吐按全接受率计账，只列独立诊断表，不混入实际生成token主表。

## 当前制品与验证（未完成项不标就绪）

以[manifests/images.json](manifests/images.json)的逐项实测状态为准。
统一standard-v1本地1,042项SHA验证通过；首发32MB文本镜像已上SWR并通过断网容器检查。
其中7×128正式请求+7×3独立预热请求，原生token-ID及对应HTTP候选均已冻结。
完整约600MB归档另存本地，Agent/CC完整源不在首发文本镜像，不冒充这些场景已可执行。
完整模型权重路径/逐文件哈希、B300驱动/互联/空闲卡和GPU脚本实机验收尚未完成。

本次仅新增实验目录和独立分支；原有工作区、集群服务和驱动未改动，未执行模型GPU压测。

## 预算与停止线

编译暂估CPU 2–8小时，现场实际记录，不当作实测保证。8卡两臂门禁/快速档暂估2–4节点小时，
标准档4–8节点小时，关键点三次交替起服另计3–6节点小时。G3后用实测耗时重估并冻结上限。
所有正式判据遵循`BENCHMARK_POLICY.md`；详见[TEST_MATRIX.md](TEST_MATRIX.md)。

当前未完成：DeepInfer build-kit未构建/未发布；DeepInfer推理镜像由现场B300脚本构建；vLLM大镜像同步未完成，保留精确源digest和同步命令。不能据此宣称整个离线环境已经准备完毕。
