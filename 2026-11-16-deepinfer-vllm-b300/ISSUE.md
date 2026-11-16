# B300：DeepInfer / vLLM 同模型对比的现场构建与测试流程

**当前交付是脚本与实验方案，不实际构建DeepInfer镜像。相关后台准备进程已停止。B300未接入，未执行模型或性能测试。**

目标：在同一B300资源预算、原厂`DeepSeek-V4.1-Flash-1017`权重与固定请求下，比较两套明确配方的性能。遵循本实验的`BENCHMARK_POLICY.md`，先冻结、门禁、快速档，再标准档与复测。

## 交付入口

- [统一shell：b300.sh](https://github.com/weetime/runbooks/blob/main/2026-11-16-deepinfer-vllm-b300/scripts/b300.sh)
- [现场操作手册](https://github.com/weetime/runbooks/blob/main/2026-11-16-deepinfer-vllm-b300/RUNBOOK.md)
- [测试矩阵、门禁与停止线](https://github.com/weetime/runbooks/blob/main/2026-11-16-deepinfer-vllm-b300/TEST_MATRIX.md)
- [官方调研/固定版本](https://github.com/weetime/runbooks/blob/main/2026-11-16-deepinfer-vllm-b300/RESEARCH.md)
- [镜像状态/digest清单](https://github.com/weetime/runbooks/blob/main/2026-11-16-deepinfer-vllm-b300/manifests/images.json)

## 构建顺序

1. 后续在联网准备机执行`b300.sh prepare`：固定源码及递归依赖、CUDA13.2/GCC13/Python3.12、Torch/cu132和构建工具，生成build-kit并由操作者推到SWR。该步骤现在未执行完成。
2. B300现场按digest导入build-kit、vLLM、数据；明确GPU范围并核对驱动、互联、实际占用及调度分配。
3. `b300.sh build`：在B300的`--network none`容器编译DeepInfer/vendor，生成推理镜像，不在A100交叉编译作最终交付。
4. 完整权重逐文件SHA、功能smoke、候选镜像发布/导出与digest冻结。
5. 门禁模板全部false起步；按同配方启动probe审核G1/G2，G3为8K/512的8请求×3组；实际输出/token/指标对账后才放行。
6. `b300.sh bench`逐点执行；关键点按A→B→A→B→A→B至少三次独立起服。每个命令的启动次数、缓存状态和配方单独记录，不冒充已完成全矩阵campaign。

```bash
cp configs/b300.env.example configs/b300.env
# 填写后续准备的build-kit digest、现场GPU_IDS、模型路径等；不放凭据。
export CONFIG="$PWD/configs/b300.env"
bash scripts/b300.sh plan
# 现场按RUNBOOK逐阶段执行，不默认自动发压。
```

## 支持与比较边界

DeepInfer固定提交`a679bb86ce36e883a98a97733492b686e5b9a08e`，其FlashMLA/DeepSelect构建目标含SM103；B300为CC10.3。这是源码支持证据，非B300验收。
原厂模型revision为`08c8657f02291e11b6f6b63e0e16014f8e748518`，不得换成W8A8/0909。

DeepInfer当前公开部分是原生离线引擎，缺完整在线服务；“离线部署”和“原生离线推理”分开。首阶段原生batch比较不冒充HTTP TTFT/TPOT、闭环容量或AIPerf成绩。G3的host chunk计时单列；不能取得可靠token/时间则不通过。
HTTP/Agent/CC未适配前not_run。官方sweep的全接受率吞吐不混入实际生成tok/s。不同并行/优化是部署对比，不归因为引擎本身。

## 制品实际状态

| 制品 | 状态 |
|---|---|
| AIPerf0.13.0 amd64 | 已推SWR并核验digest |
| CUDA13.2基础镜像 | 已推SWR并核验digest |
| 32MB文本数据镜像 | 已推SWR；按digest拉取并在断网容器校验通过 |
| vLLM0.32.0 | 已固定官方源digest，同步未完成；提供后续同步脚本，不标就绪 |
| DeepInfer build-kit | 未完成构建/发布；后续由操作者执行prepare |
| DeepInfer推理镜像 | 本轮不构建；后续由B300执行build |

数据使用固定standard-v1源；本地1,042项SHA通过。首发子集为14源文件，7×128正式请求及7×3独立预热，原生/HTTP准备记录共1,834条，预热与对应正式消息无重叠。
完整约600MB归档仅在本地保留；首发不含完整Agent/CC与权重。

## 约束、验证与待办

- GPU范围、最终官方配方比对、实质偏离确认和预算在现场冻结；脚本不默认all卡、不改驱动、不自动通过门禁。
- 扩档停止：吞吐增长<10%、TPOT P95>100ms、TTFT P95>10×同形状C1、错误率>1%、持续排队/抢占或资源不允许。native batch不据此伪造在线容量。
- 标准点128独立请求；小样本分位带n与标记；自然EOS和固定ignore_eos分表，输出逐条复核，截断不当完整回答。
- 源码和矩阵已准备，shell/Python/JSON静态检查、双引擎命令dry-run、chunk计时CPU算术及缺GPU范围拒绝检查通过。**尚未做B300实机检查**。
- vLLM已报告的DSpark/长回复风险纳入G2，不擅自关优化后沿用原配方。查询时v0.33.0rc3公开镜像tag未找到，不猜测已发布。
- 预计编译CPU2–8小时，性能机时仅候选估计，按G3实测重新冻结；不把估算写成测量。

待现场完成：剩余镜像分发、完整权重哈希、B300驱动/通信、G0–G3、客户端GPU接口适配、矩阵调度与复测、原始证据和报告。当前没有性能数据、优势或精度结论。
