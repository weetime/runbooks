# DeepInfer / vLLM × B300：离线构建与测试流程

本目录仅交付后续现场执行的脚本与方案；不实际构建DeepInfer镜像，没有B300性能结论。
实验 issue：[#1](https://github.com/weetime/runbooks/issues/1)。

统一入口为[scripts/b300.sh](scripts/b300.sh)，详细步骤见[RUNBOOK.md](RUNBOOK.md)。

## 硬件与镜像

Linux x86_64、B300/SM103，现场核对驱动对CUDA13.2的兼容性、互联和授权GPU范围。镜像digest/实际状态见[清单](manifests/images.json)。
AIPerf、CUDA基础镜像、32MB文本数据镜像已核验；vLLM同步未完成，build-kit未构建，DeepInfer推理镜像留到B300现场构建。

## 测试工具

提供Dockerfile、联网准备脚本、B300断网构建入口、模型校验、probe/G3/native batch和AIPerf候选工具。
[scripts/b300.sh](scripts/b300.sh)默认只打印plan；[配置模板](configs/b300.env.example)不含凭据。需要Docker或已配置的containerd/NVIDIA容器支持。

## 测试数据集

7组各128正式请求及7组各3独立预热，原生token-ID和对应HTTP准备记录均冻结。源文件、实际长度和SHA见[请求清单](manifests/requests.json)。
首发文本包不包含完整Agent/CC或模型权重；完整模型要另做逐文件哈希，不能用W8A8/0909代替原厂1017检查点。

## 起服务

DeepInfer当前公开部分是原生离线引擎，缺完整HTTP服务。本方案先在联网准备机生成依赖kit，再由B300的断网容器构建推理镜像。
原生引擎与在线HTTP服务分开记录；按[操作手册](RUNBOOK.md)执行build、模型校验、smoke和门禁，不把编译成功当模型验收。

## 测试命令

```bash
cp configs/b300.env.example configs/b300.env
# 按现场填写镜像digest、GPU_IDS及数据/模型路径，不放密码。
export CONFIG="$PWD/configs/b300.env"
bash scripts/b300.sh plan
# 后续才按RUNBOOK逐阶段执行prepare/pull/data/check/build/smoke/gates/probe/g3/bench。
```

完整矩阵见[TEST_MATRIX.md](TEST_MATRIX.md)，冻结方案与比较边界见[PLAN.md](PLAN.md)。本轮只验证静态/CPU和命令预演，未执行B300构建或发压。

## 判据与复算

执行[BENCHMARK_POLICY.md](BENCHMARK_POLICY.md)。固定形状/自然EOS、native batch/HTTP、host chunk/GPU纯时间分别报告。
不循环补样、不自动放行门禁；关键点三次独立起服交替复测，保留原始输出、日志、token计数与指标，可复算后再下结论。

## 已知坑

- B300为CC10.3，不能因A100编译完成而认定B300可用；正式构建和验收留在B300。
- DeepInfer官方builtin吞吐使用全接受率记账，不能混入实际生成token主表。
- vLLM候选版本有需复核的已报告正确性风险；不擅自关优化或猜测新版本镜像已发布。
- 本目录不包含原始大型数据/权重，SWR镜像可能需要操作者自己的登录；密码不得写入项目、日志或配置模板。
- GPU运行接口尚未实机验收。单点工具不是自动完成全部标准矩阵的campaign，不声称已得到性能或精度结果。
