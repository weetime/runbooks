# JEV-27B-VL：910B3 决策复现材料

这是结构化 System 1 决策适配路径，不是普通聊天生成服务。模型发布者为 AutoTrust，不能假定与 TypeSafe 托管 Jev API 是同权重。

## 权重与环境

实测 Python3.12.13，vLLM0.25.1+empty、vLLM-Ascend0.19.1rc2.dev1265+g50e0ce608、torch_npu2.10.0.post2、Transformers5.14.1；两张 Ascend910B3，BF16、TP2、eager。需要匹配的 CANN/ATB。不能视为任意镜像均可启动。

完整下载 autotrust/JEV-27B-VL 的 revision `ba3f0d584994b37998f235c0a3f6f1beff32ba1e`。`adapter_vllm` 已由模型仓库提供，含 adapter_model.safetensors、adapter_config.json、decision_head.json，不需要自行训练或重建。对照 model-download-manifest.json 校验全部文件，权重约56GB，不随本仓库分发。

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="autotrust/JEV-27B-VL",
    revision="ba3f0d584994b37998f235c0a3f6f1beff32ba1e",
    local_dir="/models/JEV-27B-VL")
```

## 启动与调用

```bash
export JEV_MODEL_ROOT=/models/JEV-27B-VL
bash start_server.sh
```

另一个终端等 `/v1/decide/info` 就绪后调用 `/v1/decide`。启动脚本仅将原实验固定目录改为环境变量和脚本目录；适配器及测试代码保持实跑版。服务器启动入口、lm_head发现、处理后logprobs以及候选mask交换的隔离兼容修改见 COMPATIBILITY.md。不改安装包或模型文件。

## 能力测试

必须在 Linux NPU 容器运行，端口8127空闲。

```bash
# gate_and_text 会自行启动/停止服务，不能与手动服务并行。
JEV_ATTEMPT=attempt-07 python3 gate_and_text.py
# 视觉测试需要服务另行保持运行；正式计分前先通过完整正确性门。
JEV_VISION_OUT=vision-results-v3 python3 run_vision.py
```

根目录fixture与vision-images是运行时输入；data/保留归档布局。CharXiv和Multimodal-Mind2Web图像来自公开数据集固定子集，原始数据许可与来源见 data/vision-manifest.json、data/mind2web-manifest.json。原题标签不发往模型。文本2,895判断、视觉128题各有图/无图一次。818条Score适配为Choice；网页64步来自14任务且正确候选事先纳入，不是正式网页任务成功率。

## 性能基准

```bash
# 每次会独立启动/停止服务；每个输出目录必须尚不存在。
JEV_ATTEMPT=perf-1 python3 gate_and_perf.py
JEV_ATTEMPT=perf-2 python3 gate_and_perf.py
JEV_ATTEMPT=perf-3 python3 gate_and_perf.py
```

先检查 gate_and_perf.py 的 JEV_ATTEMPT 环境变量定义、已安装包源码路径及 runtime 指纹路径与容器一致。若安装布局不同，应在运行前记录修改，并重新冻结指纹。一次一个Choice，64文本/16图片固定池，预热后并发1/4/8/16/32/64；C8每轮768次，其余128次。prefix缓存关闭、图像缓存已暖、固定升序闭环排空。仅C8报告P99。TTFT均值从metrics增量计算，不能当作HTTP分位数；一个输出token不报告TPOT。

analyze_capability07.py 和 analyze_performance.py 为原实验归档布局分析器，读取 runtime/attempt-07、runtime/vision-results-v3、runtime/perf-1/2/3 等原始证据。分析器要求原始服务/Pod指纹和完整证据，不能只生成 summary 后宣称完成审计。本仓库不分发内部Pod/节点日志，公开脚本是复现入口，原始数据及内部资源审计另行留存。

性能结果只适用于本配置、固定暖缓存输入和闭环协议，不代表生产容量或最优两卡部署。
