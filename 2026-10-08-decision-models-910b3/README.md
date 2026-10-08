# Jev、decider 与 Laya：多任务决策实测

复现新增四类工作流、MASSIVE中文子集、XNLI中文子集。此目录仅公开代码与公开题库，不包含集群地址、账号凭据、私有镜像或本次原始模型响应。完整历史响应保存在作者文章素材中。

## 1. 硬件与镜像

本次为昇腾910B3，单卡64GB；驱动26.0.rc1。4B和Laya单卡，27/31/35B双卡。需要自行准备可访问NPU且已安装CANN、Python3.12、torch 2.10.0、torch_npu 2.10.0.post2、Transformers5.14.1的容器。镜像使用 `<YOUR_ASCEND_IMAGE>` 占位，不提供私有镜像地址。此目录不修改或创建 Kubernetes 工作负载。

## 2. 测试工具

`run.py`直接调用SDK或HTTPS，`analyze.py`离线审计/计分。`loader.py`为NPU适配，`family_models.py`与`npu_moe.py`为双卡/视觉扩展。此公开副本仅将loader的扩展搜索目录调整为同目录；实验冻结版与运行哈希保存在原始文章素材，不能用此副本冒充原始运行文件。

在已有框架依赖的容器内：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m venv --system-site-packages venv
venv/bin/python -m pip install --no-deps decider-ai==1.9.0 laya==0.4.0
```

`--no-deps`只适用于依赖齐备的原环境。下载代码需要已有huggingface_hub，Jev需要httpx。此目录脚本使用PyTorch等第三方库，不是纯标准库脚本。

## 3. 测试数据集

`scripts/data/requests.jsonl`为1000条无答案请求；`cases.jsonl`含离线参考，禁止把gold字段发送给模型。原始公开源文件与SHA在manifest.json，题库SHA256为`b7afa7a2f921ab43ada309c59ca41236bcb56fbd330f926cdf98b3f1358a03dd`。

- [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/e039ebffcc280174dd354227424fb2b249f191de)：all/test完整400案例，每例5问，合成软教师参考。数据卡及许可随源文件保留。
- [MASSIVE转换版](https://huggingface.co/datasets/mteb/amazon_massive_intent/tree/940fd47a81eaa7f2cc7b129674d945d618ac38c2)：zh-CN test前300条，参考意图加19干扰项，seed13，不是完整60分类。本次固定test有59类，前300出现30类。原数据许可CC BY 4.0，来源为Amazon MASSIVE。
- [XNLI](https://huggingface.co/datasets/facebook/xnli/tree/b8dd5d7af51114dbda02c0e3f6133f332186418e)：zh/test前300条，三类各100。数据来源/许可以所附数据卡为准。
- 请求构造沿用固定[Laya评测脚本](https://github.com/NandhaKishorM/laya/blob/3cf26cbcb18725dbc2d127bb8bb2c4c43243ae63/research/scripts/laya_benchmark_colab.ipynb)。公开源文件保持原版权与许可，不将其重新许可为自研数据。

## 4. 起服务

质量测试使用进程内SDK，不启动HTTP服务器。默认权重根目录为`/workspace/decision-models-20261008/models`，在loader.py中修改ROOT可改路径。

在`/workspace/decision-models-20261008`内运行固定下载：

```python
from huggingface_hub import snapshot_download
snapshot_download("Mapika/decider-4b", revision="eb5fbdfc9448473ec25e399882912863afbdb70e", local_dir="models/Mapika/decider-4b")
snapshot_download("convaiinnovations/laya", revision="7b928d828b7b0e022f929d9bd2e44165aa270148", local_dir="models/convaiinnovations/laya")
```

将本目录`scripts/`内容复制至容器`/workspace/decision-models-20261008/multitask-tests/`，与venv相邻。保留所有5个脚本及完整data目录。原有目录请先备份；正式runner拒绝覆写已经存在的结果目录。单卡4B对应model参数`decider`，Laya为`laya-en`/`laya-ml`/`laya-typed`。加载打印LOADED与npu设备，若静默落CPU则失败；smoke执行三套任务各一次预热并验证输出。

## 5. 测试命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/decision-models-20261008/multitask-tests
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0
../venv/bin/python run.py --model decider --device npu:0 --smoke --output results/decider-smoke
../venv/bin/python run.py --model decider --device npu:0 --output results/decider
```

Laya改model名称和output目录。Jev在客户端安装httpx并通过环境提供`TYPESAFE_API_KEY`后：

```bash
python run.py --model jev-1.13.0 --output results/jev-1.13.0
```

本地或API每模型完整运行1000请求/2600判断；smoke只预热不计分。本地设备不可共享给同时运行的另一个模型；双卡配置使用npu:0和npu:1，需要预先下载相应固定权重，这份最小下载示例仅准备4B/Laya。HTTP失败保留为error，不重试筛选。raw.jsonl与metadata.json保存在输出目录。

## 6. 判据与复算

完整12版本结果齐备时，在multitask-tests目录：

```bash
python analyze.py
```

如果只复跑了4B或Laya，使用`python analyze.py --partial`，输出明确标为不完整。`analyze.py`只检查manifest.models中固定模型目录，不发现任意新目录；按文章示例输出`*-recheck`需在独立副本里改成该模型的标准目录才能审计。始终保留原始素材。

计分验证请求/runner哈希与响应结构，归一化舍入概率，最大概率项对原gold.label。失败留在准确率分母，Brier/KL/Score MAE只用有效判断。工作流测的是教师一致率，Laya Typed接触train，不等于业务正确率。新增并行计时不作为速度排名。原文章64消息、视觉与CPU测试未纳入此最小复现目录。

自测记录：2026-10-09，在原B3环境验证4B及LayaTyped最小调用，各返回5个问题且实际设备npu:0；离线analyze对原始12版本记录逐项复算与冻结summary一致。公开打包版runner的4B和LayaTyped各三套任务预热在同一NPU环境通过，不作为新增评测成绩。不会重新运行完整12000请求来验文档。

## 7. 已知坑

NPU兼容需要关闭MHA fastpath和显式SDPA mask，不启用CUDA Graph/FLA。四B单卡加载不可直接推广为27/31/35B双卡配方。默认候选48token限制及英文模型状态截断仍可能影响结果。NVFP4微调31B/35B预检不支持、GGUF未测，均不计入成绩。API网络耗时不等于纯推理时延。尚未覆盖JevBench、Bespoke完整套件、Banking77、CLINC、Mind2Web、完整多语MASSIVE/XNLI及TypeSafe官方完整工作流执行。
