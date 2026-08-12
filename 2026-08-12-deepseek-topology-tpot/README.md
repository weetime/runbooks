# DeepSeek-V4-Flash × 16 张昇腾 910B4:三拓扑 × 投机开关的复现手册

配套文章:《按 SLO 档位选部署拓扑:DeepSeek-V4-Flash 在 16 张 910B4 上的六格实测》

文章给的是结论与判据,这份手册给的是**从零把六格跑出来所需的全部命令**。文章里被压缩掉的启动细节、代理源码、采样脚本都在这里。

---

## 0. 硬件与镜像

两台裸金属,各 8 张昇腾 910B4(64 GB),参数面 RoCE 互通。

```
quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731
digest sha256:1a7dc241ffafad36017d74d8076a95601590bd9972304a49aa2dc6d2c316a401
```

镜像内:vLLM 0.25.1 · vllm-ascend commit `50e0ce608` · CANN 9.0.1 · torch_npu 2.10.0.post2 · mooncake-transfer-engine-npu 0.3.11.post1

**tag 可变,复现请认 digest。** DSpark 投机解码在 vLLM v0.25.0 之前需要这个专用构建才能启用,不能用发布版替代。

国内拉 quay.io 很慢(实测 8 MB/s),建议先 skopeo 搬到内网 registry:

```bash
skopeo copy --override-arch arm64 \
  docker://quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731 \
  docker://<your-registry>/vllm-ascend:DeepSeekV4-flash-0731
```

### 起容器

`manifests/ds-n0.yaml` 与 `ds-n1.yaml` 是 k8s 版本。两处关键点:

- `labels: {hami.io/webhook: "ignore"}` + `schedulerName: default-scheduler` —— 绕开 HAMi 设备插件。若集群装了 HAMi 但节点没有 ascend-docker-runtime,不绕开会一直 Pending。
- NPU 设备用 **CharDevice hostPath** 手动挂:`/dev/davinci0..7`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`,另需挂驱动目录 `/usr/local/Ascend/driver`。

```bash
kubectl apply -f manifests/ds-n0.yaml -f manifests/ds-n1.yaml
```

裸 docker 起也可以,把上面的设备与驱动目录 `--device` / `-v` 挂进去,`--privileged`、`--net=host`。

### 自检

```bash
npu-smi info                     # 8 张卡 Health OK,HBM 基线约 3.4 GB / 65536
hccn_tool -i 0 -link -g          # 参数面链路 UP
for i in $(seq 0 7); do hccn_tool -i $i -ip -g; done   # 参数面 IP 已下发
```

跨机连通性要**逐卡 ping 参数面 IP**,不能只 ping 业务面:

```bash
for ip in <对端 8 个参数面 IP>; do hccn_tool -i 0 -ping -g address $ip; done
```

> 坑:`grep -q "0.00% packet loss"` 会匹配上 `100.00% packet loss`。判通断要用锚定匹配。

---

## 1. 权重与数据集

```bash
# 权重(约 400 GB,建议放共享存储)
modelscope download --model Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  --local_dir /models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8

# 数据集
wget -O /models/_datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

本轮实测的负载画像:**平均输入约 223 token、输出约 196 token**。TTFT 与 E2E 的绝对值高度依赖这个分布,换数据集必须重测。

---

## 2. 三种拓扑怎么起

三种拓扑共用下面这段环境变量。**跨机场景少了 `HCCL_IF_IP` 或 `HCCL_SOCKET_IFNAME` 基本必挂**,`bond1` 换成你的参数面网卡名。

```bash
export HCCL_IF_IP=<本机参数面 IP>
export GLOO_SOCKET_IFNAME=bond1
export TP_SOCKET_IFNAME=bond1
export HCCL_SOCKET_IFNAME=bond1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=204
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
```

共用的引擎参数(下称 `$BASE`):

```bash
BASE="--served-model-name dsv4f \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --quantization ascend \
  --max-model-len 65536 --max-num-seqs 1024 --max-num-batched-tokens 10240 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code --seed 1024 \
  --no-disable-hybrid-kv-cache-manager --block-size 32 \
  --tokenizer-mode deepseek_v4 \
  --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 128}' \
  --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'"

MODEL=/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8
```

投机解码按需追加(**注意它同时带了 `enforce_eager`,会让草稿模型不走图模式**,这是一个混淆项):

```bash
SPEC="--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"enforce_eager\":true}'"
```

### 2.1 单机×2(两个独立实例 + 轮询代理)

两台各起一个,互不通信:

```bash
# node-A
vllm serve $MODEL $BASE --host 0.0.0.0 --port 8077
# node-B
vllm serve $MODEL $BASE --host 0.0.0.0 --port 8078
```

前面挂一个轮询代理,压测打代理端口:

```bash
python3 scripts/rr_proxy.py --port 8081 --backends <A>:8077 <B>:8078
```

> **这个代理在并发 ≥768 时会把 SSE 事件合批**,表现为 median ITL 塌到 0.01 ms、TTFT 虚高、并出现请求失败。自检方法:低并发下 `median ITL` 应约等于 `median TPOT`,若差出数倍就是代理在合批。生产环境请换支持流式透传的反向代理。

### 2.2 PD 分离(1P1D)

P 侧在 node-A、D 侧在 node-B。**先起 D 再起 P**,否则 P 连不上 KV 通道。

```bash
# D 侧(node-B,kv_consumer)
KVP='{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"30400","engine_id":"1","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'
vllm serve $MODEL $BASE --host 0.0.0.0 --port 8078 --data-parallel-size 1 \
  --kv-transfer-config "$KVP"

# 等 10 秒,再起 P 侧(node-A,kv_producer)
KVC='{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'
vllm serve $MODEL $BASE --host 0.0.0.0 --port 8077 --data-parallel-size 1 \
  --kv-transfer-config "$KVC"
```

代理用 vllm-ascend 自带的,压测打 8080:

```bash
python3 /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 8080 \
  --prefiller-hosts <A> --prefiller-ports 8077 \
  --decoder-hosts <B> --decoder-ports 8078
```

**验证 PD 通道真的生效**(缺了这条,代理绕过 P 直打 D 时前几条判据照样全过):

```bash
# 杀掉 P 侧,经 proxy 的请求必须失败
pkill -f "port 8077"
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/v1/completions -d '...'   # 期望 500
# 而直连 D 侧仍能正常输出
curl -s http://<B>:8078/v1/completions -d '...'                                          # 期望 200
```

另外在 D 侧引擎日志里应能看到 `KV cache transfer for request` 的记录。

### 2.3 跨机 EP(单实例 DP2×TP8)

一个实例横跨两机,EP 域 16 rank。**先起 headless 的那台**:

```bash
# node-B(headless,无 API server)
vllm serve $MODEL $BASE \
  --data-parallel-size 2 --data-parallel-size-local 1 \
  --data-parallel-address <A> --data-parallel-rpc-port 13389 \
  --headless --data-parallel-start-rank 1

# 等 8 秒,再起 node-A(带 API server)
vllm serve $MODEL $BASE \
  --data-parallel-size 2 --data-parallel-size-local 1 \
  --data-parallel-address <A> --data-parallel-rpc-port 13389 \
  --host 0.0.0.0 --port 8077
```

压测直连 8077,不需要代理。

---

## 3. 压测

```bash
CONC=128
NP=$(( CONC * 8 ))          # 每档发 8 倍并发条请求,保证测量窗口足够

vllm bench serve --backend openai --model dsv4f \
  --tokenizer $MODEL --tokenizer-mode deepseek_v4 --trust-remote-code \
  --base-url http://127.0.0.1:8080 \
  --dataset-name sharegpt \
  --dataset-path /models/_datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts $NP --max-concurrency $CONC \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  --save-result --result-filename result_c${CONC}.json
```

并发阶梯:`1 4 8 16 32 64 96 128 192 256 384 512 768 1024 1536`。

口径说明:闭环压测(固定并发持续灌),不设 `--request-rate`,不固定输出长度。结果里的 `output_throughput` 只算输出 token,`total_token_throughput` 含输入、约为前者的 2.1 倍——**跨文章对比时务必对齐这一项**。

`scripts/sweep_gated.sh` 是完整的自动化扫描脚本,含判据、teardown 与断言。

---

## 4. KV 占用与抢占怎么采

**必须在压测过程中后台轮询**,结束后再读会读到已回收的状态(占用率接近 0)。

```bash
# 压测前台起,采样后台起
./scripts/sample_kv.sh &          # 写 /tmp/kvsample.csv
vllm bench serve ...
pkill -f sample_kv
```

`sample_kv.sh` 的核心:

```bash
curl -sS http://$h/metrics | awk '
  /^vllm:kv_cache_usage_perc/{k+=$2}
  /^vllm:num_requests_waiting[ {]/{w+=$2}
  /^vllm:num_requests_running/{r+=$2}
  END{printf "%s,%.4f,%.1f,%.1f\n", systime(), k, w, r}'
```

三个坑:

1. **指标名**在 vLLM V1 是 `kv_cache_usage_perc`,V0 时期叫 `gpu_cache_usage_perc`。用错会全程读到 0。
2. 尽管名字带 `perc`,**它是 0 到 1 的比值不是百分数**。
3. **DP > 1 时同一端点会暴露多条带 `engine` 标签的序列**,直接求和会得到超过 100% 的假值。跨机 EP 臂需要除以 DP 度数,或改用 `avg` 而非 `sum`。

抢占计数单独取:

```bash
curl -sS http://$h/metrics | awk '/^vllm:num_preemptions_total/{s+=$2} END{print s}'
```

投机的接受率取增量:

```bash
curl -sS http://$h/metrics | grep -E "spec_decode_num_(accepted|draft)_tokens_total"
# 接受率 = Δaccepted / Δdraft,压测前后各取一次
```

---

## 5. 判据与复算

本轮不预设单一 SLO,用四档分别去卡(口径 TTFT p99 / TPOT p99):

| 档 | TTFT | TPOT | 锚点 |
|---|---|---|---|
| 交互 | 1.5 s | 15 ms | MLPerf DeepSeek-R1 v6.0 |
| 常规 Server | 2 s | 80 ms | MLPerf DeepSeek-R1 v5.1 |
| 长上下文 | 6 s | 175 ms | MLPerf Llama3.1-405B v5.0 |
| 不可用地板 | 60 s | 500 ms | 自定,非业内标准 |

复算方式:**取该配置所有通过阈值的档位,报其中吞吐最高的一档**。注意这是事后复算,不合格的中间档位会被跳过而不是就此中止阶梯——若改用「首次越线即中止」,个别配置的结果会不同(实测中单机×2 开投机那一格相差 2.2 倍)。

```python
def cap(rows, ttft_s, tpot_ms):
    ok = [r for r in rows
          if r["p99_tpot_ms"] <= tpot_ms and r["p99_ttft_ms"] <= ttft_s * 1000
          and r["median_itl_ms"] >= 1                      # 排除代理合批导致的失真
          and r["completed"] >= r["num_prompts"] * 0.99]   # 排除有失败的档位
    return max(ok, key=lambda r: r["output_throughput"]) if ok else None
```

---

## 6. 噪声底

任何差异判定之前先做这个,否则无法区分「真差异」与「抖动」。

```bash
./scripts/noise_floor.sh <twox|pd>
```

纪律:

- **必须跨服务重启复测**。同一引擎实例内反复跑只测到抖动,会严重低估噪声。
- **n ≥ 3**。实测 n=2 会低估 14 倍。
- **分档标定**。实测同一指标跨档最多差 21.8 倍,不能全表共用一个噪声底。
- **分指标标定**。并发 64 上 TTFT p99 的噪声是 TTFT p50 的 145 倍,分布类指标基本不可精读。

> 本轮的噪声底脚本里硬编码了 `--speculative-config`,因此只对**投机开启**的配置成立。要给投机关闭的配置下差异结论,需要把这行去掉重标一遍。

---

## 7. 投机解码的受控消融

```bash
./scripts/spec_ablation.sh
```

它在**同一次服务生命周期内**只改 `--speculative-config`,其余逐字固定,分别测并发 64 与 96。判据是两臂的 `Total input tokens` 必须完全相同(证明吃的是同一份数据切片),且引擎自报的 `speculative_config` 从 `SpeculativeConfig(method='dspark')` 变成 `None`。

实测结果(并发 96):开投机 374.44 tok/s / 415.53 s,关投机 1032.78 tok/s / 151.90 s,**2.76 倍**;接受率 20.6%(accepted 92,575 / draft 448,847)。

**已知混淆项**:开投机时同时带 `enforce_eager`,草稿模型不走图模式。要分离「投机算法」与「图模式」两个变量,需再加一个去掉 `enforce_eager` 的臂。

---

## 8. 踩过的坑

| 现象 | 根因 | 处理 |
|---|---|---|
| 曲线在并发 32 之后全部走平 | `--max-num-seqs` 设成 32,测到的是自己设的准入上限 | 能力边界扫描必须放到 1024 以上 |
| KV 占用全程为 0 | 用了 V0 的指标名 `gpu_cache_usage_perc` | V1 改用 `kv_cache_usage_perc` |
| KV 占用超过 100% | DP>1 时多条 `engine` 序列被求和 | 除以 DP 度数或改用 avg |
| 压测后读到的 KV 接近 0 | 结束后才采样,请求已完成、KV 已回收 | 压测过程中后台轮询 |
| median ITL = 0.01 ms | 自研代理在高并发合批转发 SSE | 换流式透传代理;低并发自检 ITL ≈ TPOT |
| 引擎没按预期加载新配置 | 改了启动参数但没验证引擎是否采用 | 起服务后 grep 引擎日志的自报配置行做断言 |
| `pkill -f <pattern>` 返回 255 | 模式匹配到了 ssh 命令自身 | 用 `[V]LLM` 这类写法,或按 PID kill |
| 跨机实例起不来、HCCL 超时 | 少了 `HCCL_IF_IP` / `HCCL_SOCKET_IFNAME` | 见第 2 节的环境变量块 |
| Pod 一直 Pending | HAMi webhook 与手动挂载冲突 | `hami.io/webhook: "ignore"` + 默认调度器 |

---

## 目录

```
scripts/
  rr_proxy.py          单机×2 用的轮询代理(注意高并发合批问题)
  sample_kv.sh         KV / 排队 / 运行数 的压测过程采样
  sweep_gated.sh       带判据的自动化并发扫描
  noise_floor.sh       噪声底标定(跨服务重启 × 3)
  spec_ablation.sh     投机解码单变量受控消融
manifests/
  ds-n0.yaml ds-n1.yaml   k8s Pod(绕开 HAMi + CharDevice 挂 NPU)
```
