# Qwen3-Omni-30B 带语音输出:实时产能测定复现手册

配套文章:《实测 Qwen3-Omni-30B 带语音输出:A100 与昇腾 910B3/910B4 三方对比》

测的是**单实例能同时服务几路实时语音**,判据 `RTF_max < 1.0`。本手册覆盖三方硬件的完整流程:
起服务 → 正确性自检 → 并发扫描 → 边界复测 → 提示模板消融。

占位符说明:`<MODEL_DIR>` 本机模型目录,`<IMAGE>` 你能拉到的 vllm-omni 镜像,
`<HOST>` 服务地址(下文默认容器内 `127.0.0.1:8000`)。

---

## 1. 硬件与镜像

本次实测的三套环境:

| | A100 | 昇腾 910B3 | 昇腾 910B4-1 |
|---|---|---|---|
| 加速卡 | A100-SXM4-80GB × 3 | 910B3-64GB × 4 | 910B4-1-64GB × 4 |
| 驱动 | 590.48.01 / CUDA 13.1 | CANN 9.0.1 / driver 26.0.rc1 | 同左 |
| 引擎 | vllm-omni 0.26.0 | vllm-omni 0.26.0 | vllm-omni 0.26.0 |

镜像用官方 `vllm-omni:v0.26.0`(昇腾需 aarch64 + CANN 版)。低于 0.26.0 的版本没验过。

**最低卡数**:thinker 权重 59.5 GB。80 GB 单卡放得下(TP=1),再加 2 张跑音频段,共 3 张;
64 GB 卡放不下,必须 TP=2 占 2 张,再加 2 张音频段,共 4 张。

---

## 2. 测试工具

三个自建脚本 + 一个数据集准备脚本,都在 `scripts/`,只依赖 Python 3 标准库,无需 pip 安装。

| 脚本 | 作用 | 产出 |
|---|---|---|
| `prep_dataset.sh` | 拉取并抽出数据集文本 | `/tmp/seedtts_en_prompts.txt`(1088 行) |
| `verify_graph.py` | 正确性自检:音频是否真的生成了 | 波形指标 + PASS/FAIL |
| `seedtts_bench.py` | 并发扫描,主采集器 | 每档一行:req/s · TTFT · TTFP · RTF · 判定 |
| `tmpl_ablation.py` | 提示模板三臂消融 | 每臂 RTF 与朗读性相关系数 |

**为什么不用引擎自带的 bench**:我们的镜像里 `vllm-omni bench serve` 缺 `audio_ttfp` /
`audio_rtf` 两个指标。自建采集器与非流式路径交叉验证过,单路 RTF 0.700 对 0.694,差 0.9%。

**指标怎么算**:

- `RTF` = 整个请求的端到端墙钟(含 prefill 与首包等待)÷ 音频时长。
  音频时长按 24 kHz / 16 bit / 单声道,由累计音频字节数反推,与解析 WAV 头的结果吻合(5.82 s 对 5.86 s)。
- `TTFP` = 发出请求到收到第一个音频包的时间。流式响应里音频包的标志是 SSE chunk 的
  `modality == "audio"`,负载在 `delta["content"]` 里(base64),**不是** `delta["audio"]`
  —— 后者是非流式响应的字段,认错会导致 TTFP 永远采不到。

---

## 3. 测试数据集

**Seed-TTS eval 的 en 子集**,1088 条,中位 11 词,合成后音频中位约 3.9 秒。

```bash
./scripts/prep_dataset.sh                    # 直连 huggingface.co
HF_ENDPOINT=https://hf-mirror.com ./scripts/prep_dataset.sh   # 国内走镜像
```

产出 `/tmp/seedtts_en_prompts.txt`,一行一句。脚本会校验行数,不足 1000 行直接报错退出。

原始 `meta.lst` 每行是 `utterance_id | prompt_text | prompt_wav_path | target_text`,
我们只取第 4 列(要被朗读的句子),不需要下载 wav。

**送给模型的提示模板**是 `Read the following sentence aloud: {text}`。这一层必须保留:
去掉指令后模型不朗读而是长篇答话(实测音频从 3.5 秒涨到 120 秒),详见第 7 节。

---

## 4. 起服务

### 昇腾(4 卡)

```bash
# 物理卡 4,5,6,7 映射为进程内的逻辑卡 0,1,2,3
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2

# devices 里的 2,3 是逻辑卡编号,对应物理卡 6,7
STAGE_OVERRIDES='{"1":{"num_replicas":2,"devices":"2,3"},"2":{"num_replicas":2,"devices":"2,3"}}'

vllm serve <MODEL_DIR>/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --served-model-name=Qwen3-Omni-30B-A3B-Instruct \
  --host=0.0.0.0 --port=8000 \
  --max-model-len=32768 \
  --init-timeout=2400 --stage-init-timeout=900 \
  --stage-overrides "$STAGE_OVERRIDES" \
  --trust-remote-code
```

起之前先让动态链接器找到驱动,否则报 `ImportError: libascend_hal.so`
(该库在 `lib64/driver/` 子目录,容易漏):

```bash
echo /usr/local/Ascend/driver/lib64      > /etc/ld.so.conf.d/ascend.conf
echo /usr/local/Ascend/driver/lib64/driver >> /etc/ld.so.conf.d/ascend.conf
ldconfig
```

### A100(3 卡)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
STAGE_OVERRIDES='{"0":{"gpu_memory_utilization":0.15},"1":{"num_replicas":2,"devices":"1,2"},"2":{"num_replicas":2,"devices":"1,2"}}'
# 其余参数同上。A100 不需要 jemalloc,也不需要 ld.so 配置。
```

`stage 0` 必须压 `gpu_memory_utilization`:vllm-omni 的 KV 池按「利用率 × 整卡容量」算且
**不扣除权重**。thinker 权重 59.5 GB,0.15 对应约 12 GB KV,合计 71.5 GB 留余量;不压会 OOM。
昇腾 TP=2 摊开后每卡权重 29.74 GB,自动算出的 KV 放得下,反而不用干预。

### 自检

```bash
curl -s http://127.0.0.1:8000/health && echo "  health OK"

# 正确性:确认音频真的生成了,不是静音或截断
python3 scripts/verify_graph.py Qwen3-Omni-30B-A3B-Instruct
```

`verify_graph.py` 判据:峰值 > 3000、RMS > 200、近静音样本占比 < 90%、时长在 3 到 15 秒之间。
它只能排除明显失效,**不是**与 eager 输出的逐样本比对。

**预热是硬性前提。** 昇腾首次请求含图捕获开销,RTF 3.804,预热后 0.694,差 5.5 倍。
采集脚本内置 5 轮预热并按目标并发打满,但跨并发档连续扫描时仍可能打到冷副本
(见第 7 节),稳妥做法是每换一档先手动打满一轮。

---

## 5. 测试命令

```bash
M=Qwen3-Omni-30B-A3B-Instruct

# 主扫描:每档产出一行,含 req/s、TTFT、TTFP、RTF 中位与最大值、实时判定
for c in 1 4 8 16 32 64; do python3 scripts/seedtts_bench.py $M $c; done

# 边界复测:先用主扫描定位「最后一个通过的档」与「第一个破线的档」,
# 再对这两档各跑 3 到 4 次。请求数默认 = 并发 × 10,可用第 3 个参数覆盖。
for r in 1 2 3; do for c in 2 3; do python3 scripts/seedtts_bench.py $M $c; done; done

# 提示模板消融(c=1,每臂 20 条同一批文本)
python3 scripts/tmpl_ablation.py $M 20
```

`tmpl_ablation.py` 三臂:当前模板 / 裸文本 / 最短指令 `朗读:{text}`。它额外输出
**输入文本长度与音频时长的相关系数**,用来检出「模型在答话而不是朗读」——
相关系数低于 0.5 的臂数据无效,不能参与比较。

---

## 6. 判据与复算

**判据**:该并发下 `RTF_max < 1.0` 即通过,即最慢的那一路也能实时。用尾部而非中位,
因为中位达标而尾部超标意味着一部分用户已经听到卡顿。

**这个判据自身有偏向**,引用前必须知道:它是极大值统计量,而样本数随并发线性增长
(并发 1 到 64 对应 10 到 640 条),同一分布下样本越多极大值越高,**它天生对高并发不利**。
所以按它得出的上限偏保守。本次采集脚本未输出 `RTF_p95`,无法换用固定分位数复核,
只能靠重复测量暴露不稳定。

**上限怎么定**:取**稳定通过**的最高档,而不是曾经通过过的最高档。实测中 910B3 的并发 3
与 A100 的并发 40 都出现「同一配置四次跑,两次过两次不过」,这类骑线档不计入上限。

**复算方法**:每行输出里 `RTF_p50` 与 `RTF_max` 是从单条请求的 `E2EL ÷ audio_dur` 算出的。
拿 `E2EL_p50` 除以 `audio_p50` 应当约等于 `RTF_p50`(不完全相等,因为两者各自取中位)。
`audio_p50` 在不同并发档之间会漂移(3.5 到 4.6 秒),原因是预热消耗的条数不同、
每档实际跑的是数据集的不同子集;该漂移对各方共模,不影响横比,但意味着
**RTF 的第三位小数不具备分辨力**。

**对称协议**(多方对比时必须遵守):各方同脚本、同输入、同预热;边界点各方复测相同次数;
所有数值照报含异常值;剔除必须写明规则,且规则定在测量之前。

---

## 7. 已知坑

**报错指向模型不被支持,实际是退出阶段崩溃。** 现象:

```
Model architectures ['Qwen3OmniMoeForConditionalGeneration'] failed to be inspected.
```

根因是 `import vllm_omni` 在解释器退出阶段 double-free,glibc 报
`corrupted size vs. prev_size` 后 SIGABRT;导入本身成功,只在 teardown 崩,
而 vLLM 的模型探测跑在子进程里并检查退出码。判别:

```bash
python3 -c "import vllm_omni, os; os._exit(0)"    # 返回 0 即属此类
```

处置:`export LD_PRELOAD=.../libjemalloc.so.2`,换掉触发该 bug 的内存清理路径。
torch / torch_npu / transformers / vllm 单独导入都不触发,只有 vllm_omni 会。

**关掉图模式直接失去实时。** `--enforce-eager` 会让单路 RTF 从 0.694 劣化到 1.883。
不要为了绕开「talker 图问题」而关图 —— 该问题在 0.26.0 上未复现。

**`--init-timeout` 什么时候必须放宽。** 四卡布局实测就绪 570 秒,只比默认 600 秒少 30 秒;
六副本以上(八卡布局)会稳定超时并进入 CrashLoop。四卡不放宽也能起,但余量过薄。

**跨并发档扫描会打到冷副本。** 两台昇腾都在**首次并发 2** 命中:RTF_max 3.825 / 3.593,
TTFP_p95 破万毫秒,数值逼近各自的冷启动 RTF。原因是紧邻的并发 1 只用到一个音频副本,
随后首个并发 2 的请求打到从未预热的第二副本。**每档预热应按全局最大副本数打满,
而不是按本档并发数。**

**裸文本输入不做 TTS。** 去掉指令后模型长篇答话:音频 3.5 秒 → 120 秒,输出文本
56 字符 → 1600 以上,相关系数 0.71 → 0.10。若只看 RTF 会误读成「去掉模板 RTF 降低 20%」,
实际是那段独白把首包固定开销摊薄了。做模板类消融时必须带朗读性检查。

**W8A8 量化目前不可用。** msmodelslim 流程能跑通,但产物只含 thinker,
talker 的 8037 个张量与 code2wav 的 230 个张量全部丢弃,语音链路直接消失;
引擎侧 `QUANT_MODEL_SUBSTR_MAPPINGS` 也缺 `qwen3_omni_moe` 条目。这是工具链缺口,
不是量化原理上不适用。
