#!/bin/bash
# Phase F —— 三臂各自跑【自己的最优配置】,同批次对照。
#
# 为什么需要这一趟:Phase C(投机全开)与 Phase D/E(投机全关)证明投机的收益【取决于拓扑】——
#   单机×2  投机开更好(并发512 关/开 = 0.59×)
#   跨机EP  投机开更好(并发128 关/开 = 0.75×)
#   PD分离  投机关更好(并发96  关/开 = 2.66×)
# 此前我把 PD 上做的投机消融结论错误推广到三臂,导致 twox/ep 的 Phase D/E 跑在次优配置上。
# 各臂最优点分散在两个批次、跨了服务重启与多日,不能直接并列。本趟同批次重测。
#
# 与 Phase E 的唯一差异:新增 SPEC 开关(SPEC=on 注入 Phase C 逐字相同的 DSpark 配置)。
#
# 与 Phase D 的差异(三处,其余逐字不变):
#   1) 硬门禁:E2E p99>300s / TTFT p99>60s / KV>=0.95 / 完成率<99% 任一命中即中止该臂
#      理由:Phase D twox 实测在并发>=256 时 E2E p99 已 342s,再往上爬没有部署意义;
#           且 >=768 时代理会把 SSE 事件合批(median ITL=0.01ms),延迟指标本身失真。
#   2) PD 臂 P/D 两侧参数可差异化(P_MNS/P_MNBT/D_MNS/D_MNBT)
#   3) 输出前缀 C2_ -> D2_(phaseD 沿用 C2_ 导致与 Phase C 撞名)
#
# 为什么:Phase C 全程开着 DSpark n=7,而受控实验证实投机在并发>=64 是净亏
#         (conc96 关掉后吞吐 2.76 倍、ITL 快 8.6 倍)。所以 Phase C 的绝对值全被压低。
#         另外跨机 EP 扫到 768 仍在涨、未见顶,需要更高档位才能找到真峰值。
#
# 其余(mns=1024/maxlen/block-size/数据集/判据)逐字不变,保证可比。
#
# 原 Phase C v2 说明 —— 修了 v1 的三个问题:
#   1) 指标名写错:gpu_cache_usage_perc 不存在,实际是 kv_cache_usage_perc(v1 全程读到 0)
#   2) 采样时机错:v1 在 bench 结束后才读,那时请求已跑完 KV 已释放 -> 改为压测【过程中】后台轮询
#   3) num_prompts 封顶 512,高并发档测量窗口太短 -> 改为 concurrency*8 不封顶
#   另:mns 32->1024,并发档延伸到 768,让 KV 而非准入成为唯一约束
#
# 用法:71_phaseC_v2.sh <twox|pd|ep>
#   ep = DP2×TP8 跨机单实例(EP 域横跨两台物理机),补齐三臂同口径
set -uo pipefail
export KUBECONFIG=/etc/kubernetes/admin.conf
K="kubectl"
ARM="${1:?twox|pd|ep}"
TAG="${TAG:-}"
OUT=/data/share/models/_exp/phaseG/$ARM${TAG:+_$TAG}
LOG=$OUT/driver.log
MODEL=/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8
SG=/models/_datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
MAXLEN=65536
MNS=1024
# PD 角色参数(可用环境变量覆盖);默认= D-1 寻优前的对称配置,便于回归对照
P_MNS=${P_MNS:-$MNS}; P_MNBT=${P_MNBT:-10240}
D_MNS=${D_MNS:-$MNS}; D_MNBT=${D_MNBT:-10240}
# 硬门禁 —— 撞到即中止该臂整条阶梯,继续爬没有部署意义
GATE_E2E_MS=${GATE_E2E_MS:-300000}   # E2E p99 > 5 min
GATE_TTFT_MS=${GATE_TTFT_MS:-60000}  # TTFT p99 > 60s(最宽松业内锚点 6s 的 10 倍)
GATE_KV=${GATE_KV:-0.95}
# TPOT 门禁 —— 原门禁漏了每 token 延迟,导致 twox 在 ITL 已达 680ms(每秒 1.5 字)时仍继续爬档。
# 阈值 500ms:业内最宽松锚点是 MLPerf Llama3.1-405B v5.0 Server 的 TPOT 175ms(且那是更大的模型),
# 取其约 3 倍作为"任何业务都无法接受"的地板,而不是当作 SLO。
GATE_TPOT_MS=${GATE_TPOT_MS:-500}
# 投机开关:逐字沿用 Phase C 的配置,保证与历史数据可比
SPEC="${SPEC:-off}"
if [ "$SPEC" = "on" ]; then
  SPECARG="--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"enforce_eager\":true}'"
else
  SPECARG=""
fi
mkdir -p $OUT
say(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

for p in ds-n0 ds-n1; do
  $K get pod $p >/dev/null 2>&1 || { say "FATAL: pod $p 不存在"; exit 1; }
done

teardown(){
  for p in ds-n0 ds-n1; do
    $K exec $p -- bash -lc 'pkill -9 -f "[V]LLM"; pkill -9 -f "[v]llm serve"; pkill -9 -f "[l]oad_balance_proxy"; pkill -9 -f "[r]r_proxy"; pkill -9 -f "[s]ample_kv"; true' >/dev/null 2>&1
  done
  for h in <NODE_A_PARAM_IP> <NODE_B_PARAM_IP>; do
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no $h 'pkill -9 -f "[V]LLM"; true' >/dev/null 2>&1
  done
  for i in $(seq 1 60); do
    local busy=0 u
    for h in <NODE_A_PARAM_IP> <NODE_B_PARAM_IP>; do
      for u in $(ssh -o BatchMode=yes -o StrictHostKeyChecking=no $h 'npu-smi info 2>/dev/null | grep -oE "[0-9]+ */ *65536" | grep -oE "^[0-9]+"' 2>/dev/null); do
        [ "$u" -ge 8000 ] && busy=$((busy+1)); done; done
    [ "$busy" -eq 0 ] && { say "  teardown OK"; return 0; }
    sleep 5; done
  say "  teardown 超时"; return 1
}

write_engine(){ # pod ip port extra
  # PD 臂角色差异化:ds-n0=P(计算密集,大 mnbt 小 mns)、ds-n1=D(访存密集,大 mns 小 mnbt)
  # 其余臂沿用对称配置。P/D 用同一套参数等于测一个被阉割的 PD。
  local EMNS=$MNS EMNBT=10240
  if [ "$ARM" = "pd" ]; then
    if [ "$1" = "ds-n0" ]; then EMNS=$P_MNS; EMNBT=$P_MNBT; else EMNS=$D_MNS; EMNBT=$D_MNBT; fi
  fi
  cat > /tmp/pc2_$1.sh <<EOF
#!/bin/bash
export HCCL_IF_IP=$2
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
vllm serve $MODEL --served-model-name dsv4f \\
  --host 0.0.0.0 --port $3 \\
  --data-parallel-size 1 --tensor-parallel-size 8 --enable-expert-parallel \\
  --quantization ascend \\
  --max-model-len $MAXLEN --max-num-seqs $EMNS --max-num-batched-tokens $EMNBT \\
  --gpu-memory-utilization 0.90 \\
  --trust-remote-code --seed 1024 \\
  --no-disable-hybrid-kv-cache-manager --block-size 32 \\
  --tokenizer-mode deepseek_v4 \\
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \\
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \\
  $SPECARG \\
  $4 > /tmp/pc.log 2>&1
EOF
  $K cp /tmp/pc2_$1.sh $1:/tmp/pc.sh >/dev/null 2>&1
}

wait_ready(){
  local ok err
  for i in $(seq 1 70); do
    ok=$($K exec $1 -- bash -lc 'grep -c "Application startup complete" /tmp/pc.log 2>/dev/null | head -1 || echo 0' 2>/dev/null | tr -d '\r')
    err=$($K exec $1 -- bash -lc 'grep -cE "ValueError|AssertionError|Engine core initialization failed" /tmp/pc.log 2>/dev/null | head -1 || echo 0' 2>/dev/null | tr -d '\r')
    [ "${ok:-0}" -ge 1 ] 2>/dev/null && { say "  $1 ready (t=$((i*20))s)"; return 0; }
    [ "${err:-0}" -ge 1 ] 2>/dev/null && { say "  $1 ENGINE FAILED"; return 1; }
    sleep 20; done
  say "  $1 TIMEOUT"; return 1
}

say "######## Phase C v2 · 臂=$ARM · mns=$MNS ########"
teardown || exit 1
for p in ds-n0 ds-n1; do $K exec $p -- bash -lc 'rm -f /tmp/pc.log' >/dev/null 2>&1; done

if [ "$ARM" = "twox" ]; then
  write_engine ds-n0 <NODE_A_PARAM_IP> 8077 ""
  write_engine ds-n1 <NODE_B_PARAM_IP> 8078 ""
  for p in ds-n0 ds-n1; do $K exec $p -- bash -lc 'chmod +x /tmp/pc.sh; setsid nohup /tmp/pc.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1; done
  wait_ready ds-n0 || exit 1; wait_ready ds-n1 || exit 1
  cat > /tmp/rr.sh <<'REOF'
#!/bin/bash
exec python3 /tmp/rr_proxy.py --port 8081 --backends <NODE_A_PARAM_IP>:8077 <NODE_B_PARAM_IP>:8078 > /tmp/rr.log 2>&1
REOF
  $K cp /tmp/rr_proxy.py ds-n0:/tmp/rr_proxy.py >/dev/null 2>&1
  $K cp /tmp/rr.sh ds-n0:/tmp/rr.sh >/dev/null 2>&1
  $K exec ds-n0 -- bash -lc 'chmod +x /tmp/rr.sh; setsid nohup /tmp/rr.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
  ENDPOINT="http://127.0.0.1:8081"
elif [ "$ARM" = "ep" ]; then
  # DP2×TP8:一个实例横跨两机,EP 域 = 16 rank 跨机。write_engine 里的 DP1×TP8 需要覆盖,
  # 所以这里单独生成启动脚本(不复用 write_engine)。
  for spec in "ds-n1|<NODE_B_PARAM_IP>|--headless --data-parallel-start-rank 1" "ds-n0|<NODE_A_PARAM_IP>|--host 0.0.0.0 --port 8077"; do
    pod=${spec%%|*}; rest=${spec#*|}; ip=${rest%%|*}; extra=${rest#*|}
    cat > /tmp/pc2_$pod.sh <<EOF
#!/bin/bash
export HCCL_IF_IP=$ip
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
vllm serve $MODEL --served-model-name dsv4f \\
  --data-parallel-size 2 --data-parallel-size-local 1 \\
  --data-parallel-address <NODE_A_PARAM_IP> --data-parallel-rpc-port 13389 \\
  --tensor-parallel-size 8 --enable-expert-parallel \\
  --quantization ascend \\
  --max-model-len $MAXLEN --max-num-seqs $MNS --max-num-batched-tokens 10240 \\
  --gpu-memory-utilization 0.90 \\
  --trust-remote-code --seed 1024 \\
  --no-disable-hybrid-kv-cache-manager --block-size 32 \\
  --tokenizer-mode deepseek_v4 \\
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \\
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \\
  $SPECARG \\
  $extra > /tmp/pc.log 2>&1
EOF
    $K cp /tmp/pc2_$pod.sh $pod:/tmp/pc.sh >/dev/null 2>&1
  done
  $K exec ds-n1 -- bash -lc 'chmod +x /tmp/pc.sh; setsid nohup /tmp/pc.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
  sleep 8
  $K exec ds-n0 -- bash -lc 'chmod +x /tmp/pc.sh; setsid nohup /tmp/pc.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
  wait_ready ds-n0 || exit 1
  ENDPOINT="http://127.0.0.1:8077"
else
  KVP='{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"30400","engine_id":"1","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'
  KVC='{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'
  write_engine ds-n1 <NODE_B_PARAM_IP> 8078 "--kv-transfer-config '$KVP'"
  $K exec ds-n1 -- bash -lc 'chmod +x /tmp/pc.sh; setsid nohup /tmp/pc.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
  sleep 10
  write_engine ds-n0 <NODE_A_PARAM_IP> 8077 "--kv-transfer-config '$KVC'"
  $K exec ds-n0 -- bash -lc 'chmod +x /tmp/pc.sh; setsid nohup /tmp/pc.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
  wait_ready ds-n1 || exit 1; wait_ready ds-n0 || exit 1
  cat > /tmp/pxy.sh <<'PEOF'
#!/bin/bash
exec python3 /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 8080 --prefiller-hosts <NODE_A_PARAM_IP> --prefiller-ports 8077 \
  --decoder-hosts <NODE_B_PARAM_IP> --decoder-ports 8078 > /tmp/proxy.log 2>&1
PEOF
  $K cp /tmp/pxy.sh ds-n0:/tmp/pxy.sh >/dev/null 2>&1
  $K exec ds-n0 -- bash -lc 'chmod +x /tmp/pxy.sh; setsid nohup /tmp/pxy.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
  ENDPOINT="http://127.0.0.1:8080"
fi
sleep 15
say "  实际采用投机: $($K exec ds-n0 -- bash -lc "grep -oE \"speculative_config=[^,]{0,40}\" /tmp/pc.log | head -1" 2>/dev/null | tr -d '\r')"
say "  实际采用 mns: $($K exec ds-n0 -- bash -lc "grep -oE \"max_num_seqs['=: ]+[0-9]+\" /tmp/pc.log | head -1" 2>/dev/null | tr -d '\r')"

# 压测【过程中】轮询 KV 占用 / 排队 / 运行数 —— v1 在结束后才读,读到的是已释放的状态
# ep 臂:ds-n1 是 --headless 没有 API server,只能读 ds-n0(DP 指标已按 engine label 聚合)。
# 【必须定义在这里】—— 下面 sed 要用它;先前定义在使用之后,set -u 直接把驱动打死(2026-08-11 空跑 48 分钟)。
if [ "$ARM" = "ep" ]; then METRIC_HOSTS="<NODE_A_PARAM_IP>:8077"; else METRIC_HOSTS="<NODE_A_PARAM_IP>:8077 <NODE_B_PARAM_IP>:8078"; fi

cat > /tmp/sample_kv.sh <<'SEOF'
#!/bin/bash
# 指标名是 kv_cache_usage_perc,不是 gpu_cache_usage_perc(v1 写错导致全程 0)
: > /tmp/kvsample.csv
while true; do
  for h in __MHOSTS__; do
    curl -sS -m 3 http://$h/metrics 2>/dev/null | awk -v H="$h" '
      /^vllm:kv_cache_usage_perc/{k+=$2}
      /^vllm:num_requests_waiting[ {]/{w+=$2}
      /^vllm:num_requests_running/{r+=$2}
      END{printf "%s,%s,%.4f,%.1f,%.1f\n", systime(), H, k, w, r}'
  done
  sleep 2
done >> /tmp/kvsample.csv
SEOF
sed -i "s|__MHOSTS__|$METRIC_HOSTS|" /tmp/sample_kv.sh
$K cp /tmp/sample_kv.sh ds-n0:/tmp/sample_kv.sh >/dev/null 2>&1


metric_sum(){ # metric name -> 端点求和
  local m=$1 s=0 v
  for h in $METRIC_HOSTS; do
    v=$($K exec ds-n0 -- bash -lc "curl -sS -m 10 http://$h/metrics 2>/dev/null | awk '/^vllm:$m/{s+=\$2} END{printf \"%.4f\", s+0}'" 2>/dev/null | tr -d '\r')
    s=$(python3 -c "print($s+${v:-0})")
  done
  echo $s
}

PREV=0; FLAT=0
TIERS="${TIERS:-1 4 8 16 32 64 96 128 192 256 384 512 768 1024 1536}"
for CONC in $TIERS; do
  # 窗口足够但有界:conc*8,下限 48、上限 6144。
  # 上限是必要的 —— conc1536 若按 12288 个请求跑,单档就要 40 分钟。
  # 6144 在 conc1536 下仍有 4 个批次,足够稳定;裁掉的部分在此显式说明,不是静默截断。
  NP=$(( CONC * 8 )); [ $NP -lt 48 ] && NP=48; [ $NP -gt 6144 ] && NP=6144
  P0=$(metric_sum "num_preemptions_total")
  $K exec ds-n0 -- bash -lc 'chmod +x /tmp/sample_kv.sh; setsid nohup /tmp/sample_kv.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
  $K exec ds-n0 -- bash -lc "vllm bench serve --backend openai --model dsv4f \
    --tokenizer $MODEL --tokenizer-mode deepseek_v4 --trust-remote-code \
    --base-url $ENDPOINT --dataset-name sharegpt --dataset-path $SG \
    --num-prompts $NP --max-concurrency $CONC \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-filename /tmp/G_${ARM}_c${CONC}.json" >> "$LOG" 2>&1
  $K exec ds-n0 -- bash -lc 'pkill -f "[s]ample_kv"; true' >/dev/null 2>&1
  $K cp ds-n0:/tmp/G_${ARM}_c${CONC}.json $OUT/G_${ARM}_c${CONC}.json >/dev/null 2>&1
  $K cp ds-n0:/tmp/kvsample.csv $OUT/kv_c${CONC}.csv >/dev/null 2>&1
  P1=$(metric_sum "num_preemptions_total")
  DP=$(python3 -c "print(round($P1-$P0,1))")
  read TP KVMAX WMAX RMAX <<< $(python3 - "$OUT/G_${ARM}_c${CONC}.json" "$OUT/kv_c${CONC}.csv" <<'PY'
import json,sys,csv
try: tp=round(json.load(open(sys.argv[1]))['output_throughput'],1)
except Exception: tp=0
kv=w=r=0.0
try:
    for row in csv.reader(open(sys.argv[2])):
        if len(row)>=5:
            kv=max(kv,float(row[2])); w=max(w,float(row[3])); r=max(r,float(row[4]))
except Exception: pass
print(tp, round(kv,4), round(w,1), round(r,1))
PY
)
  echo "$CONC,$NP,$TP,$DP,$KVMAX,$WMAX,$RMAX" >> $OUT/wall.csv
  say "conc=$CONC n=$NP 吞吐=$TP 抢占增量=$DP KV峰值=$KVMAX 排队峰值=$WMAX 运行峰值=$RMAX"
  # ---- 硬门禁 ----
  GATEHIT=$(python3 - "$OUT/G_${ARM}_c${CONC}.json" "$KVMAX" "$GATE_E2E_MS" "$GATE_TTFT_MS" "$GATE_KV" "$GATE_TPOT_MS" <<'PY'
import json,sys
f,kv,ge,gt,gk,gq=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6])
hits=[]
try: d=json.load(open(f))
except Exception: d={}
e=d.get("p99_e2el_ms"); t=d.get("p99_ttft_ms"); q=d.get("p99_tpot_ms")
comp=d.get("completed"); npr=d.get("num_prompts")
if e and e>ge: hits.append(f"E2Ep99={e/1000:.1f}s>{ge/1000:.0f}s")
if t and t>gt: hits.append(f"TTFTp99={t/1000:.1f}s>{gt/1000:.0f}s")
if q and q>gq: hits.append(f"TPOTp99={q:.0f}ms>{gq:.0f}ms")
if kv>=gk:     hits.append(f"KV={kv:.3f}>={gk}")
if comp and npr and comp<npr*0.99: hits.append(f"完成率={comp}/{npr}")
print(";".join(hits))
PY
)
  if [ -n "$GATEHIT" ]; then say "★ 硬门禁命中($GATEHIT)—— 中止该臂阶梯"; break; fi
  if python3 -c "import sys; sys.exit(0 if $DP>0 else 1)"; then say "★ 抢占>0 —— 触到 KV 墙"; break; fi
  if python3 -c "import sys; sys.exit(0 if $TP<=$PREV*1.02 else 1)"; then
    FLAT=$((FLAT+1)); [ $FLAT -ge 2 ] && { say "★ 吞吐连续两档不增 —— 触到拐点"; break; }
  else FLAT=0; fi
  PREV=$TP
done
say "######## 臂 $ARM 完成 ########"
