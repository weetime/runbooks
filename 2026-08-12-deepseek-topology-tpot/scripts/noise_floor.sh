#!/bin/bash
# 噪声底标定 —— 所有对比结论的前提,已经欠了三次。
#
# 纪律(来自项目 skill inference-perf-experiment §3):
#   * 必须【跨服务重启】复测。同一引擎实例内反复跑只测到抖动,会严重低估噪声。
#   * n>=3。实测 n=2 会低估 14 倍(0.06% vs 0.88%)。
#   * 分档量:不同并发档噪声不同,不能全表共用一个噪声底。
#   * 分指标量:分布类指标(p99)噪声可达均值类的 10 倍以上。
#
# 用法:80_noise_floor.sh <twox|pd>
set -uo pipefail
export KUBECONFIG=/etc/kubernetes/admin.conf
K="kubectl"
ARM="${1:?twox|pd}"
OUT=/data/share/models/_exp/noise/$ARM
LOG=$OUT/driver.log
MODEL=/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8
SG=/models/_datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
MAXLEN=65536
MNS=1024
TIERS="64 192"     # 一个在 PD 优势区、一个在恢复区
REPS=3
mkdir -p $OUT
say(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

for p in ds-n0 ds-n1; do
  $K get pod $p >/dev/null 2>&1 || { say "FATAL: pod $p 不存在"; exit 1; }
done

teardown(){
  for p in ds-n0 ds-n1; do
    $K exec $p -- bash -lc 'pkill -9 -f "[V]LLM"; pkill -9 -f "[v]llm serve"; pkill -9 -f "[l]oad_balance_proxy"; pkill -9 -f "[r]r_proxy"; true' >/dev/null 2>&1
  done
  for h in <NODE_A_PARAM_IP> <NODE_B_PARAM_IP>; do
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no $h 'pkill -9 -f "[V]LLM"; true' >/dev/null 2>&1
  done
  for i in $(seq 1 60); do
    local busy=0 u
    for h in <NODE_A_PARAM_IP> <NODE_B_PARAM_IP>; do
      for u in $(ssh -o BatchMode=yes -o StrictHostKeyChecking=no $h 'npu-smi info 2>/dev/null | grep -oE "[0-9]+ */ *65536" | grep -oE "^[0-9]+"' 2>/dev/null); do
        [ "$u" -ge 8000 ] && busy=$((busy+1)); done; done
    [ "$busy" -eq 0 ] && return 0
    sleep 5; done
  return 1
}

write_engine(){ # pod ip port extra
  cat > /tmp/nf_$1.sh <<EOF
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
  --max-model-len $MAXLEN --max-num-seqs $MNS --max-num-batched-tokens 10240 \\
  --gpu-memory-utilization 0.90 \\
  --trust-remote-code --seed 1024 \\
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true}' \\
  --no-disable-hybrid-kv-cache-manager --block-size 32 \\
  --tokenizer-mode deepseek_v4 \\
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \\
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \\
  $4 > /tmp/nf.log 2>&1
EOF
  $K cp /tmp/nf_$1.sh $1:/tmp/nf.sh >/dev/null 2>&1
}

wait_ready(){
  local ok err
  for i in $(seq 1 70); do
    ok=$($K exec $1 -- bash -lc 'grep -c "Application startup complete" /tmp/nf.log 2>/dev/null | head -1 || echo 0' 2>/dev/null | tr -d '\r')
    err=$($K exec $1 -- bash -lc 'grep -cE "ValueError|AssertionError|Engine core initialization failed" /tmp/nf.log 2>/dev/null | head -1 || echo 0' 2>/dev/null | tr -d '\r')
    [ "${ok:-0}" -ge 1 ] 2>/dev/null && return 0
    [ "${err:-0}" -ge 1 ] 2>/dev/null && { say "  $1 ENGINE FAILED"; return 1; }
    sleep 20; done
  say "  $1 TIMEOUT"; return 1
}

bring_up(){   # 全冷重建,这是噪声底成立的关键
  teardown || return 1
  for p in ds-n0 ds-n1; do $K exec $p -- bash -lc 'rm -f /tmp/nf.log' >/dev/null 2>&1; done
  if [ "$ARM" = "twox" ]; then
    write_engine ds-n0 <NODE_A_PARAM_IP> 8077 ""
    write_engine ds-n1 <NODE_B_PARAM_IP> 8078 ""
    for p in ds-n0 ds-n1; do $K exec $p -- bash -lc 'chmod +x /tmp/nf.sh; setsid nohup /tmp/nf.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1; done
    wait_ready ds-n0 || return 1; wait_ready ds-n1 || return 1
    cat > /tmp/rr.sh <<'REOF'
#!/bin/bash
exec python3 /tmp/rr_proxy.py --port 8081 --backends <NODE_A_PARAM_IP>:8077 <NODE_B_PARAM_IP>:8078 > /tmp/rr.log 2>&1
REOF
    $K cp /tmp/rr_proxy.py ds-n0:/tmp/rr_proxy.py >/dev/null 2>&1
    $K cp /tmp/rr.sh ds-n0:/tmp/rr.sh >/dev/null 2>&1
    $K exec ds-n0 -- bash -lc 'chmod +x /tmp/rr.sh; setsid nohup /tmp/rr.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
    ENDPOINT="http://127.0.0.1:8081"
  else
    KVP='{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"30400","engine_id":"1","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'
    KVC='{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'
    write_engine ds-n1 <NODE_B_PARAM_IP> 8078 "--kv-transfer-config '$KVP'"
    $K exec ds-n1 -- bash -lc 'chmod +x /tmp/nf.sh; setsid nohup /tmp/nf.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
    sleep 10
    write_engine ds-n0 <NODE_A_PARAM_IP> 8077 "--kv-transfer-config '$KVC'"
    $K exec ds-n0 -- bash -lc 'chmod +x /tmp/nf.sh; setsid nohup /tmp/nf.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
    wait_ready ds-n1 || return 1; wait_ready ds-n0 || return 1
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
  return 0
}

say "######## 噪声底标定 · 臂=$ARM · 档位=$TIERS · n=$REPS(每次全冷重启)########"
for REP in $(seq 1 $REPS); do
  say "--- 第 $REP 轮:全冷重建服务 ---"
  bring_up || { say "第 $REP 轮重建失败,跳过"; continue; }
  say "    服务已就绪"
  for CONC in $TIERS; do
    NP=$(( CONC * 8 ))
    $K exec ds-n0 -- bash -lc "vllm bench serve --backend openai --model dsv4f \
      --tokenizer $MODEL --tokenizer-mode deepseek_v4 --trust-remote-code \
      --base-url $ENDPOINT --dataset-name sharegpt --dataset-path $SG \
      --num-prompts $NP --max-concurrency $CONC \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --save-result --result-filename /tmp/NF_${ARM}_c${CONC}_r${REP}.json" >> "$LOG" 2>&1
    $K cp ds-n0:/tmp/NF_${ARM}_c${CONC}_r${REP}.json $OUT/NF_${ARM}_c${CONC}_r${REP}.json >/dev/null 2>&1
    if [ -s "$OUT/NF_${ARM}_c${CONC}_r${REP}.json" ]; then say "    conc=$CONC rep=$REP OK"; else say "    conc=$CONC rep=$REP FAILED"; fi
  done
done
say "######## 臂 $ARM 噪声底完成 ########"
