#!/bin/bash
# 并发 96 塌陷根因:受控单变量实验
#
# 现象:PD 在并发 64→96 时 ITL p50 从 122.3 阶跃到 591.6(4.8×),之后恒定不回落。
#       单机×2 在 96→128 出现同样 4.8× 阶跃。两臂都在【实际运行 batch ~60 → ~96】时阶跃。
# 已知:D 侧引擎日志显示 acceptance length 掉到 1.62(低并发实测 2.77)。
# 假设:DSpark 草稿质量随 batch 增大而退化,导致每步有效产出减少 → ITL 阶跃。
# 但量级对不上:2.77→1.62 只能解释 1.7×,实测 4.8× —— 所以投机最多是部分原因。
#
# 设计:只动 --speculative-config,其余逐字固定。
#   臂 SPEC:{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true}
#   臂 NOSPEC:不传 speculative-config
#   每臂测 conc=64(阶跃前)与 conc=96(阶跃后)
# 判据:NOSPEC 臂若仍出现 64→96 的 ITL 阶跃 → 投机不是主因。
set -uo pipefail
export KUBECONFIG=/etc/kubernetes/admin.conf
K="kubectl"
OUT=/data/share/models/_exp/spec96
LOG=$OUT/driver.log
MODEL=/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8
SG=/models/_datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
mkdir -p $OUT
say(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

for p in ds-n0 ds-n1; do $K get pod $p >/dev/null 2>&1 || { say "FATAL: pod $p 不存在"; exit 1; }; done

teardown(){
  for p in ds-n0 ds-n1; do
    $K exec $p -- bash -lc 'pkill -9 -f "[V]LLM"; pkill -9 -f "[v]llm serve"; pkill -9 -f "[l]oad_balance_proxy"; true' >/dev/null 2>&1
  done
  for h in <NODE_A_PARAM_IP> <NODE_B_PARAM_IP>; do
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no $h 'pkill -9 -f "[V]LLM"; true' >/dev/null 2>&1
  done
  for i in $(seq 1 60); do
    local busy=0 u
    for h in <NODE_A_PARAM_IP> <NODE_B_PARAM_IP>; do
      for u in $(ssh -o BatchMode=yes -o StrictHostKeyChecking=no $h 'npu-smi info 2>/dev/null | grep -oE "[0-9]+ */ *65536" | grep -oE "^[0-9]+"' 2>/dev/null); do
        [ "$u" -ge 8000 ] && busy=$((busy+1)); done; done
    [ "$busy" -eq 0 ] && return 0; sleep 5; done
  return 1
}

write_engine(){ # pod ip port kvcfg specflag
  cat > /tmp/s96_$1.sh <<EOF
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
  --max-model-len 65536 --max-num-seqs 1024 --max-num-batched-tokens 10240 \\
  --gpu-memory-utilization 0.90 \\
  --trust-remote-code --seed 1024 \\
  --no-disable-hybrid-kv-cache-manager --block-size 32 \\
  --tokenizer-mode deepseek_v4 \\
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \\
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \\
  $5 \\
  --kv-transfer-config '$4' > /tmp/s96.log 2>&1
EOF
  $K cp /tmp/s96_$1.sh $1:/tmp/s96.sh >/dev/null 2>&1
}

wait_ready(){
  local ok err
  for i in $(seq 1 70); do
    ok=$($K exec $1 -- bash -lc 'grep -c "Application startup complete" /tmp/s96.log 2>/dev/null | head -1 || echo 0' 2>/dev/null | tr -d '\r')
    err=$($K exec $1 -- bash -lc 'grep -cE "ValueError|AssertionError|Engine core initialization failed" /tmp/s96.log 2>/dev/null | head -1 || echo 0' 2>/dev/null | tr -d '\r')
    [ "${ok:-0}" -ge 1 ] 2>/dev/null && return 0
    [ "${err:-0}" -ge 1 ] 2>/dev/null && { say "  $1 ENGINE FAILED"; return 1; }
    sleep 20; done
  say "  $1 TIMEOUT"; return 1
}

spec_metric(){ # metric -> D 侧值
  $K exec ds-n1 -- bash -lc "curl -sS -m 10 http://127.0.0.1:8078/metrics 2>/dev/null | awk '/^vllm:$1/{s+=\$2} END{printf \"%.1f\", s+0}'" 2>/dev/null | tr -d '\r'
}

run_arm(){ # tag, specflag
  local TAG=$1 SPECFLAG=$2
  say "=== 臂 $TAG  spec=[${SPECFLAG:-无}] ==="
  teardown || { say "teardown 失败"; return 1; }
  for p in ds-n0 ds-n1; do $K exec $p -- bash -lc 'rm -f /tmp/s96.log' >/dev/null 2>&1; done
  KVP='{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"30400","engine_id":"1","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'
  KVC='{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'
  write_engine ds-n1 <NODE_B_PARAM_IP> 8078 "$KVP" "$SPECFLAG"
  $K exec ds-n1 -- bash -lc 'chmod +x /tmp/s96.sh; setsid nohup /tmp/s96.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
  sleep 10
  write_engine ds-n0 <NODE_A_PARAM_IP> 8077 "$KVC" "$SPECFLAG"
  $K exec ds-n0 -- bash -lc 'chmod +x /tmp/s96.sh; setsid nohup /tmp/s96.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
  wait_ready ds-n1 || return 1; wait_ready ds-n0 || return 1
  cat > /tmp/pxy96.sh <<'PEOF'
#!/bin/bash
exec python3 /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 8080 --prefiller-hosts <NODE_A_PARAM_IP> --prefiller-ports 8077 \
  --decoder-hosts <NODE_B_PARAM_IP> --decoder-ports 8078 > /tmp/proxy.log 2>&1
PEOF
  $K cp /tmp/pxy96.sh ds-n0:/tmp/pxy96.sh >/dev/null 2>&1
  $K exec ds-n0 -- bash -lc 'chmod +x /tmp/pxy96.sh; setsid nohup /tmp/pxy96.sh </dev/null >/dev/null 2>&1 & disown' >/dev/null 2>&1
  sleep 15
  # 断言这一臂的投机配置真的生效(配置改了 != 引擎用了)
  local adopted
  adopted=$($K exec ds-n1 -- bash -lc "grep -oE 'speculative_config=[^,]*' /tmp/s96.log | head -1" 2>/dev/null | tr -d '\r')
  say "  引擎采用: $adopted"

  for CONC in 64 96; do
    A0=$(spec_metric "spec_decode_num_accepted_tokens_total"); D0=$(spec_metric "spec_decode_num_draft_tokens_total")
    $K exec ds-n0 -- bash -lc "vllm bench serve --backend openai --model dsv4f \
      --tokenizer $MODEL --tokenizer-mode deepseek_v4 --trust-remote-code \
      --base-url http://127.0.0.1:8080 --dataset-name sharegpt --dataset-path $SG \
      --num-prompts $(( CONC * 8 )) --max-concurrency $CONC \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --save-result --result-filename /tmp/S96_${TAG}_c${CONC}.json" >> "$LOG" 2>&1
    $K cp ds-n0:/tmp/S96_${TAG}_c${CONC}.json $OUT/S96_${TAG}_c${CONC}.json >/dev/null 2>&1
    A1=$(spec_metric "spec_decode_num_accepted_tokens_total"); D1=$(spec_metric "spec_decode_num_draft_tokens_total")
    AL=$(python3 -c "
a=${A1:-0}-${A0:-0}; d=${D1:-0}-${D0:-0}
print(f'{a/d:.3f}' if d>0 else 'n/a(无投机)')")
    ITL=$(python3 -c "
import json
try: print(round(json.load(open('$OUT/S96_${TAG}_c${CONC}.json'))['median_itl_ms'],1))
except Exception: print('-')")
    say "  $TAG conc=$CONC  ITL_p50=$ITL ms  接受率=$AL  (accepted+=$(python3 -c "print(${A1:-0}-${A0:-0})") draft+=$(python3 -c "print(${D1:-0}-${D0:-0})"))"
  done
}

say "######## 并发96塌陷 · 投机单变量对照 ########"
run_arm SPEC   "--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"enforce_eager\":true}'"
run_arm NOSPEC ""
say "######## 完成 ########"
