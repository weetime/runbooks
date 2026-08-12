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
