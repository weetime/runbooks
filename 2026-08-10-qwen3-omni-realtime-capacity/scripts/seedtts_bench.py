#!/usr/bin/env python3
"""Seed-TTS 数据集口径的 omni 压测,对齐 vLLM-Omni 官方博客的指标与流程。

官方口径(2026-07-01 vLLM Blog):
  Seed-TTS `en`,prompts 10/160/320/640 对应并发 1/16/32/64,5 次预热,3 张可见 GPU。
  指标:req/s · Audio TTFP · Audio RTF。

与官方的已知差异(务必在报告中标明):
  1. 官方用其自带的 `vllm-omni bench serve`(我们的镜像缺 audio_ttfp/audio_rtf 指标),
     这里改用已交叉验证过的自建流式采集(与非流式 RTF 偏差 0.9%)。
  2. 官方未公开送给模型的具体 prompt 模板;这里用 READ_TMPL 让模型朗读 target_text,
     使音频长度与数据集文本强相关、可复现。模板在结果中原样记录。

用法: python3 seedtts_bench.py <MODEL> <concurrency> [n_prompts] [replicas]
"""
import base64, concurrent.futures, itertools, json, statistics, sys, threading, time, urllib.request

BASE = "http://127.0.0.1:8000/v1/chat/completions"
PROMPT_FILE = "/tmp/seedtts_en_prompts.txt"
READ_TMPL = "Read the following sentence aloud: {text}"

MODEL = sys.argv[1]
CC = int(sys.argv[2])
# 官方映射:并发 1/16/32/64 → 10/160/320/640 条
DEFAULT_N = {1: 10, 2: 20, 4: 40, 8: 80, 16: 160, 32: 320, 64: 640}
_a3 = sys.argv[3] if len(sys.argv) > 3 else ""
N = int(_a3) if _a3.strip() else DEFAULT_N.get(CC, CC * 10)   # 空串按默认处理
NREP = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].strip() else 2
WARM_ROUNDS = 5                 # 官方为 5 次预热
SR, WIDTH = 24000, 2

prompts = [l.rstrip("\n") for l in open(PROMPT_FILE, encoding="utf-8") if l.strip()]
_cyc = itertools.cycle(prompts)
_lock = threading.Lock()


def next_prompt():
    with _lock:
        return next(_cyc)


def once(text=None):
    text = text or next_prompt()
    body = {"model": MODEL, "max_tokens": 512, "stream": True,
            "modalities": ["text", "audio"], "speaker": "chelsie",
            "messages": [{"role": "user", "content": READ_TMPL.format(text=text)}]}
    req = urllib.request.Request(BASE, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = ttfp = None
    abytes = 0
    for ln in urllib.request.urlopen(req, timeout=900):
        ln = ln.decode().strip()
        if not ln.startswith("data:") or ln == "data: [DONE]":
            continue
        d = json.loads(ln[5:])
        mod = d.get("modality")
        delta = d.get("choices", [{}])[0].get("delta", {})
        if mod == "text" and ttft is None:
            ttft = time.time() - t0
        if mod == "audio":
            if ttfp is None:
                ttfp = time.time() - t0
            c = delta.get("content")
            if isinstance(c, str) and c:
                try:
                    abytes += len(base64.b64decode(c))
                except Exception:
                    pass
    e2el = time.time() - t0
    return ttft, ttfp, e2el, (abytes / (SR * WIDTH) if abytes else None)


def burst(n, workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda _: once(), range(n)))


# 预热:按目标并发,确保覆盖所有副本(多副本冷启动惩罚约 3.8x)
for _ in range(WARM_ROUNDS):
    burst(max(CC, NREP), max(CC, NREP))

t_start = time.time()
res = burst(N, CC)
wall = time.time() - t_start

ok = [r for r in res if r[1] is not None]
if not ok:
    print("c=%d NO audio packets" % CC)
    sys.exit(1)


def p(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


ttfp = [r[1] * 1000 for r in ok]
ttft = [r[0] * 1000 for r in ok if r[0]]
e2el = [r[2] for r in ok]
durs = [r[3] for r in ok if r[3]]
rtfs = [r[2] / r[3] for r in ok if r[3]]

print("c=%-3d n=%-4d req_s=%5.2f TTFT_p50=%6.0fms TTFP_p50=%7.0fms TTFP_p95=%7.0fms "
      "E2EL_p50=%5.2fs audio_p50=%5.2fs RTF_p50=%.3f RTF_max=%.3f realtime=%s"
      % (CC, len(ok), len(ok) / wall,
         statistics.median(ttft) if ttft else -1,
         statistics.median(ttfp), p(ttfp, 0.95),
         statistics.median(e2el),
         statistics.median(durs) if durs else -1,
         statistics.median(rtfs) if rtfs else -1,
         max(rtfs) if rtfs else -1,
         "YES" if rtfs and max(rtfs) < 1.0 else "NO"))
