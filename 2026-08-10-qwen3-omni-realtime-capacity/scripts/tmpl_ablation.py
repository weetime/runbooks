#!/usr/bin/env python3
"""提示模板消融:量化 READ_TMPL 相对裸文本输入抬高了多少 RTF,以及该抬高在 A100 与昇腾上是否等比。

背景:三方实测的 RTF 相对 vLLM-Omni 官方公布值偏高 83%,唯一在手的解释是提示模板
`Read the following sentence aloud: {text}` 让 thinker 多生成一段内容。若该系统误差在两平台
非等比(昇腾 thinker 慢约 3.8x,多生成的 token 代价更大),则昇腾的单路 RTF 0.701 被高估,
「只剩 1.43 倍余量」以及由此推出的短句实时上限全部需要重估。

三臂对照(c=1,同一批文本,同一台机器):
  A  READ  = "Read the following sentence aloud: {text}"   当前口径基线
  B  BARE  = "{text}"                                       官方口径近似
  C  MIN   = "朗读:{text}"                                  最短指令,隔离「指令长度」与「有无指令」

关键:B 臂有被模型当成问题来回答的风险(生成的不是朗读而是答话),那样 RTF 变化就不是模板
造成的。因此每臂必须同时记录 **输出 token 数** 与 **音频时长**,并对 B 臂做朗读性检查——
若 B 臂的音频时长与输入文本长度的相关性显著低于 A 臂,说明模型在答话,该臂数据无效。

用法: python3 tmpl_ablation.py <MODEL> [n] [port]
"""
import base64, json, statistics, sys, time, urllib.request

MODEL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].strip() else 20
PORT = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].strip() else "8000"
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"
PROMPT_FILE = "/tmp/seedtts_en_prompts.txt"
SR, WIDTH = 24000, 2
WARM = 5

ARMS = [
    ("READ", "Read the following sentence aloud: {text}"),
    ("BARE", "{text}"),
    ("MIN",  "朗读:{text}"),
]

texts = [l.rstrip("\n") for l in open(PROMPT_FILE, encoding="utf-8") if l.strip()]


def once(prompt_text):
    """返回 (e2el, audio_sec, out_tokens, text_out)。串行单路,c=1。"""
    body = {"model": MODEL, "max_tokens": 512, "stream": True,
            "modalities": ["text", "audio"], "speaker": "chelsie",
            "messages": [{"role": "user", "content": prompt_text}]}
    req = urllib.request.Request(BASE, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    abytes = 0
    text_out = []
    for ln in urllib.request.urlopen(req, timeout=900):
        ln = ln.decode().strip()
        if not ln.startswith("data:") or ln == "data: [DONE]":
            continue
        d = json.loads(ln[5:])
        delta = d.get("choices", [{}])[0].get("delta", {})
        c = delta.get("content")
        if d.get("modality") == "audio":
            if isinstance(c, str) and c:
                try:
                    abytes += len(base64.b64decode(c))
                except Exception:
                    pass
        elif isinstance(c, str) and c:
            text_out.append(c)
    e2el = time.time() - t0
    txt = "".join(text_out)
    # 输出 token 数无法从流式响应直接拿到,用字符数作代理(同一模型内可比)
    return e2el, (abytes / (SR * WIDTH) if abytes else None), len(txt), txt


# 预热:三臂共用同一个已热的实例,只需热一次
for i in range(WARM):
    once(ARMS[0][1].format(text=texts[i % len(texts)]))

print(f"# model={MODEL} n={N} 每臂同一批文本(前 {N} 条)")
print(f"# {'arm':5} {'RTF_p50':>8} {'RTF_max':>8} {'E2EL_p50':>9} {'audio_p50':>10} "
      f"{'textlen_p50':>12} {'corr(len,audio)':>16}")

results = {}
for name, tmpl in ARMS:
    rows = []
    for t in texts[:N]:
        e2el, dur, tl, _ = once(tmpl.format(text=t))
        if dur:
            rows.append((e2el, dur, tl, len(t)))
    if not rows:
        print(f"  {name:5} 无音频输出,跳过")
        continue
    rtfs = [r[0] / r[1] for r in rows]
    durs = [r[1] for r in rows]
    tls = [r[2] for r in rows]
    # 朗读性检查:输入文本长度与音频时长的皮尔逊相关;朗读应当强相关
    xs, ys = [r[3] for r in rows], durs
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    corr = num / den if den else float("nan")
    results[name] = (statistics.median(rtfs), max(rtfs), corr)
    print(f"  {name:5} {statistics.median(rtfs):8.3f} {max(rtfs):8.3f} "
          f"{statistics.median([r[0] for r in rows]):9.2f} {statistics.median(durs):10.2f} "
          f"{statistics.median(tls):12.0f} {corr:16.3f}")

if "READ" in results and "BARE" in results:
    r_read, r_bare = results["READ"][0], results["BARE"][0]
    print(f"\n# 模板抬高幅度 READ/BARE = {r_read / r_bare:.3f}"
          f"  (BARE 朗读性相关系数 {results['BARE'][2]:.3f};<0.5 则该臂疑为答话而非朗读,数据无效)")
