"""开图后的正确性验证:文本连贯性 + 音频不是静音/噪声,并与 eager 基线对比。"""
import array, base64, io, json, math, time, urllib.request, wave

B = "http://127.0.0.1:8000/v1/chat/completions"
M = "Qwen3-Omni-30B-A3B-Instruct"
POEM = "请朗读:床前明月光,疑是地上霜,举头望明月,低头思故乡。"


def call(mods, mt, prompt):
    body = {"model": M, "max_tokens": mt, "modalities": mods,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(B, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    return time.time() - t0, d


# --- 1. 文本连贯性 ---
w, d = call(["text"], 128, "9.11 和 9.8 哪个大？简短回答")
print("TEXT reply:", repr(d["choices"][0]["message"]["content"][:80]))
print("TEXT wall=%.2fs out_tok=%d" % (w, d["usage"]["completion_tokens"]))

# --- 2. 音频质量 ---
w, d = call(["text", "audio"], 512, POEM)
txt = d["choices"][0]["message"]["content"]
au = [c["message"].get("audio") for c in d["choices"] if c["message"].get("audio")]
print("\nAUDIO text part:", repr(txt[:80]))
if not au:
    print("NO AUDIO RETURNED")
    raise SystemExit(1)

raw = base64.b64decode(au[0]["data"])
wv = wave.open(io.BytesIO(raw))
n, sr, sw = wv.getnframes(), wv.getframerate(), wv.getsampwidth()
dur = n / sr
pcm = array.array("h", wv.readframes(n))
peak = max(abs(x) for x in pcm)
rms = math.sqrt(sum(float(x) * x for x in pcm) / len(pcm))
# 静音段占比(|x| < 1% 满量程)
quiet = sum(1 for x in pcm if abs(x) < 328) / len(pcm)

print("AUDIO wall=%.2fs dur=%.2fs RTF=%.3f  sr=%d bits=%d" % (w, dur, w / dur, sr, sw * 8))
print("AUDIO peak=%d rms=%.1f (满量程 32767)  近静音样本占比=%.1f%%" % (peak, rms, quiet * 100))
print("eager 基线: dur=6.22s RTF=1.883")
ok = peak > 3000 and rms > 200 and quiet < 0.90 and 3.0 < dur < 15.0
print("VERDICT:", "AUDIO_SANE" if ok else "AUDIO_SUSPECT")
