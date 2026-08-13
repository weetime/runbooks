#!/usr/bin/env bash
# 从 Seed-TTS eval 数据集抽出 en 子集的目标文本,产出 seedtts_bench.py / tmpl_ablation.py 吃的格式。
# 产物:一行一句纯文本,共 1088 行。
#
# 数据集: https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval  (en/meta.lst)
# meta.lst 每行: utterance_id | prompt_text | prompt_wav_path | target_text
# 我们只要第 4 列 target_text —— 那是要被朗读的句子。
#
# 用法: ./prep_dataset.sh [输出路径]
# 国内无法直连 huggingface.co 时: export HF_ENDPOINT=https://hf-mirror.com
set -euo pipefail

OUT=${1:-/tmp/seedtts_en_prompts.txt}
HF=${HF_ENDPOINT:-https://huggingface.co}
SRC="$HF/datasets/zhaochenyang20/seed-tts-eval/resolve/main/en/meta.lst"

curl -fsSL --max-time 180 "$SRC" | awk -F'|' 'NF>=4 {
    t=$4; gsub(/^[ \t]+|[ \t]+$/, "", t);
    if (length(t) > 0) print t
}' > "$OUT"

n=$(wc -l < "$OUT" | tr -d ' ')
echo "→ $OUT  共 $n 行"
[ "$n" -ge 1000 ] || { echo "ERROR: 行数异常($n),预期 1088" >&2; exit 1; }
echo "--- 前 2 行 ---"; head -2 "$OUT"
