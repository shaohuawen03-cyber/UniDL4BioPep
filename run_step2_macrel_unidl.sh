#!/usr/bin/env bash
# ==========================================================
# 第 2 步: AMP 共识预测 (Macrel + UniDL4BioPep)
# ==========================================================
# Macrel        : 随机森林, 精确率优先, 专为宏基因组低正例先验训练
#                 (Santos-Júnior et al., PeerJ 2020)
# UniDL4BioPep  : ESM-2 + CNN (Du et al., Brief Bioinform 2023)
#
# 硬件适配: NVIDIA RTX A4000 16GB
#   ESM-2 t6_8M 极小(~30MB), 显存瓶颈在 batch 内的 token 数。
#   A4000 上 esm-batch-size 512 约占 3-5GB, 留足余量。
#
# 共识模式: frac (判阳工具数 / 可评工具数)
#   两工具长度域不同(Macrel 10-100, UniDL4BioPep 11-180),
#   若用 count 模式 + min-tools 2, >100aa 的序列因 Macrel 评不了
#   最多只能拿 1 票, 会被系统性丢弃。frac 模式按"可评工具"归一化。
#
# 用法:
#   bash run_step2_macrel_unidl.sh              # 全部
#   bash run_step2_macrel_unidl.sh Cohort3      # 只跑 Cohort3
#   TEST=1 bash run_step2_macrel_unidl.sh Cohort2   # 试跑 20 万条
# ==========================================================
set -euo pipefail

ROOT="${ROOT:-$HOME/UniDL4BioPep-main}"
CLEAN="${CLEAN:-$ROOT/clean_catalog}"
OUT="${OUT:-$ROOT/Predictions_AMP_run1}"
# macrel: 直接用绝对路径调用外部二进制, 无需 conda activate
# (环境若以 -p 创建, conda activate <名字> 会失败, 这里不受影响)
if [ -z "${MACREL_BIN:-}" ]; then
  for c in "$HOME/miniconda3/envs/env_macrel/bin/macrel" \
           "$HOME/miniconda3/envs"/*/bin/macrel; do
    [ -x "$c" ] && { MACREL_BIN="$c"; break; }
  done
fi
MACREL_BIN="${MACREL_BIN:-}"
COHORT="${1:-}"

# ---- A4000 16GB 调优 ----
ESM_BS="${ESM_BS:-512}"        # ESM-2 前向 batch
PRED_BS="${PRED_BS:-8192}"     # Keras 预测 batch
CHUNK="${CHUNK:-200000}"       # 每块序列数
MACREL_THREADS="${MACREL_THREADS:-8}"

echo "=========================================================="
echo " 第 2 步: Macrel + UniDL4BioPep 共识预测"
echo " 输入: $CLEAN"
echo " 输出: $OUT"
echo " GPU : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================================="

if [ ! -d "$CLEAN" ]; then
  echo "⚠️  未找到 $CLEAN（尚未做 AntiFam 过滤）"
  echo "   建议先运行: bash run_step1_antifam.sh"
  read -rp "   是否直接用原始目录继续? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
  CLEAN="$ROOT/comparable_sorf_grouped_catalog"
fi

[ -n "$MACREL_BIN" ] && [ -x "$MACREL_BIN" ] || {
  echo "❌ 未找到 macrel 可执行文件"
  echo "   查找: ls /home/wsh/miniconda3/envs/*/bin/macrel"
  echo "   指定: MACREL_BIN=/path/to/macrel bash $0"
  exit 1; }
echo "✅ Macrel: $($MACREL_BIN --version 2>&1 | head -1)"

ARGS=(
  --catalog-dir "$CLEAN"
  --output-dir  "$OUT"
  --macrel-bin  "$MACREL_BIN"
  --macrel-threads "$MACREL_THREADS"
  --esm-batch-size "$ESM_BS"
  --predict-batch-size "$PRED_BS"
  --chunk-size "$CHUNK"
  --consensus-mode frac
  --min-consensus-frac 1.0
  --no-ampep
)
[ -n "$COHORT" ] && ARGS+=(--cohorts "$COHORT")
[ "${TEST:-0}" = "1" ] && ARGS+=(--max-seqs 200000)

echo ""
echo "命令: python run_amp_sorf_cohorts.py ${ARGS[*]}"
echo ""

if [ "${TEST:-0}" = "1" ]; then
  python "$ROOT/run_amp_sorf_cohorts.py" "${ARGS[@]}"
else
  LOG="$ROOT/amp_run1.log"
  nohup python -u "$ROOT/run_amp_sorf_cohorts.py" "${ARGS[@]}" > "$LOG" 2>&1 &
  echo $! > "$ROOT/amp_run1.pid"
  echo "🚀 后台运行中 (PID $(cat "$ROOT/amp_run1.pid"))"
  echo "   查看进度: tail -f $LOG"
  echo "   停止:     kill \$(cat $ROOT/amp_run1.pid)"
  echo "   （断点续跑：重新执行本脚本即可，会自动从上次 chunk 继续）"
fi
