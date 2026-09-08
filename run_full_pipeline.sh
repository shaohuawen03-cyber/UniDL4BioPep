#!/usr/bin/env bash
# ==========================================================
# 完整流程: AntiFam → 共识预测 → 组间统计 → AD 关联分析
# ==========================================================
# 用法:
#   # 0) 先诊断，拿到推荐阈值（3 分钟，只需做一次）
#   python diagnose_amp_model.py --sample-fasta clean_catalog/*/Cohort2_Disease_AD.fa
#
#   # 1) 用诊断给出的阈值跑全量
#   THRESHOLD=0.95 bash run_full_pipeline.sh
#
#   # 只跑某个队列
#   THRESHOLD=0.95 bash run_full_pipeline.sh Cohort3
#
# 断点续跑: 重新执行同一命令即可，已完成的组会自动跳过。
# ==========================================================
set -euo pipefail

ROOT="${ROOT:-$HOME/UniDL4BioPep-main}"
CATALOG="${CATALOG:-$ROOT/comparable_sorf_grouped_catalog}"
CLEAN="${CLEAN:-$ROOT/clean_catalog}"
OUT="${OUT:-$ROOT/Predictions_AMP_final}"
COHORT="${1:-}"

# ---- 判阳阈值 ----
# 默认 0.5 是模型在【平衡测试集】上的最优点。真实宏基因组中 AMP 先验
# 极低(百分之一量级), 沿用 0.5 会产生大量假阳性。
# 请先运行 diagnose_amp_model.py 取得校准阈值。
THRESHOLD="${THRESHOLD:-0.5}"
MACREL_TH="${MACREL_TH:-0.5}"
MIN_CHARGE="${MIN_CHARGE:-2.0}"

# ---- 性能 ----
NPROC=$(nproc)
WORKERS="${WORKERS:-$(( NPROC/4 > 16 ? 16 : (NPROC/4 < 1 ? 1 : NPROC/4) ))}"
ESM_BS="${ESM_BS:-512}"
PRED_BS="${PRED_BS:-8192}"
CHUNK="${CHUNK:-200000}"

echo "=========================================================="
echo " 抗菌肽挖掘完整流程"
echo "=========================================================="
echo " 判阳阈值 : UniDL4BioPep >= $THRESHOLD | Macrel >= $MACREL_TH"
echo " 净电荷   : >= $MIN_CHARGE"
echo " 输出     : $OUT"
echo " CPU/GPU  : $NPROC 核 | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "=========================================================="

if [ "$THRESHOLD" = "0.5" ]; then
  echo ""
  echo "⚠️  当前使用默认阈值 0.5。"
  echo "    真实宏基因组 AMP 先验极低, 0.5 通常会导致判阳率远高于"
  echo "    文献基准 (0.1-1.65%)，候选集会混入大量假阳性。"
  echo ""
  echo "    强烈建议先运行 (约 3 分钟):"
  echo "      python diagnose_amp_model.py \\"
  echo "          --sample-fasta $CLEAN/*/Cohort2_Disease_AD.fa"
  echo ""
  read -rp "    仍要用 0.5 继续? [y/N] " a
  [[ "$a" == "y" || "$a" == "Y" ]] || exit 1
fi

LOG="$ROOT/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志: $LOG"

# ---------- 第 1 步: AntiFam ----------
echo ""
echo "▶ [1/4] smORF 真实性过滤 (AntiFam)"
if [ -d "$CLEAN" ] && [ -n "$(find "$CLEAN" -name '*.fa' 2>/dev/null | head -1)" ]; then
  echo "   已存在过滤结果，跳过 (删除 $CLEAN 可重做)"
else
  WORKERS="$WORKERS" bash "$ROOT/run_step1_antifam.sh" ${COHORT:+"$COHORT"} 2>&1 | tee -a "$LOG"
fi

# ---------- 第 2 步: 共识预测 ----------
echo ""
echo "▶ [2/4] Macrel + UniDL4BioPep 共识预测"
ARGS=(
  --catalog-dir "$CLEAN"
  --output-dir  "$OUT"
  --macrel-bin  "${MACREL_BIN:-$HOME/miniconda3/envs/env_macrel/bin/macrel}"
  --macrel-threads 8
  --macrel-threshold "$MACREL_TH"
  --threshold "$THRESHOLD"
  --min-charge "$MIN_CHARGE"
  --esm-batch-size "$ESM_BS"
  --predict-batch-size "$PRED_BS"
  --chunk-size "$CHUNK"
  --consensus-mode frac
  --min-consensus-frac 1.0
  --no-ampep
)
[ -n "$COHORT" ] && ARGS+=(--cohorts "$COHORT")

python -u "$ROOT/run_amp_sorf_cohorts.py" "${ARGS[@]}" 2>&1 | tee -a "$LOG"

# ---------- 结果合理性自检 ----------
echo ""
echo "▶ 结果自检"
python - "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import sys, glob, json, os
out = sys.argv[1]
files = glob.glob(os.path.join(out, "*", "*_summary.json"))
if not files:
    print("   ⚠️ 未找到 summary")
    sys.exit(0)
n_high = n_low = 0
print(f"   {'分组':<34} {'共识率':>9}  判定")
for f in sorted(files):
    s = json.load(open(f))
    lb = s.get("literature_benchmark", {})
    r = lb.get("observed_rate_vs_raw_smorf", 0) * 100
    mark = "✅" if 0.1 <= r <= 1.65 else ("⚠️ 偏高" if r > 1.65 else "⚠️ 偏低")
    if r > 1.65:
        n_high += 1
    elif r < 0.1:
        n_low += 1
    print(f"   {os.path.basename(f)[:34]:<34} {r:8.3f}%  {mark}")

# 方向不能说反: 判阳率偏高要【提高】阈值, 偏低要【降低】阈值。
if n_high:
    print(f"\n   ⚠️ {n_high} 个分组【高于】文献基准 1.65%")
    print("      → 提高 THRESHOLD / MACREL_TH, 或提高 MIN_CHARGE 后重跑第 2 步")
if n_low:
    print(f"\n   ⚠️ {n_low} 个分组【低于】文献基准 0.1%")
    print("      → 阈值偏严。降低 MACREL_TH(官方默认 0.5)、"
          "或降低 MIN_CHARGE 后重跑第 2 步")
    print("      → 也可能是输入本身 AMP 就少: 本流程有理化预筛"
          "(净电荷/疏水比例), 分母又是【原始 smORF】,")
    print("        口径比文献更严, 偏低不一定代表出错。"
          "先看各工具单独的判阳率再决定。")
    print("      → 不要靠提高阈值来解决偏低 —— 那只会更低。")
if not (n_high or n_low):
    print("\n   ✅ 全部落在文献基准范围内")
PY

# ---------- 第 3 步: 组间统计 ----------
echo ""
echo "▶ [3/4] 组间比较统计"
python -u "$ROOT/amp_group_stats.py" "$OUT" 2>&1 | tee -a "$LOG" || \
  echo "   ⚠️ 组间统计失败(不影响已产出的预测结果)"

# ---------- 第 4 步: AD 关联 ----------
echo ""
echo "▶ [4/4] AD 疾病关联分析"
for c in Cohort1 Cohort3; do
  if [ -z "$COHORT" ] || [ "$COHORT" = "$c" ]; then
    echo "   → $c (5 阶段趋势检验)"
    python -u "$ROOT/amp_ad_association.py" "$OUT" --cohort "$c" 2>&1 | tee -a "$LOG" || \
      echo "     ⚠️ $c 关联分析跳过"
  fi
done

echo ""
echo "=========================================================="
echo "✅ 全部完成"
echo "   预测结果: $OUT"
echo "   汇总表  : $OUT/summary_all_groups.tsv"
echo "   日志    : $LOG"
echo "=========================================================="
