#!/usr/bin/env bash
# ==========================================================
# 第 2b 步: UniDL4BioPep 单工具全量预测 (关掉 Macrel, 不做共识)
# ==========================================================
# 与 run_step2a_macrel_only.sh 配对使用: 两个工具各自独立跑全量,
# 输出目录分开, 这样才能说清楚"最终候选集是被哪个工具决定的"。
#
# 用法:
#   THRESHOLD=0.99 nohup bash run_step2b_unidl_only.sh > unidl_full.log 2>&1 &
#
#   # 只跑一个队列
#   THRESHOLD=0.99 bash run_step2b_unidl_only.sh Cohort2
#
# 断点续跑: 重新执行同一命令即可, 已完成的组会自动跳过。
#
# 注意: 本步仍然要跑 ESM-2 + CNN, 是两步里慢的那一步。
#       TensorFlow 若报 "Skipping registering GPU devices",
#       CNN 就是在 CPU 上跑 —— 结果正确, 只是慢。
# ==========================================================
set -euo pipefail

ROOT="${ROOT:-$HOME/UniDL4BioPep-main}"
CLEAN="${CLEAN:-$ROOT/clean_catalog}"
OUT="${OUT:-$ROOT/Predictions_UniDL_final}"
COHORT="${1:-}"

# ---- 判阳阈值 ----
# 0.5 是模型在【平衡测试集】上的最优点; 真实宏基因组 AMP 先验极低,
# 沿用 0.5 判阳率会远高于文献基准 (0.1-1.65%)。
THRESHOLD="${THRESHOLD:-0.99}"
MIN_CHARGE="${MIN_CHARGE:-2.0}"

# ---- 性能 ----
NPROC=$(nproc)
ESM_BS="${ESM_BS:-512}"
PRED_BS="${PRED_BS:-8192}"
CHUNK="${CHUNK:-200000}"

echo "=========================================================="
echo " UniDL4BioPep 单工具全量预测 (Macrel 已关闭)"
echo "=========================================================="
echo " 判阳阈值 : UniDL4BioPep >= $THRESHOLD"
echo " 净电荷   : >= $MIN_CHARGE"
echo " 输出     : $OUT"
echo " CPU/GPU  : $NPROC 核 | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "=========================================================="

if [ ! -d "$CLEAN" ]; then
  echo "❌ 找不到 $CLEAN —— 先跑 run_step1_antifam.sh"
  exit 1
fi

LOG="$ROOT/unidl_only_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志: $LOG"

# --no-macrel : 关掉 Macrel, 只留 UniDL4BioPep 一票
# --no-ampep  : amPEPpy 未安装
# 长度域缺省=启用工具域并集 = UniDL4BioPep AMP 的 11-180 aa
ARGS=(
  --catalog-dir "$CLEAN"
  --output-dir  "$OUT"
  --threshold   "$THRESHOLD"
  --min-charge  "$MIN_CHARGE"
  --esm-batch-size    "$ESM_BS"
  --predict-batch-size "$PRED_BS"
  --chunk-size  "$CHUNK"
  --no-macrel
  --no-ampep
  --consensus-mode frac
  --min-consensus-frac 1.0
)
[ -n "$COHORT" ] && ARGS+=(--cohorts "$COHORT")

python -u "$ROOT/run_amp_sorf_cohorts.py" "${ARGS[@]}" 2>&1 | tee -a "$LOG"

# ---------- 结果自检 ----------
echo ""
echo "▶ 结果自检 (UniDL4BioPep 单工具)"
python - "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import sys, glob, json, os
out = sys.argv[1]
files = glob.glob(os.path.join(out, "*", "*_summary.json"))
if not files:
    print("   ⚠️ 未找到 summary")
    sys.exit(0)
print(f"   {'分组':<34} {'判阳率(vs raw)':>14} {'P99 概率':>9}  判定")
bad = 0
for f in sorted(files):
    s = json.load(open(f))
    lb = s.get("literature_benchmark", {})
    r = lb.get("observed_rate_vs_raw_smorf", 0) * 100
    pd_ = (s.get("prob_distribution", {}).get("AMP") or {})
    p99 = (pd_.get("quantiles") or {}).get("P99")
    mark = "✅" if 0.1 <= r <= 1.65 else ("⚠️ 偏高" if r > 1.65 else "⚠️ 偏低")
    if not (0.1 <= r <= 1.65):
        bad += 1
    print(f"   {os.path.basename(f)[:34]:<34} {r:13.4f}% "
          f"{(p99 if p99 is not None else float('nan')):9.4f}  {mark}")

print("\n   📊 多档阈值下的判阳条数 (换阈值不用重跑):")
print(f"   {'分组':<34} " + " ".join(f"{t:>12}" for t in ("0.5", "0.9", "0.95", "0.99", "0.999")))
for f in sorted(files):
    s = json.load(open(f))
    d = (s.get("prob_distribution", {}).get("AMP") or {})
    c = d.get("counts_at_threshold") or {}
    if not c:
        continue
    print(f"   {os.path.basename(f)[:34]:<34} "
          + " ".join(f"{c.get(t, 0):>12,}" for t in ("0.5", "0.9", "0.95", "0.99", "0.999")))

if bad:
    print(f"\n   ⚠️ {bad} 个分组超出文献基准 0.1-1.65%")
    print("      若把阈值提到 0.999 判阳率仍远高于 1.65%,")
    print("      说明 UniDL4BioPep 在本数据上不具备判别力, 而非阈值没调好。")
PY

echo ""
echo "=========================================================="
echo "✅ UniDL4BioPep 单工具完成"
echo "   结果: $OUT"
echo "   下一步 (务必带 --min-tools 1):"
echo "     python $ROOT/amp_group_stats.py $OUT"
echo "     python $ROOT/amp_ad_association.py $OUT --min-tools 1 --cohort Cohort3"
echo "=========================================================="
