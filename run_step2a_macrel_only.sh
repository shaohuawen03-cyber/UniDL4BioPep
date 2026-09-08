#!/usr/bin/env bash
# ==========================================================
# 第 2a 步: Macrel 单工具全量预测 (不加载 ESM-2 / TensorFlow)
# ==========================================================
# 与 run_step2b_unidl_only.sh 配对使用。
#
# 这一步【快】: 不加载 ESM-2、不跑 CNN, 只有 Macrel 的 ONNX 前向 + 理化预筛。
#
# 用法:
#   nohup bash run_step2a_macrel_only.sh > macrel_full.log 2>&1 &
#   echo $! > macrel_full.pid
#   tail -f macrel_full.log
#
#   # 只跑一个队列
#   bash run_step2a_macrel_only.sh Cohort2
#
# 断点续跑: 重新执行同一命令即可, 已完成的组会自动跳过。
# 强制重跑: 加 FORCE=1
# ==========================================================
set -euo pipefail

ROOT="${ROOT:-$HOME/UniDL4BioPep-main}"
CLEAN="${CLEAN:-$ROOT/clean_catalog}"
OUT="${OUT:-$ROOT/Predictions_Macrel_final}"
COHORT="${1:-}"

# ---- Macrel ----
# 官方默认判阳阈值 0.5 (AMP_probability >= 0.5)。
# 之前共识流程里用的 0.9 是精确率优先, 单工具全量建议回到 0.5,
# 再用 summary 里的多档阈值表回看 0.9/0.95 各剩多少 —— 不用重跑。
MACREL_TH="${MACREL_TH:-0.5}"
MACREL_BIN="${MACREL_BIN:-$HOME/miniconda3/envs/env_macrel/bin/macrel}"
MACREL_THREADS="${MACREL_THREADS:-8}"
MIN_CHARGE="${MIN_CHARGE:-2.0}"

# ---- 长度 ----
# Macrel 在 (meta)genome 中只输出 10-100 aa 的 smORF (PeerJ 2020)。
# 缺省就锁在 10-100; 传 MAX_LEN=180 可与 UniDL4BioPep 同口径,
# 但 100-180 段对 Macrel 属域外, 不计入其判阳。
MAX_LEN="${MAX_LEN:-100}"
CHUNK="${CHUNK:-200000}"

echo "=========================================================="
echo " Macrel 单工具全量预测 (无 ESM-2 / 无 TensorFlow)"
echo "=========================================================="
echo " 判阳阈值 : Macrel AMP_probability >= $MACREL_TH (官方默认 0.5)"
echo " 净电荷   : >= $MIN_CHARGE"
echo " 长度     : 10-$MAX_LEN aa"
echo " 输出     : $OUT"
echo "=========================================================="

if [ ! -d "$CLEAN" ]; then
  echo "❌ 找不到 $CLEAN —— 先跑 run_step1_antifam.sh"
  exit 1
fi

LOG="$ROOT/macrel_only_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志: $LOG"

ARGS=(
  --catalog-dir "$CLEAN"
  --output-dir  "$OUT"
  --macrel-bin  "$MACREL_BIN"
  --macrel-threads "$MACREL_THREADS"
  --macrel-threshold "$MACREL_TH"
  --min-charge "$MIN_CHARGE"
  --max-len "$MAX_LEN"
  --chunk-size "$CHUNK"
)
[ -n "$COHORT" ] && ARGS+=(--cohorts "$COHORT")
[ "${FORCE:-0}" = "1" ] && ARGS+=(--force)

python -u "$ROOT/run_macrel_only.py" "${ARGS[@]}" 2>&1 | tee -a "$LOG"

echo ""
echo "▶ 结果自检 (Macrel 单工具)"
python - "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import sys, glob, json, os
out = sys.argv[1]
files = glob.glob(os.path.join(out, "*", "*_summary.json"))
if not files:
    print("   ⚠️ 未找到 summary")
    sys.exit(0)
print(f"   {'分组':<34} {'判阳率(vs raw)':>14}  判定")
bad = 0
for f in sorted(files):
    s = json.load(open(f))
    r = s.get("literature_benchmark", {}).get("observed_rate_vs_raw_smorf", 0) * 100
    mark = "✅" if 0.1 <= r <= 1.65 else ("⚠️ 偏高" if r > 1.65 else "⚠️ 偏低")
    if not (0.1 <= r <= 1.65):
        bad += 1
    print(f"   {os.path.basename(f)[:34]:<34} {r:13.4f}%  {mark}")

print("\n   📊 多档阈值下的判阳条数 (换阈值不用重跑):")
print(f"   {'分组':<34} " + " ".join(f"{t:>10}" for t in ("0.5", "0.7", "0.9", "0.95", "0.99")))
for f in sorted(files):
    s = json.load(open(f))
    d = (s.get("prob_distribution", {}).get("Macrel") or {})
    c = d.get("counts_at_threshold") or {}
    if not c:
        continue
    print(f"   {os.path.basename(f)[:34]:<34} "
          + " ".join(f"{c.get(t, 0):>10,}" for t in ("0.5", "0.7", "0.9", "0.95", "0.99")))

if bad:
    print(f"\n   ⚠️ {bad} 个分组超出文献基准 0.1-1.65%")
else:
    print("\n   ✅ 全部落在文献基准范围内")
PY

echo ""
echo "=========================================================="
echo "✅ Macrel 单工具完成"
echo "   结果: $OUT"
echo "   下一步 (务必带 --min-tools 1):"
echo "     python $ROOT/amp_group_stats.py $OUT"
echo "     python $ROOT/amp_ad_association.py $OUT --min-tools 1 --cohort Cohort3"
echo "=========================================================="
