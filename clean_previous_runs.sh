#!/usr/bin/env bash
# ==========================================================
# 清理之前跑出来的中间产物 / 结果目录
# ==========================================================
# 默认【只列出】要删什么, 不删。确认无误后加 APPLY=1 真删。
#
#   bash clean_previous_runs.sh              # 干跑, 只列出
#   APPLY=1 bash clean_previous_runs.sh      # 真的删
#   APPLY=1 KEEP_CLEAN=1 bash clean_previous_runs.sh   # 连 clean_catalog 一起删
#
# clean_catalog 是 AntiFam 过滤后的 sORF 目录, 重建很贵(要重跑 AntiFam),
# 默认【保留】。只有确定要连第 1 步一起重做时才加 KEEP_CLEAN=0。
# ==========================================================
set -uo pipefail

ROOT="${ROOT:-$HOME/UniDL4BioPep-main}"
APPLY="${APPLY:-0}"
KEEP_CLEAN="${KEEP_CLEAN:-1}"

# 要清理的目标(存在才处理)
TARGETS=(
  "$ROOT/Predictions_AMP_final"      # 原双工具共识跑
  "$ROOT/Predictions_Macrel_final"   # Macrel 单工具跑
  "$ROOT/Predictions_UniDL_final"    # UniDL4BioPep 单工具跑
  "$ROOT"/Predictions_AMP_run*       # 早期试跑
  "$ROOT"/Predictions_Macrel_*
  "$ROOT"/Predictions_UniDL_*
  "$ROOT"/pipeline_*.log
  "$ROOT"/macrel_only_*.log
  "$ROOT"/unidl_only_*.log
  "$ROOT"/full_run.log
  "$ROOT"/full_run.pid
  "$ROOT"/amp_run*.log
  "$ROOT"/amp_run*.pid
  "$ROOT/__pycache__"
)
if [ "$KEEP_CLEAN" != "1" ]; then
  TARGETS+=("$ROOT/clean_catalog")
fi

echo "=========================================================="
echo " 清理之前的运行产物"
echo " ROOT      : $ROOT"
echo " APPLY     : $APPLY  (0 = 只列出, 1 = 真的删)"
echo " KEEP_CLEAN: $KEEP_CLEAN (1 = 保留 clean_catalog)"
echo "=========================================================="

n=0
for pat in "${TARGETS[@]}"; do
  for t in $pat; do
    [ -e "$t" ] || continue
    if [ -d "$t" ]; then
      sz=$(du -sh "$t" 2>/dev/null | cut -f1)
      nf=$(find "$t" -type f 2>/dev/null | wc -l)
      echo "  [目录] $t   ($sz, $nf 个文件)"
    else
      sz=$(du -h "$t" 2>/dev/null | cut -f1)
      echo "  [文件] $t   ($sz)"
    fi
    n=$((n + 1))
    if [ "$APPLY" = "1" ]; then
      rm -rf -- "$t"
    fi
  done
done

echo "----------------------------------------------------------"
if [ "$n" -eq 0 ]; then
  echo "✅ 没有需要清理的东西。"
elif [ "$APPLY" = "1" ]; then
  echo "🗑 已删除 $n 项。"
  echo "  注意: clean_catalog $([ "$KEEP_CLEAN" = "1" ] && echo '已保留' || echo '也已删除')。"
else
  echo "👀 以上是 $n 项待删目标 —— 什么都没删。"
  echo "   确认无误后执行:  APPLY=1 bash clean_previous_runs.sh"
fi
echo "=========================================================="
