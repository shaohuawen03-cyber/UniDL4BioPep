#!/usr/bin/env bash
# ==========================================================
# 一键跑完剩余全部任务, 并组装成总结果
# ==========================================================
# 前提: 共识跑(Predictions_AMP_final)和 Macrel 单工具跑
#       (Predictions_Macrel_final)已经完成。
#
# 用法:
#   nohup bash run_all_remaining.sh > all_remaining.log 2>&1 &
#   echo $! > all_remaining.pid
#   tail -f all_remaining.log
#
# 可选:
#   CORES=16 bash run_all_remaining.sh          # 改并发上限(默认 8, 硬上限 32)
#   RUN_UNIDL_ONLY=1 bash run_all_remaining.sh  # 额外单独跑一次 UniDL4BioPep
#   FORCE=1 bash run_all_remaining.sh           # 忽略"已完成"标记强制重跑
#
# 特性:
#   - 并发核数硬上限 32, 超过自动压到 32
#   - 每一步失败不中断, 继续跑下一步, 最后汇总哪些成了哪些没成
#   - 每步单独写日志到 logs/
# ==========================================================
set -uo pipefail

ROOT="${ROOT:-$HOME/UniDL4BioPep-main}"
MACREL_OUT="${MACREL_OUT:-$ROOT/Predictions_Macrel_final}"
UNIDL_OUT="${UNIDL_OUT:-$ROOT/Predictions_AMP_final}"
UNIDL_ONLY_OUT="${UNIDL_ONLY_OUT:-$ROOT/Predictions_UniDL_final}"
REPORT="${REPORT:-$ROOT/FINAL_REPORT}"
LOGD="$ROOT/logs"
mkdir -p "$LOGD"

MACREL_BIN="${MACREL_BIN:-$HOME/miniconda3/envs/env_macrel/bin/macrel}"
# 仅当 Macrel 那步还没跑时才用到(已经跑完就会跳过)
MACREL_TH="${MACREL_TH:-0.5}"
# 第 4 步对比表里"认定 Macrel 命中"用的阈值 —— 必须等于【共识跑当时】用的值。
# 你那次是 MACREL_TH=0.9, 所以这里默认 0.9, 不要跟着上面的 0.5 走。
CMP_MACREL_TH="${CMP_MACREL_TH:-0.9}"
MIN_CHARGE="${MIN_CHARGE:-2.0}"
THRESHOLD="${THRESHOLD:-0.99}"
RUN_UNIDL_ONLY="${RUN_UNIDL_ONLY:-0}"
FORCE="${FORCE:-0}"

# ---- 并发: 硬上限 32 核 ----
NPROC=$(nproc)
REQ="${CORES:-8}"
if [ "$REQ" -gt 32 ]; then
  echo "⚠️ CORES=$REQ 超过 32, 压到 32"
  REQ=32
fi
if [ "$REQ" -gt "$NPROC" ]; then
  echo "⚠️ CORES=$REQ 超过本机 $NPROC 核, 压到 $NPROC"
  REQ=$NPROC
fi
[ "$REQ" -lt 1 ] && REQ=1
# CD-HIT 内存(MB): 留足余量, 别超过物理内存的一半
MEM_TOTAL_MB=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 16000)
CDHIT_MEM="${CDHIT_MEM:-$(( MEM_TOTAL_MB / 2 ))}"

export OMP_NUM_THREADS="$REQ"
export OPENBLAS_NUM_THREADS="$REQ"
export MKL_NUM_THREADS="$REQ"
export NUMEXPR_NUM_THREADS="$REQ"
export TF_NUM_INTRAOP_THREADS="$REQ"
export TF_NUM_INTEROP_THREADS=2

echo "=========================================================="
echo " 剩余任务总跑"
echo "=========================================================="
echo " 并发核数   : $REQ  (本机 $NPROC 核, 硬上限 32)"
echo " CD-HIT 内存: ${CDHIT_MEM} MB  (物理内存 ${MEM_TOTAL_MB} MB)"
echo " Macrel 结果: $MACREL_OUT"
echo " 对比口径   : Macrel >= $CMP_MACREL_TH (共识跑当时的值) | UniDL >= $THRESHOLD"
echo " 共识跑结果 : $UNIDL_OUT"
echo " 报告输出   : $REPORT"
echo " 分步日志   : $LOGD/"
echo "=========================================================="

STEP_NO=0
declare -a OK_LIST=()
declare -a FAIL_LIST=()

run_step() {
  # run_step "名字" 命令...
  local name="$1"; shift
  STEP_NO=$((STEP_NO + 1))
  local log="$LOGD/$(printf '%02d' "$STEP_NO")_$(echo "$name" | tr ' /' '__').log"
  echo ""
  echo "▶ [$STEP_NO] $name"
  echo "   日志: $log"
  local t0=$SECONDS
  if "$@" >"$log" 2>&1; then
    echo "   ✅ 完成 ($((SECONDS - t0)) 秒)"
    OK_LIST+=("$name")
  else
    echo "   ❌ 失败 (退出码 $?) —— 详见 $log, 继续下一步"
    FAIL_LIST+=("$name")
  fi
}

# ---------- 0. 前置检查 ----------
echo ""
echo "▶ [0] 前置检查"
miss=0
[ -d "$MACREL_OUT" ] || { echo "   ❌ 缺 $MACREL_OUT (Macrel 单工具跑)"; miss=1; }
[ -d "$UNIDL_OUT" ]  || { echo "   ❌ 缺 $UNIDL_OUT (共识跑)"; miss=1; }
if [ "$miss" = 1 ]; then
  echo "   两个预测目录都得先跑完才能组装。退出。"
  exit 1
fi
echo "   ✅ Macrel 结果: $(find "$MACREL_OUT" -name '*_summary.json' | wc -l) 个分组"
echo "   ✅ 共识跑结果: $(find "$UNIDL_OUT"  -name '*_summary.json' | wc -l) 个分组"

# 探测本次数据里有哪些队列(用来决定 AD 关联跑哪些)
# 只认 Cohort* 开头的目录 —— 输出目录里还会有 .ckpt / AD_association
# 这类非队列目录, 不筛掉会被当成队列名传给 --cohort, 导致匹配不到任何分组而报错。
COHORTS=$(find "$MACREL_OUT" "$UNIDL_OUT" -mindepth 1 -maxdepth 1 -type d \
            -name 'Cohort*' -printf '%f\n' 2>/dev/null | sort -u)
if [ -z "$COHORTS" ]; then
  # 兜底: 目录名不含 Cohort 时, 从 summary.json 的 cohort 字段读
  COHORTS=$(find "$MACREL_OUT" "$UNIDL_OUT" -name '*_summary.json' \
              -exec python -c 'import json,sys;print(json.load(open(sys.argv[1]))["cohort"])' {} \; \
              2>/dev/null | sort -u)
fi
echo "   探测到队列:"
for c in $COHORTS; do echo "     - $c"; done

# ---------- 1. (可选) UniDL4BioPep 单工具 ----------
if [ "$RUN_UNIDL_ONLY" = "1" ]; then
  if [ -f "$UNIDL_ONLY_OUT/summary_all_groups.tsv" ] && [ "$FORCE" != "1" ]; then
    echo ""
    echo "▶ UniDL4BioPep 单工具已完成, 跳过 (FORCE=1 可重跑)"
  else
    run_step "UniDL4BioPep 单工具全量 (最慢的一步)" \
      env THRESHOLD="$THRESHOLD" MIN_CHARGE="$MIN_CHARGE" \
          OUT="$UNIDL_ONLY_OUT" ROOT="$ROOT" CHUNK=200000 ESM_BS=512 PRED_BS=8192 \
      bash "$ROOT/run_step2b_unidl_only.sh"
  fi
else
  echo ""
  echo "ℹ️ 未单独跑 UniDL4BioPep —— 它的判阳率/阈值敏感性从共识跑的 summary 里取,"
  echo "   不需要重跑 ESM-2 + CNN。"
  echo "   只有需要 UniDL4BioPep 【单独的候选肽清单】时才加 RUN_UNIDL_ONLY=1。"
fi

# ---------- 2. 组间统计 ----------
for d in "$MACREL_OUT" "$UNIDL_OUT" "$UNIDL_ONLY_OUT"; do
  [ -d "$d" ] || continue
  [ -f "$d/group_stats_rhoAMP.tsv" ] && [ "$FORCE" != "1" ] && {
    echo ""; echo "▶ 组间统计已存在, 跳过: $d"; continue; }
  run_step "组间统计 $(basename "$d")" \
    python -u "$ROOT/amp_group_stats.py" "$d"
done

# ---------- 3. AD 关联 (每个队列 × 每个结果目录) ----------
for d in "$MACREL_OUT" "$UNIDL_OUT" "$UNIDL_ONLY_OUT"; do
  [ -d "$d" ] || continue
  # 单工具跑必须 --min-tools 1, 共识跑用 2
  if [ -f "$d/.ckpt" ] || find "$d" -name '*_summary.json' -print -quit | \
       xargs -r grep -l '"single_tool_macrel_only"' >/dev/null 2>&1; then
    MT=1
  else
    MT=2
  fi
  for c in $COHORTS; do
    if [ -d "$d/AD_association" ] && [ "$FORCE" != "1" ]; then
      echo ""; echo "▶ AD 关联已存在, 跳过: $d (FORCE=1 可重跑)"; break
    fi
    run_step "AD 关联 $(basename "$d") / $c (min-tools=$MT)" \
      python -u "$ROOT/amp_ad_association.py" "$d" \
        --min-tools "$MT" --cohort "$c" \
        --cdhit-threads "$REQ" --cdhit-memory "$CDHIT_MEM"
  done
done

# ---------- 4. 对比表 ----------
CMP_ARGS=(--macrel-dir "$MACREL_OUT" --unidl-dir "$UNIDL_OUT"
          --consensus-dir "$UNIDL_OUT" --macrel-th "$CMP_MACREL_TH"
          --consensus-th "$THRESHOLD" --out "$REPORT/cmp")
if [ -d "$UNIDL_ONLY_OUT" ]; then
  CMP_ARGS=(--macrel-dir "$MACREL_OUT" --unidl-dir "$UNIDL_ONLY_OUT"
            --consensus-dir "$UNIDL_OUT" --macrel-th "$CMP_MACREL_TH"
            --consensus-th "$THRESHOLD" --out "$REPORT/cmp")
fi
mkdir -p "$REPORT"
run_step "Macrel vs UniDL4BioPep 对比表" \
  python -u "$ROOT/compare_macrel_vs_unidl.py" "${CMP_ARGS[@]}"

# ---------- 5. 组装总表 + 报告 ----------
ASM_ARGS=(--macrel-dir "$MACREL_OUT" --unidl-dir "$UNIDL_OUT"
          --out-dir "$REPORT")
[ -d "$UNIDL_ONLY_OUT" ] && ASM_ARGS[3]="$UNIDL_ONLY_OUT"
run_step "组装总表 + 报告" \
  python -u "$ROOT/assemble_results.py" "${ASM_ARGS[@]}"

# ---------- 汇总 ----------
echo ""
echo "=========================================================="
echo " 全部步骤结束"
echo "=========================================================="
echo " ✅ 成功 ${#OK_LIST[@]} 步:"
for s in "${OK_LIST[@]:-}"; do [ -n "$s" ] && echo "     - $s"; done
if [ "${#FAIL_LIST[@]}" -gt 0 ]; then
  echo " ❌ 失败 ${#FAIL_LIST[@]} 步:"
  for s in "${FAIL_LIST[@]}"; do echo "     - $s"; done
  echo "    逐个看 $LOGD/ 下对应日志。"
fi
echo ""
echo " 📁 总结果: $REPORT"
ls -1 "$REPORT" 2>/dev/null | sed 's/^/     /'
echo ""
echo " 先看: $REPORT/README_结果汇总.md"
echo "=========================================================="
[ "${#FAIL_LIST[@]}" -eq 0 ]
