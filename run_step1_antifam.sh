#!/usr/bin/env bash
# ==========================================================
# 第 1 步: smORF 真实性过滤 (AntiFam)
# ==========================================================
# 依据: 短肽基因预测未过滤时假阳性可达 61.2%
#       (MACREL, bioRxiv 2019.12.17.880385)
#       AntiFam = EBI 专门收录"伪基因预测产物"的 HMM 库
#       (Eberhardt et al., Database 2012)
#
# 用法:
#   bash run_step1_antifam.sh                 # 全部分组
#   bash run_step1_antifam.sh Cohort3         # 只处理 Cohort3
# ==========================================================
set -euo pipefail

ROOT="${ROOT:-$HOME/UniDL4BioPep-main}"
CATALOG="${CATALOG:-$ROOT/comparable_sorf_grouped_catalog}"
CLEAN="${CLEAN:-$ROOT/clean_catalog}"
ANTIFAM="${ANTIFAM:-$HOME/db/antifam/AntiFam.hmm}"
THREADS="${THREADS:-16}"
FILTER="${1:-}"

echo "=========================================================="
echo " 第 1 步: AntiFam smORF 真实性过滤"
echo " 输入: $CATALOG"
echo " 输出: $CLEAN"
echo " 线程: $THREADS"
echo "=========================================================="

command -v hmmsearch >/dev/null 2>&1 || {
  echo "❌ 未找到 hmmsearch，请先: conda install -c bioconda hmmer"; exit 1; }
[ -f "$ANTIFAM" ] || {
  echo "❌ 未找到 $ANTIFAM"
  echo "   mkdir -p ~/db/antifam && cd ~/db/antifam"
  echo "   wget https://ftp.ebi.ac.uk/pub/databases/Pfam/AntiFam/current/Antifam.tar.gz"
  echo "   tar xzf Antifam.tar.gz && hmmpress AntiFam.hmm"
  exit 1; }

LOG="$ROOT/antifam_filter.log"
: > "$LOG"

find "$CATALOG" -mindepth 2 -name '*.fa' | sort | while read -r fa; do
  sub=$(basename "$(dirname "$fa")")
  name=$(basename "$fa" .fa)

  if [ -n "$FILTER" ] && [[ "$sub" != *"$FILTER"* ]]; then
    continue
  fi

  outdir="$CLEAN/$sub"
  mkdir -p "$outdir"
  out="$outdir/$name.fa"
  rep="$outdir/$name.spurious.txt"

  if [ -s "$out" ]; then
    echo "⏭  已存在，跳过: $sub/$name"
    continue
  fi

  echo ""
  echo "▶ $sub/$name  ($(du -h "$fa" | cut -f1))"
  /usr/bin/time -f "   耗时 %E  峰值内存 %MKB" \
    python "$ROOT/smorf_authenticity.py" antifam \
      --input "$fa" \
      --antifam-db "$ANTIFAM" \
      --output "$out" \
      --report "$rep" \
      --threads "$THREADS" 2>&1 | tee -a "$LOG"
done

echo ""
echo "=========================================================="
echo "✅ 第 1 步完成"
echo "   过滤后目录: $CLEAN"
echo "   日志: $LOG"
echo ""
echo "下一步:"
echo "   bash run_step2_macrel_unidl.sh"
echo "=========================================================="
