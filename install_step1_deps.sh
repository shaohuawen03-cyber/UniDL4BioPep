#!/usr/bin/env bash
# ==========================================================
# 第 1 步依赖安装: hmmer + AntiFam 数据库
# ==========================================================
# 自适应处理:
#   - 环境用 -p 路径创建、conda 按名字找不到 → 用 $CONDA_PREFIX 按路径装
#   - 优先 mamba > micromamba > conda
#
# 用法:
#   bash install_step1_deps.sh              # 装进当前激活的环境
#   PREFIX=/path/to/env bash install_step1_deps.sh   # 指定环境路径
# ==========================================================
set -euo pipefail

# ---------- 1. 定位目标环境 ----------
PREFIX="${PREFIX:-${CONDA_PREFIX:-}}"
if [ -z "$PREFIX" ]; then
  echo "❌ 未检测到激活的 conda 环境，且未指定 PREFIX"
  echo "   请先激活环境，或: PREFIX=/home/wsh/miniconda3/envs/xxx bash $0"
  exit 1
fi
if [ ! -d "$PREFIX" ]; then
  echo "❌ 环境路径不存在: $PREFIX"; exit 1
fi
echo "🎯 目标环境: $PREFIX"
echo "   Python  : $("$PREFIX/bin/python" -V 2>&1 || echo '(无)')"

# ---------- 2. 选择包管理器 ----------
if command -v mamba >/dev/null 2>&1; then
  PM="mamba"
elif command -v micromamba >/dev/null 2>&1; then
  PM="micromamba"
else
  echo ""
  echo "⚠️  未找到 mamba，正在装入 base（一次性，之后所有环境都快）..."
  conda install -n base -c conda-forge mamba -y
  # 重新加载 conda 的 shell 函数以便识别 mamba
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [ -f "$(conda info --base)/etc/profile.d/mamba.sh" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/mamba.sh"
  fi
  PM=$(command -v mamba >/dev/null 2>&1 && echo mamba || echo conda)
fi
echo "📦 包管理器: $PM"

# ---------- 3. 安装 hmmer（按路径，不按名字） ----------
echo ""
echo "▶ 安装 hmmer / cd-hit / diamond ..."
$PM install -p "$PREFIX" -c conda-forge -c bioconda \
    hmmer cd-hit diamond -y

# ---------- 4. 校验 ----------
echo ""
if [ -x "$PREFIX/bin/hmmsearch" ]; then
  echo "✅ hmmsearch: $("$PREFIX/bin/hmmsearch" -h | head -2 | tail -1)"
else
  echo "❌ hmmsearch 安装失败"; exit 1
fi

# ---------- 5. 下载 AntiFam ----------
DB="${DB:-$HOME/db/antifam}"
echo ""
echo "▶ 准备 AntiFam 数据库 → $DB"
mkdir -p "$DB" && cd "$DB"

if [ -f "$DB/AntiFam.hmm.h3i" ]; then
  echo "✅ AntiFam 已就绪（已 hmmpress），跳过下载"
else
  if [ ! -f Antifam.tar.gz ]; then
    echo "   下载中..."
    wget -c --tries=3 \
      https://ftp.ebi.ac.uk/pub/databases/Pfam/AntiFam/current/Antifam.tar.gz
  fi
  tar xzf Antifam.tar.gz
  # 发行包里可能是分库(AntiFam_Bacteria.hmm 等)，合并成单一 AntiFam.hmm
  if [ ! -f AntiFam.hmm ]; then
    echo "   合并分库为 AntiFam.hmm ..."
    cat AntiFam_*.hmm > AntiFam.hmm
  fi
  echo "   hmmpress 建索引..."
  "$PREFIX/bin/hmmpress" -f AntiFam.hmm
fi

NMODEL=$(grep -c '^NAME' AntiFam.hmm || echo '?')
echo "✅ AntiFam 就绪: $DB/AntiFam.hmm  (共 $NMODEL 个 HMM 模型)"

# ---------- 6. 定位 macrel ----------
echo ""
MACREL=$(ls "$(dirname "$PREFIX")"/*/bin/macrel 2>/dev/null | head -1 || true)
if [ -n "$MACREL" ]; then
  echo "✅ 找到 macrel: $MACREL"
  echo ""
  echo "把这行加到 ~/.bashrc 省得每次输:"
  echo "   export MACREL_BIN=$MACREL"
else
  echo "⚠️  未自动找到 macrel，请手动: conda activate env_macrel && which macrel"
fi

echo ""
echo "=========================================================="
echo "✅ 第 1 步依赖安装完成"
echo ""
echo "下一步:"
echo "   cd ~/UniDL4BioPep-main"
echo "   ANTIFAM=$DB/AntiFam.hmm bash run_step1_antifam.sh Cohort2"
echo "=========================================================="
