#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smORF 真实性过滤 —— "这是不是一个真实的编码基因"

============================================================
为什么需要这一步
============================================================
短肽的基因预测假阳性率极高: 降低长度阈值而不加过滤时,
**可达 61.2% 的预测 smORF 是假阳性**
(MACREL, bioRxiv 2019.12.17.880385, 引 Sberro et al. 2019)。

"是不是 AMP" 与 "是不是真实编码基因" 是两道独立关卡。
Cell 2024 的 SEP 研究即在 AmPEP 活性预测之外, 另用 SmORFinder
确认候选是高置信编码基因 (DOI 10.1016/j.cell.2024.05.031)。

============================================================
两条路径
============================================================
【路径 A】SmORFinder —— 需要**核酸 contigs**
    Durrant & Bhatt, Cell Host Microbe 2020, DOI 10.1016/j.chom.2020.11.002
    组合 pHMM(4,500+ smORF 家族) + 两个深度学习模型(DSN1/DSN2)。
    作者建议: 用**宽松阈值但要求三个模型同时满足**来收紧候选
    (原文: "using lenient significance cutoffs that must be met by
     all three models is another good strategy")。
    默认判据: pHMM E-value < 1.0 或 DSN1 > 0.5 或 DSN2 > 0.5。
    注意: SmORFinder 是 Prodigal 之上的过滤层, 只接受核酸序列。

【路径 B】AntiFam —— 只需**蛋白序列**
    Eberhardt et al., Database 2012 (EBI)。
    专门收录"伪基因预测产物"(spurious ORF)的 HMM 库,
    被 Pfam/UniProt/GMSC 等用于清理错误的 ORF 预测。
    适用于本项目输入已是氨基酸 sORF 的情形。

============================================================
用法
============================================================
    # 安装
    pip install smorfinder && smorf          # 路径 A(需 contigs)
    conda install -c bioconda hmmer          # 路径 B
    # AntiFam:
    #   wget https://ftp.ebi.ac.uk/pub/databases/Pfam/AntiFam/current/Antifam.tar.gz
    #   mkdir -p antifam && tar xzf Antifam.tar.gz -C antifam
    #   hmmpress antifam/AntiFam.hmm

    # 路径 B: 蛋白 sORF 去伪
    python smorf_authenticity.py antifam \
        --input comparable_sorf_grouped_catalog/Cohort3_Full476_5Stage/Cohort3_AD.fa \
        --antifam-db antifam/AntiFam.hmm \
        --output Cohort3_AD.clean.fa

    # 路径 A: 核酸 contigs 真实性注释
    python smorf_authenticity.py smorfinder \
        --input contigs.fna --output smorf_out/
"""

import os
import re
import sys
import glob
import gzip
import shutil
import argparse
import subprocess
import tempfile

# ==========================================
# 通用
# ==========================================


def have(binary):
    return shutil.which(binary) is not None


def open_maybe_gz(path, mode="rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def iter_fasta(path):
    header, buf = None, []
    with open_maybe_gz(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(buf)
                header = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        if header is not None:
            yield header, "".join(buf)


# ==========================================
# 路径 B: AntiFam (蛋白序列)
# ==========================================

def run_antifam(input_fa, antifam_hmm, output_fa, threads=8,
                evalue=1e-3, report=None, chunk=500000):
    """
    用 hmmsearch 对 AntiFam 比对, 移除命中的 spurious ORF。
    返回 (n_total, n_spurious)。
    """
    if not have("hmmsearch"):
        sys.exit("❌ 未找到 hmmsearch (conda install -c bioconda hmmer)")
    if not os.path.exists(antifam_hmm):
        sys.exit(f"❌ 找不到 AntiFam HMM: {antifam_hmm}")

    spurious = set()
    n_total = 0
    tmpdir = tempfile.mkdtemp(prefix="antifam_")
    try:
        buf, part = [], 0
        def flush(buf, part):
            if not buf:
                return
            fa = os.path.join(tmpdir, f"p{part}.faa")
            with open(fa, "w") as fh:
                for h, s in buf:
                    fh.write(f">{h}\n{s}\n")
            tbl = os.path.join(tmpdir, f"p{part}.tbl")
            cmd = ["hmmsearch", "--cut_ga", "--cpu", str(threads),
                   "--tblout", tbl, "-o", os.devnull, antifam_hmm, fa]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                # --cut_ga 需要 HMM 带 GA 阈值, 回退到 E-value
                cmd = ["hmmsearch", "-E", str(evalue), "--cpu", str(threads),
                       "--tblout", tbl, "-o", os.devnull, antifam_hmm, fa]
                subprocess.run(cmd, capture_output=True, text=True)
            if os.path.exists(tbl):
                with open(tbl) as fh:
                    for line in fh:
                        if line.startswith("#"):
                            continue
                        f = line.split()
                        if f:
                            spurious.add(f[0])

        for h, s in iter_fasta(input_fa):
            n_total += 1
            buf.append((h.split()[0], s))
            if len(buf) >= chunk:
                flush(buf, part)
                buf, part = [], part + 1
        flush(buf, part)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    n_kept = 0
    with open(output_fa, "w") as out:
        for h, s in iter_fasta(input_fa):
            if h.split()[0] in spurious:
                continue
            out.write(f">{h}\n{s}\n")
            n_kept += 1

    rate = len(spurious) / n_total * 100 if n_total else 0
    print(f"   总数 {n_total:,} | 判为 spurious {len(spurious):,} ({rate:.2f}%) "
          f"| 保留 {n_kept:,}")
    if report:
        with open(report, "w") as fh:
            fh.write("spurious_id\n")
            for s in sorted(spurious):
                fh.write(s + "\n")
    return n_total, len(spurious)


# ==========================================
# 路径 A: SmORFinder (核酸 contigs)
# ==========================================

def run_smorfinder(input_fna, out_dir, mode="single", strict=False,
                   threads=8):
    """
    调用 SmORFinder。
    判据(Cell Host Microbe 2020):
      默认  —— pHMM E<1.0 或 DSN1>0.5 或 DSN2>0.5 (任一满足)
      strict —— 要求三者同时满足(作者推荐的收紧策略)
    """
    if not have("smorf"):
        sys.exit("❌ 未找到 smorf (pip install smorfinder && smorf)")
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["smorf", mode, input_fna, "-o", out_dir, "-f"]
    print(f"🧬 SmORFinder: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit("❌ SmORFinder 运行失败")

    tsvs = glob.glob(os.path.join(out_dir, "*.tsv"))
    if not tsvs:
        print(f"⚠️ 未找到输出表: {os.listdir(out_dir)}")
        return None
    print(f"✅ SmORFinder 输出: {tsvs}")

    if strict:
        try:
            import pandas as pd
        except ImportError:
            print("⚠️ 需要 pandas 才能应用 strict 过滤")
            return tsvs[0]
        df = pd.read_csv(tsvs[0], sep="\t")
        cols = {c.lower(): c for c in df.columns}
        e = cols.get("pfam_evalue") or cols.get("hmm_evalue") or cols.get("evalue")
        d1 = cols.get("dsn1_prob") or cols.get("dsn1")
        d2 = cols.get("dsn2_prob") or cols.get("dsn2")
        if e and d1 and d2:
            keep = (df[e].astype(float) < 1.0) & \
                   (df[d1].astype(float) > 0.5) & \
                   (df[d2].astype(float) > 0.5)
            out = os.path.join(out_dir, "smorf_strict.tsv")
            df[keep].to_csv(out, sep="\t", index=False)
            print(f"   strict(三模型同时满足): {keep.sum():,} / {len(df):,} "
                  f"→ {out}")
            return out
        print(f"⚠️ 未识别出三模型列, 现有列: {list(df.columns)}")
    return tsvs[0]


# ==========================================
# CLI
# ==========================================

def main():
    ap = argparse.ArgumentParser(
        description="smORF 真实性过滤",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("antifam", help="蛋白序列去伪 (AntiFam + hmmsearch)")
    a.add_argument("--input", required=True, help="蛋白 FASTA (可 .gz)")
    a.add_argument("--antifam-db", required=True, help="AntiFam.hmm (已 hmmpress)")
    a.add_argument("--output", required=True, help="过滤后 FASTA")
    a.add_argument("--report", default=None, help="被剔除 ID 清单")
    a.add_argument("--threads", type=int, default=8)
    a.add_argument("--evalue", type=float, default=1e-3)

    b = sub.add_parser("smorfinder", help="核酸 contigs 真实性注释")
    b.add_argument("--input", required=True, help="contigs FASTA (核酸)")
    b.add_argument("--output", required=True, help="输出目录")
    b.add_argument("--mode", default="single",
                   choices=["single", "meta"],
                   help="single=单基因组, meta=宏基因组")
    b.add_argument("--strict", action="store_true",
                   help="要求 pHMM/DSN1/DSN2 三模型同时满足")
    b.add_argument("--threads", type=int, default=8)

    args = ap.parse_args()

    if args.cmd == "antifam":
        print("=" * 70)
        print("smORF 真实性过滤 — AntiFam (蛋白序列路径)")
        print("  AntiFam: 专门收录伪基因预测产物的 HMM 库 (EBI, Database 2012)")
        print("  背景: 短肽基因预测未过滤时假阳性可达 61.2% (MACREL 2019)")
        print("=" * 70)
        run_antifam(args.input, args.antifam_db, args.output,
                    args.threads, args.evalue, args.report)
        print(f"✅ 输出: {args.output}")

    else:
        print("=" * 70)
        print("smORF 真实性过滤 — SmORFinder (核酸 contigs 路径)")
        print("  Durrant & Bhatt, Cell Host Microbe 2020")
        print("  pHMM + DSN1 + DSN2; 预测结果富集 Ribo-seq 翻译信号")
        print("=" * 70)
        run_smorfinder(args.input, args.output, args.mode,
                       args.strict, args.threads)


if __name__ == "__main__":
    main()
