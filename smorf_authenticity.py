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
import time
import glob
import gzip
import shutil
import argparse
import concurrent.futures as cf
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

def _dedup_pass(input_fa, tmpdir):
    """
    第 1 遍: 按序列去重。
    宏基因组 sORF 目录中同一短肽常被大量样本重复收录, 去重后送检序列
    通常只剩原来的一小部分, 这是最有效的加速手段。
    返回 (uniq_fa 路径, n_total, n_uniq)。
    """
    uniq_fa = os.path.join(tmpdir, "uniq.faa")
    seen = {}
    n_total = 0
    t0 = time.time()
    fsize = os.path.getsize(input_fa)
    nxt = 2_000_000
    print(f"   [1/3] 去重中 (读取 {fsize/1024**3:.2f} GB, 单线程, 请耐心)...",
          flush=True)
    with open(uniq_fa, "w") as out:
        for h, seq in iter_fasta(input_fa):
            n_total += 1
            if seq not in seen:
                idx = len(seen)
                seen[seq] = idx
                out.write(f">u{idx}\n{seq}\n")
            if n_total >= nxt:
                el = time.time() - t0
                print(f"         已读 {n_total/1e6:.0f}M 条 | 唯一 "
                      f"{len(seen)/1e6:.1f}M ({len(seen)/n_total*100:.0f}%) "
                      f"| {el/60:.1f} 分 | {n_total/el/1000:.0f}k 条/秒",
                      flush=True)
                nxt += 2_000_000
    print(f"   [1/3] 去重完成: {n_total:,} 条, 耗时 {(time.time()-t0)/60:.1f} 分",
          flush=True)
    return uniq_fa, n_total, seen


def run_antifam(input_fa, antifam_hmm, output_fa, threads=8,
                evalue=1e-3, report=None, chunk=500000,
                hmmsearch_bin="hmmsearch", workers=0, dedup=True,
                keep_tmp=False):
    """
    用 hmmsearch 对 AntiFam 比对, 移除命中的 spurious ORF。

    性能说明
    --------
    hmmsearch 的 --cpu 只在 MSV 阶段并行, 超过 ~8 线程几乎不再加速。
    真正有效的加速是:
      1) 序列去重 (dedup)  —— 宏基因组目录冗余极高, 常能省掉大部分工作量
      2) 分块 + 多进程并行 (workers) —— 每个进程独立跑 hmmsearch, 近线性扩展
      3) 单遍写出 (不再二次读取输入)

    返回 (n_total, n_spurious)。
    """
    hs = hmmsearch_bin or "hmmsearch"
    if os.path.isabs(hs):
        if not os.access(hs, os.X_OK):
            sys.exit(f"❌ hmmsearch 不可执行: {hs}")
    elif not have(hs):
        sys.exit("❌ 未找到 hmmsearch，先运行 bash install_step1_deps.sh")
    if not os.path.exists(antifam_hmm):
        sys.exit(f"❌ 找不到 AntiFam HMM: {antifam_hmm}")
    if not os.path.exists(antifam_hmm + ".h3i"):
        print(f"   ⚠️ {antifam_hmm} 未 hmmpress, 建议先建索引以加速")

    ncpu = os.cpu_count() or 8
    if workers <= 0:
        # hmmsearch 单进程超过 ~4 线程收益很低, 用多进程铺满 CPU
        workers = max(1, min(16, ncpu // 4))
    per_cpu = max(1, min(4, ncpu // workers))

    tmpdir = tempfile.mkdtemp(prefix="antifam_")
    t0 = time.time()
    try:
        # ---------- 第 1 遍: 去重 ----------
        if dedup:
            uniq_fa, n_total, seen = _dedup_pass(input_fa, tmpdir)
            n_uniq = len(seen)
            saved = (1 - n_uniq / n_total) * 100 if n_total else 0
            print(f"   去重: {n_total:,} → {n_uniq:,} 条唯一序列 "
                  f"(省去 {saved:.1f}% 的比对量)")
            search_fa = uniq_fa
            n_search = n_uniq
        else:
            seen = None
            n_total = sum(1 for _ in iter_fasta(input_fa))
            search_fa = input_fa
            n_search = n_total

        # ---------- 切分 ----------
        parts, buf, part = [], [], 0
        for h, seq in iter_fasta(search_fa):
            buf.append((h.split()[0], seq))
            if len(buf) >= chunk:
                fp = os.path.join(tmpdir, f"p{part}.faa")
                with open(fp, "w") as fh:
                    for a, b in buf:
                        fh.write(f">{a}\n{b}\n")
                parts.append(fp)
                buf, part = [], part + 1
        if buf:
            fp = os.path.join(tmpdir, f"p{part}.faa")
            with open(fp, "w") as fh:
                for a, b in buf:
                    fh.write(f">{a}\n{b}\n")
            parts.append(fp)
        del buf

        print(f"   [2/3] 比对: {n_search:,} 条 / {len(parts)} 块 "
              f"/ {workers} 进程 × {per_cpu} 线程", flush=True)

        # ---------- 并行 hmmsearch ----------
        def _one(fp):
            tbl = fp + ".tbl"
            base = [hs, "--cpu", str(per_cpu), "--noali",
                    "--tblout", tbl, "-o", os.devnull]
            r = subprocess.run(base + ["--cut_ga", antifam_hmm, fp],
                               capture_output=True, text=True)
            if r.returncode != 0:
                r = subprocess.run(base + ["-E", str(evalue), antifam_hmm, fp],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"hmmsearch 失败: {r.stderr[:400]}")
            hit = set()
            if os.path.exists(tbl):
                with open(tbl) as fh:
                    for line in fh:
                        if line.startswith("#"):
                            continue
                        f = line.split()
                        if f:
                            hit.add(f[0])
                os.remove(tbl)
            os.remove(fp)
            return hit

        spurious = set()
        done = 0
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for hit in ex.map(_one, parts):
                spurious |= hit
                done += 1
                el = time.time() - t0
                rate = done / len(parts)
                eta = el / rate - el if rate > 0 else 0
                print(f"      [{done}/{len(parts)}] 累计 spurious "
                      f"{len(spurious):,} | 已用 {el/60:.1f} 分 "
                      f"| 预计剩余 {eta/60:.1f} 分", flush=True)

        # ---------- 写出 ----------
        if dedup:
            # spurious 是 u<idx> 形式, 映射回序列
            bad_idx = {int(x[1:]) for x in spurious if x.startswith("u")}
            bad_seq = {sq for sq, i in seen.items() if i in bad_idx}
            n_spur_uniq = len(bad_seq)
            del seen
            print(f"   [3/3] 写出过滤后 FASTA ...", flush=True)
            n_kept = 0
            n_removed = 0
            with open(output_fa, "w") as out:
                for h, sq in iter_fasta(input_fa):
                    if sq in bad_seq:
                        n_removed += 1
                        continue
                    out.write(f">{h}\n{sq}\n")
                    n_kept += 1
            rate = n_removed / n_total * 100 if n_total else 0
            print(f"   总数 {n_total:,} | spurious {n_removed:,} ({rate:.2f}%) "
                  f"[{n_spur_uniq:,} 条唯一序列] | 保留 {n_kept:,}")
            if report:
                with open(report, "w") as fh:
                    fh.write("spurious_sequence\n")
                    for sq in sorted(bad_seq):
                        fh.write(sq + "\n")
            return n_total, n_removed
        else:
            n_kept = 0
            with open(output_fa, "w") as out:
                for h, sq in iter_fasta(input_fa):
                    if h.split()[0] in spurious:
                        continue
                    out.write(f">{h}\n{sq}\n")
                    n_kept += 1
            rate = len(spurious) / n_total * 100 if n_total else 0
            print(f"   总数 {n_total:,} | spurious {len(spurious):,} "
                  f"({rate:.2f}%) | 保留 {n_kept:,}")
            if report:
                with open(report, "w") as fh:
                    fh.write("spurious_id\n")
                    for x in sorted(spurious):
                        fh.write(x + "\n")
            return n_total, len(spurious)
    finally:
        if not keep_tmp:
            shutil.rmtree(tmpdir, ignore_errors=True)


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
    a.add_argument("--chunk", type=int, default=100000,
                   help="每块序列数(越小并行粒度越细)")
    a.add_argument("--workers", type=int, default=0,
                   help="并行 hmmsearch 进程数(0=按 CPU 自动)。"
                        "注意: hmmsearch --cpu 超过 ~8 几乎不再加速, "
                        "多进程才能铺满 CPU")
    a.add_argument("--no-dedup", action="store_true",
                   help="关闭序列去重(默认开启; 宏基因组冗余高, 去重能大幅提速)")
    a.add_argument("--hmmsearch-bin", default="hmmsearch",
                   help="hmmsearch 可执行文件路径(环境按 -p 创建时需指定绝对路径)")
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
        run_antifam(input_fa=args.input, antifam_hmm=args.antifam_db,
                    output_fa=args.output, threads=args.threads,
                    evalue=args.evalue, report=args.report,
                    chunk=args.chunk, hmmsearch_bin=args.hmmsearch_bin,
                    workers=args.workers, dedup=not args.no_dedup)
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
