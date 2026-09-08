#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AMP 谱的阿尔茨海默病(AD)关联分析

面向课题:「基于深度学习从阿尔茨海默症和健康人群肠道宏基因组中挖掘候选抗菌肽」
落脚点为【疾病关联】——AMP 谱在 NC → SCS → SCD → MCI → AD 进程中的变化规律。

============================================================
为什么必须先做家族级去冗余
============================================================
序列级计数会把同一 AMP 家族的多个变体重复计入, 系统性夸大组间差异。
AMPSphere (Cell 2024) 用 CD-HIT 在 **100% / 85% / 75%** 三个层级聚类,
每一层称为一个 SPHERE, 并在家族层面开展分析。本脚本沿用该做法, 默认在
**75% identity / 90% 覆盖度**(AMPSphere 参数)聚成家族后再做统计。

============================================================
分析内容
============================================================
① 家族级去冗余         CD-HIT (默认 c=0.75, aS=0.90), 对齐 AMPSphere
② 家族 × 分组 矩阵     每个家族在各阶段的检出数与 ρ(归一化密度)
③ 阶段趋势检验         Cochran-Armitage trend test
                       —— 利用 NC→SCS→SCD→MCI→AD 的【有序性】,
                          检验单调趋势, 结论强于两两比较
④ 阶段特异家族         AD 特异 / NC 特异 / 进程性富集家族
⑤ 理化性质漂移         各阶段候选肽的电荷/长度/pI 分布变化
⑥ 新颖性(可选)         与已知 AMP 库比对, 标注是否为新家族

============================================================
用法
============================================================
    # 1) 安装依赖
    conda install -c bioconda cd-hit diamond

    # 2) 运行(输入为 run_amp_sorf_cohorts.py 的输出目录)
    python amp_ad_association.py Predictions_AMP_run1/

    # 3) 加入新颖性比对
    python amp_ad_association.py Predictions_AMP_run1/ \
        --known-amp-db known_amps.fasta
"""

import os
import re
import sys
import glob
import json
import shutil
import argparse
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

STAGE_ORDER = ["NC", "SCS", "SCD", "MCI", "AD"]
STAGE_ALIAS = {"Healthy_NC": "NC", "Disease_AD": "AD"}


def parse_args():
    p = argparse.ArgumentParser(
        description="AMP 谱的 AD 关联分析",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("results_dir", help="run_amp_sorf_cohorts.py 输出目录")
    p.add_argument("--out", default=None, help="输出目录, 默认 <results>/AD_association")
    p.add_argument("--min-tools", type=int, default=2,
                   help="纳入分析所需的最少判阳工具数")
    p.add_argument("--cohort", default=None,
                   help="只分析指定队列(建议先用 Cohort3, 5 阶段最全)")

    c = p.add_argument_group("家族聚类 (CD-HIT, 对齐 AMPSphere)")
    c.add_argument("--cdhit-bin", default="cd-hit")
    c.add_argument("--identity", type=float, default=0.75,
                   help="序列一致性阈值 (AMPSphere: 100/85/75%%)")
    c.add_argument("--coverage", type=float, default=0.90,
                   help="较短序列的覆盖度阈值 (AMPSphere: 90%%)")
    c.add_argument("--cdhit-memory", type=int, default=8000)
    c.add_argument("--cdhit-threads", type=int, default=8)
    c.add_argument("--skip-cdhit", action="store_true",
                   help="跳过聚类(仅做序列级分析, 不推荐)")

    n = p.add_argument_group("新颖性比对 (可选)")
    n.add_argument("--known-amp-db", default=None,
                   help="已知 AMP 的 FASTA (APD3/dbAMP/DRAMP/AMPSphere 合并)")
    n.add_argument("--diamond-bin", default="diamond")
    n.add_argument("--novelty-identity", type=float, default=0.40,
                   help="低于此 identity 视为新颖 (Nat Biotechnol 2022 用 40%%)")

    p.add_argument("--min-family-count", type=int, default=5,
                   help="家族在全体中的最少出现次数(过滤单例噪声)")
    return p.parse_args()


# ==========================================
# 工具函数
# ==========================================

def stage_of(group_name):
    g = group_name.replace("Cohort", "")
    for s in ["Healthy_NC", "Disease_AD", "SCS", "SCD", "MCI", "AD", "NC"]:
        if re.search(rf"(^|_){re.escape(s)}($|_)", g):
            return STAGE_ALIAS.get(s, s)
    return g


def have(binary):
    return shutil.which(binary) is not None


# ==========================================
# 1. 载入候选肽
# ==========================================

def load_hits(results_dir, min_tools, cohort_filter=None):
    """读取所有分组的候选肽, 返回长表。"""
    frames = []
    meta = {}
    for sj in sorted(glob.glob(os.path.join(results_dir, "*", "*_summary.json"))):
        with open(sj) as fh:
            s = json.load(fh)
        if cohort_filter and cohort_filter.lower() not in s["cohort"].lower():
            continue
        group = s["group"]
        stage = stage_of(group)
        meta[(s["cohort"], group)] = {
            "stage": stage,
            "n_raw": s["funnel"]["n_raw"],
            "n_scored": s["funnel"]["n_scored"],
        }
        hf = os.path.join(os.path.dirname(sj), f"{group}_AMP_hits.csv")
        if not os.path.exists(hf):
            print(f"   ⚠️ 缺少 hits 文件: {hf}")
            continue
        df = pd.read_csv(hf)
        if "n_tools_positive" in df.columns:
            df = df[df["n_tools_positive"] >= min_tools]
        df["Cohort"] = s["cohort"]
        df["Group"] = group
        df["stage"] = stage
        frames.append(df)
        print(f"   {s['cohort']}/{group} ({stage}): {len(df):,} 条 (≥{min_tools} 票)")
    if not frames:
        raise SystemExit("❌ 未找到任何候选肽")
    return pd.concat(frames, ignore_index=True), meta


# ==========================================
# 2. CD-HIT 家族聚类
# ==========================================

def run_cdhit(seqs, out_dir, args):
    """
    对唯一序列做 CD-HIT 聚类, 返回 {sequence: family_id}。
    参数对齐 AMPSphere: 75% identity, 90% 覆盖度。
    """
    os.makedirs(out_dir, exist_ok=True)
    fa = os.path.join(out_dir, "candidates.faa")
    with open(fa, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">s{i}\n{s}\n")

    out = os.path.join(out_dir, f"cdhit_{int(args.identity*100)}")
    # 短序列需要相应调小 word size
    n = 5 if args.identity >= 0.7 else (4 if args.identity >= 0.6 else 3)
    cmd = [args.cdhit_bin, "-i", fa, "-o", out,
           "-c", str(args.identity), "-aS", str(args.coverage),
           "-n", str(n), "-M", str(args.cdhit_memory),
           "-T", str(args.cdhit_threads), "-d", "0", "-g", "1"]
    print(f"🧬 CD-HIT 聚类: identity={args.identity}, coverage={args.coverage}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   ⚠️ CD-HIT 失败: {r.stderr[-500:]}")
        return None

    # 解析 .clstr
    fam = {}
    clstr = out + ".clstr"
    cur = -1
    with open(clstr) as fh:
        for line in fh:
            if line.startswith(">Cluster"):
                cur = int(line.split()[1])
            else:
                m = re.search(r">s(\d+)\.\.\.", line)
                if m:
                    fam[int(m.group(1))] = cur
    print(f"   → {len(seqs):,} 条唯一序列 聚为 {len(set(fam.values())):,} 个家族")
    return {seqs[i]: f"FAM{f:07d}" for i, f in fam.items()}


# ==========================================
# 3. Cochran-Armitage 趋势检验
# ==========================================

def cochran_armitage(counts, totals, scores=None):
    """
    Cochran-Armitage trend test。
    检验比例是否随有序分组(NC→AD)单调变化。

    counts : 各组的阳性数
    totals : 各组的总数
    scores : 各组的序数得分, 默认 0,1,2,...

    返回 (z, p_two_sided, direction)
    """
    counts = np.asarray(counts, float)
    totals = np.asarray(totals, float)
    ok = totals > 0
    counts, totals = counts[ok], totals[ok]
    if len(counts) < 3:
        return np.nan, np.nan, ""
    if scores is None:
        scores = np.arange(len(counts), dtype=float)
    else:
        scores = np.asarray(scores, float)[ok]

    N = totals.sum()
    R = counts.sum()
    if N == 0 or R == 0 or R == N:
        return np.nan, np.nan, ""
    p_bar = R / N
    s_bar = (totals * scores).sum() / N

    num = (counts * (scores - s_bar)).sum()
    var = p_bar * (1 - p_bar) * (totals * (scores - s_bar) ** 2).sum()
    if var <= 0:
        return np.nan, np.nan, ""
    z = num / np.sqrt(var)
    p = 2 * (1 - _norm_cdf(abs(z)))
    return float(z), float(p), ("increasing" if z > 0 else "decreasing")


def _norm_cdf(x):
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x):
    sign = np.sign(x)
    x = abs(x)
    a1, a2, a3, a4, a5, pp = (0.254829592, -0.284496736, 1.421413741,
                              -1.453152027, 1.061405429, 0.3275911)
    t = 1.0 / (1.0 + pp * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR。"""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    q = np.full(len(p), np.nan)
    m = ok.sum()
    if m == 0:
        return q
    idx = np.argsort(p[ok])
    sp = p[ok][idx]
    qv = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, sp[i] * m / (i + 1))
        qv[i] = prev
    tmp = np.empty(m)
    tmp[idx] = qv
    q[ok] = tmp
    return q


# ==========================================
# 4. 新颖性比对
# ==========================================

def run_novelty(seqs, db_fasta, out_dir, args):
    """用 DIAMOND 比对已知 AMP 库, 返回 {seq: (best_identity, best_hit)}。"""
    if not have(args.diamond_bin):
        print("   ⚠️ 未安装 diamond, 跳过新颖性比对")
        return {}
    os.makedirs(out_dir, exist_ok=True)
    q = os.path.join(out_dir, "query.faa")
    with open(q, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">s{i}\n{s}\n")
    dbp = os.path.join(out_dir, "knownamp")
    subprocess.run([args.diamond_bin, "makedb", "--in", db_fasta,
                    "-d", dbp, "--quiet"], check=False)
    out = os.path.join(out_dir, "novelty.tsv")
    subprocess.run([args.diamond_bin, "blastp", "-q", q, "-d", dbp,
                    "-o", out, "--max-target-seqs", "1", "--quiet",
                    "--outfmt", "6", "qseqid", "sseqid", "pident",
                    "--sensitive", "-e", "10"], check=False)
    res = {}
    if os.path.exists(out):
        df = pd.read_csv(out, sep="\t", header=None,
                         names=["q", "s", "pident"])
        for _, r in df.iterrows():
            i = int(str(r["q"])[1:])
            if i < len(seqs):
                prev = res.get(seqs[i], (0.0, ""))
                if r["pident"] > prev[0]:
                    res[seqs[i]] = (float(r["pident"]), str(r["s"]))
    return res


# ==========================================
# 5. 主流程
# ==========================================

def main():
    args = parse_args()
    out_dir = args.out or os.path.join(args.results_dir, "AD_association")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("载入候选肽")
    print("=" * 72)
    hits, meta = load_hits(args.results_dir, args.min_tools, args.cohort)
    print(f"\n合计 {len(hits):,} 条 (含跨组重复)")

    uniq = hits["sequence"].drop_duplicates().tolist()
    print(f"唯一序列 {len(uniq):,} 条")

    # ---------- ① 家族聚类 ----------
    print("\n" + "=" * 72)
    print("① 家族级去冗余 (对齐 AMPSphere: CD-HIT 75% identity / 90% coverage)")
    print("   序列级计数会重复计入同家族变体, 系统性夸大组间差异")
    print("=" * 72)
    fam_map = None
    if not args.skip_cdhit:
        if have(args.cdhit_bin):
            fam_map = run_cdhit(uniq, os.path.join(out_dir, "cdhit"), args)
        else:
            print("   ⚠️ 未安装 cd-hit (conda install -c bioconda cd-hit)")
            print("      降级为序列级分析, 组间差异可能被夸大")
    if fam_map is None:
        fam_map = {s: f"SEQ{i:07d}" for i, s in enumerate(uniq)}
    hits["family"] = hits["sequence"].map(fam_map)

    # ---------- ② 家族 × 阶段 矩阵 ----------
    print("\n" + "=" * 72)
    print("② 家族 × 阶段 检出矩阵")
    print("=" * 72)
    stages = [s for s in STAGE_ORDER if s in set(hits["stage"])]
    if len(stages) < 2:
        raise SystemExit(f"❌ 有效阶段不足: {stages}")
    print(f"   阶段: {' → '.join(stages)}")

    # 各阶段的 smORF 总数(归一化分母)
    denom = defaultdict(int)
    for (_, _g), m in meta.items():
        if m["stage"] in stages:
            denom[m["stage"]] += m["n_raw"]

    # 家族在各阶段的【检出数】(丰度) —— 趋势检验的计数单元
    # 另存唯一序列数(多样性), 两者含义不同, 分别输出
    fam_tab = (hits.groupby(["family", "stage"]).size()
               .unstack(fill_value=0))
    fam_rich = (hits.drop_duplicates(["family", "stage", "sequence"])
                .groupby(["family", "stage"]).size().unstack(fill_value=0))
    for s in stages:
        if s not in fam_tab.columns:
            fam_tab[s] = 0
    fam_tab = fam_tab[stages]
    for s_ in stages:
        if s_ not in fam_rich.columns:
            fam_rich[s_] = 0
    fam_rich = fam_rich[stages]
    fam_tab["total"] = fam_tab.sum(axis=1)
    fam_tab = fam_tab[fam_tab["total"] >= args.min_family_count]
    print(f"   家族总数 {hits['family'].nunique():,}, "
          f"保留(≥{args.min_family_count} 次) {len(fam_tab):,}")

    # ---------- ③ 趋势检验 ----------
    print("\n" + "=" * 72)
    print("③ Cochran-Armitage 阶段趋势检验")
    print("   利用 NC→SCS→SCD→MCI→AD 的有序性检验【单调趋势】,")
    print("   结论强于单纯的 AD vs NC 两两比较。")
    print("=" * 72)

    totals = np.array([denom[s] for s in stages], float)
    rows = []
    for fam, r in fam_tab.iterrows():
        cnt = np.array([r[s] for s in stages], float)
        z, p, direction = cochran_armitage(cnt, totals)
        rows.append({
            "family": fam,
            **{f"n_{s}": int(r[s]) for s in stages},
            **{f"rho_{s}": (r[s] / denom[s] if denom[s] else 0.0)
               for s in stages},
            **{f"nseq_{s}": int(fam_rich.loc[fam, s])
               if fam in fam_rich.index else 0 for s in stages},
            "total": int(r["total"]),
            "CA_z": round(z, 3) if z == z else None,
            "CA_p": p, "trend": direction,
        })
    tdf = pd.DataFrame(rows)
    if tdf.empty:
        # 家族表为空(候选太少, 或 --min-family-count 过滤后无家族留存)。
        # pd.DataFrame([]) 没有列, 直接取 CA_p 会 KeyError。
        print(f"   ⚠️ 无家族通过 --min-family-count "
              f"{args.min_family_count} 的过滤, 跳过趋势检验。")
        print("      候选量小时这是正常现象; 可下调 --min-family-count "
              "(如 2 或 1)后重跑。")
        pd.DataFrame(columns=["family"] + [f"n_{s}" for s in stages]
                     + ["total", "CA_z", "CA_p", "CA_q_BH", "trend"]
                     ).to_csv(os.path.join(out_dir, "family_trend_test.tsv"),
                              sep="\t", index=False)
    else:
        tdf["CA_q_BH"] = bh_fdr(tdf["CA_p"].values)
        tdf = tdf.sort_values("CA_p")
        tdf["CA_p"] = tdf["CA_p"].map(lambda v: f"{v:.3g}" if v == v else None)
        tdf["CA_q_BH"] = tdf["CA_q_BH"].map(
            lambda v: f"{v:.3g}" if v == v else None)

    sig = tdf[pd.to_numeric(tdf["CA_q_BH"], errors="coerce") < 0.05] \
        if len(tdf) else tdf
    print(f"   显著趋势家族 (BH q<0.05): {len(sig):,} / {len(tdf):,}")
    if len(sig):
        inc = (sig["trend"] == "increasing").sum()
        print(f"     随疾病进程上升: {inc:,}")
        print(f"     随疾病进程下降: {len(sig)-inc:,}")
        print("\n   Top 10 显著趋势家族:")
        cols = ["family"] + [f"n_{s}" for s in stages] + \
               ["CA_z", "trend", "CA_q_BH"]
        print(sig[cols].head(10).to_string(index=False))
    tdf.to_csv(os.path.join(out_dir, "family_trend_test.tsv"),
               sep="\t", index=False)

    # ---------- ④ 阶段特异家族 ----------
    print("\n" + "=" * 72)
    print("④ 阶段特异家族")
    print("=" * 72)
    spec_rows = []
    for fam, r in fam_tab.iterrows():
        present = [s for s in stages if r[s] > 0]
        if len(present) == 1:
            spec_rows.append({"family": fam, "specific_to": present[0],
                              "count": int(r[present[0]])})
    sdf = pd.DataFrame(spec_rows)
    if len(sdf):
        print(sdf.groupby("specific_to")["family"].count()
              .reindex(stages).fillna(0).astype(int).to_string())
        ad_only = sdf[sdf["specific_to"] == "AD"].sort_values(
            "count", ascending=False)
        nc_only = sdf[sdf["specific_to"] == "NC"].sort_values(
            "count", ascending=False)
        print(f"\n   ★ AD 特异家族 {len(ad_only):,} 个 —— 本课题重点候选")
        print(f"   ★ NC 特异家族 {len(nc_only):,} 个 —— 健康相关/保护性候选")
        sdf.to_csv(os.path.join(out_dir, "stage_specific_families.tsv"),
                   sep="\t", index=False)
    else:
        print("   (无单阶段特异家族)")

    # ---------- ⑤ 理化性质漂移 ----------
    print("\n" + "=" * 72)
    print("⑤ 候选肽理化性质的阶段漂移")
    print("   AMPSphere 基准: 平均长 37 aa, 平均净电荷 +4.7, pI ~10.9")
    print("=" * 72)
    props = [c for c in ["length", "net_charge", "pI", "hydrophobic_frac"]
             if c in hits.columns]
    if props:
        pdf = (hits.drop_duplicates(["sequence", "stage"])
               .groupby("stage")[props].agg(["mean", "median"]).round(3))
        pdf = pdf.reindex([s for s in stages if s in pdf.index])
        print(pdf.to_string())
        pdf.to_csv(os.path.join(out_dir, "physchem_by_stage.tsv"), sep="\t")

    # ---------- ⑥ 新颖性 ----------
    if args.known_amp_db:
        print("\n" + "=" * 72)
        print("⑥ 新颖性比对")
        print(f"   identity < {args.novelty_identity*100:.0f}% 视为新颖")
        print("   (Nat Biotechnol 2022 以 <40% 同源性论证新颖性)")
        print("=" * 72)
        nov = run_novelty(uniq, args.known_amp_db,
                          os.path.join(out_dir, "novelty"), args)
        if nov:
            hits["best_known_identity"] = hits["sequence"].map(
                lambda s: nov.get(s, (0.0, ""))[0])
            hits["best_known_hit"] = hits["sequence"].map(
                lambda s: nov.get(s, (0.0, ""))[1])
            novel = hits[hits["best_known_identity"]
                         < args.novelty_identity * 100]
            n_uniq_novel = novel["sequence"].nunique()
            print(f"   新颖序列: {n_uniq_novel:,} / {len(uniq):,} "
                  f"({n_uniq_novel/len(uniq)*100:.1f}%)")
            print(f"   新颖家族: {novel['family'].nunique():,}")

    # ---------- 汇总 ----------
    keep = ["Cohort", "Group", "stage", "family", "id", "sequence"] + props
    keep += [c for c in hits.columns
             if c.endswith("_prob") or c == "n_tools_positive"
             or c.startswith("best_known") or c == "Macrel_hemolytic"]
    keep = [c for c in dict.fromkeys(keep) if c in hits.columns]
    hits[keep].to_csv(os.path.join(out_dir, "candidates_annotated.tsv.gz"),
                      sep="\t", index=False, compression="gzip")

    print("\n" + "=" * 72)
    print(f"✅ 输出目录: {out_dir}")
    print("   family_trend_test.tsv        阶段趋势检验(核心结果)")
    print("   stage_specific_families.tsv  AD/NC 特异家族")
    print("   physchem_by_stage.tsv        理化性质漂移")
    print("   candidates_annotated.tsv.gz  带家族注释的候选肽全表")
    print("=" * 72)
    print("\n⚠️ 解读要点:")
    print("  1. 以【家族】为统计单元, 不用序列绝对数(避免冗余夸大)。")
    print("  2. 趋势检验(CA)结论强于 AD vs NC 两两比较, 优先报告。")
    print("  3. 显著性需配合效应量: 样本量千万级时 p 值极易显著。")
    print("  4. Cohort1/2 与 Cohort3/4 样本重叠, 不是独立验证。")
    print("  5. 本分析为计算预测, 无实验验证, 结论应表述为'候选'。")


if __name__ == "__main__":
    main()
