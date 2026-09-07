#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AMP 组间比较统计 (NC → SCS → SCD → MCI → AD)

解决"绝对数量不可比"的问题。三层做法, 均有文献依据:

  ① 组内归一化 —— AMP density (ρAMP)
     用 "c_AMP 数 / smORF 总数" 而非绝对条数。
     AMPSphere (Cell 2024) 即以 ρAMP(每个基因组/样本的 AMP 密度)
     做跨物种、跨生境比较, 正是为了消除采样深度不均。
     宏基因组数据是 compositional 的, 直接比绝对计数会有偏
     [Gloor et al.; Nearing et al. Nat Commun 2022]。

  ② 跨工具不可比 —— 只做工具内比较 + 秩相关一致性检验
     同一批 51 条高置信肽, AxPEP 报 70.6%、AMP Scanner 45.1%、
     Macrel 9.8% (Springer Microb Ecol 2025) —— 阈值尺度完全不同。
     因此: 每个工具独立算组间趋势, 再用 Spearman 检验各工具
     给出的组间排序是否一致(方向一致 = 结论稳健)。

  ③ 分布层面比较 —— 不依赖阈值
     除比例外, 同时比较概率分布(均值/分位数/直方图 KS 检验),
     避免结论被单一阈值绑架。

用法:
    python amp_group_stats.py Predictions_AMP_run1/
    python amp_group_stats.py Predictions_AMP_run1/ --order NC SCS SCD MCI AD
"""

import os
import re
import json
import glob
import argparse
import itertools

import numpy as np
import pandas as pd

DEFAULT_ORDER = ["NC", "SCS", "SCD", "MCI", "AD",
                 "Healthy_NC", "Disease_AD"]


def parse_args():
    p = argparse.ArgumentParser(description="AMP 组间比较统计")
    p.add_argument("results_dir", help="run_amp_sorf_cohorts.py 的输出目录")
    p.add_argument("--order", nargs="*", default=None,
                   help="疾病阶段顺序, 默认 NC SCS SCD MCI AD")
    p.add_argument("--out", default=None, help="输出前缀")
    return p.parse_args()


def load_summaries(results_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*", "*_summary.json"))):
        with open(f) as fh:
            s = json.load(fh)
        base = {
            "Cohort": s["cohort"],
            "Group": s["group"],
            "stage": stage_of(s["group"]),
            "N_raw": s["funnel"]["n_raw"],
            "N_after_len": s["funnel"]["n_after_len"],
            "N_scored": s["funnel"]["n_scored"],
            "mean_len": s["input_properties"]["mean_length"],
            "mean_charge": s["input_properties"]["mean_net_charge"],
            "mean_pI": s["input_properties"]["mean_pI"],
            "tools": ",".join(s["tools"]),
            "lit_verdict": s["literature_benchmark"]["verdict"],
            "_dir": os.path.dirname(f),
        }
        for t, v in s["n_hits"].items():
            base[f"N_{t}"] = v
            # ρAMP: 归一化到原始 smORF 总数(消除测序深度差异)
            base[f"rhoAMP_{t}"] = v / max(s["funnel"]["n_raw"], 1)
        for k, v in s["n_consensus"].items():
            base[f"N_consensus{k}"] = v
            base[f"rhoAMP_consensus{k}"] = v / max(s["funnel"]["n_raw"], 1)
        for t, d in s["prob_distribution"].items():
            if d:
                base[f"{t}_mean_prob"] = d["mean_prob"]
                for q, val in d["quantiles"].items():
                    base[f"{t}_{q}"] = val
        rows.append(base)
    return pd.DataFrame(rows)


def stage_of(group_name):
    """从组名提取疾病阶段。"""
    g = group_name.replace("Cohort", "")
    for s in ["Healthy_NC", "Disease_AD", "SCS", "SCD", "MCI", "AD", "NC"]:
        if re.search(rf"(^|_){re.escape(s)}($|_)", g):
            return s
    return g


def norm_stage(s):
    """把 Healthy_NC / Disease_AD 归并到 NC / AD 以便排序。"""
    return {"Healthy_NC": "NC", "Disease_AD": "AD"}.get(s, s)


# ==========================================
# 统计检验(纯 numpy/scipy-free 实现)
# ==========================================

def two_proportion_ztest(x1, n1, x2, n2):
    """两比例 z 检验, 返回 (z, p, 双侧)。"""
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    return z, 2 * (1 - _norm_cdf(abs(z)))


def _norm_cdf(x):
    """标准正态 CDF(Abramowitz-Stegun 近似)。"""
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x):
    sign = np.sign(x)
    x = abs(x)
    a1, a2, a3, a4, a5, pp = (0.254829592, -0.284496736, 1.421413741,
                              -1.453152027, 1.061405429, 0.3275911)
    t = 1.0 / (1.0 + pp * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def spearman(a, b):
    """Spearman 秩相关。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return np.nan
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else np.nan


def _rank(x):
    order = np.argsort(x)
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1)
    return r


def hist_ks(h1, h2):
    """基于直方图的 KS 统计量 + 近似 p 值。"""
    n1, n2 = h1.sum(), h2.sum()
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    c1, c2 = np.cumsum(h1) / n1, np.cumsum(h2) / n2
    d = float(np.max(np.abs(c1 - c2)))
    ne = np.sqrt(n1 * n2 / (n1 + n2))
    lam = (ne + 0.12 + 0.11 / ne) * d
    p = 2 * sum((-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam)
                for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


# ==========================================
# 主流程
# ==========================================

def main():
    args = parse_args()
    order = args.order or DEFAULT_ORDER
    out_prefix = args.out or os.path.join(args.results_dir, "group_stats")

    df = load_summaries(args.results_dir)
    if df.empty:
        raise SystemExit(f"❌ 未找到 summary.json: {args.results_dir}")

    df["stage_norm"] = df["stage"].map(norm_stage)
    rank_map = {s: i for i, s in enumerate(
        [s for s in ["NC", "SCS", "SCD", "MCI", "AD"] if s in order]
        or ["NC", "SCS", "SCD", "MCI", "AD"])}
    df["stage_rank"] = df["stage_norm"].map(rank_map)
    df = df.sort_values(["Cohort", "stage_rank"])

    tools = sorted({c[len("rhoAMP_"):] for c in df.columns
                    if c.startswith("rhoAMP_")})

    print("=" * 72)
    print("① 组内归一化: AMP density (ρAMP = c_AMP 数 / smORF 总数)")
    print("   依据: AMPSphere (Cell 2024) 用 ρAMP 做跨样本比较;")
    print("         宏基因组为 compositional 数据, 绝对计数有偏。")
    print("=" * 72)
    show = ["Cohort", "Group", "stage_norm", "N_raw", "N_scored"] + \
           [f"rhoAMP_{t}" for t in tools if not t.startswith("consensus")]
    show = [c for c in show if c in df.columns]
    print(df[show].to_string(index=False))

    # ---------- ② 跨工具一致性 ----------
    print("\n" + "=" * 72)
    print("② 跨工具一致性检验 (Spearman, 组间排序是否方向一致)")
    print("   跨工具绝对数量不可比 —— 同一批肽各工具命中率可差 7 倍")
    print("   (Macrel 9.8% vs AxPEP 70.6%, Microb Ecol 2025)。")
    print("   因此只比较各工具给出的【组间排序】是否一致。")
    print("=" * 72)
    base_tools = [t for t in tools if not t.startswith("consensus")]
    con_rows = []
    for cohort, sub in df.groupby("Cohort"):
        if len(sub) < 3:
            continue
        for t1, t2 in itertools.combinations(base_tools, 2):
            c1, c2 = f"rhoAMP_{t1}", f"rhoAMP_{t2}"
            if c1 in sub and c2 in sub:
                r = spearman(sub[c1], sub[c2])
                con_rows.append({"Cohort": cohort, "tool_1": t1,
                                 "tool_2": t2, "spearman_rho": round(r, 4)
                                 if r == r else None})
    if con_rows:
        cdf = pd.DataFrame(con_rows)
        print(cdf.to_string(index=False))
        print("\n   解读: rho > 0.8 = 各工具组间趋势高度一致, 结论稳健;")
        print("         rho < 0.5 = 工具间分歧大, 不应下结论。")
        cdf.to_csv(f"{out_prefix}_tool_concordance.tsv", sep="\t", index=False)
    else:
        cdf = pd.DataFrame()
        print("   (工具数或组数不足, 跳过)")

    # ---------- ③ 组间比例检验 ----------
    print("\n" + "=" * 72)
    print("③ 组间 ρAMP 差异检验 (两比例 z 检验, 以各组 NC 为参照)")
    print("=" * 72)
    test_rows = []
    for cohort, sub in df.groupby("Cohort"):
        ref = sub[sub["stage_norm"] == "NC"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        for _, r in sub.iterrows():
            if r["Group"] == ref["Group"]:
                continue
            for t in tools:
                cn, cr = f"N_{t}", f"rhoAMP_{t}"
                if cn not in df.columns:
                    cn = f"N_consensus{t[len('consensus'):]}" \
                        if t.startswith("consensus") else cn
                if cn not in df.columns:
                    continue
                z, p = two_proportion_ztest(
                    r[cn], r["N_raw"], ref[cn], ref["N_raw"])
                fold = (r[cr] / ref[cr]) if ref[cr] else np.nan
                test_rows.append({
                    "Cohort": cohort, "Group": r["Group"],
                    "vs": ref["Group"], "tool": t,
                    "rhoAMP": round(r[cr], 8),
                    "rhoAMP_ref": round(ref[cr], 8),
                    "fold_change": round(fold, 3) if fold == fold else None,
                    "z": round(z, 3) if z == z else None,
                    "p_value": f"{p:.3g}" if p == p else None,
                })
    if test_rows:
        tdf = pd.DataFrame(test_rows)
        # Benjamini-Hochberg FDR
        pv = pd.to_numeric(tdf["p_value"], errors="coerce").values
        ok = ~np.isnan(pv)
        q = np.full(len(pv), np.nan)
        if ok.sum():
            idx = np.argsort(pv[ok])
            m = ok.sum()
            qv = np.empty(m)
            sp = pv[ok][idx]
            prev = 1.0
            for i in range(m - 1, -1, -1):
                prev = min(prev, sp[i] * m / (i + 1))
                qv[i] = prev
            tmp = np.full(m, np.nan)
            tmp[idx] = qv
            q[ok] = tmp
        tdf["q_value_BH"] = [f"{v:.3g}" if v == v else None for v in q]
        print(tdf.to_string(index=False))
        tdf.to_csv(f"{out_prefix}_group_tests.tsv", sep="\t", index=False)
    else:
        print("   (未找到 NC 参照组, 跳过)")

    # ---------- ④ 概率分布 KS ----------
    print("\n" + "=" * 72)
    print("④ 概率分布比较 (KS 检验, 不依赖阈值)")
    print("=" * 72)
    ks_rows = []
    for cohort, sub in df.groupby("Cohort"):
        ref = sub[sub["stage_norm"] == "NC"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        for t in base_tools:
            rh = os.path.join(ref["_dir"], f"{ref['Group']}_{t}_prob_hist.npy")
            if not os.path.exists(rh):
                continue
            h_ref = np.load(rh)
            for _, r in sub.iterrows():
                if r["Group"] == ref["Group"]:
                    continue
                hp = os.path.join(r["_dir"], f"{r['Group']}_{t}_prob_hist.npy")
                if not os.path.exists(hp):
                    continue
                d, p = hist_ks(np.load(hp), h_ref)
                ks_rows.append({"Cohort": cohort, "Group": r["Group"],
                                "vs": ref["Group"], "tool": t,
                                "KS_D": round(d, 4),
                                "p_value": f"{p:.3g}"})
    if ks_rows:
        kdf = pd.DataFrame(ks_rows)
        print(kdf.to_string(index=False))
        kdf.to_csv(f"{out_prefix}_ks_tests.tsv", sep="\t", index=False)
    else:
        print("   (未找到直方图文件, 跳过)")

    # ---------- 保存 ----------
    df.drop(columns=["_dir"]).to_csv(f"{out_prefix}_rhoAMP.tsv",
                                     sep="\t", index=False)
    print("\n" + "=" * 72)
    print(f"✅ 输出: {out_prefix}_rhoAMP.tsv")
    if len(con_rows):
        print(f"         {out_prefix}_tool_concordance.tsv")
    if test_rows:
        print(f"         {out_prefix}_group_tests.tsv")
    if ks_rows:
        print(f"         {out_prefix}_ks_tests.tsv")
    print("=" * 72)
    print("\n⚠️ 报告要点:")
    print("  1. 只报告 ρAMP(归一化密度), 不报告绝对条数。")
    print("  2. 固定同一套工具+阈值; 换工具需重跑全部组。")
    print("  3. 结论需在 ≥2 个工具上方向一致(见 Spearman 表)。")
    print("  4. Cohort1/2 与 Cohort3/4 样本重叠, 不是独立重复。")


if __name__ == "__main__":
    main()
