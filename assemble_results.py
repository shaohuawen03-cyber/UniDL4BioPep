#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 Macrel 单工具跑 / UniDL4BioPep(共识跑里的那一栏) / 组间统计
组装成一张总表 + 一份可直接引用的报告。

只做汇总, 不做任何预测 —— 读现成的 summary.json / TSV, 秒级完成。

用法:
    python assemble_results.py \
        --macrel-dir Predictions_Macrel_final \
        --unidl-dir  Predictions_AMP_final \
        --out-dir    FINAL_REPORT
"""

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd

LIT_LO, LIT_HI = 0.001, 0.0165           # Macrel, PeerJ 2020
MACREL_THS = [0.5, 0.7, 0.9, 0.95, 0.99]
UNIDL_THS = [0.5, 0.9, 0.95, 0.99, 0.999]


def load(results_dir):
    out = {}
    if not results_dir:
        return out
    for f in sorted(glob.glob(os.path.join(results_dir, "*", "*_summary.json"))):
        with open(f) as fh:
            s = json.load(fh)
        s["_dir"] = os.path.dirname(f)
        out[s["group"]] = s
    return out


def rate(s, tool):
    return s["n_hits"].get(tool, 0) / max(s["funnel"]["n_raw"], 1)


def dist(s, tool):
    return ((s.get("prob_distribution") or {}).get(tool) or {})


def pick_th_in_range(s, tool, ths):
    """找出判阳率落在文献基准 0.1-1.65% 内的那个阈值。"""
    d = dist(s, tool)
    c = d.get("counts_at_threshold") or {}
    n = max(s["funnel"]["n_raw"], 1)
    hit = []
    for t in ths:
        v = c.get(str(t))
        if v is None:
            continue
        r = v / n
        if LIT_LO <= r <= LIT_HI:
            hit.append((t, v, r))
    return hit


def headline(mac, uni, out_dir):
    print("=" * 78)
    print("① 总表: 两个工具各自的判阳率 (分母 = 原始 smORF, 与文献基准同口径)")
    print("=" * 78)
    rows = []
    groups = sorted(set(mac) | set(uni))
    for g in groups:
        m, u = mac.get(g), uni.get(g)
        ref = m or u
        row = {"分组": g,
               "原始 smORF": ref["funnel"]["n_raw"],
               "打分(过滤后)": ref["funnel"]["n_scored"]}
        if m:
            row["Macrel@0.5"] = m["n_hits"].get("Macrel", 0)
            row["Macrel 判阳率"] = f"{rate(m, 'Macrel') * 100:.3f}%"
        if u:
            row["UniDL@0.99"] = u["n_hits"].get("AMP", 0)
            row["UniDL 判阳率"] = f"{rate(u, 'AMP') * 100:.2f}%"
            row["UniDL/文献上限"] = f"{rate(u, 'AMP') / LIT_HI:.1f} 倍"
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def threshold_table(summ, tool, ths, label):
    print("\n" + "=" * 78)
    print(f"② {label}: 多档阈值下的判阳条数 / 判阳率(vs 原始 smORF)")
    print(f"   文献基准 {LIT_LO * 100:g}%-{LIT_HI * 100:g}%, 标 ← 的档位落在区间内")
    print("=" * 78)
    rows = []
    for g, s in summ.items():
        d = dist(s, tool)
        c = d.get("counts_at_threshold") or {}
        n = max(s["funnel"]["n_raw"], 1)
        row = {"分组": g}
        for t in ths:
            v = c.get(str(t))
            if v is None:
                row[f"{t}"] = "-"
                continue
            r = v / n
            mark = " ←" if LIT_LO <= r <= LIT_HI else ""
            row[f"{t}"] = f"{v:,} ({r * 100:.4f}%){mark}"
        rows.append(row)
    if not rows:
        print("   (无数据)")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def discriminative(mac, uni):
    print("\n" + "=" * 78)
    print("③ 判别力对比: 阈值从最宽推到最严, 判阳率塌缩了多少倍")
    print("   塌缩倍数大 = 概率有区分度, 阈值是有意义的旋钮;")
    print("   塌缩倍数≈1 = 概率挤在一起, 阈值拧不动, 该工具不构成筛选。")
    print("=" * 78)
    rows = []
    for g in sorted(set(mac) | set(uni)):
        m, u = mac.get(g), uni.get(g)
        row = {"分组": g}
        if m:
            c = dist(m, "Macrel").get("counts_at_threshold") or {}
            lo, hi = c.get("0.5"), c.get("0.99")
            if lo and hi is not None:
                row["Macrel 0.5→0.99 塌缩"] = f"{lo / max(hi, 1):,.0f} 倍"
        if u:
            c = dist(u, "AMP").get("counts_at_threshold") or {}
            lo, hi = c.get("0.5"), c.get("0.999")
            if lo and hi:
                row["UniDL 0.5→0.999 塌缩"] = f"{lo / max(hi, 1):,.2f} 倍"
        rows.append(row)
    if not rows:
        print("   (无数据)")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def group_diff(mac, uni, out_dir):
    """读 amp_group_stats.py 已经算好的组间检验, 按效应量重排。"""
    print("\n" + "=" * 78)
    print("④ 组间差异 (AD vs NC) —— 必须看效应量, 不能只看 p 值")
    print("   分母是千万级, p 值几乎必然显著; fold_change 才说明有没有实际差异。")
    print("=" * 78)
    frames = []
    for tag, d in (("Macrel 单工具", mac), ("共识跑", uni)):
        if not d:
            continue
        f = os.path.join(next(iter(d.values()))["_dir"], "..",
                         "group_stats_group_tests.tsv")
        f = os.path.normpath(f)
        if not os.path.exists(f):
            continue
        t = pd.read_csv(f, sep="\t")
        t.insert(0, "来源", tag)
        frames.append(t)
    if not frames:
        print("   (未找到 group_stats_group_tests.tsv —— 先跑 amp_group_stats.py)")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in ("来源", "Cohort", "Group", "vs", "tool", "rhoAMP",
                        "rhoAMP_ref", "fold_change", "p_value", "q_value_BH")
            if c in df.columns]
    df = df[cols]
    if "fold_change" in df.columns:
        df["偏离 1 的幅度"] = (pd.to_numeric(df["fold_change"], errors="coerce")
                              - 1).abs()
        df = df.sort_values("偏离 1 的幅度", ascending=False)
    print(df.to_string(index=False))
    print("\n   解读: 偏离 1 的幅度 < 0.05 (即 fold_change 在 0.95-1.05 之间)")
    print("         即使 p < 0.05 也应报为【无实际差异】。")
    return df


def write_report(mac, uni, t_mac, t_uni, t_dis, t_grp, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "README_结果汇总.md")
    L = []
    L.append("# 结果汇总\n")
    L.append(f"文献基准: 真实人肠道宏基因组 smORF 中 AMP 占比 "
             f"{LIT_LO * 100:g}–{LIT_HI * 100:g}% "
             f"(Santos-Júnior et al., *PeerJ* 2020)\n")

    L.append("\n## 1. 各工具判阳率 (分母 = 原始 smORF)\n")
    L.append("| 分组 | 原始 smORF | Macrel@0.5 | Macrel 判阳率 | "
             "UniDL@0.99 | UniDL 判阳率 |")
    L.append("|---|---|---|---|---|---|")
    for g in sorted(set(mac) | set(uni)):
        m, u = mac.get(g), uni.get(g)
        ref = m or u
        L.append("| {} | {:,} | {} | {} | {} | {} |".format(
            g, ref["funnel"]["n_raw"],
            f"{m['n_hits'].get('Macrel', 0):,}" if m else "-",
            f"{rate(m, 'Macrel') * 100:.3f}%" if m else "-",
            f"{u['n_hits'].get('AMP', 0):,}" if u else "-",
            f"{rate(u, 'AMP') * 100:.2f}%" if u else "-"))

    L.append("\n## 2. 阈值敏感性\n")
    L.append("### Macrel\n")
    L.append("| 分组 | " + " | ".join(str(t) for t in MACREL_THS) + " |")
    L.append("|---" * (len(MACREL_THS) + 1) + "|")
    for g, s in mac.items():
        c = dist(s, "Macrel").get("counts_at_threshold") or {}
        n = max(s["funnel"]["n_raw"], 1)
        cells = []
        for t in MACREL_THS:
            v = c.get(str(t))
            cells.append("-" if v is None
                         else f"{v:,} ({v / n * 100:.4f}%)")
        L.append(f"| {g} | " + " | ".join(cells) + " |")

    L.append("\n### UniDL4BioPep (取自共识跑 summary 的 AMP 那一栏)\n")
    L.append("| 分组 | " + " | ".join(str(t) for t in UNIDL_THS) + " |")
    L.append("|---" * (len(UNIDL_THS) + 1) + "|")
    for g, s in uni.items():
        c = dist(s, "AMP").get("counts_at_threshold") or {}
        n = max(s["funnel"]["n_raw"], 1)
        cells = []
        for t in UNIDL_THS:
            v = c.get(str(t))
            cells.append("-" if v is None
                         else f"{v:,} ({v / n * 100:.4f}%)")
        L.append(f"| {g} | " + " | ".join(cells) + " |")

    L.append("\n## 3. 可直接引用的结论\n")
    for g in sorted(set(mac) | set(uni)):
        m, u = mac.get(g), uni.get(g)
        if u:
            r = rate(u, "AMP")
            L.append(f"- **{g}**: UniDL4BioPep 在 p ≥ 0.99 下判阳 "
                     f"{r * 100:.2f}% 的原始 smORF, 是文献基准上限 "
                     f"{LIT_HI * 100:g}% 的 {r / LIT_HI:.1f} 倍。")
        if m:
            hit = pick_th_in_range(m, "Macrel", MACREL_THS)
            if hit:
                t, v, r = hit[0]
                L.append(f"- **{g}**: Macrel 在阈值 {t} 下判阳 {v:,} 条 "
                         f"({r * 100:.4f}%), 落在文献基准区间内。")
            else:
                L.append(f"- **{g}**: Macrel 在 "
                         f"{MACREL_THS[0]}–{MACREL_THS[-1]} 各档下判阳率"
                         f"均未落入文献基准区间, 需按第 2 节的表自行选档。")

    # 判别力: 用实际数字说, 不写死结论
    ratios = []
    for g, s in mac.items():
        c = dist(s, "Macrel").get("counts_at_threshold") or {}
        lo, hi = c.get("0.5"), c.get("0.99")
        if lo and hi is not None:
            ratios.append(("Macrel 0.5→0.99", g, lo / max(hi, 1)))
    for g, s in uni.items():
        c = dist(s, "AMP").get("counts_at_threshold") or {}
        lo, hi = c.get("0.5"), c.get("0.999")
        if lo and hi:
            ratios.append(("UniDL4BioPep 0.5→0.999", g, lo / max(hi, 1)))
    if ratios:
        L.append("\n- **判别力** (阈值从最宽推到最严, 判阳条数塌缩倍数):")
        for tag, g, v in ratios:
            L.append(f"  - {g} / {tag}: **{v:,.0f} 倍**")
        mac_r = [v for t, _, v in ratios if t.startswith("Macrel")]
        uni_r = [v for t, _, v in ratios if t.startswith("UniDL")]
        if mac_r and uni_r:
            L.append(f"\n  Macrel 塌缩 {min(mac_r):,.0f}–{max(mac_r):,.0f} 倍, "
                     f"UniDL4BioPep 塌缩 {min(uni_r):,.2f}–{max(uni_r):,.2f} 倍。"
                     if min(mac_r) != max(mac_r) or min(uni_r) != max(uni_r)
                     else f"\n  Macrel 塌缩 {mac_r[0]:,.0f} 倍, "
                          f"UniDL4BioPep 塌缩 {uni_r[0]:,.2f} 倍。")
            if max(uni_r) < 100 and min(mac_r) > 100:
                L.append("  → 两者相差悬殊: Macrel 的概率有区分度, 阈值是有意义的旋钮; "
                         "UniDL4BioPep 的概率挤在高位, 阈值拧不动, 不构成筛选。")
            else:
                L.append("  → 两者塌缩倍数接近, 不宜下\"某工具不构成筛选\"的结论, "
                         "需结合第 ③ 张表逐档核对。")
    L.append("- **组间差异**: 见第 ④ 张表。fold_change 接近 1 时, "
             "即使 p < 0.05 也应报为无实际差异 (分母千万级, p 值必然显著)。")
    L.append("\n## 4. 报告注意事项\n")
    L.append("1. 只报 ρAMP (归一化密度), 不报绝对条数。")
    L.append("2. 两个工具长度域不同 (Macrel 10–100 aa, UniDL4BioPep 11–180 aa), "
             "绝对数不可直接相比。")
    L.append("3. Cohort2 是 Cohort1 的两个组, 与 Cohort3/4 样本高度重叠, "
             "不是独立重复验证。")
    L.append("4. 这是计算预测, 无实验验证, 结论应表述为\"候选\"。")
    L.append("5. UniDL4BioPep 的绝对概率不可解释为后验概率 "
             "(先验错配 + softmax 未校准), 仅可用于排序。")

    with open(p, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n📄 报告: {p}")


def main():
    ap = argparse.ArgumentParser(
        description="把各步结果组装成总表 + 报告",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--macrel-dir", default="Predictions_Macrel_final")
    ap.add_argument("--unidl-dir", default="Predictions_AMP_final",
                    help="共识跑目录即可, 脚本自动取其中 AMP 那一栏")
    ap.add_argument("--out-dir", default="FINAL_REPORT")
    args = ap.parse_args()

    mac, uni = load(args.macrel_dir), load(args.unidl_dir)
    if not mac and not uni:
        raise SystemExit(f"❌ {args.macrel_dir} 和 {args.unidl_dir} "
                         f"下都没有 *_summary.json")

    t_h = headline(mac, uni, args.out_dir)
    t_mac = threshold_table(mac, "Macrel", MACREL_THS, "Macrel 单工具")
    t_uni = threshold_table(uni, "AMP", UNIDL_THS,
                            "UniDL4BioPep (共识跑里的 AMP 栏)")
    t_dis = discriminative(mac, uni)
    t_grp = group_diff(mac, uni, args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    for tag, df in (("01_headline", t_h), ("02_macrel_thresholds", t_mac),
                    ("03_unidl_thresholds", t_uni),
                    ("04_discriminative", t_dis), ("05_group_diff", t_grp)):
        if df is not None and not df.empty:
            df.to_csv(os.path.join(args.out_dir, f"{tag}.tsv"),
                      sep="\t", index=False)
    write_report(mac, uni, t_mac, t_uni, t_dis, t_grp, args.out_dir)
    print(f"📊 表格: {args.out_dir}/*.tsv")


if __name__ == "__main__":
    main()
