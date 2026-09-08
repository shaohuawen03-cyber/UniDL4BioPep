#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比 Macrel 单工具跑 / UniDL4BioPep 单工具跑 / (可选)原共识跑

用来回答一个问题: 在原来的"双工具共识"里, 最终候选集到底是被谁决定的?
如果 UniDL4BioPep 在极高阈值下仍然放行绝大多数序列, 那么"共识"实际上
等价于 Macrel 单筛, UniDL4BioPep 那一票不构成筛选。

三张表
------
① 各工具单独跑全量的判阳率, 与文献基准 0.1-1.65% 对照
② UniDL4BioPep 多档阈值敏感性 —— 看把阈值推到 0.999 判阳率是否仍远高于基准
③ (可选, 需要共识跑的 hits.csv) 在原共识候选里, UniDL4BioPep 各阈值
   分别淘汰掉多少 Macrel 命中。淘汰率≈0 即"无筛选作用"的直接证据。

用法
----
    python compare_macrel_vs_unidl.py \
        --macrel-dir Predictions_Macrel_final \
        --unidl-dir  Predictions_UniDL_final

    # 加上原来那次共识跑, 输出第 ③ 张表
    python compare_macrel_vs_unidl.py \
        --macrel-dir Predictions_Macrel_final \
        --unidl-dir  Predictions_UniDL_final \
        --consensus-dir Predictions_AMP_final

依赖: numpy / pandas
"""

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd

LIT_LO, LIT_HI = 0.001, 0.0165          # Macrel, PeerJ 2020
UNIDL_THS = ("0.5", "0.9", "0.95", "0.99", "0.999")
MACREL_THS = ("0.5", "0.7", "0.9", "0.95", "0.99")


def load(results_dir):
    """{group: summary.json 内容}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*", "*_summary.json"))):
        with open(f) as fh:
            s = json.load(fh)
        out[(s["cohort"], s["group"])] = s
    return out


def tool_of(s, prefer=None):
    """挑出这份 summary 里要看的工具。

    单工具跑只有一个; 共识跑(run_full_pipeline.sh 第 2 步)的 summary 里
    每个工具的计数和多档阈值分布【都是分开存的】, 所以共识跑的目录也能
    直接当作某个工具的单工具结果来读 —— 不必为了拿 UniDL4BioPep 的
    判阳率再单独跑一遍 ESM-2 + CNN。
    """
    t = s.get("tools") or []
    if not t:
        return None
    if prefer and prefer in t:
        return prefer
    return t[0] if len(t) == 1 else None


def verdict_mark(r):
    return "✅" if LIT_LO <= r <= LIT_HI else ("⚠️ 偏高" if r > LIT_HI else "⚠️ 偏低")


def table_single(summ, label, ths, tool=None):
    print("\n" + "=" * 78)
    print(f"① {label}: 各分组判阳率 (对照文献基准 0.1-1.65%)")
    print("=" * 78)
    rows = []
    multi = any(len(s.get("tools") or []) > 1 for s in summ.values())
    if multi:
        print("   ⚠️ 这是【共识跑】的目录。下表取的是该工具【自己】的判阳条数")
        print("      (summary 的 n_hits 是分工具存的), 判阳率也按该工具重算,")
        print("      不是 summary 里 literature_benchmark 那个共识数。")
    rows = []
    for (coh, grp), s in summ.items():
        t = tool_of(s, prefer=tool) or (s.get("tools") or ["?"])[0]
        n_raw = max(s["funnel"]["n_raw"], 1)
        n_pos = s["n_hits"].get(t, 0)
        # 分母用 n_raw, 与文献基准 0.1-1.65% 同口径。
        # 共识跑的 literature_benchmark.observed_rate_vs_raw_smorf 是【共识】
        # 计数算出来的, 不能拿来当单个工具的判阳率, 这里一律重算。
        r = n_pos / n_raw
        d = (s.get("prob_distribution", {}).get(t) or {})
        c = d.get("counts_at_threshold") or {}
        row = {"Cohort": coh, "Group": grp, "N_raw": s["funnel"]["n_raw"],
               "N_scored": s["funnel"]["n_scored"],
               "N_pos": n_pos,
               "判阳率_vs_raw": f"{r * 100:.4f}%",
               "判定": verdict_mark(r)}
        for th in ths:
            row[f"≥{th}"] = c.get(th)
        rows.append(row)
    if not rows:
        print("   (无数据)")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def table_sensitivity(summ, label, tool, ths, note=""):
    print("\n" + "=" * 78)
    print(f"② {label}: 多档阈值下的判阳率 (占【打分序列】比例)")
    print(f"   若把阈值推到最高档, 判阳率仍远高于 1.65%,")
    print(f"   说明该工具在本数据上不具备判别力, 而不是阈值没调好。")
    print("=" * 78)
    if note:
        print(f"   {note}")
    rows = []
    for (coh, grp), s in summ.items():
        d = (s.get("prob_distribution", {}).get(tool) or {})
        n = d.get("n") or 0
        c = d.get("counts_at_threshold") or {}
        if not n:
            continue
        row = {"Cohort": coh, "Group": grp, "域内可评": n}
        for th in ths:
            v = c.get(th)
            row[f"≥{th}"] = f"{v:,} ({100 * v / n:.2f}%)" if v is not None else "-"
        rows.append(row)
    if not rows:
        print("   (无数据)")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def table_overlap(consensus_dir, unidl_prob_col="UniDL_AMP_prob",
                  macrel_prob_col="Macrel_prob", macrel_th=0.5,
                  consensus_th="0.99"):
    """在共识跑的 hits.csv 上, 看 UniDL 各阈值淘汰掉多少 Macrel 命中。"""
    print("\n" + "=" * 78)
    print("③ 共识跑里 UniDL4BioPep 那一票的实际筛选强度")
    print(f"   口径: 取 Macrel_prob >= {macrel_th} 的序列为 Macrel 命中,")
    print("         再看其中有多少条被 UniDL4BioPep 各阈值淘汰。")
    print("   淘汰率 ≈ 0 → UniDL4BioPep 不构成筛选, 共识 ≈ Macrel 单筛。")
    print("=" * 78)
    files = sorted(glob.glob(os.path.join(consensus_dir, "*", "*_AMP_hits.csv")))
    if not files:
        print("   (没找到 hits.csv, 跳过)")
        return pd.DataFrame(), []
    rows = []
    all_rates = []
    for f in files:
        df = pd.read_csv(f)
        if macrel_prob_col not in df.columns or unidl_prob_col not in df.columns:
            print(f"   ⚠️ {os.path.basename(f)} 缺少概率列, 跳过")
            continue
        m = df[macrel_prob_col] >= macrel_th
        n_m = int(m.sum())
        row = {"文件": os.path.basename(f).replace("_AMP_hits.csv", ""),
               f"Macrel≥{macrel_th}": n_m}
        up = df.loc[m, unidl_prob_col]
        dropped_rates = []
        for th in ("0.5", "0.9", "0.95", "0.99", "0.999"):
            t = float(th)
            kept = int((up >= t).sum())
            dropped = n_m - kept
            rate = 100 * dropped / n_m if n_m else float("nan")
            dropped_rates.append(rate)
            row[f"UniDL≥{th} 保留"] = kept
            row[f"UniDL≥{th} 淘汰率"] = f"{rate:.2f}%" if n_m else "-"
        rows.append(row)
        all_rates.append(dropped_rates)
    if not rows:
        return pd.DataFrame(), []
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df, all_rates


def main():
    ap = argparse.ArgumentParser(
        description="Macrel / UniDL4BioPep 单工具跑结果对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--macrel-dir", default=None,
                    help="Macrel 结果目录。可以是 Macrel 单工具跑"
                         "(run_step2a_macrel_only.sh), 也可以是共识跑目录"
                         " —— 共识跑的 summary 里 Macrel 的计数是单独存的")
    ap.add_argument("--unidl-dir", default=None,
                    help="UniDL4BioPep 结果目录。同样可以是共识跑目录: "
                         "共识跑的 summary 里 UniDL4BioPep 的判阳条数和 "
                         "0.5/0.9/0.95/0.99/0.999 五档分布都是单独存的, "
                         "不必为了这张表再单独跑一遍 ESM-2 + CNN")
    ap.add_argument("--consensus-dir", default=None,
                    help="原双工具共识跑的输出目录(可选, 用于第 ③ 张表)")
    ap.add_argument("--consensus-th", default="0.99",
                    choices=UNIDL_THS,
                    help="共识跑当时 UniDL4BioPep 用的阈值(仅用于标注)")
    ap.add_argument("--macrel-th", type=float, default=0.5,
                    help="第 ③ 张表里认定 Macrel 命中所用的阈值 "
                         "(要与共识跑当时的 MACREL_TH 一致)")
    ap.add_argument("--out", default=None, help="TSV 输出前缀")
    args = ap.parse_args()

    print("=" * 78)
    print("Macrel vs UniDL4BioPep 单工具全量结果对比")
    print("=" * 78)
    if not args.macrel_dir and not args.unidl_dir:
        raise SystemExit("❌ --macrel-dir / --unidl-dir 至少给一个")
    print(f"Macrel  结果目录: {args.macrel_dir or '(未提供)'}")
    print(f"UniDL   结果目录: {args.unidl_dir or '(未提供)'}")
    if args.consensus_dir:
        print(f"共识跑 结果目录: {args.consensus_dir}")
    print(f"文献基准: 真实人肠道宏基因组 smORF 中 AMP 占比 "
          f"{LIT_LO * 100:g}-{LIT_HI * 100:g}% (Macrel, PeerJ 2020)")

    mac = load(args.macrel_dir) if args.macrel_dir else {}
    uni = load(args.unidl_dir) if args.unidl_dir else {}
    for tag, d, path in (("macrel", mac, args.macrel_dir),
                         ("unidl", uni, args.unidl_dir)):
        if path and not d:
            raise SystemExit(f"❌ {path} 下没有 *_summary.json")

    note = ("   (数据取自共识跑目录 —— 其中该工具的计数是单独统计的)"
            if args.unidl_dir and uni
            and any(len(s.get("tools") or []) > 1 for s in uni.values())
            else "")
    t_mac = table_single(mac, "Macrel", MACREL_THS, tool="Macrel") if mac \
        else pd.DataFrame()
    t_uni = table_single(uni, "UniDL4BioPep", UNIDL_THS, tool="AMP") if uni \
        else pd.DataFrame()
    if uni:
        table_sensitivity(uni, "UniDL4BioPep", "AMP", UNIDL_THS, note=note)
    if mac:
        table_sensitivity(mac, "Macrel", "Macrel", MACREL_THS)

    t_ovl, drop_rates = pd.DataFrame(), []
    if args.consensus_dir:
        t_ovl, drop_rates = table_overlap(args.consensus_dir,
                                          macrel_th=args.macrel_th,
                                          consensus_th=args.consensus_th)

    # ---------- 结论 ----------
    print("\n" + "=" * 78)
    print("④ 结论")
    print("=" * 78)
    for name, summ, tool in (("Macrel", mac, "Macrel"),
                             ("UniDL4BioPep", uni, "AMP")):
        if not summ:
            continue
        # 按【该工具自己】的判阳数重算, 不用 literature_benchmark
        # (共识跑里那个字段是共识计数)
        rs = [s["n_hits"].get(tool_of(s, prefer=tool) or tool, 0)
              / max(s["funnel"]["n_raw"], 1) for s in summ.values()]
        if not rs:
            continue
        lo, hi = min(rs) * 100, max(rs) * 100
        inside = sum(1 for r in rs if LIT_LO <= r <= LIT_HI)
        print(f"   {name:<14} 判阳率 {lo:.4f}% - {hi:.4f}%  "
              f"({inside}/{len(rs)} 个分组落在文献基准内)")

    if drop_rates:
        # drop_rates: 每个文件一行, 5 个阈值各一个淘汰率(%)
        arr = np.asarray(drop_rates, dtype=float)
        finite = arr[~np.isnan(arr)]
        if finite.size == 0:
            print("\n   ⚠️ 在给定 --macrel-th 下没有任何 Macrel 命中, "
                  "无法计算淘汰率。")
            print("      用 --macrel-th 指定共识跑当时真正用的阈值。")
        else:
            print("\n   UniDL4BioPep 各阈值对 Macrel 命中的淘汰率 "
                  "(全部文件中的最小-最大):")
        for i, th in enumerate(UNIDL_THS if finite.size else ()):
            col = arr[:, i]
            col = col[~np.isnan(col)]
            if not col.size:
                continue
            mark = ("  ← 共识跑当时用的阈值"
                    if th == args.consensus_th else "")
            print(f"     ≥{th:<6} {col.min():6.2f}% - {col.max():6.2f}%{mark}")
            if th == args.consensus_th:
                if col.max() < 10:
                    print("       → 淘汰率 < 10%: UniDL4BioPep 那一票几乎不改变候选集,")
                    print("         原'双工具共识'在效果上等价于 Macrel 单筛。")
                else:
                    print("       → 淘汰率不可忽略, 两个工具都有贡献。")

    if args.out:
        for tag, df in (("macrel", t_mac), ("unidl", t_uni),
                        ("overlap", t_ovl)):
            if df is not None and not df.empty:
                df.to_csv(f"{args.out}_{tag}.tsv", sep="\t", index=False)
        print(f"\n📊 已写出: {args.out}_*.tsv")


if __name__ == "__main__":
    main()
