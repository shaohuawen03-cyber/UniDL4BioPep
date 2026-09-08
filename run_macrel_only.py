#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Macrel 单工具全量预测 (不加载 ESM-2 / TensorFlow, 不做共识)

为什么要单独跑
--------------
run_amp_sorf_cohorts.py 是"多工具共识"流程: 序列先过 ESM-2 编码 + UniDL4BioPep
打分, 再和 Macrel 取交集。这样一来:
  * 两个工具的判阳率绑在一起, 无法判断最终候选集是被谁决定的;
  * Macrel 白等 ESM/CNN 那一段时间(实测占总耗时的大头)。

本脚本把 Macrel 拆出来独立跑全量, 输出与 run_amp_sorf_cohorts.py
**完全同构** 的 <group>_AMP_hits.csv / <group>_summary.json, 因此
amp_group_stats.py 和 amp_ad_association.py 可以直接读。

依赖: numpy / pandas + macrel 二进制。不需要 torch / esm / tensorflow / keras。

用法
----
    # 全量(14 组), Macrel 官方阈值 0.5
    nohup bash run_step2a_macrel_only.sh > macrel_full.log 2>&1 &

    # 只跑一个队列
    python run_macrel_only.py --cohorts Cohort2

    # 关掉理化预筛, 得到"纯 Macrel"判阳率(用于和 UniDL 单独跑的结果对齐口径)
    python run_macrel_only.py --no-physchem-filter

注意
----
下游 amp_ad_association.py 的 --min-tools 默认是 2(为双工具共识设计)。
单工具跑完后必须显式传 --min-tools 1, 否则候选会被全部过滤掉。
"""

import os
import gc
import sys
import json
import time
import glob
import shutil
import hashlib
import argparse
import subprocess
import tempfile

import numpy as np
import pandas as pd

import amp_physchem as physchem

# ==========================================
# 1. 配置 (与 run_amp_sorf_cohorts.py 保持一致)
# ==========================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_DIR = os.path.join(ROOT_DIR, "comparable_sorf_grouped_catalog")

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Macrel 原生训练域: 在 (meta)genome 中只输出 10-100 aa 的 smORF
# (Santos-Júnior et al., PeerJ 2020, 10.7717/peerj.10555)
MACREL_DOMAIN = (10, 100)

N_BINS = 1000
REPORT_THRESHOLDS = [0.5, 0.7, 0.9, 0.95, 0.99]

# 文献基准: 真实人肠道宏基因组 smORF 中 AMP 占比 (Macrel, PeerJ 2020)
LIT_HIT_RATE_RANGE = (0.001, 0.0165)

TOOL = "Macrel"


# ==========================================
# 2. 常数内存统计器
# ==========================================

class ProbStats:
    """概率分布直方图 + 分位数 + 多档阈值计数, 内存占用与数据量无关。"""

    def __init__(self, n_bins=N_BINS):
        self.n_bins = n_bins
        self.hist = np.zeros(n_bins, dtype=np.int64)
        self.n = 0
        self.sum = 0.0

    def update(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs[~np.isnan(probs)]
        if probs.size == 0:
            return
        idx = np.clip((probs * self.n_bins).astype(int), 0, self.n_bins - 1)
        self.hist += np.bincount(idx, minlength=self.n_bins)
        self.n += probs.size
        self.sum += float(probs.sum())

    def to_dict(self):
        if self.n == 0:
            return {}
        edges = (np.arange(self.n_bins) + 0.5) / self.n_bins
        cum = np.cumsum(self.hist)
        qs = {}
        for q in (0.5, 0.75, 0.9, 0.99, 0.999, 0.9999):
            k = int(np.searchsorted(cum, q * self.n))
            qs[f"P{q*100:g}"] = round(float(edges[min(k, self.n_bins - 1)]), 4)
        counts = {str(t): int(self.hist[int(t * self.n_bins):].sum())
                  for t in REPORT_THRESHOLDS}
        return {
            "n": self.n,
            "mean_prob": round(self.sum / self.n, 6),
            "quantiles": qs,
            "counts_at_threshold": counts,
            "rates_at_threshold": {k: round(v / self.n, 8)
                                   for k, v in counts.items()},
        }

    def state(self):
        return {"hist": self.hist.tolist(), "n": self.n, "sum": self.sum}

    @classmethod
    def from_state(cls, st):
        o = cls(len(st["hist"]))
        o.hist = np.array(st["hist"], dtype=np.int64)
        o.n, o.sum = st["n"], st["sum"]
        return o


class RunningMean:
    def __init__(self):
        self.n, self.sum = 0, 0.0

    def update(self, v):
        v = np.asarray(v, dtype=np.float64)
        self.n += v.size
        self.sum += float(v.sum())

    @property
    def mean(self):
        return round(self.sum / self.n, 3) if self.n else None

    def state(self):
        return {"n": self.n, "sum": self.sum}

    @classmethod
    def from_state(cls, st):
        o = cls()
        o.n, o.sum = st["n"], st["sum"]
        return o


# ==========================================
# 3. FASTA 流式读取
# ==========================================

def iter_fasta(path):
    header, buf = None, []
    with open(path, "r", buffering=1024 * 1024) as fh:
        for line in fh:
            if line[0] == ">":
                if header is not None:
                    yield header, "".join(buf)
                header, buf = line[1:].strip(), []
            elif header is not None:
                buf.append(line.strip())
    if header is not None:
        yield header, "".join(buf)


def iter_chunks(path, chunk_size, min_len, max_len, dedup, seen, max_seqs=0):
    """流式产出 (ids, seqs, n_raw_seen), 含长度/字符过滤与去重。"""
    ids, seqs = [], []
    n_taken = 0
    n_raw = 0
    for sid, raw in iter_fasta(path):
        n_raw += 1
        s = raw.strip().upper().replace("*", "")
        if not s:
            continue
        if not (min_len <= len(s) <= max_len):
            continue
        if not set(s) <= VALID_AA:
            continue
        if dedup:
            h = hashlib.blake2b(s.encode(), digest_size=16).digest()
            if h in seen:
                continue
            seen.add(h)
        ids.append(sid)
        seqs.append(s)
        n_taken += 1
        if len(seqs) >= chunk_size:
            yield ids, seqs, n_raw
            ids, seqs, n_raw = [], [], 0
        if max_seqs and n_taken >= max_seqs:
            break
    if seqs:
        yield ids, seqs, n_raw


# ==========================================
# 4. Macrel 调用
# ==========================================

def check_macrel(binary):
    p = shutil.which(binary)
    if p is None and os.path.exists(binary):
        p = binary
    if p is None:
        return None
    try:
        r = subprocess.run([p, "--version"], capture_output=True, text=True,
                           timeout=60)
        return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else p
    except Exception:
        return p


def run_macrel_peptides(binary, seqs, threads=4, threshold=0.5):
    """
    对一批肽调用 `macrel peptides`, 返回 {行内序号: (prob, is_amp, hemolytic)}。
    用 --keep-negatives 拿到全部序列的分数, 而不是只拿阳性。
    """
    res = {}
    with tempfile.TemporaryDirectory(prefix="macrel_") as td:
        fa = os.path.join(td, "in.faa")
        # macrel 对 header 有长度/字符限制, 用序号作 id 再映回
        with open(fa, "w") as fh:
            for i, s in enumerate(seqs):
                fh.write(f">p{i}\n{s}\n")
        outdir = os.path.join(td, "out")
        cmd = [binary, "peptides", "--fasta", fa, "--output", outdir,
               "--keep-negatives", "-t", str(threads)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"   ⚠️ Macrel 返回码 {r.returncode}: "
                      f"{r.stderr.strip()[-300:]}")
                return res
        except Exception as e:
            print(f"   ⚠️ Macrel 调用失败: {e}")
            return res

        tsvs = (glob.glob(os.path.join(outdir, "*.prediction.gz"))
                + glob.glob(os.path.join(outdir, "*.prediction"))
                + glob.glob(os.path.join(outdir, "*.tsv.gz"))
                + glob.glob(os.path.join(outdir, "*.tsv")))
        if not tsvs:
            print(f"   ⚠️ 未找到 Macrel 输出: {os.listdir(outdir)}")
            return res
        df = pd.read_csv(tsvs[0], sep="\t", comment="#")
        cols = {c.lower(): c for c in df.columns}
        c_acc = cols.get("access") or cols.get("sequence") or df.columns[0]
        c_prob = cols.get("amp_probability") or cols.get("probability")
        c_cls = cols.get("amp_family") or cols.get("is_amp") or cols.get("class")
        c_hem = cols.get("hemolytic")
        if c_prob is None:
            print(f"   ⚠️ Macrel 输出缺少概率列, 现有列: {list(df.columns)}")
        for _, row in df.iterrows():
            key = str(row[c_acc])
            if not key.startswith("p"):
                continue
            try:
                i = int(key[1:])
            except ValueError:
                continue
            prob = float(row[c_prob]) if c_prob else np.nan
            # 判阳优先用概率: Macrel 官方判据是 AMP_probability >= 0.5。
            # 不能用 AMP_family —— 它是【家族名】(ADP/ALP/...) 而非二分类标签,
            # 配合 --keep-negatives 时任何非 "NAMP" 值都会被误判为阳性。
            if not np.isnan(prob):
                is_amp = prob >= threshold
            elif c_cls is not None:
                v = str(row[c_cls]).strip().upper()
                is_amp = v not in ("NAMP", "NAN", "0", "FALSE", "", "-")
            else:
                is_amp = False
            hem = str(row[c_hem]) if c_hem else ""
            res[i] = (prob, bool(is_amp), hem)
    return res


# ==========================================
# 5. 分组发现
# ==========================================

def collect_groups(catalog_dir, cohort_filter):
    groups = []
    for d in sorted(os.listdir(catalog_dir)):
        full = os.path.join(catalog_dir, d)
        if not os.path.isdir(full) or d.startswith("."):
            continue
        if cohort_filter and not any(c.lower() in d.lower()
                                     for c in cohort_filter):
            continue
        for fa in sorted(glob.glob(os.path.join(full, "*.fa"))
                         + glob.glob(os.path.join(full, "*.fasta"))):
            groups.append((d, os.path.splitext(os.path.basename(fa))[0], fa))
    return groups


# ==========================================
# 6. 单组预测
# ==========================================

def run_group(cohort, group, fasta, out_dir, ckpt_dir, args):
    gout = os.path.join(out_dir, cohort)
    os.makedirs(gout, exist_ok=True)

    hits_file = os.path.join(gout, f"{group}_AMP_hits.csv")
    summ_json = os.path.join(gout, f"{group}_summary.json")
    ckpt_file = os.path.join(ckpt_dir, f"{cohort}__{group}.ckpt.json")

    if os.path.exists(summ_json) and not args.force:
        print(f"⏭ 已完成, 跳过: {cohort}/{group} (加 --force 可重做)")
        with open(summ_json) as fh:
            return json.load(fh)
    if args.force:
        for f in (summ_json, hits_file, ckpt_file):
            if os.path.exists(f):
                os.remove(f)

    start_chunk = 0
    funnel = {"n_raw": 0, "n_after_len": 0, "n_after_physchem": 0,
              "n_scored": 0}
    n_hits = {TOOL: 0}
    n_eval = {TOOL: 0}          # 域内可评序列数
    n_consensus = {"1": 0}
    strata = sorted(set(args.length_strata))
    strata_labels = ([f"<{strata[0]}"]
                     + [f"{strata[i]}-{strata[i+1]}"
                        for i in range(len(strata) - 1)]
                     + [f">{strata[-1]}"])
    n_stratum = {lb: 0 for lb in strata_labels}
    n_stratum_hit = {lb: {TOOL: 0} for lb in strata_labels}
    pstats = ProbStats()
    charge_m, len_m, pi_m = RunningMean(), RunningMean(), RunningMean()

    if os.path.exists(ckpt_file):
        with open(ckpt_file) as fh:
            ck = json.load(fh)
        start_chunk = ck["next_chunk"]
        funnel = ck["funnel"]
        n_hits = {TOOL: ck["n_hits"].get(TOOL, 0)}
        n_eval = {TOOL: ck.get("n_eval", {}).get(TOOL, 0)}
        n_consensus = ck.get("n_consensus", n_consensus)
        n_stratum = ck.get("n_stratum", n_stratum)
        n_stratum_hit = ck.get("n_stratum_hit", n_stratum_hit)
        if "pstats" in ck:
            pstats = ProbStats.from_state(ck["pstats"])
        charge_m = RunningMean.from_state(ck["charge"])
        len_m = RunningMean.from_state(ck["len"])
        pi_m = RunningMean.from_state(ck["pi"])
        print(f"🔁 断点续跑: 从 chunk {start_chunk} 继续")
    elif os.path.exists(hits_file):
        os.remove(hits_file)

    def save_ckpt(ci):
        with open(ckpt_file, "w") as fh:
            json.dump({"next_chunk": ci + 1, "funnel": funnel,
                       "n_hits": n_hits, "n_eval": n_eval,
                       "n_consensus": n_consensus,
                       "n_stratum": n_stratum,
                       "n_stratum_hit": n_stratum_hit,
                       "pstats": pstats.state(),
                       "charge": charge_m.state(), "len": len_m.state(),
                       "pi": pi_m.state()}, fh)

    seen = set()
    t0 = time.time()
    header_written = os.path.exists(hits_file)

    for ci, (ids, seqs, n_raw) in enumerate(iter_chunks(
            fasta, args.chunk_size, args.min_len, args.max_len,
            args.dedup, seen, args.max_seqs)):

        if ci < start_chunk:
            continue

        funnel["n_raw"] += n_raw
        funnel["n_after_len"] += len(seqs)

        # ---------- 理化预筛 (与共识流程同参数, 保证口径可比) ----------
        props = physchem.compute_all(seqs)
        if args.physchem_filter:
            keep = physchem.physchem_pass(
                props, min_charge=args.min_charge,
                min_len=args.min_len, max_len=args.max_len,
                min_hyd_frac=args.min_hydrophobic_frac)
            kidx = np.flatnonzero(keep)
            if kidx.size == 0:
                print(f"   chunk{ci}: 理化预筛后为空, 跳过")
                save_ckpt(ci)
                continue
            ids = [ids[i] for i in kidx]
            seqs = [seqs[i] for i in kidx]
            props = {k: v[kidx] for k, v in props.items()}
        funnel["n_after_physchem"] += len(seqs)
        funnel["n_scored"] += len(seqs)

        charge_m.update(props["net_charge"])
        len_m.update(props["length"])
        pi_m.update(props["pI"])

        df = pd.DataFrame({
            "id": ids, "sequence": seqs,
            "length": props["length"],
            "net_charge": props["net_charge"].round(2),
            "pI": props["pI"].round(2),
            "hydrophobic_frac": props["hydrophobic_frac"].round(3),
        })

        # ---------- Macrel ----------
        dmask = ((props["length"] >= MACREL_DOMAIN[0])
                 & (props["length"] <= MACREL_DOMAIN[1]))
        didx = np.flatnonzero(dmask)
        mprob = np.full(len(seqs), np.nan, dtype=np.float32)
        mcls = np.zeros(len(seqs), dtype=np.int8)
        mhem = [""] * len(seqs)
        if didx.size:
            sub = [seqs[i] for i in didx]
            mres = run_macrel_peptides(args.macrel_bin, sub,
                                       args.macrel_threads,
                                       args.macrel_threshold)
            for j, (pr, isamp, hem) in mres.items():
                if j < len(didx):
                    g = didx[j]
                    mprob[g] = pr
                    mcls[g] = int(isamp)
                    mhem[g] = hem

        df["Macrel_prob"] = mprob
        df["Macrel_class"] = mcls
        df["Macrel_hemolytic"] = mhem
        df["Macrel_in_domain"] = dmask.astype(np.int8)
        # 单工具: 这两列保留是为了让 amp_ad_association.py 的
        # `n_tools_positive >= min_tools` 过滤(默认 2)不至于把结果清空。
        # 单工具跑完下游必须传 --min-tools 1。
        df["n_tools_positive"] = mcls
        df["n_tools_evaluable"] = dmask.astype(np.int8)
        df["consensus_frac"] = np.where(dmask, mcls.astype(float), 0.0)

        pstats.update(mprob[~np.isnan(mprob)])
        n_hits[TOOL] += int(mcls.sum())
        n_eval[TOOL] += int(dmask.sum())
        n_consensus["1"] += int(mcls.sum())

        # 长度分层统计
        L = props["length"]
        bins = np.digitize(L, strata)
        for bi, lb in enumerate(strata_labels):
            m = bins == bi
            if not m.any():
                continue
            n_stratum[lb] += int(m.sum())
            n_stratum_hit[lb][TOOL] += int(mcls[m].sum())

        hits = df[df["Macrel_class"] == 1]
        if len(hits):
            hits.to_csv(hits_file, mode="a", index=False,
                        header=not header_written)
            header_written = True

        save_ckpt(ci)

        del df, hits, props
        gc.collect()

        print(f"   chunk{ci}: 打分 {funnel['n_scored']:,} | "
              f"域内可评 {n_eval[TOOL]:,} | Macrel {n_hits[TOOL]:,}",
              flush=True)

    elapsed = time.time() - t0
    n = funnel["n_scored"]
    n_raw = max(funnel["n_raw"], 1)

    rate_vs_raw = n_hits[TOOL] / n_raw
    lo, hi = LIT_HIT_RATE_RANGE
    verdict = ("in_range" if lo <= rate_vs_raw <= hi
               else ("above_range" if rate_vs_raw > hi else "below_range"))

    pd_ = pstats.to_dict()
    summary = {
        "cohort": cohort, "group": group, "fasta": fasta,
        "tools": [TOOL],
        "mode": "single_tool_macrel_only",
        "params": {
            "macrel_threshold": args.macrel_threshold,
            "macrel_version": args.macrel_version,
            "min_len": args.min_len, "max_len": args.max_len,
            "physchem_filter": args.physchem_filter,
            "min_charge": args.min_charge,
            "min_hydrophobic_frac": args.min_hydrophobic_frac,
            "dedup": args.dedup,
        },
        "funnel": funnel,
        "n_hits": n_hits,
        "n_evaluable_per_tool": n_eval,
        "tool_domains": {TOOL: list(MACREL_DOMAIN)},
        "hit_rate_in_domain": {TOOL: round(n_hits[TOOL] / n_eval[TOOL], 8)
                               if n_eval.get(TOOL) else 0.0},
        "hit_rate_vs_scored": {TOOL: round(n_hits[TOOL] / n, 8) if n else 0.0},
        "length_strata": {"bins": strata,
                          "n_per_stratum": n_stratum,
                          "n_hits_per_stratum": n_stratum_hit},
        "n_consensus": n_consensus,
        "consensus_rate_vs_raw": {"1": round(n_hits[TOOL] / n_raw, 8)},
        "input_properties": {
            "mean_length": len_m.mean,
            "mean_net_charge": charge_m.mean,
            "mean_pI": pi_m.mean,
        },
        "prob_distribution": {TOOL: pd_},
        "literature_benchmark": {
            "reference": "Macrel, PeerJ 2020 (10.7717/peerj.10555): "
                         "AMP fraction of smORFs in real gut metagenomes "
                         "ranged 0.1-1.65%",
            "expected_range": list(LIT_HIT_RATE_RANGE),
            "observed_rate_vs_raw_smorf": round(rate_vs_raw, 8),
            "verdict": verdict,
        },
        "elapsed_sec": round(elapsed, 1),
        "hits_file": hits_file,
    }
    with open(summ_json, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    np.save(os.path.join(gout, f"{group}_{TOOL}_prob_hist.npy"), pstats.hist)

    row = {"Cohort": cohort, "Group": group,
           "N_raw": funnel["n_raw"], "N_scored": n,
           "mean_len": len_m.mean, "mean_charge": charge_m.mean}
    row[f"N_{TOOL}"] = n_hits[TOOL]
    row["N_consensus>=1"] = n_hits[TOOL]
    pd.DataFrame([row]).to_csv(
        os.path.join(gout, f"{group}_summary.tsv"), sep="\t", index=False)

    print(f"💾 {cohort}/{group} 完成: 原始 {funnel['n_raw']:,} → "
          f"打分 {n:,} → 域内 {n_eval[TOOL]:,} → Macrel≥"
          f"{args.macrel_threshold} {n_hits[TOOL]:,} "
          f"({rate_vs_raw * 100:.4f}% of raw, 文献基准 0.1-1.65% → "
          f"{verdict}) | {elapsed / 60:.1f} 分钟", flush=True)

    if pd_:
        print("   📊 阈值敏感性 (域内序列, 换阈值不用重跑):")
        print("      " + "  ".join(
            f"≥{t}: {pd_['counts_at_threshold'][str(t)]:,}"
            for t in REPORT_THRESHOLDS))
    return summary


# ==========================================
# 7. 主流程
# ==========================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Macrel 单工具全量预测(不含 ESM/UniDL4BioPep)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--catalog-dir", default=CATALOG_DIR)
    p.add_argument("--output-dir", default=None,
                   help="固定输出目录(断点续跑必须固定); 缺省用时间戳")
    p.add_argument("--cohorts", nargs="*", default=None,
                   help="只跑指定队列, 如 Cohort2")
    p.add_argument("--macrel-bin", default=os.environ.get(
        "MACREL_BIN",
        os.path.expanduser("~/miniconda3/envs/env_macrel/bin/macrel")))
    p.add_argument("--macrel-threads", type=int, default=8)
    p.add_argument("--macrel-threshold", type=float, default=0.5,
                   help="Macrel AMP_probability 判阳阈值(官方默认 0.5)")

    p.add_argument("--min-len", type=int, default=MACREL_DOMAIN[0],
                   help="全局最小长度; 缺省=Macrel 域下界")
    p.add_argument("--max-len", type=int, default=None,
                   help="全局最大长度; 缺省=Macrel 域上界(100), "
                        "传 180 可与 UniDL4BioPep 同口径但 100-180 段为域外")
    p.add_argument("--length-strata", nargs="*", type=int,
                   default=[10, 30, 50, 100])

    g = p.add_argument_group("理化预筛")
    g.add_argument("--physchem-filter", action="store_true", default=True)
    g.add_argument("--no-physchem-filter", dest="physchem_filter",
                   action="store_false",
                   help="关掉预筛, 得到纯 Macrel 判阳率")
    g.add_argument("--min-charge", type=float, default=2.0)
    g.add_argument("--min-hydrophobic-frac", type=float, default=0.30)

    p.add_argument("--chunk-size", type=int, default=200000)
    p.add_argument("--dedup", action="store_true", default=True)
    p.add_argument("--no-dedup", dest="dedup", action="store_false")
    p.add_argument("--max-seqs", type=int, default=0,
                   help=">0 时每组只处理前 N 条(试跑)")
    p.add_argument("--force", action="store_true",
                   help="忽略已有 summary/检查点, 强制重跑")
    return p.parse_args()


def main():
    args = parse_args()

    if args.max_len is None:
        args.max_len = MACREL_DOMAIN[1]

    out_dir = args.output_dir or os.path.join(
        ROOT_DIR, f"Predictions_Macrel_{time.strftime('%Y%m%d_%H%M%S')}")
    ckpt_dir = os.path.join(out_dir, ".ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 70)
    print("Macrel 单工具全量预测 (无 ESM-2 / 无 TensorFlow)")
    print("=" * 70)
    ver = check_macrel(args.macrel_bin)
    if ver is None:
        sys.exit(f"❌ 找不到 macrel: {args.macrel_bin}\n"
                 f"   conda install -c bioconda macrel\n"
                 f"   或用 --macrel-bin / MACREL_BIN 指定路径")
    args.macrel_version = ver
    print(f"✅ Macrel: {ver}")
    print(f"📁 输入: {args.catalog_dir}")
    print(f"📁 输出: {out_dir}")
    print(f"📏 长度: {args.min_len}-{args.max_len} aa "
          f"(Macrel 训练域 {MACREL_DOMAIN[0]}-{MACREL_DOMAIN[1]})")
    print(f"🎚 判阳阈值: AMP_probability >= {args.macrel_threshold} "
          f"(官方默认 0.5)")
    print(f"🧪 理化预筛: {'开启' if args.physchem_filter else '关闭'} "
          f"(净电荷 >= {args.min_charge}, "
          f"疏水比例 >= {args.min_hydrophobic_frac})")
    print(f"🧵 线程: {args.macrel_threads} | chunk: {args.chunk_size:,}")

    groups = collect_groups(args.catalog_dir, args.cohorts)
    if not groups:
        sys.exit(f"❌ 在 {args.catalog_dir} 下没找到任何 .fa 分组")
    print(f"✅ 待预测分组 {len(groups)} 个:")
    for c, g, f in groups:
        print(f"   - {c}/{g}  ({os.path.getsize(f) / 1024**3:.2f} GB)")

    all_sum = []
    t_start = time.time()
    for c, g, f in groups:
        print("\n" + "=" * 70)
        print(f"🚀 {c} / {g}")
        print("=" * 70, flush=True)
        try:
            all_sum.append(run_group(c, g, f, out_dir, ckpt_dir, args))
        except Exception as e:
            import traceback
            print(f"❌ {c}/{g} 失败: {e}")
            traceback.print_exc()

    if all_sum:
        rows = []
        for s in all_sum:
            r = {"Cohort": s["cohort"], "Group": s["group"],
                 "N_raw": s["funnel"]["n_raw"],
                 "N_after_len": s["funnel"]["n_after_len"],
                 "N_scored": s["funnel"]["n_scored"],
                 "mean_len": s["input_properties"]["mean_length"],
                 "mean_charge": s["input_properties"]["mean_net_charge"],
                 "mean_pI": s["input_properties"]["mean_pI"],
                 f"N_{TOOL}": s["n_hits"][TOOL],
                 f"rate_{TOOL}": s["hit_rate_vs_scored"][TOOL],
                 f"rho_{TOOL}": s["consensus_rate_vs_raw"]["1"],
                 "lit_verdict": s["literature_benchmark"]["verdict"]}
            d = s["prob_distribution"].get(TOOL) or {}
            if d:
                r[f"{TOOL}_mean_prob"] = d["mean_prob"]
                r[f"{TOOL}_P99"] = d["quantiles"].get("P99")
            rows.append(r)
        path = os.path.join(out_dir, "summary_all_groups.tsv")
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
        print(f"\n📊 汇总表: {path}")

    print("\n" + "=" * 70)
    print("🎉 Macrel 单工具全量预测完成")
    print(f"总耗时: {(time.time() - t_start) / 3600:.2f} 小时")
    print(f"结果目录: {out_dir}")
    print("\n下一步:")
    print(f"  python amp_group_stats.py {out_dir}")
    print(f"  python amp_ad_association.py {out_dir} --min-tools 1 "
          f"--cohort Cohort3")
    print("  ⚠️ 单工具必须传 --min-tools 1, 否则候选会被默认值 2 清空")
    print("=" * 70)


if __name__ == "__main__":
    main()
