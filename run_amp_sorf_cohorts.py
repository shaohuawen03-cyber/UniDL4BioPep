#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分组 / 分阶段 宏基因组 sORF 抗菌肽(AMP)共识预测流水线

============================================================
流程设计(依据文献)
============================================================
输入 sORF FASTA
  ① 长度过滤 10-100 aa
     Macrel 在 (meta)genome 中只输出 10-100 aa 的 smORF
     [Santos-Júnior et al. PeerJ 2020, DOI 10.7717/peerj.10555]
  ② 理化预筛: 净电荷 >= +2, 疏水残基比例 >= 0.30
     多数 AMP 净电荷 +2~+11、约 50% 疏水残基 [Zhang & Gallo 2016];
     AMPSphere c_AMP 平均长 37 aa、平均电荷 +4.7、pI ~10.9
     [Santos-Júnior et al. Cell 2024, DOI 10.1016/j.cell.2024.05.013]
  ③ Macrel 分类 (随机森林, 以精确率优先, 专为宏基因组低正例先验训练)
     [PeerJ 2020] —— 主筛
  ④ UniDL4BioPep AMP 模型 (ESM-2 + CNN) —— 第二票
     [Du et al. Brief Bioinform 2023, DOI 10.1093/bib/bbad135]
  ⑤ 共识: 记录每条序列被几个工具判阳 (n_tools_positive)
     AMPSphere 质控即采用多工具共预测: 98.4% 候选被至少一个其他
     工具支持; 合成前要求 7 个方法全部判阳 [Cell 2024]
     共识策略可提高特异性与精确率、降低假阳性
     [DAMPC, Sci Data 2026, DOI 10.1038/s41597-026-07521-8]
  ⑥ 组间比较: 输出各组命中比例, 并与文献基准区间对照
     真实人肠道宏基因组中 AMP 占 smORF 的比例约 0.1-1.65%
     [Santos-Júnior et al. PeerJ 2020]

============================================================
输出
============================================================
Predictions_AMP_<tag>/
    <Cohort>/<Group>_AMP_hits.csv      共识候选(含各工具分数与理化性质)
    <Cohort>/<Group>_summary.json      统计汇总(含概率分布/漏斗/文献对照)
    <Cohort>/<Group>_summary.tsv
    <Cohort>/<Group>_AMP_prob_hist.npy 概率直方图(任意阈值可重算)
    summary_all_groups.tsv             所有分组横向汇总
    .ckpt/<Cohort>__<Group>.ckpt.json  断点续跑检查点

============================================================
用法
============================================================
    # 试跑
    python run_amp_sorf_cohorts.py --cohorts Cohort2 --max-seqs 200000

    # 正式(推荐后台)
    nohup python -u run_amp_sorf_cohorts.py \
        --output-dir ~/UniDL4BioPep-main/Predictions_AMP_run1 \
        > amp_run1.log 2>&1 &

    # 关闭 Macrel(只用 UniDL4BioPep)
    python run_amp_sorf_cohorts.py --no-macrel
"""

import os
import gc
import json
import time
import glob
import pickle
import shutil
import hashlib
import argparse
import subprocess
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import esm
import tensorflow as tf
from tqdm import tqdm
from keras.models import load_model

import amp_physchem as physchem

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ==========================================
# 1. 配置
# ==========================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_DIR = os.path.join(ROOT_DIR, "comparable_sorf_grouped_catalog")

AMP_MODEL_DIRS = {
    "AMP": "5. Antimicrobial activity",
    "AB": "14. antibacterial AB",
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

N_BINS = 1000
REPORT_THRESHOLDS = [0.5, 0.9, 0.95, 0.99, 0.999]

# 文献基准: 真实人肠道宏基因组 smORF 中 AMP 占比 (Macrel, PeerJ 2020)
LIT_HIT_RATE_RANGE = (0.001, 0.0165)


def parse_args():
    p = argparse.ArgumentParser(
        description="分组宏基因组 sORF 抗菌肽共识预测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--catalog-dir", default=CATALOG_DIR)
    p.add_argument("--output-dir", default=None,
                   help="固定输出目录(断点续跑必须固定); 缺省用时间戳")
    p.add_argument("--cohorts", nargs="*", default=None,
                   help="只跑指定队列, 如 Cohort1")
    p.add_argument("--models", nargs="*", default=["AMP"],
                   choices=list(AMP_MODEL_DIRS.keys()))
    p.add_argument("--include-total", action="store_true",
                   help="同时预测 sORF_All_Total.fa (9.8 GB, 默认跳过)")
    p.add_argument("--chunk-size", type=int, default=200000)
    p.add_argument("--esm-batch-size", type=int, default=256)
    p.add_argument("--predict-batch-size", type=int, default=4096)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="UniDL4BioPep 判阳阈值(概率分布会同时报告多档)")

    # --- 长度: 对齐 Macrel 的 smORF 范围 ---
    p.add_argument("--min-len", type=int, default=10)
    p.add_argument("--max-len", type=int, default=100)

    # --- 理化预筛 (AMPSphere / Zhang & Gallo 2016) ---
    g = p.add_argument_group("理化预筛")
    g.add_argument("--physchem-filter", action="store_true", default=True,
                   help="启用净电荷/疏水性预筛")
    g.add_argument("--no-physchem-filter", dest="physchem_filter",
                   action="store_false")
    g.add_argument("--min-charge", type=float, default=2.0,
                   help="最小净电荷(pH 7)")
    g.add_argument("--min-hydrophobic-frac", type=float, default=0.30,
                   help="最小疏水残基比例")

    # --- Macrel ---
    m = p.add_argument_group("Macrel")
    m.add_argument("--macrel", action="store_true", default=True,
                   help="启用 Macrel 作为主筛(需已安装 macrel)")
    m.add_argument("--no-macrel", dest="macrel", action="store_false")
    m.add_argument("--macrel-bin", default="macrel")
    m.add_argument("--macrel-threads", type=int, default=4)

    p.add_argument("--dedup", action="store_true", default=True)
    p.add_argument("--no-dedup", dest="dedup", action="store_false")
    p.add_argument("--max-seqs", type=int, default=0,
                   help=">0 时每组只处理前 N 条(试跑)")
    p.add_argument("--min-tools", type=int, default=1,
                   help="写入 hits 所需的最少判阳工具数(2 = 取交集)")
    return p.parse_args()


# ==========================================
# 2. 统计器(常数内存)
# ==========================================

class ProbStats:
    """概率分布直方图 + 分位数, 内存占用与数据量无关。"""

    def __init__(self, n_bins=N_BINS):
        self.n_bins = n_bins
        self.hist = np.zeros(n_bins, dtype=np.int64)
        self.n = 0
        self.sum = 0.0

    def update(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
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
            "mean_prob": round(self.sum / self.n, 4),
            "quantiles": qs,
            "counts_at_threshold": counts,
            "rates_at_threshold": {k: round(v / self.n, 6)
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
            if not line:
                continue
            if line[0] == ">":
                if header is not None:
                    yield header, "".join(buf)
                header = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if header is not None:
            yield header, "".join(buf)


def iter_chunks(path, chunk_size, min_len, max_len, dedup, seen, max_seqs=0):
    """流式产出 (ids, seqs, n_raw_seen)，含长度/字符过滤与去重。"""
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
# 4. ESM-2 编码
# ==========================================

class ESMEncoder:
    def __init__(self, device):
        print("📦 正在加载 ESM-2 (esm2_t6_8M_UR50D) ...")
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.model = model.eval().to(device)
        self.batch_converter = alphabet.get_batch_converter()
        self.device = device

    @torch.no_grad()
    def encode(self, sequences, batch_size=256, desc="ESM-2"):
        feats = np.empty((len(sequences), 320), dtype=np.float32)
        order = np.argsort([len(s) for s in sequences], kind="stable")
        for i in tqdm(range(0, len(order), batch_size), desc=desc, leave=False):
            idx = order[i:i + batch_size]
            batch = [(f"s{k}", sequences[k]) for k in idx]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)
            out = self.model(tokens, repr_layers=[6], return_contacts=False)
            reps = out["representations"][6]
            for j, k in enumerate(idx):
                L = len(sequences[k])
                feats[k] = reps[j, 1:L + 1].mean(0).float().cpu().numpy()
            del out, reps, tokens
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return feats


# ==========================================
# 5. UniDL4BioPep 模型
# ==========================================

def _try_load_scaler(path):
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception:
        try:
            import joblib
            obj = joblib.load(path)
        except Exception:
            return None
    return obj if hasattr(obj, "transform") else None


def load_amp_models(root_dir, model_keys):
    loaded = {}
    for key in model_keys:
        folder = AMP_MODEL_DIRS[key]
        fpath = os.path.join(root_dir, folder)
        if not os.path.isdir(fpath):
            print(f"⚠️ 模型目录不存在, 跳过: {folder}")
            continue
        files = sorted(os.listdir(fpath))
        keras_files = [f for f in files if f.endswith(".keras")]
        if not keras_files:
            print(f"⚠️ 无 .keras 权重, 跳过: {folder}")
            continue
        k2 = [f for f in keras_files if "keras_2" in f]
        model_file = k2[0] if k2 else keras_files[0]
        prefix = model_file.split("_")[0]

        cands = [c for c in [
            f"{prefix}_working_scaler.pkl",
            f"{prefix}_keras_2_minmax_scaler.pkl" if "keras_2" in model_file else None,
            f"{prefix}_minmax_scaler.pkl",
            "minmax_scaler.pkl",
            f"{prefix}.joblib",
        ] if c]
        cands += [f for f in files
                  if f.endswith((".pkl", ".joblib")) and f not in cands]

        scaler = scaler_path = None
        for c in cands:
            p = os.path.join(fpath, c)
            if not os.path.exists(p):
                continue
            s = _try_load_scaler(p)
            if s is not None:
                scaler, scaler_path = s, p
                break
            print(f"   ↷ scaler 不可用, 尝试下一个: {c}")
        if scaler is None:
            print(f"⚠️ 无可用 scaler, 跳过: {folder}")
            continue

        nfeat = getattr(scaler, "n_features_in_", None)
        if nfeat is not None and nfeat != 320:
            print(f"⚠️ {folder}: scaler 期望 {nfeat} 维 != 320, 跳过")
            continue

        model = load_model(os.path.join(fpath, model_file))
        loaded[key] = (scaler, model)
        print(f"✅ UniDL4BioPep {key}: {model_file} | "
              f"scaler: {os.path.basename(scaler_path)}")
    if not loaded:
        raise RuntimeError("没有可用的 UniDL4BioPep 抗菌肽模型")
    return loaded


def to_prob(pred):
    """UniDL4BioPep 末层为 Dense(2, softmax), 取第 2 列为 AMP 概率。"""
    pred = np.asarray(pred)
    if pred.ndim == 2:
        if pred.shape[1] == 1:
            return pred[:, 0]
        if pred.shape[1] == 2:
            return pred[:, 1]
        raise ValueError(f"输出维度异常: {pred.shape}")
    return pred.reshape(-1)


# ==========================================
# 6. Macrel
# ==========================================

def check_macrel(binary):
    """检测 macrel 是否可用, 返回版本字符串或 None。"""
    if shutil.which(binary) is None:
        return None
    try:
        r = subprocess.run([binary, "--version"], capture_output=True,
                           text=True, timeout=120)
        return (r.stdout + r.stderr).strip().splitlines()[0]
    except Exception:
        return None


def run_macrel_peptides(binary, ids, seqs, threads=4):
    """
    对一批肽调用 `macrel peptides`, 返回 {id: (prob, is_amp, hemolytic)}。
    使用 --keep-negatives 以获得全部序列的分数。
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
        for _, row in df.iterrows():
            key = str(row[c_acc])
            if not key.startswith("p"):
                continue
            try:
                i = int(key[1:])
            except ValueError:
                continue
            prob = float(row[c_prob]) if c_prob else np.nan
            if c_cls is not None:
                v = str(row[c_cls])
                is_amp = v not in ("NAMP", "nan", "0", "False")
            else:
                is_amp = prob >= 0.5
            hem = str(row[c_hem]) if c_hem else ""
            res[i] = (prob, bool(is_amp), hem)
    return res


# ==========================================
# 7. 分组枚举
# ==========================================

def collect_groups(catalog_dir, cohort_filter, include_total):
    groups = []
    for d in sorted(os.listdir(catalog_dir)):
        full = os.path.join(catalog_dir, d)
        if not os.path.isdir(full):
            continue
        if cohort_filter and not any(c.lower() in d.lower()
                                     for c in cohort_filter):
            continue
        for fa in sorted(glob.glob(os.path.join(full, "*.fa"))
                         + glob.glob(os.path.join(full, "*.fasta"))):
            groups.append((d, os.path.splitext(os.path.basename(fa))[0], fa))
    if include_total:
        total = os.path.join(catalog_dir, "sORF_All_Total.fa")
        if os.path.exists(total):
            groups.append(("All", "sORF_All_Total", total))
    return groups


# ==========================================
# 8. 单组预测
# ==========================================

def run_group(cohort, group, fasta, encoder, models, use_macrel,
              out_dir, ckpt_dir, args):
    gout = os.path.join(out_dir, cohort)
    os.makedirs(gout, exist_ok=True)

    hits_file = os.path.join(gout, f"{group}_AMP_hits.csv")
    summ_json = os.path.join(gout, f"{group}_summary.json")
    ckpt_file = os.path.join(ckpt_dir, f"{cohort}__{group}.ckpt.json")

    if os.path.exists(summ_json):
        print(f"⏭ 已完成, 跳过: {cohort}/{group}")
        with open(summ_json) as fh:
            return json.load(fh)

    tool_names = list(models) + (["Macrel"] if use_macrel else [])

    start_chunk = 0
    funnel = {"n_raw": 0, "n_after_len": 0, "n_after_physchem": 0,
              "n_scored": 0}
    n_hits = {t: 0 for t in tool_names}
    n_consensus = {str(k): 0 for k in range(1, len(tool_names) + 1)}
    pstats = {t: ProbStats() for t in tool_names}
    charge_m, len_m, pi_m = RunningMean(), RunningMean(), RunningMean()

    if os.path.exists(ckpt_file):
        with open(ckpt_file) as fh:
            ck = json.load(fh)
        start_chunk = ck["next_chunk"]
        funnel = ck["funnel"]
        n_hits = {t: ck["n_hits"].get(t, 0) for t in tool_names}
        n_consensus = ck.get("n_consensus", n_consensus)
        for t in tool_names:
            if t in ck.get("pstats", {}):
                pstats[t] = ProbStats.from_state(ck["pstats"][t])
        charge_m = RunningMean.from_state(ck["charge"])
        len_m = RunningMean.from_state(ck["len"])
        pi_m = RunningMean.from_state(ck["pi"])
        print(f"🔁 断点续跑: 从 chunk {start_chunk} 继续")
    elif os.path.exists(hits_file):
        os.remove(hits_file)

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

        # ---------- ② 理化预筛 ----------
        props = physchem.compute_all(seqs)
        if args.physchem_filter:
            keep = physchem.physchem_pass(
                props, min_charge=args.min_charge,
                min_len=args.min_len, max_len=args.max_len,
                min_hyd_frac=args.min_hydrophobic_frac)
            kidx = np.flatnonzero(keep)
            if kidx.size == 0:
                print(f"   chunk{ci}: 理化预筛后为空, 跳过")
                with open(ckpt_file, "w") as fh:
                    json.dump({"next_chunk": ci + 1, "funnel": funnel,
                               "n_hits": n_hits, "n_consensus": n_consensus,
                               "pstats": {k: v.state() for k, v in pstats.items()},
                               "charge": charge_m.state(), "len": len_m.state(),
                               "pi": pi_m.state()}, fh)
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

        n_pos = np.zeros(len(seqs), dtype=np.int8)

        # ---------- ③ Macrel ----------
        if use_macrel:
            mres = run_macrel_peptides(args.macrel_bin, ids, seqs,
                                       args.macrel_threads)
            mprob = np.full(len(seqs), np.nan, dtype=np.float32)
            mcls = np.zeros(len(seqs), dtype=np.int8)
            mhem = [""] * len(seqs)
            for i, (pr, isamp, hem) in mres.items():
                if i < len(seqs):
                    mprob[i] = pr
                    mcls[i] = int(isamp)
                    mhem[i] = hem
            df["Macrel_prob"] = mprob
            df["Macrel_class"] = mcls
            df["Macrel_hemolytic"] = mhem
            valid = ~np.isnan(mprob)
            pstats["Macrel"].update(mprob[valid])
            n_hits["Macrel"] += int(mcls.sum())
            n_pos += mcls

        # ---------- ④ UniDL4BioPep ----------
        feats = encoder.encode(seqs, batch_size=args.esm_batch_size,
                               desc=f"{cohort}/{group} c{ci} ({len(seqs)})")
        for name, (scaler, model) in models.items():
            X = scaler.transform(feats)
            prob = to_prob(model.predict(X, batch_size=args.predict_batch_size,
                                         verbose=0))
            df[f"UniDL_{name}_prob"] = prob.astype(np.float32)
            cls = (prob >= args.threshold).astype(np.int8)
            df[f"UniDL_{name}_class"] = cls
            pstats[name].update(prob)
            n_hits[name] += int(cls.sum())
            n_pos += cls
            del X
        del feats

        # ---------- ⑤ 共识 ----------
        df["n_tools_positive"] = n_pos
        df["n_tools_total"] = len(tool_names)
        for k in range(1, len(tool_names) + 1):
            n_consensus[str(k)] += int((n_pos >= k).sum())

        hits = df[n_pos >= args.min_tools]
        if len(hits):
            hits.to_csv(hits_file, mode="a", index=False,
                        header=not header_written)
            header_written = True

        with open(ckpt_file, "w") as fh:
            json.dump({"next_chunk": ci + 1, "funnel": funnel,
                       "n_hits": n_hits, "n_consensus": n_consensus,
                       "pstats": {k: v.state() for k, v in pstats.items()},
                       "charge": charge_m.state(), "len": len_m.state(),
                       "pi": pi_m.state()}, fh)

        del df, hits, props
        gc.collect()
        tf.keras.backend.clear_session()

        print(f"   chunk{ci}: 打分 {funnel['n_scored']:,} | "
              + " | ".join(f"{t} {n_hits[t]:,}" for t in tool_names)
              + (f" | 共识≥2 {n_consensus.get('2', 0):,}"
                 if len(tool_names) > 1 else ""))

    elapsed = time.time() - t0
    n = funnel["n_scored"]
    n_raw = max(funnel["n_raw"], 1)

    # 文献基准对照: AMP 占 smORF 比例 0.1-1.65% (Macrel PeerJ 2020)
    strict_key = str(min(2, len(tool_names)))
    n_strict = n_consensus.get(strict_key, 0)
    rate_vs_raw = n_strict / n_raw
    lo, hi = LIT_HIT_RATE_RANGE
    verdict = ("in_range" if lo <= rate_vs_raw <= hi
               else ("above_range" if rate_vs_raw > hi else "below_range"))

    summary = {
        "cohort": cohort, "group": group, "fasta": fasta,
        "tools": tool_names,
        "params": {
            "threshold": args.threshold,
            "min_len": args.min_len, "max_len": args.max_len,
            "physchem_filter": args.physchem_filter,
            "min_charge": args.min_charge,
            "min_hydrophobic_frac": args.min_hydrophobic_frac,
            "dedup": args.dedup, "min_tools": args.min_tools,
        },
        "funnel": funnel,
        "n_hits": n_hits,
        "hit_rate_vs_scored": {t: round(v / n, 6) if n else 0.0
                               for t, v in n_hits.items()},
        "n_consensus": n_consensus,
        "consensus_rate_vs_raw": {k: round(v / n_raw, 8)
                                  for k, v in n_consensus.items()},
        "input_properties": {
            "mean_length": len_m.mean,
            "mean_net_charge": charge_m.mean,
            "mean_pI": pi_m.mean,
        },
        "prob_distribution": {t: v.to_dict() for t, v in pstats.items()},
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

    for t, v in pstats.items():
        np.save(os.path.join(gout, f"{group}_{t}_prob_hist.npy"), v.hist)

    row = {"Cohort": cohort, "Group": group,
           "N_raw": funnel["n_raw"], "N_scored": n,
           "mean_len": len_m.mean, "mean_charge": charge_m.mean}
    for t in tool_names:
        row[f"N_{t}"] = n_hits[t]
    for k, v in n_consensus.items():
        row[f"N_consensus>={k}"] = v
    pd.DataFrame([row]).to_csv(
        os.path.join(gout, f"{group}_summary.tsv"), sep="\t", index=False)

    print(f"💾 {cohort}/{group} 完成: 原始 {funnel['n_raw']:,} → "
          f"打分 {n:,} → 共识≥{strict_key} {n_strict:,} "
          f"({rate_vs_raw*100:.3f}%, 文献基准 0.1-1.65% → {verdict}) | "
          f"{elapsed/60:.1f} 分钟")
    return summary


# ==========================================
# 9. 主流程
# ==========================================

def main():
    args = parse_args()

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(
        ROOT_DIR, f"Predictions_AMP_{run_tag}")
    ckpt_dir = os.path.join(out_dir, ".ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"📁 结果目录: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 设备: {device}")

    groups = collect_groups(args.catalog_dir, args.cohorts, args.include_total)
    if not groups:
        raise SystemExit(f"❌ 在 {args.catalog_dir} 未找到任何 FASTA 分组")
    print(f"✅ 待预测分组 {len(groups)} 个:")
    for c, g, f in groups:
        print(f"   - {c}/{g}  ({os.path.getsize(f)/1e9:.2f} GB)")

    use_macrel = False
    if args.macrel:
        ver = check_macrel(args.macrel_bin)
        if ver:
            print(f"✅ Macrel 可用: {ver}")
            use_macrel = True
        else:
            print("⚠️ 未检测到 Macrel, 将只用 UniDL4BioPep(单工具, 假阳性偏高)。")
            print("   安装: conda install -c bioconda macrel")

    print(f"🧪 理化预筛: {'开启' if args.physchem_filter else '关闭'} "
          f"(净电荷 >= {args.min_charge}, 疏水比例 >= "
          f"{args.min_hydrophobic_frac}, 长度 {args.min_len}-{args.max_len})")

    encoder = ESMEncoder(device)
    models = load_amp_models(ROOT_DIR, args.models)

    all_sum = []
    t_start = time.time()
    for c, g, f in groups:
        print("\n" + "=" * 70)
        print(f"🚀 {c} / {g}")
        print("=" * 70)
        try:
            all_sum.append(run_group(c, g, f, encoder, models, use_macrel,
                                     out_dir, ckpt_dir, args))
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
                 "mean_pI": s["input_properties"]["mean_pI"]}
            for t, v in s["n_hits"].items():
                r[f"N_{t}"] = v
                r[f"rate_{t}"] = s["hit_rate_vs_scored"][t]
            for k, v in s["n_consensus"].items():
                r[f"N_consensus>={k}"] = v
                r[f"rate_consensus>={k}"] = s["consensus_rate_vs_raw"][k]
            for t, d in s["prob_distribution"].items():
                if d:
                    r[f"{t}_mean_prob"] = d["mean_prob"]
                    r[f"{t}_P99"] = d["quantiles"].get("P99")
            r["lit_verdict"] = s["literature_benchmark"]["verdict"]
            rows.append(r)
        path = os.path.join(out_dir, "summary_all_groups.tsv")
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
        print(f"\n📊 汇总表: {path}")

    print("\n" + "=" * 70)
    print("🎉 全部抗菌肽共识预测完成")
    print(f"总耗时: {(time.time()-t_start)/3600:.2f} 小时")
    print(f"结果目录: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
