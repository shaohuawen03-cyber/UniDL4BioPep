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

# ==========================================
# 各工具的原生训练长度域 (aa)
# 每个工具只对自己域内的序列打分, 域外输出 NaN 不外推。
#   Macrel        : 在 (meta)genome 中只输出 10-100 aa 的 smORF (PeerJ 2020)
#   UniDL4BioPep  : AMP 训练集长度 11-180 aa (本地 AMP_train.csv 核实)
#   Ma-ATT/LSTM   : getorf -maxsize 150 nt = 50 aa; 仓库标明 "under 50AA"
#                   (Nat Biotechnol 2022, github.com/mayuefine/c_AMPs-prediction)
#   amPEPpy       : 无严格上限, 沿用 AmPEP 常用范围
# ==========================================
TOOL_DOMAIN = {
    "Macrel":  (10, 100),
    "AMP":     (11, 180),   # UniDL4BioPep AMP
    "AB":      (11, 180),   # UniDL4BioPep antibacterial
    "MaATT":   (5,  50),
    "MaLSTM":  (5,  50),
    "amPEPpy": (10, 200),
}


def domain_mask(lengths, tool):
    lo, hi = TOOL_DOMAIN.get(tool, (0, 10 ** 6))
    return (lengths >= lo) & (lengths <= hi)


def union_domain(tools):
    """所有启用工具的长度域并集, 作为全局预过滤范围。"""
    los = [TOOL_DOMAIN.get(t, (0, 10 ** 6))[0] for t in tools]
    his = [TOOL_DOMAIN.get(t, (0, 10 ** 6))[1] for t in tools]
    return (min(los) if los else 10), (max(his) if his else 180)

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

    # --- 长度: 默认取所有启用工具的域【并集】, 各工具再各自按域打分 ---
    p.add_argument("--min-len", type=int, default=None,
                   help="全局最小长度; 缺省=启用工具长度域并集的下界")
    p.add_argument("--max-len", type=int, default=None,
                   help="全局最大长度; 缺省=启用工具长度域并集的上界")
    p.add_argument("--length-strata", nargs="*", type=int,
                   default=[10, 50, 100, 180],
                   help="分层统计的长度分界(各工具域不同, 需分层比较)")

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
    m.add_argument("--macrel-threshold", type=float, default=0.5,
                   help="Macrel AMP_probability 判阳阈值(官方默认 0.5)")

    a = p.add_argument_group("amPEPpy (第三票, 可选)")
    a.add_argument("--ampep", action="store_true", default=True,
                   help="启用 amPEPpy(需已安装 ampep 且已 train 出模型)")
    a.add_argument("--no-ampep", dest="ampep", action="store_false")
    a.add_argument("--ampep-bin", default="ampep")
    a.add_argument("--ampep-model", default=None,
                   help="amPEPpy 模型 .pkl 路径(未提供则跳过该工具)")
    a.add_argument("--ampep-threshold", type=float, default=0.5)

    y = p.add_argument_group("Ma et al. 2022 模型 (ATT / LSTM, 不含 BERT)")
    y.add_argument("--ma-models", nargs="*", default=[],
                   choices=["ATT", "LSTM"],
                   help="启用马跃模型, 如 --ma-models ATT LSTM")
    y.add_argument("--ma-att-h5", default=None, help="10att.h5 路径")
    y.add_argument("--ma-lstm-h5", default=None, help="10lstm.h5 路径")
    y.add_argument("--ma-threshold", type=float, default=0.5,
                   help="判阳阈值 (原文与后续研究均用 0.5)")
    y.add_argument("--ma-max-len", type=int, default=50,
                   help="马跃模型训练域上限(50 aa); 超长序列标记为 NA 而非外推")

    p.add_argument("--dedup", action="store_true", default=True)
    p.add_argument("--no-dedup", dest="dedup", action="store_false")
    p.add_argument("--max-seqs", type=int, default=0,
                   help=">0 时每组只处理前 N 条(试跑)")
    p.add_argument("--min-tools", type=int, default=1,
                   help="写入 hits 所需的最少判阳工具数(2 = 取交集)")
    p.add_argument("--consensus-mode", default="count",
                   choices=["count", "frac"],
                   help="count=按绝对票数; frac=按'判阳/可评'比例。"
                        "各工具长度域不同, 长肽可评工具数更少, "
                        "count 模式会系统性丢弃长肽, 建议用 frac")
    p.add_argument("--min-consensus-frac", type=float, default=1.0,
                   help="frac 模式下的阈值(1.0=所有可评工具都判阳)")
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
# 5b. Ma et al. 2022 模型 (ATT / LSTM)
# ==========================================

# 官方 format.pl 的整数编码表(A=1 ... Y=20), 左侧 0 填充至 300 维
MA_AACODE = {a: i + 1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
MA_MAXLEN = 300


def ma_encode(seqs):
    """复刻 c_AMPs-prediction/format.pl 的编码: 整数映射 + 左侧零填充到 300。"""
    X = np.zeros((len(seqs), MA_MAXLEN), dtype=np.float32)
    for i, s in enumerate(seqs):
        codes = [MA_AACODE.get(ch, 0) for ch in s[-MA_MAXLEN:]]
        X[i, MA_MAXLEN - len(codes):] = codes
    return X


def load_ma_models(args):
    """
    加载马跃 ATT / LSTM 模型。
    仅用 ATT+LSTM 而不含 BERT 有直接文献先例:
    J Gerontol A Biol Sci 2024 (百岁老人肠道 AMP) 明确
    "选择其中两个(ATT 和 LSTM)作为本研究使用的模型", 阈值同为 0.5。
    """
    loaded = {}
    if not args.ma_models:
        return loaded
    paths = {"ATT": args.ma_att_h5, "LSTM": args.ma_lstm_h5}
    for key in args.ma_models:
        p_ = paths.get(key)
        if not p_ or not os.path.exists(p_):
            print(f"⚠️ 未提供/找不到 {key} 权重, 跳过 (--ma-{key.lower()}-h5)")
            continue
        try:
            if key == "ATT":
                # 自定义 Attention_layer, 需与仓库的 Attention.py 同目录或可导入
                try:
                    from Attention import Attention_layer
                except ImportError:
                    sys.path.insert(0, os.path.dirname(os.path.abspath(p_)))
                    from Attention import Attention_layer
                m = load_model(p_, custom_objects={
                    "Attention_layer": Attention_layer})
            else:
                m = load_model(p_)
            loaded[f"Ma{key}"] = m
            print(f"✅ Ma et al. {key}: {p_}")
        except Exception as e:
            print(f"⚠️ 加载 Ma {key} 失败: {e}")
            print("   提示: 该模型为旧版 Keras .h5, 建议用独立环境")
    return loaded


def ma_predict(model, seqs, max_len, batch_size=1024):
    """
    返回概率数组; 超出训练域(>max_len aa)的序列置 NaN, 不做外推。
    马跃模型训练数据为 <=50 aa (getorf -maxsize 150 nt)。
    """
    probs = np.full(len(seqs), np.nan, dtype=np.float32)
    idx = [i for i, s in enumerate(seqs) if len(s) <= max_len]
    if not idx:
        return probs
    X = ma_encode([seqs[i] for i in idx])
    pred = np.asarray(model.predict(X, batch_size=batch_size, verbose=0))
    if pred.ndim == 2 and pred.shape[1] == 2:
        pred = pred[:, 1]
    else:
        pred = pred.reshape(-1)
    probs[np.array(idx)] = pred
    return probs


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


def run_macrel_peptides(binary, ids, seqs, threads=4, threshold=0.5):
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
            # 配合 --keep-negatives 时任何非 "NAMP" 值都会被误判为阳性,
            # 导致 100% 判阳。
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


def check_ampep(binary, model):
    """检测 amPEPpy 是否可用(需要二进制 + 已训练模型)。"""
    if shutil.which(binary) is None:
        return None
    if not model or not os.path.exists(model):
        return "no_model"
    return "ok"


def run_ampep(binary, model, seqs, threshold=0.5, threads=4):
    """调用 amPEPpy predict, 返回 {index: prob}。"""
    res = {}
    with tempfile.TemporaryDirectory(prefix="ampep_") as td:
        fa = os.path.join(td, "in.faa")
        with open(fa, "w") as fh:
            for i, s in enumerate(seqs):
                fh.write(f">p{i}\n{s}\n")
        out = os.path.join(td, "out.tsv")
        cmd = [binary, "predict", "-m", model, "-i", fa, "-o", out,
               "-d", "-t", str(threads)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(out):
                print(f"   ⚠️ amPEPpy 失败: {r.stderr.strip()[-300:]}")
                return res
            df = pd.read_csv(out, sep="\t")
        except Exception as e:
            print(f"   ⚠️ amPEPpy 调用失败: {e}")
            return res
        cols = {c.lower(): c for c in df.columns}
        c_id = cols.get("seq_id") or df.columns[0]
        c_pr = cols.get("probability_amp") or cols.get("probability") or df.columns[-1]
        for _, row in df.iterrows():
            k = str(row[c_id])
            if not k.startswith("p"):
                continue
            try:
                res[int(k[1:])] = float(row[c_pr])
            except (ValueError, TypeError):
                continue
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
              use_ampep, ma_models, out_dir, ckpt_dir, args):
    gout = os.path.join(out_dir, cohort)
    os.makedirs(gout, exist_ok=True)

    hits_file = os.path.join(gout, f"{group}_AMP_hits.csv")
    summ_json = os.path.join(gout, f"{group}_summary.json")
    ckpt_file = os.path.join(ckpt_dir, f"{cohort}__{group}.ckpt.json")

    if os.path.exists(summ_json):
        print(f"⏭ 已完成, 跳过: {cohort}/{group}")
        with open(summ_json) as fh:
            return json.load(fh)

    tool_names = (list(models) + (["Macrel"] if use_macrel else [])
                  + (["amPEPpy"] if use_ampep else [])
                  + list(ma_models))

    start_chunk = 0
    funnel = {"n_raw": 0, "n_after_len": 0, "n_after_physchem": 0,
              "n_scored": 0}
    n_hits = {t: 0 for t in tool_names}
    n_eval = {t: 0 for t in tool_names}   # 各工具"域内可评"序列数
    n_consensus = {str(k): 0 for k in range(1, len(tool_names) + 1)}
    # 按长度分层的命中统计(各工具域不同, 必须分层比较)
    strata = sorted(set(args.length_strata))
    strata_labels = ([f"<{strata[0]}"]
                     + [f"{strata[i]}-{strata[i+1]}" for i in range(len(strata)-1)]
                     + [f">{strata[-1]}"])
    n_stratum = {lb: 0 for lb in strata_labels}
    n_stratum_hit = {lb: {t: 0 for t in tool_names} for lb in strata_labels}
    pstats = {t: ProbStats() for t in tool_names}
    charge_m, len_m, pi_m = RunningMean(), RunningMean(), RunningMean()

    if os.path.exists(ckpt_file):
        with open(ckpt_file) as fh:
            ck = json.load(fh)
        start_chunk = ck["next_chunk"]
        funnel = ck["funnel"]
        n_hits = {t: ck["n_hits"].get(t, 0) for t in tool_names}
        n_eval = {t: ck.get("n_eval", {}).get(t, 0) for t in tool_names}
        n_consensus = ck.get("n_consensus", n_consensus)
        n_stratum = ck.get("n_stratum", n_stratum)
        n_stratum_hit = ck.get("n_stratum_hit", n_stratum_hit)
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
                               "n_hits": n_hits, "n_eval": n_eval,
                               "n_consensus": n_consensus,
                               "n_stratum": n_stratum,
                               "n_stratum_hit": n_stratum_hit,
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
        n_evaluable = np.zeros(len(seqs), dtype=np.int8)

        # ---------- ③ Macrel ----------
        if use_macrel:
            dmask = domain_mask(props["length"], "Macrel")
            didx = np.flatnonzero(dmask)
            mprob = np.full(len(seqs), np.nan, dtype=np.float32)
            mcls = np.zeros(len(seqs), dtype=np.int8)
            mhem = [""] * len(seqs)
            if didx.size:
                sub = [seqs[i] for i in didx]
                mres = run_macrel_peptides(args.macrel_bin,
                                           [ids[i] for i in didx], sub,
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
            pstats["Macrel"].update(mprob[~np.isnan(mprob)])
            n_hits["Macrel"] += int(mcls.sum())
            n_eval["Macrel"] += int(dmask.sum())
            n_pos += mcls
            n_evaluable += dmask.astype(np.int8)

        # ---------- ③b amPEPpy ----------
        if use_ampep:
            dmask = domain_mask(props["length"], "amPEPpy")
            didx = np.flatnonzero(dmask)
            aprob = np.full(len(seqs), np.nan, dtype=np.float32)
            if didx.size:
                ares = run_ampep(args.ampep_bin, args.ampep_model,
                                 [seqs[i] for i in didx],
                                 args.ampep_threshold, args.macrel_threads)
                for j, pr in ares.items():
                    if j < len(didx):
                        aprob[didx[j]] = pr
            acls = (np.nan_to_num(aprob) >= args.ampep_threshold).astype(np.int8)
            df["amPEPpy_prob"] = aprob
            df["amPEPpy_class"] = acls
            df["amPEPpy_in_domain"] = dmask.astype(np.int8)
            pstats["amPEPpy"].update(aprob[~np.isnan(aprob)])
            n_hits["amPEPpy"] += int(acls.sum())
            n_eval["amPEPpy"] += int(dmask.sum())
            n_pos += acls
            n_evaluable += dmask.astype(np.int8)

        # ---------- ④ UniDL4BioPep ----------
        feats = encoder.encode(seqs, batch_size=args.esm_batch_size,
                               desc=f"{cohort}/{group} c{ci} ({len(seqs)})")
        for name, (scaler, model) in models.items():
            dmask = domain_mask(props["length"], name)
            X = scaler.transform(feats)
            praw = to_prob(model.predict(X, batch_size=args.predict_batch_size,
                                         verbose=0))
            prob = np.where(dmask, praw, np.nan).astype(np.float32)
            df[f"UniDL_{name}_prob"] = prob
            cls = ((np.nan_to_num(prob) >= args.threshold) & dmask).astype(np.int8)
            df[f"UniDL_{name}_class"] = cls
            df[f"UniDL_{name}_in_domain"] = dmask.astype(np.int8)
            pstats[name].update(prob[~np.isnan(prob)])
            n_hits[name] += int(cls.sum())
            n_eval[name] += int(dmask.sum())
            n_pos += cls
            n_evaluable += dmask.astype(np.int8)
            del X
        del feats

        # ---------- ④b Ma et al. ATT / LSTM ----------
        for mname, mmodel in ma_models.items():
            dmask = domain_mask(props["length"], mname)
            mp = ma_predict(mmodel, seqs, TOOL_DOMAIN[mname][1],
                            args.predict_batch_size)
            df[f"{mname}_prob"] = mp
            mc = ((np.nan_to_num(mp) >= args.ma_threshold) & dmask).astype(np.int8)
            df[f"{mname}_class"] = mc
            df[f"{mname}_in_domain"] = dmask.astype(np.int8)
            pstats[mname].update(mp[~np.isnan(mp)])
            n_hits[mname] += int(mc.sum())
            n_eval[mname] += int(dmask.sum())
            n_pos += mc
            n_evaluable += dmask.astype(np.int8)

        # ---------- ⑤ 共识 ----------
        df["n_tools_positive"] = n_pos
        # 分母改为"该序列有几个工具够格评", 而非固定工具总数
        df["n_tools_evaluable"] = n_evaluable
        df["consensus_frac"] = np.where(n_evaluable > 0,
                                        n_pos / np.maximum(n_evaluable, 1), 0.0)
        for k in range(1, len(tool_names) + 1):
            n_consensus[str(k)] += int((n_pos >= k).sum())

        # 长度分层统计
        L = props["length"]
        bins = np.digitize(L, strata)
        for bi, lb in enumerate(strata_labels):
            m = bins == bi
            if not m.any():
                continue
            n_stratum[lb] += int(m.sum())
            for t in tool_names:
                col = (f"UniDL_{t}_class" if f"UniDL_{t}_class" in df.columns
                       else f"{t}_class")
                if col in df.columns:
                    n_stratum_hit[lb][t] += int(df[col].values[m].sum())

        if args.consensus_mode == "frac":
            frac = np.where(n_evaluable > 0,
                            n_pos / np.maximum(n_evaluable, 1), 0.0)
            sel = (frac >= args.min_consensus_frac) & (n_evaluable > 0)
        else:
            sel = n_pos >= args.min_tools
        hits = df[sel]
        if len(hits):
            hits.to_csv(hits_file, mode="a", index=False,
                        header=not header_written)
            header_written = True

        with open(ckpt_file, "w") as fh:
            json.dump({"next_chunk": ci + 1, "funnel": funnel,
                       "n_hits": n_hits, "n_eval": n_eval,
                       "n_consensus": n_consensus,
                       "n_stratum": n_stratum,
                       "n_stratum_hit": n_stratum_hit,
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
        "n_evaluable_per_tool": n_eval,
        "tool_domains": {t: list(TOOL_DOMAIN.get(t, (0, 0)))
                         for t in tool_names},
        # 命中率分母用"该工具域内可评序列数", 而非全部打分序列
        "hit_rate_in_domain": {t: round(n_hits[t] / n_eval[t], 6)
                               if n_eval.get(t) else 0.0 for t in n_hits},
        "hit_rate_vs_scored": {t: round(v / n, 6) if n else 0.0
                               for t, v in n_hits.items()},
        "length_strata": {"bins": strata,
                          "n_per_stratum": n_stratum,
                          "n_hits_per_stratum": n_stratum_hit},
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

    use_ampep = False
    if args.ampep:
        st = check_ampep(args.ampep_bin, args.ampep_model)
        if st == "ok":
            print(f"✅ amPEPpy 可用: {args.ampep_model}")
            use_ampep = True
        elif st == "no_model":
            print("⚠️ 检测到 ampep 但未提供 --ampep-model, 跳过该工具")
        # 未安装时静默跳过

    ma_models = load_ma_models(args)
    n_tools = (len(args.models) + int(use_macrel) + int(use_ampep)
               + len(ma_models))

    # 依据启用工具自动确定全局长度范围(各工具域的并集)
    active = (list(args.models) + (["Macrel"] if use_macrel else [])
              + (["amPEPpy"] if use_ampep else []) + list(ma_models))
    u_lo, u_hi = union_domain(active)
    if args.min_len is None:
        args.min_len = u_lo
    if args.max_len is None:
        args.max_len = u_hi
    print(f"📏 全局长度范围(工具域并集): {args.min_len}-{args.max_len} aa")
    for t in active:
        lo, hi = TOOL_DOMAIN.get(t, (0, 0))
        print(f"     {t:<10} 训练域 {lo}-{hi} aa"
              + ("  ← 域外输出 NaN, 不外推" if (lo, hi) != (args.min_len,
                                                          args.max_len) else ""))
    print(f"🗳 共识工具数: {n_tools}  (模式: {args.consensus_mode})")
    if args.consensus_mode == "count" and args.min_tools > 1:
        narrow = [t for t in active
                  if TOOL_DOMAIN.get(t, (0, 1e6))[1] < u_hi]
        if narrow:
            print(f"   ⚠️ count 模式 + --min-tools {args.min_tools}: "
                  f"长度超出 {narrow} 训练域的序列可评工具数不足, "
                  f"将被系统性丢弃")
            print(f"      如需保留长肽, 改用 --consensus-mode frac")
    if n_tools < 2:
        print("   ⚠️ 单工具模式假阳性偏高, 强烈建议至少加装 Macrel")

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
                                     use_ampep, ma_models, out_dir,
                                     ckpt_dir, args))
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
