#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分组 / 分阶段 宏基因组 sORF 抗菌肽(AMP)预测

输入:
    ROOT_DIR/comparable_sorf_grouped_catalog/
        Cohort1_Matched265_5Stage/Cohort1_NC.fa ...
        Cohort2_Matched265_NCvsAD/...
        Cohort3_Full476_5Stage/...
        Cohort4_Full476_NCvsAD/...
        group_manifest.tsv
        sORF_All_Total.fa

模型:
    只使用抗菌肽相关模型:
        "5. Antimicrobial activity"  (AMP)
        "14. antibacterial AB"       (AB, 可选)

输出:
    Predictions_AMP_<时间戳>/
        <Cohort>/<Group>_AMP_hits.csv        阳性(prob>=阈值)序列明细
        <Cohort>/<Group>_summary.json        统计汇总
        <Cohort>/<Group>_summary.tsv
        summary_all_groups.tsv               所有分组汇总表
        .ckpt/<Cohort>__<Group>.ckpt.json    断点续跑检查点

设计要点(针对千万级序列 / GB 级 FASTA):
    * FASTA 流式读取, 不整文件载入内存
    * 序列去重(可选, 按 chunk 内 + 全局哈希集合)
    * 分 chunk 提取 ESM-2 特征 -> 立即预测 -> 只写出阳性, 释放内存
    * 每个 chunk 结束写检查点, 中断后可从断点继续
    * 长度过滤(AMP 模型训练集为短肽)
用法:
    python run_amp_sorf_cohorts.py                      # 全部分组
    python run_amp_sorf_cohorts.py --cohorts Cohort1    # 指定队列
    python run_amp_sorf_cohorts.py --models AMP AB      # 指定模型
    python run_amp_sorf_cohorts.py --threshold 0.9 --max-seqs 100000   # 试跑
"""

import os
import gc
import json
import time
import glob
import pickle
import hashlib
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import esm
import tensorflow as tf
from tqdm import tqdm
from keras.models import load_model

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ==========================================
# 1. 配置
# ==========================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_DIR = os.path.join(ROOT_DIR, "comparable_sorf_grouped_catalog")

# 抗菌肽相关模型: key -> (目录名, 权重文件关键字前缀)
AMP_MODEL_DIRS = {
    "AMP": "5. Antimicrobial activity",
    "AB": "14. antibacterial AB",
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args():
    p = argparse.ArgumentParser(description="分组宏基因组 sORF 抗菌肽预测")
    p.add_argument("--catalog-dir", default=CATALOG_DIR)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--cohorts", nargs="*", default=None,
                   help="只跑指定队列目录, 如 Cohort1_Matched265_5Stage 或 Cohort1")
    p.add_argument("--models", nargs="*", default=["AMP"],
                   choices=list(AMP_MODEL_DIRS.keys()),
                   help="使用的抗菌肽模型, 默认只用 AMP")
    p.add_argument("--include-total", action="store_true",
                   help="同时预测 sORF_All_Total.fa (非常大, 默认跳过)")
    p.add_argument("--chunk-size", type=int, default=200000,
                   help="每个处理块的序列数(决定内存占用与检查点粒度)")
    p.add_argument("--esm-batch-size", type=int, default=256,
                   help="ESM-2 前向 batch 大小")
    p.add_argument("--predict-batch-size", type=int, default=4096)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="判为阳性的概率阈值")
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--max-len", type=int, default=100,
                   help="AMP 模型面向短肽, 过长序列建议过滤")
    p.add_argument("--dedup", action="store_true", default=True,
                   help="组内序列去重(默认开启)")
    p.add_argument("--no-dedup", dest="dedup", action="store_false")
    p.add_argument("--max-seqs", type=int, default=0,
                   help=">0 时每组只处理前 N 条(用于试跑)")
    p.add_argument("--save-negatives", action="store_true",
                   help="同时保存全部序列的概率(输出会非常大)")
    return p.parse_args()


# ==========================================
# 2. FASTA 流式读取
# ==========================================

def iter_fasta(path):
    """流式产出 (header_id, sequence)。"""
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


def clean_seq(seq):
    s = seq.strip().upper().replace("*", "")
    return s


def iter_chunks(path, chunk_size, min_len, max_len, dedup, seen, max_seqs=0):
    """流式产出 (ids, seqs) 块, 并做长度/字符过滤与去重。"""
    ids, seqs = [], []
    n_taken = 0
    for sid, raw in iter_fasta(path):
        s = clean_seq(raw)
        if not s:
            continue
        L = len(s)
        if L < min_len or L > max_len:
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
            yield ids, seqs
            ids, seqs = [], []
        if max_seqs and n_taken >= max_seqs:
            break
    if seqs:
        yield ids, seqs


# ==========================================
# 3. ESM-2 特征提取
# ==========================================

class ESMEncoder:
    def __init__(self, device):
        print("📦 正在加载 ESM-2 (esm2_t6_8M_UR50D) ...")
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.model = model.eval().to(device)
        self.batch_converter = alphabet.get_batch_converter()
        self.device = device

    @torch.no_grad()
    def encode(self, sequences, batch_size=256, desc="提取 ESM-2 特征"):
        feats = np.empty((len(sequences), 320), dtype=np.float32)
        # 按长度排序可显著减少 padding 开销, 结束后还原顺序
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
# 4. 抗菌肽模型加载
# ==========================================

def all_files_of(d):
    return sorted(os.listdir(d))


def _try_load_scaler(path):
    """依次尝试 pickle / joblib 加载 scaler, 失败返回 None。"""
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception:
        try:
            import joblib
            obj = joblib.load(path)
        except Exception:
            return None
    if not hasattr(obj, "transform"):
        return None
    return obj


def load_amp_models(root_dir, model_keys):
    """返回 {name: (scaler, keras_model)}, 优先使用 keras_2 版本。"""
    loaded = {}
    for key in model_keys:
        folder = AMP_MODEL_DIRS[key]
        fpath = os.path.join(root_dir, folder)
        if not os.path.isdir(fpath):
            print(f"⚠️ 模型目录不存在, 跳过: {folder}")
            continue
        keras_files = sorted(f for f in os.listdir(fpath) if f.endswith(".keras"))
        if not keras_files:
            print(f"⚠️ 无 .keras 权重, 跳过: {folder}")
            continue
        k2 = [f for f in keras_files if "keras_2" in f]
        model_file = k2[0] if k2 else keras_files[0]
        prefix = model_file.split("_")[0]
        # 与模型对应的 scaler: 按优先级逐个尝试, 有的文件可能损坏/是 joblib 格式
        cands = [
            f"{prefix}_working_scaler.pkl",
            f"{prefix}_keras_2_minmax_scaler.pkl" if "keras_2" in model_file else None,
            f"{prefix}_minmax_scaler.pkl",
            "minmax_scaler.pkl",
            f"{prefix}.joblib",
        ]
        cands = [c for c in cands if c]
        # 兜底: 目录下其余所有 pkl/joblib
        cands += sorted(f for f in all_files_of(fpath)
                        if f.endswith((".pkl", ".joblib")) and f not in cands)

        scaler, scaler_path = None, None
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
            print(f"⚠️ 该目录没有可用 scaler, 跳过: {folder}")
            continue

        nfeat = getattr(scaler, "n_features_in_", None)
        if nfeat is not None and nfeat != 320:
            print(f"⚠️ {folder}: scaler 期望 {nfeat} 维, 与 ESM-2 320 维不符, 跳过")
            continue

        model = load_model(os.path.join(fpath, model_file))
        loaded[key] = (scaler, model)
        print(f"✅ 载入模型 {key}: {model_file} | scaler: {os.path.basename(scaler_path)}")
    if not loaded:
        raise RuntimeError("没有可用的抗菌肽模型")
    return loaded


def to_prob(pred):
    pred = np.asarray(pred)
    if pred.ndim == 2:
        if pred.shape[1] == 1:
            return pred[:, 0]
        if pred.shape[1] == 2:
            return pred[:, 1]
        raise ValueError(f"输出维度异常: {pred.shape}")
    return pred.reshape(-1)


# ==========================================
# 5. 分组枚举
# ==========================================

def collect_groups(catalog_dir, cohort_filter, include_total):
    """返回 [(cohort_dir_name, group_name, fasta_path), ...]"""
    groups = []
    for d in sorted(os.listdir(catalog_dir)):
        full = os.path.join(catalog_dir, d)
        if not os.path.isdir(full):
            continue
        if cohort_filter and not any(c.lower() in d.lower() for c in cohort_filter):
            continue
        for fa in sorted(glob.glob(os.path.join(full, "*.fa")) +
                         glob.glob(os.path.join(full, "*.fasta"))):
            name = os.path.splitext(os.path.basename(fa))[0]
            groups.append((d, name, fa))
    if include_total:
        total = os.path.join(catalog_dir, "sORF_All_Total.fa")
        if os.path.exists(total):
            groups.append(("All", "sORF_All_Total", total))
    return groups


# ==========================================
# 6. 单组预测
# ==========================================

def run_group(cohort, group, fasta, encoder, models, out_dir, ckpt_dir, args):
    gout = os.path.join(out_dir, cohort)
    os.makedirs(gout, exist_ok=True)

    hits_file = os.path.join(gout, f"{group}_AMP_hits.csv")
    all_file = os.path.join(gout, f"{group}_all_probs.csv.gz")
    summ_json = os.path.join(gout, f"{group}_summary.json")
    ckpt_file = os.path.join(ckpt_dir, f"{cohort}__{group}.ckpt.json")

    if os.path.exists(summ_json):
        print(f"⏭ 已完成, 跳过: {cohort}/{group}")
        with open(summ_json) as fh:
            return json.load(fh)

    start_chunk = 0
    stats = {"n_scored": 0, "n_hits": {k: 0 for k in models}}
    if os.path.exists(ckpt_file):
        with open(ckpt_file) as fh:
            ck = json.load(fh)
        start_chunk = ck["next_chunk"]
        stats = ck["stats"]
        stats["n_hits"] = {k: stats["n_hits"].get(k, 0) for k in models}
        print(f"🔁 断点续跑: 从 chunk {start_chunk} 继续")
    else:
        # 新建输出文件(写表头)
        for f in (hits_file, all_file):
            if os.path.exists(f):
                os.remove(f)

    seen = set()
    t0 = time.time()
    header_written = os.path.exists(hits_file)
    all_header_written = os.path.exists(all_file)

    for ci, (ids, seqs) in enumerate(iter_chunks(
            fasta, args.chunk_size, args.min_len, args.max_len,
            args.dedup, seen, args.max_seqs)):

        if ci < start_chunk:
            continue  # 已处理过(去重集合仍已同步更新)

        feats = encoder.encode(
            seqs, batch_size=args.esm_batch_size,
            desc=f"{cohort}/{group} chunk{ci} ({len(seqs)})")

        df = pd.DataFrame({"id": ids, "sequence": seqs,
                           "length": [len(s) for s in seqs]})

        hit_mask = np.zeros(len(seqs), dtype=bool)
        for name, (scaler, model) in models.items():
            X = scaler.transform(feats)
            prob = to_prob(model.predict(
                X, batch_size=args.predict_batch_size, verbose=0))
            df[f"{name}_prob"] = prob.astype(np.float32)
            cls = (prob >= args.threshold)
            df[f"{name}_class"] = cls.astype(np.int8)
            stats["n_hits"][name] += int(cls.sum())
            hit_mask |= cls
            del X
        stats["n_scored"] += len(seqs)

        hits = df[hit_mask]
        if len(hits):
            hits.to_csv(hits_file, mode="a", index=False,
                        header=not header_written)
            header_written = True
        if args.save_negatives:
            df.to_csv(all_file, mode="a", index=False,
                      header=not all_header_written, compression="gzip")
            all_header_written = True

        with open(ckpt_file, "w") as fh:
            json.dump({"next_chunk": ci + 1, "stats": stats}, fh)

        del df, feats, hits
        gc.collect()
        tf.keras.backend.clear_session()

        print(f"   chunk{ci}: 累计 {stats['n_scored']:,} 条, "
              + ", ".join(f"{k}阳性 {v:,}" for k, v in stats["n_hits"].items()))

    elapsed = time.time() - t0
    summary = {
        "cohort": cohort,
        "group": group,
        "fasta": fasta,
        "threshold": args.threshold,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "dedup": args.dedup,
        "n_scored": stats["n_scored"],
        "n_hits": stats["n_hits"],
        "hit_rate": {k: (v / stats["n_scored"] if stats["n_scored"] else 0.0)
                     for k, v in stats["n_hits"].items()},
        "elapsed_sec": round(elapsed, 1),
        "hits_file": hits_file,
    }
    with open(summ_json, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    pd.DataFrame([{
        "Cohort": cohort, "Group": group, "N_scored": stats["n_scored"],
        **{f"N_hits_{k}": v for k, v in stats["n_hits"].items()},
        **{f"HitRate_{k}": round(summary["hit_rate"][k], 6) for k in stats["n_hits"]},
    }]).to_csv(os.path.join(gout, f"{group}_summary.tsv"), sep="\t", index=False)

    print(f"💾 {cohort}/{group} 完成: {stats['n_scored']:,} 条, "
          f"耗时 {elapsed/60:.1f} 分钟")
    return summary


# ==========================================
# 7. 主流程
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

    encoder = ESMEncoder(device)
    models = load_amp_models(ROOT_DIR, args.models)

    all_summaries = []
    t_start = time.time()
    for c, g, f in groups:
        print("\n" + "=" * 70)
        print(f"🚀 {c} / {g}")
        print("=" * 70)
        try:
            all_summaries.append(
                run_group(c, g, f, encoder, models, out_dir, ckpt_dir, args))
        except Exception as e:
            print(f"❌ {c}/{g} 失败: {e}")

    if all_summaries:
        rows = []
        for s in all_summaries:
            row = {"Cohort": s["cohort"], "Group": s["group"],
                   "N_scored": s["n_scored"]}
            for k, v in s["n_hits"].items():
                row[f"N_hits_{k}"] = v
                row[f"HitRate_{k}"] = round(s["hit_rate"][k], 6)
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "summary_all_groups.tsv"),
            sep="\t", index=False)

    print("\n" + "=" * 70)
    print("🎉 全部抗菌肽预测完成")
    print(f"总耗时: {(time.time()-t_start)/3600:.2f} 小时")
    print(f"结果目录: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
