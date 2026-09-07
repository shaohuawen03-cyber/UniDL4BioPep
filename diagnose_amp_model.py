#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断 UniDL4BioPep AMP 模型为何判阳率异常偏高。

思路: 拿模型自己的【带标签测试集】(AMP_test.csv, 2583 正 / 6369 负) 跑一遍。
  - 若能复现论文精度(准确率~0.90, 判阳率~29%) → 模型/scaler 正确,
    问题出在真实数据的分布或阈值上。
  - 若判阳率也接近 80% → scaler 或模型加载有问题, 是 pipeline 的 bug。

同时对比所有候选 scaler, 找出哪一个才是正确的。

用法:
    python diagnose_amp_model.py
    python diagnose_amp_model.py --repo-root . --sample-fasta clean_catalog/xxx.fa
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

warn = []
try:
    import joblib
except Exception:
    joblib = None


def load_any(path):
    """依次尝试 pickle / joblib 载入。"""
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as e1:
        if joblib is not None:
            try:
                return joblib.load(path)
            except Exception as e2:
                raise RuntimeError(f"pickle: {e1} | joblib: {e2}")
        raise


def metrics(y, prob, th=0.5):
    pred = (prob >= th).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    sn = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return dict(acc=acc, sn=sn, sp=sp, prec=prec, mcc=mcc,
                posrate=pred.mean(), tp=tp, tn=tn, fp=fp, fn=fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--amp-dir", default="5. Antimicrobial activity")
    ap.add_argument("--sample-fasta", default=None,
                    help="可选: 附带跑一份真实 sORF 数据做分布对比")
    ap.add_argument("--sample-n", type=int, default=5000)
    args = ap.parse_args()

    root = os.path.abspath(args.repo_root)
    ampdir = os.path.join(root, args.amp_dir)
    if not os.path.isdir(ampdir):
        cands = glob.glob(os.path.join(root, "*Antimicrobial*"))
        if cands:
            ampdir = cands[0]
        else:
            sys.exit(f"❌ 找不到 AMP 目录: {ampdir}")
    print(f"📁 AMP 目录: {ampdir}")

    # ---------- 读取带标签测试集 ----------
    import pandas as pd
    testcsv = os.path.join(ampdir, "AMP_test.csv")
    if not os.path.exists(testcsv):
        sys.exit(f"❌ 找不到 {testcsv}")
    df = pd.read_csv(testcsv)
    df.columns = [c.strip().lower() for c in df.columns]
    seqs = df["sequence"].astype(str).str.strip().tolist()
    y = df["label"].astype(str).str.strip().astype(int).values
    print(f"✅ 测试集: {len(seqs)} 条 | 正 {int((y==1).sum())} "
          f"负 {int((y==0).sum())} | 正例先验 {y.mean()*100:.1f}%")
    L = np.array([len(s) for s in seqs])
    print(f"   长度: min {L.min()} / 中位 {int(np.median(L))} / max {L.max()}")

    # ---------- ESM-2 编码 ----------
    print("\n📦 加载 ESM-2 (esm2_t6_8M_UR50D) ...")
    import torch
    import esm
    model_esm, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    bc = alphabet.get_batch_converter()
    model_esm.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_esm = model_esm.to(dev)
    print(f"   设备: {dev}")

    def encode(seq_list, bs=256):
        out = []
        for i in range(0, len(seq_list), bs):
            batch = [(f"p{j}", s) for j, s in enumerate(seq_list[i:i + bs])]
            _, _, toks = bc(batch)
            toks = toks.to(dev)
            with torch.no_grad():
                rep = model_esm(toks, repr_layers=[6])["representations"][6]
            for k, (_, s) in enumerate(batch):
                out.append(rep[k, 1:len(s) + 1].mean(0).cpu().numpy())
        return np.asarray(out, dtype=np.float32)

    print("   编码中 ...")
    X = encode(seqs)
    print(f"   特征: {X.shape}  范围 [{X.min():.3f}, {X.max():.3f}]")

    # ---------- 载入模型 ----------
    from keras.models import load_model
    mfiles = (sorted(glob.glob(os.path.join(ampdir, "*keras_2*.keras")))
              + sorted(glob.glob(os.path.join(ampdir, "*.keras"))))
    if not mfiles:
        sys.exit("❌ 找不到 .keras 模型")
    mpath = mfiles[0]
    print(f"\n🧠 模型: {os.path.basename(mpath)}")
    kmodel = load_model(mpath, compile=False)
    try:
        print(f"   输入 {kmodel.input_shape} → 输出 {kmodel.output_shape}")
    except Exception:
        pass

    # ---------- 遍历所有候选 scaler ----------
    scalers = sorted(set(
        glob.glob(os.path.join(ampdir, "*scaler*.pkl"))
        + glob.glob(os.path.join(ampdir, "*scaler*.joblib"))))
    print(f"\n🔍 发现 {len(scalers)} 个候选 scaler, 逐个评测:\n")

    hdr = (f"{'scaler':<34} {'判阳率':>7} {'准确率':>7} {'敏感度':>7} "
           f"{'特异度':>7} {'精确率':>7} {'MCC':>7}")
    print(hdr)
    print("-" * len(hdr))

    results = []

    def evaluate(tag, Xs):
        pred = kmodel.predict(Xs, batch_size=4096, verbose=0)
        pred = np.asarray(pred)
        prob = pred[:, 1] if (pred.ndim == 2 and pred.shape[1] == 2) \
            else pred.reshape(-1)
        m = metrics(y, prob)
        print(f"{tag:<34} {m['posrate']*100:6.1f}% {m['acc']*100:6.1f}% "
              f"{m['sn']*100:6.1f}% {m['sp']*100:6.1f}% "
              f"{m['prec']*100:6.1f}% {m['mcc']:7.3f}")
        results.append((tag, m, prob))
        return m, prob

    # 不做任何缩放
    evaluate("(不缩放, 原始 ESM 特征)", X)

    for sp in scalers:
        name = os.path.basename(sp)
        try:
            sc = load_any(sp)
        except Exception as e:
            print(f"{name:<34}   ❌ 载入失败: {str(e)[:40]}")
            continue
        nfi = getattr(sc, "n_features_in_", None)
        if nfi is not None and nfi != X.shape[1]:
            print(f"{name:<34}   ⚠️ 特征数不符 ({nfi} vs {X.shape[1]}), 跳过")
            continue
        try:
            Xs = sc.transform(X)
        except Exception as e:
            print(f"{name:<34}   ❌ transform 失败: {str(e)[:40]}")
            continue
        evaluate(name, Xs)

    # ---------- 结论 ----------
    print("\n" + "=" * 78)
    print("📊 结论")
    print("=" * 78)
    if not results:
        sys.exit("❌ 没有任何组合成功运行")

    best = max(results, key=lambda r: r[1]["mcc"])
    print(f"\n✅ 表现最好的组合: {best[0]}")
    b = best[1]
    print(f"   准确率 {b['acc']*100:.2f}% | MCC {b['mcc']:.4f} | "
          f"判阳率 {b['posrate']*100:.1f}%")
    print(f"   论文报告 AMP 任务准确率约 90%, 测试集真实正例率 "
          f"{y.mean()*100:.1f}%")

    if b["acc"] >= 0.85:
        print("\n   → 模型与 scaler 组合【正确】, 能复现论文精度。")
        print("     真实数据上判阳率高, 属于【分布偏移 + 先验差异】,")
        print("     应通过提高阈值/共识来控制, 而非修 pipeline。")
    else:
        print("\n   ⚠️ 即使最优组合也远低于论文精度 → pipeline 存在 bug,")
        print("     可能是 ESM 特征提取方式与训练时不一致。")

    # 当前 pipeline 用的是哪个
    cur = [r for r in results if "working" in r[0].lower()]
    if cur:
        c = cur[0][1]
        print(f"\n📌 当前 pipeline 使用 *_working_scaler.pkl:")
        print(f"   准确率 {c['acc']*100:.2f}% | MCC {c['mcc']:.4f} | "
              f"判阳率 {c['posrate']*100:.1f}%")
        if c["mcc"] < b["mcc"] - 0.02:
            print(f"   ❗ 明显劣于 {best[0]} —— 建议改用后者")
        else:
            print("   ✅ 与最优组合相当")

    # ---------- 阈值扫描 ----------
    print("\n" + "=" * 78)
    print("🎚 阈值扫描 (用最优 scaler, 在带标签测试集上)")
    print("=" * 78)
    prob = best[2]
    print(f"{'阈值':>6} {'判阳率':>8} {'精确率':>8} {'敏感度':>8} {'MCC':>8}")
    for th in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        m = metrics(y, prob, th)
        print(f"{th:6.2f} {m['posrate']*100:7.1f}% {m['prec']*100:7.1f}% "
              f"{m['sn']*100:7.1f}% {m['mcc']:8.3f}")

    # ---------- 真实数据对比 ----------
    if args.sample_fasta and os.path.exists(args.sample_fasta):
        print("\n" + "=" * 78)
        print(f"🧬 真实 sORF 数据分布对比 ({args.sample_fasta})")
        print("=" * 78)
        rs, h = [], None
        with open(args.sample_fasta) as fh:
            for line in fh:
                if line.startswith(">"):
                    h = 1
                elif h:
                    s = line.strip()
                    if 11 <= len(s) <= 180:
                        rs.append(s)
                    if len(rs) >= args.sample_n:
                        break
        print(f"   取样 {len(rs)} 条")
        Xr = encode(rs)
        scb = [s for s in scalers if os.path.basename(s) == best[0]]
        if scb:
            Xr = load_any(scb[0]).transform(Xr)
        pr = np.asarray(kmodel.predict(Xr, batch_size=4096, verbose=0))
        pr = pr[:, 1] if (pr.ndim == 2 and pr.shape[1] == 2) else pr.reshape(-1)
        print(f"\n   {'阈值':>6} {'真实数据判阳率':>14} {'测试集正例判阳率':>16}")
        for th in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
            tr = (prob[y == 1] >= th).mean()
            print(f"   {th:6.2f} {(pr>=th).mean()*100:13.1f}% "
                  f"{tr*100:15.1f}%")
        print("\n   ↑ 若真实数据判阳率远高于文献基准(0.1-1.65%),")
        print("     选一个能把判阳率压到该量级的阈值。")


if __name__ == "__main__":
    main()
