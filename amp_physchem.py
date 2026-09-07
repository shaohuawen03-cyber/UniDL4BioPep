#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肽段理化性质计算(纯 numpy, 无额外依赖)

用于宏基因组 sORF 的 AMP 预筛。参数依据:
  * Macrel 文档 / Zhang & Gallo 2016: 多数 AMP 长 10-50 aa(可达 100),
    净电荷 +2 ~ +11, 约 50% 疏水残基。
  * AMPSphere (Cell 2024): c_AMP 平均长 37 aa, 平均净电荷 +4.7, pI ~10.9。
  * Macrel 在 (meta)genome 中只输出 10-100 aa 的 smORF。
"""

import numpy as np

# pH 7 下的近似净电荷(Lys/Arg +1, Asp/Glu -1, His ~+0.1)
CHARGE = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}

# Eisenberg 共识疏水性标度
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

HYDROPHOBIC = set("AVLIMFWCY")

# pKa (Bjellqvist), 用于等电点计算
PKA_POS = {"N_term": 9.69, "K": 10.5, "R": 12.5, "H": 6.0}
PKA_NEG = {"C_term": 2.34, "D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07}

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
_AA_INDEX = {a: i for i, a in enumerate(AA_LIST)}


def composition_matrix(seqs):
    """返回 (n_seq, 20) 的氨基酸计数矩阵 + 长度数组。"""
    n = len(seqs)
    counts = np.zeros((n, 20), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int32)
    for i, s in enumerate(seqs):
        lengths[i] = len(s)
        for ch in s:
            j = _AA_INDEX.get(ch)
            if j is not None:
                counts[i, j] += 1
    return counts, lengths


def net_charge(counts):
    """pH 7 近似净电荷。"""
    v = np.zeros(20, dtype=np.float32)
    for a, c in CHARGE.items():
        v[_AA_INDEX[a]] = c
    return counts @ v


def mean_hydrophobicity(counts, lengths):
    """Eisenberg 平均疏水性。"""
    v = np.array([EISENBERG[a] for a in AA_LIST], dtype=np.float32)
    return (counts @ v) / np.maximum(lengths, 1)


def hydrophobic_fraction(counts, lengths):
    v = np.zeros(20, dtype=np.float32)
    for a in HYDROPHOBIC:
        v[_AA_INDEX[a]] = 1.0
    return (counts @ v) / np.maximum(lengths, 1)


def _charge_at_ph(counts, ph):
    """给定 pH 下的净电荷(用于二分求 pI)。"""
    n = counts.shape[0]
    q = np.zeros(n, dtype=np.float64)
    # N 端 / C 端
    q += 1.0 / (1.0 + 10 ** (ph - PKA_POS["N_term"]))
    q -= 1.0 / (1.0 + 10 ** (PKA_NEG["C_term"] - ph))
    for a, pka in (("K", PKA_POS["K"]), ("R", PKA_POS["R"]), ("H", PKA_POS["H"])):
        q += counts[:, _AA_INDEX[a]] / (1.0 + 10 ** (ph - pka))
    for a in ("D", "E", "C", "Y"):
        pka = PKA_NEG[a]
        q -= counts[:, _AA_INDEX[a]] / (1.0 + 10 ** (pka - ph))
    return q


def isoelectric_point(counts, n_iter=40):
    """二分法求等电点。"""
    lo = np.full(counts.shape[0], 0.0)
    hi = np.full(counts.shape[0], 14.0)
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        q = _charge_at_ph(counts, mid)
        pos = q > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return ((lo + hi) / 2.0).astype(np.float32)


def compute_all(seqs):
    """一次性计算全部理化描述符。"""
    counts, lengths = composition_matrix(seqs)
    return {
        "length": lengths,
        "net_charge": net_charge(counts).astype(np.float32),
        "pI": isoelectric_point(counts),
        "hydrophobicity": mean_hydrophobicity(counts, lengths).astype(np.float32),
        "hydrophobic_frac": hydrophobic_fraction(counts, lengths).astype(np.float32),
    }


def physchem_pass(props, min_charge=2.0, min_len=10, max_len=100,
                  min_hyd_frac=0.30):
    """AMP 样理化性质筛选掩码。"""
    return (
        (props["net_charge"] >= min_charge)
        & (props["length"] >= min_len)
        & (props["length"] <= max_len)
        & (props["hydrophobic_frac"] >= min_hyd_frac)
    )
