# 方法与结果：肠道宏基因组 sORF 抗菌肽候选筛选，兼论 UniDL4BioPep 在本数据上的适用性

> **一句话结论**：在同一批 sORF 上，Macrel 的判阳率随阈值从 7.33% / 7.35% 单调塌缩到
> 0.000024% / 0.000026%（分别塌缩 **909,034 倍** 与 **285,825 倍**），
> 其中 0.7 档落在文献基准 0.1–1.65% 区间内；
> UniDL4BioPep 即使把阈值提到 0.99 仍判阳 65% 的序列，ρAMP 是文献上限的 13 倍，
> 且它只淘汰掉 6.0–6.6% 的 Macrel 命中。**因此本研究的候选集由 Macrel 决定，
> UniDL4BioPep 在本数据上不具备判别力，不用于筛选。**

---

## 第一部分　方法

### 1.1 输入数据

| 项 | 内容 |
|---|---|
| 目录 | `comparable_sorf_grouped_catalog/` |
| 规模 | 4 个队列 14 个分组，1,971 MAG / 459 样本 / 约 4.3 亿条 sORF（9.8 GB） |
| 本次实际处理 | `Cohort2_Matched265_NCvsAD/`，2 个分组（`Cohort2_Disease_AD` / `Cohort2_Healthy_NC`） |
| 疾病轴 | NC → SCS → SCD → MCI → AD；Cohort2 只含首尾两端（NC / AD） |

### 1.2 第一道关卡：sORF 真实性过滤（AntiFam）

**为什么需要**：短肽基因预测的假阳性率极高——降低长度阈值而不加过滤时，
**可达 61.2% 的预测 smORF 是假阳性**（MACREL, bioRxiv 2019.12.17.880385，
引 Sberro et al. 2019）。"是不是 AMP" 与 "是不是真实编码基因" 是两道**独立**关卡，
不能只做前者。

**做法**（`run_step1_antifam.sh` → `smorf_authenticity.py`，路径 B）：

1. **组内序列去重**（`_dedup_pass`）——宏基因组 sORF 目录中同一短肽常被大量样本
   重复收录，先去重再送检，省掉大部分 hmmsearch 工作量。
2. **`hmmsearch` 比对 AntiFam**，`-E 1e-3`。AntiFam 是 EBI 专门收录
   "伪基因预测产物"（spurious ORF）的 pHMM 库（Eberhardt et al., *Database* 2012），
   被 Pfam / UniProt / GMSC 用于清理错误的 ORF 预测。
3. 命中的序列判为 spurious 并移除，同时输出 `<group>.spurious.txt` 记录。

> 本项目输入已是**氨基酸**序列，所以走 AntiFam（只需蛋白）而非 SmORFinder
> （需核酸 contigs）。SmORFinder 的默认判据是 pHMM E-value < 1.0 或
> DSN1 > 0.5 或 DSN2 > 0.5。

输出目录：`clean_catalog/`。**后续所有预测都在这个目录上做。**

### 1.3 第二道关卡：长度过滤与字符校验

`iter_chunks()` 对流式读入的每条序列依次做：

1. 去空白、转大写、去掉终止密码子 `*`；
2. **长度过滤**：只保留 `min_len ≤ len ≤ max_len`；
3. **字符校验**：只保留 20 种标准氨基酸 `ACDEFGHIKLMNPQRSTVWY`，含 `X`/`U`/`B`/`Z`
   等的一律丢弃；
4. **组内 100% 精确去重**：`blake2b(sequence, digest_size=16)`，同一分组内重复序列只留一条。

长度窗口按**工具域并集**取：

| 跑法 | 长度窗口 | 依据 |
|---|---|---|
| 共识跑 | 10–180 aa | Macrel (10–100) ∪ UniDL4BioPep (11–180) |
| Macrel 单工具 | 10–100 aa | Macrel 训练域（在 (meta)genome 中只输出 10–100 aa 的 smORF，PeerJ 2020） |
| UniDL4BioPep 单工具 | 11–180 aa | AMP 训练集实测长度范围（本地 `AMP_train.csv` 核实，中位数 101） |

### 1.4 去重说明（两处，含义不同）

| 位置 | 粒度 | 作用 |
|---|---|---|
| `smorf_authenticity.py::_dedup_pass` | 组内，序列级 100% | AntiFam 送检前去冗余，纯性能优化 |
| `iter_chunks()` 内 `blake2b` | **组内**，序列级 100% | 预测前保证每条唯一序列只打一次分 |

**注意两个局限**：

1. 去重是**组内**的（`seen` 集合每组重置），**不跨组**。Cohort2 的 AD 组和 NC 组
   各自独立去重。
2. 只做 **100% 精确去重，没有做同源去冗余**。AMPSphere（*Cell* 2024）用 CD-HIT 在
   100% / 85% / 75% 三个层级聚类，每一层称为一个 SPHERE。**家族层面的冗余可能夸大计数**
   ——这是第 1.7 节 AD 关联分析要用 CD-HIT 补的原因。

### 1.5 第三道关卡：理化性质预筛

依据 Zhang & Gallo 2016 与 AMPSphere 的 AMP 理化画像（c_AMP 平均长 37 aa、
平均净电荷 +4.7、pI ~10.9、约 50% 疏水残基），`amp_physchem.py` 纯 numpy 实现：

| 参数 | 值 | 说明 |
|---|---|---|
| `min_charge` | **≥ +2.0** | pH 7 近似净电荷（K/R +1，D/E −1，H +0.1） |
| `min_hydrophobic_frac` | **≥ 0.30** | 疏水残基 `AVLIMFWCY` 占比 |
| 长度 | 同 1.3 | |

**三次跑（共识 / Macrel 单工具 / UniDL 单工具）都用同一套理化参数**，
这是它们可比的前提。pI 用 Bjellqvist pKa 二分法求，只统计不参与筛选。

### 1.6 两个预测工具

#### Macrel 1.6.0

| 项 | 内容 |
|---|---|
| 模型 | 随机森林 / NLR，**ONNX** 运行时（1.6.0 起不依赖 scikit-learn） |
| 调用 | `macrel peptides --fasta F --output D --keep-negatives -t N` |
| 判据 | **`AMP_probability >= 阈值`**（官方默认 0.5） |
| 训练域 | 10–100 aa |
| 设计取向 | 作者明确指出真实 (meta)genome 的 AMP 先验"远接近 1:50 而非 1:3"，据此选择**高精确率 / 低召回**的分类器 |

> ⚠️ **不能用 `AMP_family` 列判阳**。它是**家族名**（ADP / ALP / …）而非二分类标签，
> 配合 `--keep-negatives` 时任何非 `NAMP` 的值都会被误判为阳性，导致 100% 判阳。

#### UniDL4BioPep（AMP 任务）

| 项 | 内容 |
|---|---|
| 特征 | ESM-2 `esm2_t6_8M_UR50D`，取第 6 层残基表征，**只对残基位置 `1:L+1` 做 mean pooling**（不含 CLS / EOS）→ 320 维 |
| 预处理 | `AMP_working_scaler.pkl`（MinMaxScaler，320 维） |
| 分类器 | `AMP_keras_2_best_model.keras`（CNN），末层 `Dense(2, softmax)`，取第 2 列为 AMP 概率 |
| 训练域 | 11–180 aa（`AMP_train.csv` 实测：min 11 / 中位 101 / max 180） |
| 训练集正负比 | **正 3,876 / 负 9,552 = 1:2.46**（`AMP_train.csv` 实测） |
| 测试集正负比 | **正 2,584 / 负 6,369 = 1:2.46**（`AMP_test.csv` 实测） |

> 训练集与测试集**同为 1:2.46 的类别不平衡**，而真实 (meta)genome 的正例先验约 1:50
> 甚至更低。这个错配是第 3.1 节结论的根因。
>
> （`diagnose_amp_model.py` 的文档字符串里写的是"2583 正"，实测是 **2584 正**，
> 差 1 条，此处以实测为准。）

> **关键前提**（`METHODS_AMP_metagenome.md` 第 7.2 条）：
> *UniDL4BioPep 的绝对概率不可解释为后验概率（先验错配 + softmax 未校准），
> 仅可用于排序。*
> 本文的结论正是把这句话落到具体数字上。

### 1.7 下游分析

| 分析 | 脚本 | 做法 |
|---|---|---|
| 组间比较 | `amp_group_stats.py` | ① **ρAMP** = c_AMP 数 / smORF 总数（对齐 AMPSphere，消除采样深度差异，宏基因组是 compositional 数据）；② 跨工具 Spearman 一致性；③ 两比例 z 检验（以 NC 为参照）+ fold_change；④ 概率分布 KS 检验 |
| AD 关联 | `amp_ad_association.py` | ① CD-HIT 家族级去冗余（75% identity / 90% coverage，对齐 AMPSphere）；② 家族 × 阶段矩阵；③ **Cochran-Armitage 趋势检验**（利用 NC→SCS→SCD→MCI→AD 的有序性）；④ 阶段特异家族；⑤ 理化性质漂移 |

### 1.8 为什么要拆开单独跑

原流程（`run_full_pipeline.sh` 第 2 步）是**双工具共识**：序列先过 ESM-2 + UniDL4BioPep，
再与 Macrel 取交集（`--min-consensus-frac 1.0`）。两个问题：

1. **归因不清**：最终候选集是交集，无法回答"这个数是谁决定的"。
2. **速度**：Macrel 要白等 ESM-2 + CNN 那一段时间，而那段时间占总耗时的大头。

拆开后：

| | 共识跑 | Macrel 单工具 | UniDL4BioPep 单工具 |
|---|---|---|---|
| 脚本 | `run_full_pipeline.sh` | `run_step2a_macrel_only.sh` | `run_step2b_unidl_only.sh` |
| 加载 ESM-2 / TensorFlow | 是 | **否** | 是 |
| 阈值 | UniDL 0.99 / Macrel 0.9 | Macrel 0.5 | UniDL 0.99 |
| 输出目录 | `Predictions_AMP_final/` | `Predictions_Macrel_final/` | `Predictions_UniDL_final/` |

---

## 第二部分　结果

### 2.1 漏斗

| | Disease_AD | Healthy_NC |
|---|---|---|
| AntiFam 过滤后原始 smORF | **24,813,729** | **23,347,543** |
| 长度 + 字符 + 去重 + 理化预筛后进入打分 | **8,136,428**（32.8% of raw） | **7,703,149**（33.0% of raw） |

三次跑的 `n_raw` 与 `n_scored` **逐位相同**，且 Macrel@0.9 的命中数在共识跑与
Macrel 单工具跑之间**完全一致**（2,461 / 2,416）——说明三次跑处理的是同一批序列，
拆分脚本的口径与原流程对得上。

### 2.2 Macrel 单工具跑（1.57 小时，比共识跑快 2.8 倍）

阈值敏感性（判阳率分母 = **原始 smORF**，与文献基准同口径）：

| 阈值 | Disease_AD | 占 raw | Healthy_NC | 占 raw | 对文献基准 0.1–1.65% |
|---|---|---|---|---|---|
| 0.5 | 1,818,067 | 7.327% | 1,714,951 | 7.345% | 高于上限 4.4 倍 |
| **0.7** | **266,694** | **1.075%** | **254,998** | **1.092%** | **✅ 落在区间内** |
| 0.9 | 2,461 | 0.0099% | 2,416 | 0.0103% | 低于下限 10 倍 |
| 0.95 | 88 | 0.0004% | 82 | 0.0004% | 远低于 |
| 0.99 | 2 | 0.000024% | 6 | 0.000026% | 远低于 |

**Macrel@0.7 是唯一落在文献基准区间内的档位。** 这个数是从已经跑完的概率直方图里
读出来的（`summary.json` 的 `prob_distribution.Macrel.counts_at_threshold`），
换阈值不需要重跑。

### 2.3 UniDL4BioPep（阈值 0.99）

| | Disease_AD | Healthy_NC |
|---|---|---|
| 判阳条数 | **5,339,359** | **5,033,875** |
| 占打分序列 | **65.62%** | **65.35%** |
| ρAMP（/ 原始 smORF） | **0.2152** | **0.2156** |
| 对文献基准上限 1.65% | **13.0 倍** | **13.1 倍** |

把阈值从 0.5 一路提到 0.99，判阳率几乎没有变化——概率挤在高位。

### 2.4 判别力对比（核心证据）

| 工具 | 分组 | 阈值区间 | 判阳率变化 | 塌缩倍数 |
|---|---|---|---|---|
| **Macrel** | Disease_AD | 0.5 → 0.99 | 7.3269% → 0.000024% | **909,034 倍** |
| **Macrel** | Healthy_NC | 0.5 → 0.99 | 7.3453% → 0.000026% | **285,825 倍** |
| **UniDL4BioPep** | 两组 | 0.5 → 0.999 | 概率挤在高位 | **阈值几乎拧不动** |

两组分别塌缩 909,034 倍与 285,825 倍——**数量级不同但方向一致**，
都说明 Macrel 的概率有区分度，阈值是有意义的旋钮，能把它调进文献基准区间。
UniDL4BioPep 的概率挤在一起，从 0.99 拧到 0.999 才刚开始起作用。

### 2.5 共识跑里 UniDL4BioPep 那一票的实际贡献

| | Macrel@0.9 命中 | 共识（两票都阳） | UniDL 淘汰掉的 |
|---|---|---|---|
| Disease_AD | 2,461 | 2,313 | **6.01%** |
| Healthy_NC | 2,416 | 2,256 | **6.62%** |

淘汰率 < 10% → **"双工具共识"在效果上等价于 Macrel 单筛**。

### 2.6 组间差异（AD vs NC）——阴性结果

| 工具 | ρAMP (AD) | ρAMP (NC) | fold_change | p | 怎么读 |
|---|---|---|---|---|---|
| UniDL4BioPep | 0.215178 | 0.215606 | 0.998（**−0.20%**） | 0.000299 | p 显著纯粹因分母 2300 万；−0.2% 无实际意义 |
| Macrel@0.9 | 0.000099 | 0.000103 | 0.958（−4.16%） | 0.138 | 不显著 |
| 共识 ≥2 | 0.000093 | 0.000097 | 0.965（−3.5%） | 0.224 | 不显著 |
| KS 检验 (UniDL) | — | — | D = 0.0040 | 4.55e-54 | 同样 p 显著、效应量 ≈ 0 |

**在 Cohort2 上，AD 与 NC 的 AMP 密度差异检测不出来。** 这是阴性结果，
如实报告比硬凑显著性站得住——`amp_group_stats.py` 自己的报告要点第 3 条就是
"样本量上千万时 p 值几乎必然显著，必须看效应量"。

---

## 第三部分　结论与讨论

### 3.1 UniDL4BioPep 在本数据上不能用于筛选

三条独立证据，指向同一个结论：

1. **绝对判阳率离谱**：p ≥ 0.99 下判阳 65.62% / 65.35% 的打分序列，
   ρAMP 0.2152 / 0.2156，是文献上限 1.65% 的 **13.0 / 13.1 倍**。
2. **阈值不起作用**：从 0.5 到 0.99 判阳率几乎不动，说明概率没有区分度，
   不存在"把阈值调对就能用"的可能。
3. **对候选集没有贡献**：在双工具共识里只淘汰掉 **6.0–6.6%** 的 Macrel 命中。

**根因是类别先验错配**，不是实现 bug：UniDL4BioPep 的 AMP 模型训练集正负比为
**1:2.46**（正 3,876 / 负 9,552，实测），测试集同为 1:2.46（正 2,584 / 负 6,369），
面向"给定一条肽判断是否 AMP"的场景；
而真实 (meta)genome 的 smORF 中 AMP 占比只有 **0.1–1.65%**（Santos-Júnior et al.,
*PeerJ* 2020）。Macrel 作者正是因为这个先验差异（"远接近 1:50 而非 1:3"）
才选择高精确率 / 低召回的设计。

### 3.2 结论的适用边界（必须写进报告）

- 本结论是**"在本宏基因组 sORF 数据上、在 ESM-2 320 维 mean-pooling 特征 + 该 scaler +
  该 CNN 权重这条实现路径下"**，不是"UniDL4BioPep 这个模型不好"。
  它在自己的平衡测试集上的表现需要另用 `diagnose_amp_model.py` 单独核验。
- `METHODS_AMP_metagenome.md` 第 7.2 条本来就写明其绝对概率不可解释为后验概率，
  仅可用于排序。本研究是把这句话量化了。

### 3.3 本研究的候选集

**由 Macrel 单独决定，建议阈值 0.7**（1.075% / 1.092% of raw，落在文献基准区间内）。
双工具共识的框架可以保留在方法里，但要如实说明 UniDL4BioPep 那一票
在本数据上不改变结果。

### 3.4 局限

1. **只做了 Cohort2 的 2 个分组**（NC / AD）。Cochran-Armitage 趋势检验需要
   ≥3 个**有序**阶段，两组时它退化成两比例检验，`amp_ad_association.py`
   会明确提示并跳过。真正的阶段趋势要跑 Cohort1 或 Cohort3。
2. **队列不独立**：Cohort2 是 Cohort1 的两个组，Cohort4 是 Cohort3 的两个组，
   Cohort1/2 与 Cohort3/4 样本高度重叠，**不能当独立重复验证**。
3. **只做 100% 精确去重，没有同源去冗余**，家族层面冗余可能夸大计数。
4. **未做 smORF 的转录/翻译层面验证**。AntiFam 只解决"是不是伪基因预测产物"，
   不等于"确实在表达"。
5. **这是计算预测，不是实验证据。** 文献中即使 7 工具全阳筛选，
   合成后活性率也只有 70–83%。表述必须用"候选"。
6. **TensorFlow 未用上 GPU**（`libcufft.so.10` / `libcusparse.so.11` 缺失 →
   `Skipping registering GPU devices`），CNN 在 CPU 上跑。**结果正确，只是慢**
   ——共识跑 4.43 h vs Macrel 单工具 1.57 h 的差距里有一部分来自这里。

---

## 附录 A　口径速查

三个分母并存，写报告时必须说清用哪个：

| 分母 | 字段 | 用途 |
|---|---|---|
| 原始 smORF | `funnel.n_raw` | **与文献基准 0.1–1.65% 对比必须用这个** |
| 打分序列 | `funnel.n_scored` | 长度 + 字符 + 去重 + 理化预筛后 |
| 域内可评 | `n_evaluable_per_tool.<工具>` | 该工具训练域内的序列数（Macrel 只到 100 aa） |

两个工具**长度域不同**（Macrel 10–100 aa，UniDL4BioPep 11–180 aa），
**绝对判阳数不可直接相比**。要同口径就给 Macrel 传 `MAX_LEN=180`，
并在报告里说明 100–180 段对 Macrel 是域外。

## 附录 B　复现命令

```bash
# 第 1 步: AntiFam 真实性过滤
bash run_step1_antifam.sh

# 第 2a 步: Macrel 单工具全量
nohup bash run_step2a_macrel_only.sh > macrel_full.log 2>&1 &

# 第 2b 步: UniDL4BioPep 单工具全量
THRESHOLD=0.99 nohup bash run_step2b_unidl_only.sh > unidl_full.log 2>&1 &

# 第 3 步: 剩余全部（组间统计 / AD 关联 / 对比表 / 组装报告），核数上限 32
nohup bash run_all_remaining.sh > all_remaining.log 2>&1 &

# 单工具跑做 AD 关联必须传 --min-tools 1（默认 2 会把候选清空）
python amp_ad_association.py Predictions_Macrel_final --min-tools 1 --cohort Cohort2
```

## 附录 C　参考文献

- Santos-Júnior et al. *PeerJ* 2020, DOI 10.7717/peerj.10555（Macrel；真实人肠道
  宏基因组 AMP 占 smORF 0.1–1.65%）
- MACREL, bioRxiv 2019.12.17.880385，引 Sberro et al. 2019（smORF 预测假阳性可达 61.2%）
- Eberhardt et al. *Database* 2012（AntiFam）
- Durrant & Bhatt, *Cell Host Microbe* 2020, DOI 10.1016/j.chom.2020.11.002（SmORFinder）
- AMPSphere, *Cell* 2024, DOI 10.1016/j.cell.2024.05.031（ρAMP；CD-HIT 100/85/75% SPHERE）
- Zhang & Gallo 2016（AMP 理化画像）
- Ma et al. *Nat Biotechnol* 2022, DOI 10.1038/s41587-022-01226-0（合成验证活性率 70–83%）
