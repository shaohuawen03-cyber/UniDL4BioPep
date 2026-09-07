# 方法学记录：分组宏基因组 sORF 抗菌肽预测与阶段间比较

> 项目：阿尔茨海默病进程（NC → SCS → SCD → MCI → AD）肠道宏基因组
> sORF 抗菌肽（AMP）候选筛选与组间差异分析
> 数据：`comparable_sorf_grouped_catalog/`，4 个队列 14 个分组，
> 1,971 MAG / 459 样本 / 约 4.3 亿条 sORF（9.8 GB）

---

## 1. 问题定义与核心风险

### 1.1 类别先验不匹配（最关键）

UniDL4BioPep 的 AMP 模型训练集正负比约 **1:2.5**（正 3,876 / 负 9,552，
经本地 `AMP_train.csv` 核实），面向"给定一条肽，判断是否 AMP"的**平衡场景**。

而真实 (meta)genome 的 smORF 中 AMP 占比极低。Macrel 作者明确指出，真实先验
"远接近 1:50 而非 1:3"，并据此选择高精确率/低召回的分类器；实测 182 个真实人
肠道宏基因组中，**AMP 占 smORF 的比例仅 0.1–1.65%**
（Santos-Júnior et al., *PeerJ* 2020, DOI 10.7717/peerj.10555）。

AMPSphere 的规模可作旁证：45.99 亿条 smORF 去冗余至 27.2 亿条，经 Macrel 筛选
后仅得 **863,498 条 c_AMP**（Santos-Júnior et al., *Cell* 2024,
DOI 10.1016/j.cell.2024.05.013）。

**后果**：本项目初次单模型试跑（阈值 0.9）得到 86–90% 的"阳性率"，比文献基准
高约两个数量级。这不是软件缺陷，而是先验错配的必然结果，**调阈值无法修正**，
只能通过多工具共识与归一化比较来控制。

### 1.2 smORF 本身的假阳性

短肽的基因预测本身假阳性率就很高——不加过滤时可达 **61.2%**
（MACREL, *bioRxiv* 2019.12.17.880385）。因此"是不是 AMP"与"是不是真实编码
基因"是**两道独立关卡**。Cell 2024 的 SEP 研究即用 AmPEP 预测活性后，再用
SmORFinder 确认编码基因真实性（DOI 10.1016/j.cell.2024.05.031）。

---

## 2. 流程设计

```
sORF FASTA（流式读取）
  ├─ ① 长度过滤          10–100 aa
  ├─ ② 理化预筛          净电荷 ≥ +2，疏水残基比例 ≥ 0.30
  ├─ ③ Macrel            随机森林，精确率优先          ← 主筛
  ├─ ③b amPEPpy          随机森林，CTD 特征（可选）    ← 第三票
  ├─ ④ UniDL4BioPep      ESM-2 + CNN                  ← 第二票
  ├─ ⑤ 共识计数          n_tools_positive
  └─ ⑥ 基准对照          与 0.1–1.65% 区间比较
```

### 2.1 各步骤依据

| 步骤 | 参数 | 文献依据 |
|---|---|---|
| ① 长度 | 10–100 aa | Macrel 在 (meta)genome 中只输出 10–100 aa 的 smORF（*PeerJ* 2020） |
| ② 净电荷 | ≥ +2 | 多数 AMP 净电荷 +2~+11、约 50% 疏水残基（Zhang & Gallo 2016，Macrel 文档引用）；AMPSphere c_AMP 平均长 37 aa、平均电荷 **+4.7**、pI ~**10.9**（*Cell* 2024） |
| ② 疏水比例 | ≥ 0.30 | 同上；两亲性有序结构是膜结合/破膜的驱动力 |
| ③ Macrel | 默认阈值 | 22 个混合描述符 + 随机森林，**以极低正负比训练**，模拟真实 (meta)genome 条件（*PeerJ* 2020） |
| ④ UniDL4BioPep | softmax `[:,1]` | ESM-2 (esm2_t6_8M, 320 维) 均值池化 + CNN，20 个数据集中 15 个超 SOTA（Du et al., *Brief Bioinform* 2023, DOI 10.1093/bib/bbad135） |
| ⑤ 共识 | `n_tools_positive` | 见 2.2 |

### 2.2 为什么必须多工具共识

这是该领域的标准做法，**没有任何一篇代表性文献用单一通用分类器直接下结论**：

- **AMPSphere（*Cell* 2024）**：质控阶段用 6 个额外工具（AMPScanner v2、ampir、
  amPEPpy、APIN、AI4AMP、AMPlify）共预测，**98.4% 候选被至少一个其他工具支持**，
  仅 <2% 为 Macrel 独有；挑选合成时**要求 7 个方法全部判阳**。100 条合成肽中
  79 条有活性。
- **Ma et al.（*Nat Biotechnol* 2022, DOI 10.1038/s41587-022-01226-0）**：
  LSTM + Attention + BERT **多模型联合**流水线，2,349 条候选 → 合成 216 条 →
  181 条有活性（>83%）。
- **双模型交集实例**（海洋 MAG）：Hybrid model 阈值 0.9 **交** Macrel 阈值 0.8。
- **三工具投票实例**（肠道 metatranscriptome, *Microb Ecol* 2025,
  DOI 10.1007/s00248-025-02620-2）：Macrel + AxPEP + AMP Scanner v2。
- **DAMPC（*Sci Data* 2026, DOI 10.1038/s41597-026-07521-8）**：定量证明
  consensus 策略"一致提升特异性与精确率、牺牲召回，可有效降低假阳性"。

---

## 3. 跨工具不可比问题及其解决方案

### 3.1 问题的量化证据

同一批 51 条高置信 AMP，三个工具的报出率分别为：
**AxPEP 70.6%、AMP Scanner v2 45.1%、Macrel 9.8%**（*Microb Ecol* 2025）。
差异达 7 倍。原因是各工具的训练负样本构造、阈值尺度、正负比设定完全不同——
已有系统性研究指出 **AMP 预测基准本身因负样本选择而存在偏倚**
（*Brief Bioinform* 2022, PMC9487607），下表节选其对负样本构造的梳理：

| 工具 | 负样本长度范围 | CD-HIT | 类别平衡 |
|---|---|---|---|
| AMPScanner V2 | 与正样本长度分布相同 | 无 | 平衡 |
| ampir-mature | 10–40 | 0.9 | 不平衡 |
| dbAMP | 10–100 | 0.4 | 不平衡 |
| MACREL | — | — | **极低正例比（模拟真实宏基因组）** |

**结论：跨工具的绝对命中数没有任何可比性。**

### 3.2 解决方案（三层，均已实现于 `amp_group_stats.py`）

#### ① 组内归一化：AMP density（ρAMP）

用 **ρAMP = c_AMP 数 / smORF 总数** 替代绝对条数。

依据：AMPSphere 正是以 ρAMP（AMP 密度）做跨物种、跨生境比较，并明确说明是为
"避免生境采样不均带来的偏倚"（*Cell* 2024，"More transmissible species have
lower c_AMP density"一节）。

同时，宏基因组数据是 **compositional（成分型）**数据，直接比较绝对计数会产生
系统性偏倚，这是微生物组统计的共识（McMurdie & Holmes, *PLoS Comput Biol*
2014；Nearing et al., *Nat Commun* 2022 建议"跨方法取共识"）。本项目各组
sORF 总数差异达 2.9 倍（23.4M vs 53.0M），不归一化必然得出错误结论。

#### ② 跨工具：只比较组间**排序**，用 Spearman 检验一致性

不比较工具 A 与工具 B 的数值，而是各工具**独立**计算 NC→AD 的 ρAMP 序列，
再用 Spearman 秩相关检验这些序列的**方向是否一致**。

- ρ > 0.8：各工具组间趋势高度一致 → 结论稳健
- ρ < 0.5：工具间分歧大 → 不应下结论

这与 Nearing et al.（*Nat Commun* 2022）对差异丰度分析的建议一致：
**跨方法取共识（consensus across methods is advised）**。

#### ③ 分布层面比较：KS 检验，不依赖阈值

保存每组完整的概率直方图（1000 bins），用 KS 检验比较各组概率分布与 NC 的
差异。这样即使阈值选择有争议，分布层面的差异仍然成立；换阈值也**无需重跑**。

### 3.3 统计检验

- **两比例 z 检验**：比较各组 ρAMP 与 NC 的差异
- **Benjamini-Hochberg FDR 校正**：多组多工具比较必须校正
- **fold change**：报告效应量，不只报 p 值（样本量极大时 p 值几乎必然显著）

---

## 4. 模型选择：两个够不够？

### 4.1 现状

当前配置为 **Macrel + UniDL4BioPep** 两票，代码已预留 **amPEPpy** 第三票接口。

### 4.2 评估

**两个是可用的下限，三个是推荐配置。** 理由：

1. **两工具只能得到"交集/并集"，无法投票。** 当两者分歧时无法裁决。三个工具
   可用"≥2 票"这一稳健规则。
2. **两工具的 Spearman 只有一个配对**，一致性检验的说服力弱。三工具有 3 个配对。
3. 文献惯例是 3 个及以上：AMPSphere 用 7 个；*Microb Ecol* 2025 用 3 个；
   海洋 MAG 研究用 2 个（属下限）。

**但也不必追求 7 个**。AMPSphere 用 7 个是因为要挑选合成候选（成本极高，需要
最大化精确率）；本项目是**组间比较**，关键是"同一套工具固定不变"，而非工具越多
越好。**3 个工具 + ρAMP 归一化 + 一致性检验**是成本与稳健性的合理平衡。

### 4.3 工具互补性

当前组合在方法学上是互补的，这一点很重要（同质工具堆叠无意义）：

| 工具 | 特征 | 算法 | 训练正负比 |
|---|---|---|---|
| Macrel | 22 个混合局部/全局描述符 | 随机森林 | 极低正例（贴近真实） |
| UniDL4BioPep | ESM-2 蛋白语言模型嵌入（320 维） | CNN | 1:2.5（平衡） |
| amPEPpy | 105 个 CTD 描述符 | 随机森林 | 1:10 |

三者分别代表**理化描述符**、**深度语言模型**、**组成分布描述符**三条独立技术
路线，误差不相关，共识才有意义。

### 4.4 建议的备选第三方工具

若 amPEPpy 安装不便，可替换为（按易用性排序）：

- **ampir**（R 包，`mature` 模型专为成熟肽设计，10–40 aa）
- **AMPScanner v2**（网页/本地，Bi-LSTM，AMPSphere 质控采用）
- **AMPlify**（Bi-LSTM + 注意力，在多个基准中表现最好，
  *BMC Genomics* 2022；ESCAPE 基准 F1 68.5% 为序列模型最高）

---

## 5. 实现要点

### 5.1 工程约束

数据规模约 4.3 亿条序列、9.8 GB，因此：

- **流式 FASTA 读取**：不整文件载入内存
- **分 chunk 处理**（默认 20 万条）：特征提取 → 预测 → 只落盘阳性 → 释放
- **按长度排序做 ESM 前向**：大幅减少 padding，显著提速
- **常数内存统计**：概率直方图（1000 bins）+ 运行均值，与数据量无关
- **断点续跑**：每 chunk 写检查点，中断后自动续跑
- **组内去重**：blake2b 哈希

### 5.2 关键校正记录

| 问题 | 修正 |
|---|---|
| `AMP_minmax_scaler.pkl` 文件损坏（`invalid load key '\x0b'`） | scaler 加载改为多候选依次尝试 + pickle/joblib 双路 + 320 维校验 |
| 初版 `--max-len 100` 过滤掉半数训练分布内序列（训练集实为 11–180，中位数 101） | 改为对齐 Macrel smORF 范围 10–100，并在报告中说明这是**宏基因组场景**的选择而非训练分布 |
| 只统计阳性、无法回溯阈值 | 增加全序列概率直方图，换阈值无需重跑 |

### 5.3 Macrel 版本

Macrel 已更新至 **1.6.0**（2025-11-24，bioconda 1.6.1）。相较旧版：
小基因预测改用 pyrodigal；模型以 **ONNX** 运行（运行时不需要 scikit-learn，
避免与 TensorFlow/PyTorch 环境的依赖冲突）；`macrel peptides` 支持
`--keep-negatives` 输出全部序列分数。

```bash
conda install -c bioconda macrel
macrel --version
```

---

## 6. 使用流程

```bash
# 1) 试跑，确认 lit_verdict
python run_amp_sorf_cohorts.py --cohorts Cohort2 --max-seqs 200000

# 2) 正式运行（后台，固定输出目录以支持断点续跑）
nohup python -u run_amp_sorf_cohorts.py \
    --output-dir ~/UniDL4BioPep-main/Predictions_AMP_run1 \
    --min-tools 2 \
    > amp_run1.log 2>&1 &
echo $! > amp_run1.pid

# 3) 组间统计
python amp_group_stats.py ~/UniDL4BioPep-main/Predictions_AMP_run1/
```

### 6.1 结果自检清单

1. **`lit_verdict` 是否为 `in_range`**（0.1–1.65%）。若 `above_range`，
   提高 `--min-tools` 或 `--min-charge`。
2. **Spearman ρ 是否 > 0.8**。若否，工具间分歧大，不下结论。
3. **fold change 是否有实际意义**。样本量上千万时 p 值几乎必然显著，
   必须看效应量。
4. **`mean_len` / `mean_charge` 是否与 AMPSphere 基准接近**
   （37 aa / +4.7）。偏离过大说明输入或过滤有问题。

---

## 7. 局限性与报告注意事项

1. **这是计算预测，不是实验证据。** 文献中即使经过 7 工具全阳筛选，合成后
   活性率也只有 70–83%。本项目的候选未经任何实验验证。
2. **UniDL4BioPep 的绝对概率不可解释为后验概率**（先验错配 + softmax 未校准），
   仅可用于排序。
3. **绝对条数不可报告**，只报 ρAMP；跨工具数值不可比较。
4. **队列不独立**：Cohort2 是 Cohort1 的两个组；Cohort4 是 Cohort3 的两个组；
   Cohort1/2 与 Cohort3/4 样本高度重叠。报告时不可当作独立重复验证。
5. **未做 smORF 真实性验证**。若要进一步提高可信度，应参照 Cell 2024 SEP 研究
   加入 SmORFinder，或用宏转录组/宏蛋白组交叉验证（AMPSphere 有 20% 的 c_AMP
   获得了此类实验信号支持）。
6. **未做同源去冗余**。AMPSphere 用 CD-HIT 在 100%/85%/75% 三个层级聚类；
   本流程仅做 100% 精确去重，家族层面的冗余可能夸大计数。

---

## 8. 参考文献

1. Santos-Júnior CD, Pan S, Zhao XM, Coelho LP. **Macrel: antimicrobial peptide
   screening in genomes and metagenomes.** *PeerJ* 2020;8:e10555.
   DOI: 10.7717/peerj.10555
2. Santos-Júnior CD, et al. **Discovery of antimicrobial peptides in the global
   microbiome with machine learning.** *Cell* 2024;187(14):3761.
   DOI: 10.1016/j.cell.2024.05.013
3. Du Z, Ding X, Xu Y, Li Y. **UniDL4BioPep: a universal deep learning
   architecture for binary classification in peptide bioactivity.**
   *Brief Bioinform* 2023;24(3):bbad135. DOI: 10.1093/bib/bbad135
4. Ma Y, Guo Z, Xia B, et al. **Identification of antimicrobial peptides from
   the human gut microbiome using deep learning.** *Nat Biotechnol*
   2022;40:921–931. DOI: 10.1038/s41587-022-01226-0
5. **Mining human microbiomes reveals an untapped source of peptide
   antibiotics.** *Cell* 2024;187:3761-3778.e16.
   DOI: 10.1016/j.cell.2024.05.031
6. **Unveiling the evolution of antimicrobial peptides in gut microbes via
   foundation-model-powered framework (AMP-SEMiner).** *Cell Reports* 2025.
7. Sidorczuk K, et al. **Benchmarks in antimicrobial peptide prediction are
   biased due to the selection of negative data.** *Brief Bioinform* 2022.
   PMC9487607
8. **Bioactive Plasmid- and Phage-Encoded Antimicrobial Peptides in the Human
   Gut.** *Microb Ecol* 2025. DOI: 10.1007/s00248-025-02620-2
9. **Dual Activity Microbial Peptides Catalog (DAMPC).** *Sci Data* 2026.
   DOI: 10.1038/s41597-026-07521-8
10. McMurdie PJ, Holmes S. **Waste not, want not: why rarefying microbiome data
    is inadmissible.** *PLoS Comput Biol* 2014;10(4):e1003531.
11. Nearing JT, et al. **Microbiome differential abundance methods produce
    different results across 38 datasets.** *Nat Commun* 2022;13:342.
12. Li C, et al. **AMPlify: attentive deep learning model for discovery of novel
    antimicrobial peptides.** *BMC Genomics* 2022;23:77.

---


---

*文档随代码维护，对应提交见 `git log -- METHODS_AMP_metagenome.md`*

---

## 9. AD 关联分析（`amp_ad_association.py`）

课题落脚点为**疾病关联**，故在候选筛选之上补充以下分析。

### 9.1 家族级去冗余（前置必需步骤）

序列级计数会把同一 AMP 家族的多个变体重复计入，**系统性夸大组间差异**。
AMPSphere（*Cell* 2024）用 CD-HIT 在 **100% / 85% / 75%** 三个层级聚类
（每层称一个 SPHERE，90% 覆盖度），并在家族层面开展生态学分析。

本流程默认采用其 **75% identity / 90% coverage** 参数，统计单元由"序列"
改为"家族"。未安装 cd-hit 时自动降级为序列级并给出警告。

```bash
conda install -c bioconda cd-hit diamond
```

### 9.2 Cochran-Armitage 趋势检验（核心方法）

本数据的独特优势在于 **NC → SCS → SCD → MCI → AD 是有序的 5 阶段**，
而非简单二分类。因此采用 Cochran-Armitage trend test 检验**单调趋势**：

- 结论形式为"AMP 家族密度随疾病进程单调上升"，**强于**"AD 组高于 NC 组"
- 充分利用中间三个阶段的信息，而两两比较会浪费这部分数据
- 配合 BH FDR 校正控制多重比较

**统计单元说明**：趋势检验以家族**检出数**（丰度）为计数单元；同时输出各
阶段的**唯一序列数**（`nseq_*` 列，代表家族内多样性）。两者生物学含义不同，
分别报告。

### 9.3 输出

| 文件 | 内容 |
|---|---|
| `family_trend_test.tsv` | 各家族的 CA 趋势检验（核心结果），含各阶段 ρ |
| `stage_specific_families.tsv` | AD 特异 / NC 特异家族 |
| `physchem_by_stage.tsv` | 候选肽理化性质的阶段漂移 |
| `candidates_annotated.tsv.gz` | 带家族注释的候选肽全表 |

### 9.4 新颖性比对（可选）

提供 `--known-amp-db` 时用 DIAMOND 比对已知 AMP 库，以 **identity < 40%**
判定新颖——该阈值取自 Ma et al.（*Nat Biotechnol* 2022），该研究即以
"多数肽与训练集 AMP 同源性 <40%"论证其发现的新颖性。

可用数据库：APD3、dbAMP、DRAMP、AMPSphere（https://ampsphere.big-data-biology.org/）。

### 9.5 验证记录

- Cochran-Armitage 实现经算例验证：单调上升 z=3.95（p=7.7e-5）、
  无趋势 z=0、单调下降 z=-3.95，符号与量值均正确。
- 端到端验证：在 200 个背景家族中植入 5 个递增家族（2→8→20→45→90），
  全部检出且**零假阳性**。
- BH FDR 实现经标准算例核对。

---

## 10. smORF 真实性过滤（前置第一步）

### 10.1 必要性

短肽的基因预测假阳性率极高：降低长度阈值而不加过滤时，**可达 61.2% 的预测
smORF 是假阳性**（MACREL, *bioRxiv* 2019.12.17.880385，引 Sberro et al. 2019）。

"是不是 AMP"与"是不是真实编码基因"是**两道独立关卡**。Cell 2024 的 SEP 研究
即在 AmPEP 活性预测之外，另用 SmORFinder 确认候选是高置信编码基因
（DOI 10.1016/j.cell.2024.05.031）。

### 10.2 为什么放在第一步

- **算力**：假阳性最高 61.2%，先过滤能让后续所有模型少跑一半以上序列
- **逻辑**：Cell 2024 是先 AmPEP 后 SmORFinder，因其需先把 44 万家族缩到 1.1 万；
  本项目瓶颈在算力（4.3 亿条），倒序更划算

### 10.3 两条路径

| 路径 | 工具 | 输入 | 依据 |
|---|---|---|---|
| A | SmORFinder | **核酸 contigs** | Durrant & Bhatt, *Cell Host Microbe* 2020, DOI 10.1016/j.chom.2020.11.002 |
| B | AntiFam | **蛋白序列** | Eberhardt et al., *Database* 2012 (EBI) |

**SmORFinder** 组合 pHMM（4,500+ smORF 家族）+ 两个深度学习模型（DSN1/DSN2），
预测结果富集 Ribo-seq 翻译信号。默认判据为 pHMM E<1.0 **或** DSN1>0.5 **或**
DSN2>0.5；作者另建议"用宽松阈值但要求三个模型同时满足"来收紧候选，脚本以
`--strict` 实现。**限制**：SmORFinder 是 Prodigal 之上的过滤层，只接受核酸序列。

**AntiFam** 是专门收录"伪基因预测产物"的 HMM 库，被 Pfam/UniProt/GMSC 等用于
清理错误 ORF 预测。本项目输入已是氨基酸 sORF，故以路径 B 为主。

---

## 11. Ma et al. 2022 模型（ATT / LSTM）的引入

### 11.1 只用 ATT+LSTM 而不含 BERT 的依据

原研究（Ma et al., *Nat Biotechnol* 2022）以 ATT+LSTM+BERT 组合取得最高
AUPRC（0.9244）。但**去掉 BERT 有直接文献先例**：

《Antimicrobial Peptides From the Gut Microbiome of the Centenarians》
（*J Gerontol A Biol Sci* 2024）明确写道：在 ATT、LSTM、BERT 三个高精度模型中
**"我们选择其中两个（ATT 和 LSTM）作为本研究使用的模型"**，同样以 score > 0.5
判阳，用于百岁老人肠道宏基因组的 c_AMP 挖掘与人群分组比较——与本项目场景高度
一致。

另有 *Nature* 2024（全球海洋微生物）用全部三个模型，要求**三者 score 均 > 0.5**。
两种做法均有据可依。

### 11.2 实现要点

- **编码方式**复刻官方 `format.pl`：整数映射（A=1 … Y=20）+ **左侧零填充至 300 维**。
  已验证与官方脚本输出一致。
- **长度域限制**：马跃模型训练数据为 **≤50 aa**（`getorf -maxsize 150` nt，
  仓库亦标明"Test set data under 50AA"）。而 Macrel 为 10–100 aa、
  UniDL4BioPep 为 11–180 aa。**三者长度域不一致**，故对 >50 aa 的序列
  输出 `NaN` 而非外推，避免超出训练域的不可信预测。
- **环境隔离**：模型为旧版 Keras `.h5` + 自定义 `Attention_layer`
  （依赖 `keras.engine.topology`），需 TF 1.15 / Keras 2.2.4 独立环境。

### 11.3 当前工具组合

| 工具 | 特征 | 算法 | 长度域 | 训练正负比 |
|---|---|---|---|---|
| Macrel | 22 混合理化描述符 | 随机森林 | 10–100 aa | 极低正例（贴近真实） |
| UniDL4BioPep | ESM-2 嵌入（320 维） | CNN | 11–180 aa | 1:2.5 |
| Ma-ATT | 整数编码序列 | Attention | ≤50 aa | 1:10 |
| Ma-LSTM | 整数编码序列 | LSTM | ≤50 aa | 1:10 |

四条独立技术路线（理化描述符 / 蛋白语言模型 / 注意力 / 循环网络），
误差不相关，共识才有统计意义。

**注意**：因长度域不同，`n_tools_positive` 对 >50 aa 的序列最多只能得 2 票。
统计时应**按长度分层**，或将分析限定在 ≤50 aa 区间以保证各组可比。

---

## 12. 各工具长度域的分别处理

### 12.1 问题

四个工具的原生训练长度域互不相同：

| 工具 | 训练长度域 | 依据 |
|---|---|---|
| Macrel | **10–100 aa** | 在 (meta)genome 中只输出 10–100 aa 的 smORF（*PeerJ* 2020） |
| UniDL4BioPep | **11–180 aa** | 本地 `AMP_train.csv` 核实（中位数 101） |
| Ma-ATT / Ma-LSTM | **≤50 aa** | `getorf -maxsize 150` nt；仓库标明 "Test set data under 50AA" |
| amPEPpy | ~10–200 aa | AmPEP 常用范围 |

若统一取交集，会丢弃大量可分析序列；若统一取并集而不加区分，则等于让模型
在训练域外**外推**，预测不可信。

### 12.2 处理方式

按各工具**自己的域**分别处理：

1. **全局预过滤**取所有启用工具域的**并集**（如 Macrel+UniDL4BioPep → 10–180 aa），
   保证不提前丢数据。
2. **每个工具只对自己域内的序列打分**，域外输出 `NaN`，并记录
   `<tool>_in_domain` 标志列。绝不外推。
3. **命中率分母改为"该工具域内可评序列数"**（`hit_rate_in_domain`），
   而非全部打分序列——否则窄域工具的命中率会被无端稀释。
4. **共识票数按"可评工具数"归一化**：输出 `n_tools_evaluable` 与
   `consensus_frac = n_tools_positive / n_tools_evaluable`。
5. **长度分层统计**：`length_strata` 记录各长度区间（<10 / 10–50 / 50–100 /
   100–180 / >180）内每个工具的命中数，便于分层比较。

### 12.3 共识模式的重要副作用

`--consensus-mode count --min-tools 2` 配合 Macrel + UniDL4BioPep 两工具时，
**>100 aa 的序列永远无法入选**（Macrel 评不了，最多 1 票），会静默丢弃整个
长度区间。实测验证：count 模式下 hits 最大长度被截断在 100 aa。

因此提供 **`--consensus-mode frac`**（默认推荐）：按"判阳工具数 / 可评工具数"
判定。实测该模式正确保留了 166 条 >100 aa 候选（`consensus_frac = 1.0`，
即唯一可评工具判阳）。

**报告时须注意**：不同长度区间的候选，其证据强度不同（可评工具数不同）。
应在结果中标注 `n_tools_evaluable`，或将主分析限定在 **10–100 aa**
（所有工具均可评的区间）以保证证据强度一致，长肽作为补充分析。
