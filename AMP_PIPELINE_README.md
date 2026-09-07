# 宏基因组 sORF 抗菌肽(AMP)共识预测流水线

针对 `comparable_sorf_grouped_catalog/` 下分队列、分阶段(NC→SCS→SCD→MCI→AD)
的 sORF FASTA，进行抗菌肽候选筛选与组间比较。

## 为什么不能只用 UniDL4BioPep

UniDL4BioPep 的 AMP 模型训练集正负比约 **1:2.5**(正 3876 / 负 9552)，
面向"给定一条肽判断是否 AMP"的平衡场景。而真实 (meta)genome 的 smORF 中
AMP 占比远低：Macrel 论文指出真实先验"远接近 1:50 而非 1:3"，实测真实人肠道
宏基因组中 AMP 占 smORF 的比例仅 **0.1–1.65%**
(Santos-Júnior et al., *PeerJ* 2020, DOI 10.7717/peerj.10555)。

先验不匹配会导致输出概率系统性高估——直接单模型跑会得到 80%+ 的"阳性率"，
比文献基准高两个数量级。因此本流水线采用**多工具共识**，这也是该领域的标准做法。

## 流程与文献依据

| 步骤 | 做法 | 依据 |
|---|---|---|
| ① 长度过滤 | 10–100 aa | Macrel 在 (meta)genome 中只输出 10–100 aa 的 smORF, PeerJ 2020 |
| ② 理化预筛 | 净电荷 ≥ +2, 疏水残基比例 ≥ 0.30 | 多数 AMP 电荷 +2~+11、约 50% 疏水残基 (Zhang & Gallo 2016); AMPSphere c_AMP 平均长 37 aa、电荷 +4.7、pI ~10.9 (Cell 2024) |
| ③ Macrel | 随机森林, 精确率优先, 专为低正例先验训练 | PeerJ 2020 |
| ④ UniDL4BioPep | ESM-2 + CNN, 第二票 | Du et al., *Brief Bioinform* 2023, DOI 10.1093/bib/bbad135 |
| ⑤ 共识计数 | `n_tools_positive` 列 | AMPSphere 质控用 6 个额外工具共预测, 98.4% 候选被至少一个支持; 合成前要求 7 法全阳 (Cell 2024)。共识提升特异性/精确率、降低假阳性 (DAMPC, *Sci Data* 2026) |
| ⑥ 基准对照 | 与 0.1–1.65% 区间比较, 输出 verdict | PeerJ 2020 |

同类流程参考：
- **AMPSphere / Cell 2024** (DOI 10.1016/j.cell.2024.05.013): 45.99 亿 smORF → 去冗余 27.2 亿 → Macrel → 863,498 c_AMP; 合成 100 条, 79 条有活性。
- **Ma et al., Nat Biotechnol 2022** (DOI 10.1038/s41587-022-01226-0): LSTM+Attention+BERT 联合流水线, 2,349 候选 → 合成 216, 181 有活性 (>83%)。
- **Cell 2024 SEP** (DOI 10.1016/j.cell.2024.05.031): AmPEP ≥0.5 → 再用 SmORFinder 确认是真实编码基因 → 323 条; 合成 78, 70.5% 有活性。("是不是 AMP"与"是不是真基因"是两道独立关卡)
- **AMP-SEMiner, Cell Reports 2025**: 微调 ESM-2 残基级分类器从 MAG 挖 AMP (与本项目技术路线最接近)。
- **双模型交集实例**: Hybrid model 阈值 0.9 + Macrel 阈值 0.8。
- **三工具投票实例**: Macrel + AxPEP + AMP Scanner v2 → 1,095 smORF 得 51 条高置信; 其中 Macrel 只报 9.8%，说明**不同工具绝对数量不可直接比较**。

## 安装 Macrel

Macrel 已更新至 **1.6.x**(2025-11)，改用 pyrodigal 预测小基因、模型以 ONNX 运行
(运行时不需要 scikit-learn)，`macrel peptides` 可直接接收肽 FASTA。

```bash
conda install -c bioconda macrel
macrel --version
```

未安装时脚本会自动降级为单工具模式并给出警告。

## 用法

```bash
# 试跑
python run_amp_sorf_cohorts.py --cohorts Cohort2 --max-seqs 200000

# 正式(后台, 固定输出目录以支持断点续跑)
nohup python -u run_amp_sorf_cohorts.py \
    --output-dir ~/UniDL4BioPep-main/Predictions_AMP_run1 \
    > amp_run1.log 2>&1 &
echo $! > amp_run1.pid

# 只保留两个工具都判阳的交集
python run_amp_sorf_cohorts.py --min-tools 2

# 关闭理化预筛 / 关闭 Macrel
python run_amp_sorf_cohorts.py --no-physchem-filter --no-macrel
```

## 输出

```
Predictions_AMP_*/
  <Cohort>/<Group>_AMP_hits.csv       候选肽: 序列/长度/净电荷/pI/疏水比例/
                                       Macrel_prob/Macrel_hemolytic/
                                       UniDL_AMP_prob/n_tools_positive
  <Cohort>/<Group>_summary.json       漏斗统计、各工具命中、共识分布、
                                       概率分位数、文献基准对照 verdict
  <Cohort>/<Group>_*_prob_hist.npy    概率直方图(换阈值无需重跑)
  summary_all_groups.tsv              各队列各阶段横向汇总
  .ckpt/                              断点续跑检查点
```

## 组间比较注意事项

1. **用比例，不用绝对条数**——各组测序深度和 sORF 总数差数倍。
2. **优先看 `rate_consensus>=2`**，其次看概率分布(`*_mean_prob`, `*_P99`)。
3. **检查 `lit_verdict`**：若为 `above_range`，说明筛选仍偏松，应提高
   `--min-tools` 或 `--min-charge`。
4. Cohort1/2 与 Cohort3/4 样本高度重叠，Cohort2 是 Cohort1 的两个组，
   报告时不要当作独立重复。

---

## 组间统计分析

预测跑完后运行：

```bash
python amp_group_stats.py Predictions_AMP_run1/
```

输出四张表，解决"绝对数量不可比"：

| 文件 | 内容 |
|---|---|
| `group_stats_rhoAMP.tsv` | ρAMP 归一化密度(c_AMP 数 / smORF 总数) |
| `group_stats_tool_concordance.tsv` | 跨工具 Spearman 一致性(ρ>0.8 才可下结论) |
| `group_stats_group_tests.tsv` | 各组 vs NC 的两比例 z 检验 + BH FDR + fold change |
| `group_stats_ks_tests.tsv` | 概率分布 KS 检验(不依赖阈值) |

详细方法学与文献依据见 **`METHODS_AMP_metagenome.md`**。

## 第三个工具(推荐)

两工具是下限，三工具才能"≥2 票"投票。代码已预留 amPEPpy 接口：

```bash
python run_amp_sorf_cohorts.py --ampep-model /path/to/ampep_model.pkl
```

备选：ampir (R, mature 模型) / AMPScanner v2 / AMPlify。

## AD 关联分析

```bash
conda install -c bioconda cd-hit diamond
python amp_ad_association.py Predictions_AMP_run1/ --cohort Cohort3
```

做四件事：CD-HIT 家族级去冗余(对齐 AMPSphere 75%/90%) → Cochran-Armitage
阶段趋势检验(利用 NC→SCS→SCD→MCI→AD 有序性) → AD/NC 特异家族 →
理化性质阶段漂移。可选 `--known-amp-db` 做新颖性比对。
