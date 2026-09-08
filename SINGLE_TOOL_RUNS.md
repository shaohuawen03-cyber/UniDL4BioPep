# 单工具拆分跑：Macrel 与 UniDL4BioPep 各自全量

> 目的：把原来"双工具共识"流程拆成两条独立的单工具全量跑，
> 让每个工具的判阳率、阈值敏感性、以及它对最终候选集的**实际贡献**
> 都能单独量化。

---

## 0. 为什么要拆

原来 `run_full_pipeline.sh` 第 2 步是共识流程：序列先过 ESM-2 编码 +
UniDL4BioPep 打分，再与 Macrel 取交集（`--min-consensus-frac 1.0`）。
这带来两个问题：

1. **归因不清**：最终候选集是两个工具的交集，无法回答"这个数是谁决定的"。
2. **速度**：Macrel 要白等 ESM-2 + CNN 那一段时间，而那段时间在总耗时里占大头。

拆开之后：

| | Macrel 单工具 | UniDL4BioPep 单工具 |
|---|---|---|
| 脚本 | `run_step2a_macrel_only.sh` → `run_macrel_only.py` | `run_step2b_unidl_only.sh` → `run_amp_sorf_cohorts.py --no-macrel` |
| 依赖 | numpy / pandas + macrel 二进制 | 额外需要 torch / esm / tensorflow / keras |
| 是否加载 ESM-2 | **否** | 是 |
| 是否跑 CNN | **否** | 是 |
| 长度域 | 10–100 aa（Macrel 训练域） | 11–180 aa（AMP 训练域） |
| 输出目录 | `Predictions_Macrel_final/` | `Predictions_UniDL_final/` |
| 相对耗时 | 快（只有 Macrel 的 ONNX 前向 + 理化预筛） | 慢（ESM-2 编码 + CNN 预测） |

两者的输出文件结构**完全同构**，`amp_group_stats.py` 与
`amp_ad_association.py` 可以直接读任意一个目录。

---

## 1. 跑之前：清理上一次的结果

```bash
# 先看要删什么（默认不删）
bash clean_previous_runs.sh

# 确认后真删
APPLY=1 bash clean_previous_runs.sh
```

默认**保留** `clean_catalog/`（AntiFam 过滤后的 sORF，重建要重跑第 1 步，很贵）。
只有确定要连第 1 步一起重做时才加 `KEEP_CLEAN=0`。

清理范围：`Predictions_AMP_final/`（原共识跑）、`Predictions_Macrel_final/`、
`Predictions_UniDL_final/`、`Predictions_*_run*/`、`pipeline_*.log`、
`macrel_only_*.log`、`unidl_only_*.log`、`full_run.log`、`full_run.pid`、`__pycache__/`。

> ⚠️ 原共识跑的 `Predictions_AMP_final/` 里的 `*_AMP_hits.csv` 保存了
> `Macrel_prob` 和 `UniDL_AMP_prob` 两列原始概率。**如果还想用它做第 3 节的
> 淘汰率分析，先把它挪到别处再清理**：
> ```bash
> mv ~/UniDL4BioPep-main/Predictions_AMP_final ~/UniDL4BioPep-main/Predictions_AMP_consensus_keep
> ```

---

## 2. 第 2a 步：Macrel 单工具全量（先跑这个）

```bash
cd ~/UniDL4BioPep-main

nohup bash run_step2a_macrel_only.sh > macrel_full.log 2>&1 &
echo $! > macrel_full.pid
tail -f macrel_full.log
```

只跑一个队列：

```bash
bash run_step2a_macrel_only.sh Cohort2
```

可调参数（都是环境变量）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MACREL_TH` | `0.5` | Macrel 官方判阳阈值。之前共识流程用的 0.9 是精确率优先；单工具全量建议回到 0.5，其余档位看 summary 里的多档阈值表即可，**不用重跑** |
| `MACREL_BIN` | `~/miniconda3/envs/env_macrel/bin/macrel` | |
| `MACREL_THREADS` | `8` | |
| `MIN_CHARGE` | `2.0` | 理化预筛净电荷下限 |
| `MAX_LEN` | `100` | Macrel 训练域上界。传 `180` 可与 UniDL4BioPep 同口径，但 100–180 段对 Macrel 属**域外**，不计入其判阳 |
| `CHUNK` | `200000` | 每 chunk 的序列数（长度过滤后计） |
| `OUT` | `$ROOT/Predictions_Macrel_final` | |
| `FORCE` | `0` | `1` = 忽略已有 summary/检查点强制重跑 |

想看**纯 Macrel**判阳率（不做理化预筛）：

```bash
python -u run_macrel_only.py --catalog-dir ~/UniDL4BioPep-main/clean_catalog \
    --output-dir ~/UniDL4BioPep-main/Predictions_Macrel_nofilter \
    --macrel-bin ~/miniconda3/envs/env_macrel/bin/macrel \
    --macrel-threads 8 --no-physchem-filter
```

断点续跑：重新执行同一条命令即可，已完成的组会自动跳过，
未完成的组从最后一个 chunk 之后继续（检查点在 `<OUT>/.ckpt/`）。

---

## 3. 第 2b 步：UniDL4BioPep 单工具全量

```bash
cd ~/UniDL4BioPep-main

THRESHOLD=0.99 nohup bash run_step2b_unidl_only.sh > unidl_full.log 2>&1 &
echo $! > unidl_full.pid
tail -f unidl_full.log
```

只跑一个队列：

```bash
THRESHOLD=0.99 bash run_step2b_unidl_only.sh Cohort2
```

这一步就是原来的 `run_amp_sorf_cohorts.py` 加 `--no-macrel --no-ampep`，
逻辑没有改动，长度域自动变成 UniDL4BioPep 的 11–180 aa。

跑完自检会打印**多档阈值下的判阳条数**（0.5 / 0.9 / 0.95 / 0.99 / 0.999）。
这张表是关键证据：

> 如果把阈值推到 0.999，判阳率仍然远高于文献基准 1.65%，
> 说明 UniDL4BioPep 在本数据上不具备判别力，而不是阈值没调好。

`METHODS_AMP_metagenome.md` 第 7 节其实已经写明：
*"UniDL4BioPep 的绝对概率不可解释为后验概率（先验错配 + softmax 未校准），
仅可用于排序。"* 多档阈值表就是把这句话落到具体数字上。

---

## 4. 出对比结论

```bash
python compare_macrel_vs_unidl.py \
    --macrel-dir ~/UniDL4BioPep-main/Predictions_Macrel_final \
    --unidl-dir  ~/UniDL4BioPep-main/Predictions_UniDL_final \
    --out        ~/UniDL4BioPep-main/cmp

# 加上原来那次共识跑（如果保留了），多出一张"淘汰率"表
python compare_macrel_vs_unidl.py \
    --macrel-dir ~/UniDL4BioPep-main/Predictions_Macrel_final \
    --unidl-dir  ~/UniDL4BioPep-main/Predictions_UniDL_final \
    --consensus-dir ~/UniDL4BioPep-main/Predictions_AMP_consensus_keep \
    --macrel-th 0.9 --consensus-th 0.99 \
    --out ~/UniDL4BioPep-main/cmp
```

输出四张表：

| 表 | 内容 | 用来说明什么 |
|---|---|---|
| ① | 两个工具各自的分组判阳率 + 是否落在文献基准 0.1–1.65% | 哪个工具的绝对判阳率合理 |
| ② | 多档阈值敏感性（占域内可评序列比例） | 提高阈值能否把判阳率压进基准区间 |
| ③ | 共识跑里 UniDL4BioPep 各阈值对 Macrel 命中的**淘汰率** | **淘汰率≈0 就是"UniDL4BioPep 那一票不构成筛选"的直接证据** |
| ④ | 结论汇总 | 直接可引 |

`--macrel-th` 和 `--consensus-th` 要填**共识跑当时真正用的阈值**
（那次是 `MACREL_TH=0.9`、`THRESHOLD=0.99`），否则第 ③ 张表口径不对。

第 ③ 张表不需要重跑任何预测——它直接读共识跑
`*_AMP_hits.csv` 里已经保存的 `Macrel_prob` / `UniDL_AMP_prob` 两列。

---

## 5. 下游统计（组间比较 / AD 关联）

```bash
# Macrel 单工具
python amp_group_stats.py      ~/UniDL4BioPep-main/Predictions_Macrel_final
python amp_ad_association.py   ~/UniDL4BioPep-main/Predictions_Macrel_final \
    --min-tools 1 --cohort Cohort3

# UniDL4BioPep 单工具
python amp_group_stats.py      ~/UniDL4BioPep-main/Predictions_UniDL_final
python amp_ad_association.py   ~/UniDL4BioPep-main/Predictions_UniDL_final \
    --min-tools 1 --cohort Cohort3
```

> ### ⚠️ 单工具跑必须传 `--min-tools 1`
> `amp_ad_association.py` 的 `--min-tools` **默认是 2**（为双工具共识设计）。
> 单工具跑的 `n_tools_positive` 最大只有 1，用默认值会把候选**全部过滤成 0 条**。

> ### ⚠️ `run_full_pipeline.sh` 第 4 步不含 Cohort2
> `run_full_pipeline.sh` 里 AD 关联那一步写死了 `for c in Cohort1 Cohort3`。
> 现在跑的是 `Cohort2_Matched265_NCvsAD`，要单独补：
> ```bash
> python amp_ad_association.py <结果目录> --min-tools 1 --cohort Cohort2
> ```
> （`--cohort` 是子串匹配，`Cohort2` 能命中 `Cohort2_Matched265_NCvsAD`。）

> ### ⚠️ 跨工具一致性检验会被跳过
> `amp_group_stats.py` 的 Spearman 一致性检验要求每个队列 ≥3 个分组
> 且 ≥2 个工具。单工具跑 + Cohort2 只有 2 组，这一项会打印
> "(工具数或组数不足, 跳过)"。这是预期的，不是报错。

---

## 6. 结果文件夹结构

两个目录结构一致：

```
Predictions_Macrel_final/                     Predictions_UniDL_final/
├── summary_all_groups.tsv                    ├── summary_all_groups.tsv
├── group_stats_*.tsv          (第 5 步产出)   ├── group_stats_*.tsv
├── AD_association/            (第 5 步产出)   ├── AD_association/
├── .ckpt/                                    ├── .ckpt/
│   └── <Cohort>__<Group>.ckpt.json           │   └── <Cohort>__<Group>.ckpt.json
└── <Cohort>/                                 └── <Cohort>/
    ├── <Group>_AMP_hits.csv     ← 候选肽全表      ├── <Group>_AMP_hits.csv
    ├── <Group>_summary.json     ← 核心汇总        ├── <Group>_summary.json
    ├── <Group>_summary.tsv                        ├── <Group>_summary.tsv
    └── <Group>_Macrel_prob_hist.npy               └── <Group>_AMP_prob_hist.npy
```

`<Group>_summary.json` 里可以直接引用的字段：

| 字段 | 含义 |
|---|---|
| `funnel.n_raw` → `n_after_len` → `n_after_physchem` → `n_scored` | 逐级漏斗条数 |
| `n_hits.<工具>` | 判阳条数 |
| `n_evaluable_per_tool.<工具>` | 该工具**域内可评**条数（分母，不是 n_scored） |
| `hit_rate_in_domain` | 判阳 / 域内可评 |
| `literature_benchmark.observed_rate_vs_raw_smorf` | 判阳 / **原始** smORF 数（与文献基准同口径） |
| `literature_benchmark.verdict` | `in_range` / `above_range` / `below_range` |
| `prob_distribution.<工具>.counts_at_threshold` | **多档阈值下的判阳条数**（换阈值不用重跑） |
| `length_strata` | 按长度分层的条数与命中数 |

---

## 7. 口径与报告注意事项

1. **判阳率的分母要说清楚。** 三个分母并存：`n_raw`（原始）、`n_scored`
   （长度+理化过滤后）、`n_evaluable`（工具域内）。与文献基准 0.1–1.65%
   对比时必须用 `observed_rate_vs_raw_smorf`（分母 = `n_raw`）。
2. **两个工具的长度域不同**，绝对判阳数不可直接相比。Macrel 只看 10–100 aa，
   UniDL4BioPep 看 11–180 aa。要同口径就把 `MAX_LEN=180` 传给 Macrel，
   并在报告里说明 100–180 段对 Macrel 是域外。
3. **只报 ρAMP，不报绝对条数**（`amp_group_stats.py` 的既有约定）。
4. **队列不独立**：Cohort2 是 Cohort1 的两个组，Cohort4 是 Cohort3 的两个组，
   Cohort1/2 与 Cohort3/4 样本高度重叠，不能当独立重复验证
   （`METHODS_AMP_metagenome.md` 第 7.4 条）。
5. **这是计算预测，不是实验证据。** 文献中即使 7 工具全阳筛选，
   合成后活性率也只有 70–83%。表述应为"候选"。
6. **"UniDL4BioPep 筛选没效果"这个结论要用第 ③ 张表的淘汰率来支撑**，
   并且要说清是在**哪个阈值**下成立的。阈值不同结论可能不同：
   淘汰率在 0.99 下可能接近 0，在 0.999 下可能不可忽略。
   把两个数都写进报告，比只写一个更站得住。

---

## 8. 已知问题记录

| 问题 | 处理 |
|---|---|
| `amp_ad_association.py` 在候选量小、家族表为空时崩在 `KeyError: 'CA_p'` | 已修：家族表为空时打印提示并写出带表头的空 TSV，不再崩溃。可下调 `--min-family-count`（默认 5）到 2 或 1 后重跑 |
| 单工具跑被 `amp_ad_association.py` 默认的 `--min-tools 2` 清空 | 必须显式传 `--min-tools 1`（见第 5 节） |
| TensorFlow 报 `Could not load dynamic library 'libcufft.so.10'` / `libcusparse.so.11` → `Skipping registering GPU devices` | CNN 落到 CPU 上跑。**结果正确，只是慢。** `LD_LIBRARY_PATH` 里是 cuda-12.8，而 TF 找的是 CUDA 11 时代的 soname，版本对不上 |
| 日志里 `🖥 设备: cuda` 但 TF 没用上 GPU | 那行是 `torch.cuda.is_available()`，只对 PyTorch/ESM-2 成立，与 TensorFlow 无关 |
| Macrel 命中率偏低（此前 `MACREL_TH=0.9` 时约 0.029%，低于文献基准下限 0.1%） | 0.9 是精确率优先。单工具全量回到官方默认 0.5，再用 summary 的多档阈值表回看 0.9/0.95 |
