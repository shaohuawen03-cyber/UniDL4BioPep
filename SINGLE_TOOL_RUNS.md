# Macrel 单工具补跑 + 与 UniDL4BioPep 对比

> 你的计划：**正在跑的那个任务不停**，等它跑完后**单独补一次 Macrel 全量**，
> 然后对比两个工具。本文档就是这个流程。
>
> **不清空任何已有结果。** 所有输出都写到新目录，互不覆盖。

---

## 0. 那次跑的不是"UniDL4BioPep 单工具"，是双工具共识

`run_full_pipeline.sh` 第 2 步的标题就是 `▶ [2/4] Macrel + UniDL4BioPep 共识预测`，
参数里同时有 `--macrel-threshold` 和 `--threshold`，**没有 `--no-macrel`**。
你日志里这两行也对得上：

```
🗳 共识工具数: 2  (模式: frac)
   chunk32: 打分 2,935,609 | AMP 1,932,099 | Macrel 860 | 共识≥2 810
```

`AMP` 和 `Macrel` 是**每个工具各自的判阳数**（`run_amp_sorf_cohorts.py:931`），
`共识≥2` 才是交集。所以它是一次双工具共识跑。

### 但这反而省了你一大步

共识跑的 `summary.json` 里，**两个工具的计数和多档阈值分布是分开存的**：

| 字段 | 内容 |
|---|---|
| `n_hits.AMP` | UniDL4BioPep 自己的判阳数 |
| `n_hits.Macrel` | Macrel 自己的判阳数 |
| `n_evaluable_per_tool.AMP` / `.Macrel` | 各自的域内可评分母 |
| `prob_distribution.AMP.counts_at_threshold` | UniDL4BioPep 在 **0.5 / 0.9 / 0.95 / 0.99 / 0.999** 五档下的判阳数 |
| `prob_distribution.Macrel.counts_at_threshold` | Macrel 在 **0.5 / 0.7 / 0.9 / 0.95 / 0.99** 五档下的判阳数 |

也就是说：**"UniDL4BioPep 在各阈值下判阳多少、判阳率多高"这张表，
现在这次跑完就有了，不需要再单独跑一遍 ESM-2 + CNN。**

`compare_macrel_vs_unidl.py` 已经支持直接把共识跑目录当 `--unidl-dir` 传进去，
它会自动从里面取 `AMP` 那一栏，并且**按该工具自己的计数重算判阳率**
（不会误用 `literature_benchmark` 里那个共识数）。

### 唯一还缺的东西

共识跑的 `*_AMP_hits.csv` **只存交集**（`共识≥2`，那 810 条），
不含"UniDL4BioPep 判阳但 Macrel 判阴"的 193 万条。

所以要不要再单独跑一次 UniDL4BioPep（`run_step2b_unidl_only.sh`），
取决于你要不要那份 **UniDL4BioPep 单独的候选肽序列清单**：

| 你的目的 | 要不要单独跑 UniDL4BioPep |
|---|---|
| 只要判阳率、阈值敏感性、"筛选有没有效果"的结论 | **不用**，共识跑的 summary 就够 |
| 要拿 UniDL4BioPep 单独的候选肽去做 CD-HIT 家族聚类 / AD 关联 | 要，`run_step2b_unidl_only.sh` |

---

## 1. 等当前任务跑完

```bash
# 看进度
tail -f ~/UniDL4BioPep-main/full_run.log

# 确认是否已结束
ps -p "$(cat ~/UniDL4BioPep-main/full_run.pid)" && echo "还在跑" || echo "已结束"
```

**别删 `Predictions_AMP_final/`。** 它是后面对比的输入。

---

## 2. 单独跑 Macrel 全量

不加载 ESM-2、不加载 TensorFlow，只依赖 numpy / pandas + macrel 二进制，
比共识跑快得多。

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

输出目录 `Predictions_Macrel_final/`，**与共识跑的 `Predictions_AMP_final/` 分开**，
不会互相覆盖。

可调参数（都是环境变量）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MACREL_TH` | `0.5` | Macrel 官方判阳阈值。共识跑用的是 0.9（精确率优先）。这里回到 0.5，其余档位看 summary 的多档阈值表，**不用重跑** |
| `MACREL_BIN` | `~/miniconda3/envs/env_macrel/bin/macrel` | |
| `MACREL_THREADS` | `8` | |
| `MIN_CHARGE` | `2.0` | 理化预筛净电荷下限，与共识跑一致 |
| `MAX_LEN` | `100` | Macrel 训练域上界。传 `180` 可与 UniDL4BioPep 同口径，但 100–180 段对 Macrel 属**域外**，不计入其判阳 |
| `CHUNK` | `200000` | 每 chunk 序列数（长度过滤后计） |
| `OUT` | `$ROOT/Predictions_Macrel_final` | |
| `FORCE` | `0` | `1` = 忽略已有 summary/检查点强制重跑 |

断点续跑：重新执行同一条命令即可，已完成的组自动跳过，
未完成的组从最后一个 chunk 之后继续（检查点在 `<OUT>/.ckpt/`）。

想看**纯 Macrel**判阳率（关掉理化预筛）：

```bash
python -u run_macrel_only.py \
    --catalog-dir ~/UniDL4BioPep-main/clean_catalog \
    --output-dir  ~/UniDL4BioPep-main/Predictions_Macrel_nofilter \
    --macrel-bin  ~/miniconda3/envs/env_macrel/bin/macrel \
    --macrel-threads 8 --no-physchem-filter
```

---

## 3. （可选）单独跑 UniDL4BioPep

只在需要 UniDL4BioPep 单独的候选肽清单时才跑。这一步要重新过一遍
ESM-2 + CNN，是最慢的一步。

```bash
cd ~/UniDL4BioPep-main
THRESHOLD=0.99 nohup bash run_step2b_unidl_only.sh > unidl_full.log 2>&1 &
echo $! > unidl_full.pid
```

输出目录 `Predictions_UniDL_final/`。内部就是 `run_amp_sorf_cohorts.py`
加 `--no-macrel --no-ampep`，长度域自动变成 UniDL4BioPep 的 11–180 aa。

---

## 4. 出对比结论

```bash
cd ~/UniDL4BioPep-main

# 常规用法：Macrel 用单工具跑的结果，UniDL 直接复用共识跑
python compare_macrel_vs_unidl.py \
    --macrel-dir Predictions_Macrel_final \
    --unidl-dir  Predictions_AMP_final \
    --consensus-dir Predictions_AMP_final \
    --macrel-th 0.9 --consensus-th 0.99 \
    --out cmp

# 如果第 3 步也跑了，就把 --unidl-dir 换成单工具跑的目录
python compare_macrel_vs_unidl.py \
    --macrel-dir Predictions_Macrel_final \
    --unidl-dir  Predictions_UniDL_final \
    --consensus-dir Predictions_AMP_final \
    --macrel-th 0.9 --consensus-th 0.99 \
    --out cmp
```

`--macrel-th 0.9` / `--consensus-th 0.99` 要填**共识跑当时真正用的阈值**
（你那次是 `MACREL_TH=0.9`、`THRESHOLD=0.99`），否则第 ③ 张表口径不对。

输出四张表：

| 表 | 内容 | 用来说明什么 |
|---|---|---|
| ① | 两个工具各自的分组判阳率 + 是否落在文献基准 0.1–1.65% | 哪个工具的绝对判阳率合理 |
| ② | 多档阈值敏感性（占域内可评序列比例） | 把阈值推到最高档能否压进基准区间 |
| ③ | 共识跑里 UniDL4BioPep 各阈值对 Macrel 命中的**淘汰率** | **淘汰率≈0 就是"那一票不构成筛选"的直接证据** |
| ④ | 结论汇总 | 直接可引 |

第 ③ 张表**不需要重跑任何预测**——它直接读共识跑 `*_AMP_hits.csv` 里
已经保存的 `Macrel_prob` / `UniDL_AMP_prob` 两列原始概率。

> **关于"UniDL4BioPep 筛选没效果"这个结论**：要用第 ③ 张表的淘汰率支撑，
> 并且**必须写明是在哪个阈值下成立的**。阈值不同结论可能反过来——
> 淘汰率在 0.99 下可能接近 0，在 0.999 下可能不可忽略。
> 两个数都写进报告，比只写一个更站得住。

---

## 5. 下游统计（组间比较 / AD 关联）

```bash
# Macrel 单工具
python amp_group_stats.py    ~/UniDL4BioPep-main/Predictions_Macrel_final
python amp_ad_association.py ~/UniDL4BioPep-main/Predictions_Macrel_final \
    --min-tools 1 --cohort Cohort2

# 共识跑
python amp_group_stats.py    ~/UniDL4BioPep-main/Predictions_AMP_final
python amp_ad_association.py ~/UniDL4BioPep-main/Predictions_AMP_final \
    --min-tools 2 --cohort Cohort2
```

> ### ⚠️ 单工具跑必须传 `--min-tools 1`
> `amp_ad_association.py` 的 `--min-tools` **默认是 2**（为双工具共识设计）。
> 单工具跑的 `n_tools_positive` 最大只有 1，用默认值会把候选**全部过滤成 0 条**。
> 共识跑用默认的 2 是对的。

> ### ⚠️ `run_full_pipeline.sh` 第 4 步不含 Cohort2
> 那一步写死了 `for c in Cohort1 Cohort3`（`run_full_pipeline.sh:133`）。
> 你现在跑的是 `Cohort2_Matched265_NCvsAD`，AD 关联要单独补：
> ```bash
> python amp_ad_association.py <结果目录> --min-tools 1 --cohort Cohort2
> ```
> （`--cohort` 是子串匹配，`Cohort2` 能命中 `Cohort2_Matched265_NCvsAD`。）

> ### ⚠️ 跨工具一致性检验会被跳过
> `amp_group_stats.py` 的 Spearman 一致性检验要求每个队列 ≥3 个分组
> 且 ≥2 个工具。单工具跑 + Cohort2 只有 2 组，这一项会打印
> "(工具数或组数不足, 跳过)"。这是预期的，不是报错。

---

## 6. 结果文件夹

```
Predictions_AMP_final/        ← 正在跑的共识任务（别删）
Predictions_Macrel_final/     ← 第 2 步新增
Predictions_UniDL_final/      ← 第 3 步（可选）新增
```

三个目录结构一致：

```
<结果目录>/
├── summary_all_groups.tsv
├── group_stats_*.tsv           (第 5 步产出)
├── AD_association/             (第 5 步产出)
├── cmp_*.tsv                   (第 4 步产出)
├── .ckpt/
│   └── <Cohort>__<Group>.ckpt.json
└── <Cohort>/
    ├── <Group>_AMP_hits.csv        ← 候选肽全表(含原始概率列)
    ├── <Group>_summary.json        ← 核心汇总
    ├── <Group>_summary.tsv
    └── <Group>_<工具>_prob_hist.npy
```

`<Group>_summary.json` 里可直接引用的字段见第 0 节的表。
另外：`funnel.n_raw` → `n_after_len` → `n_after_physchem` → `n_scored`
是逐级漏斗条数；`length_strata` 是按长度分层的条数与命中数。

---

## 7. 口径与报告注意事项

1. **判阳率的分母要说清楚。** 三个分母并存：`n_raw`（原始）、`n_scored`
   （长度+理化过滤后）、`n_evaluable`（工具域内）。与文献基准 0.1–1.65%
   对比时必须用**分母 = `n_raw`** 的那个数。
   `compare_macrel_vs_unidl.py` 的表 ① 已按此重算，不用自己换算。
2. **两个工具的长度域不同**（Macrel 10–100 aa，UniDL4BioPep 11–180 aa），
   绝对判阳数不可直接相比。要同口径就给 Macrel 传 `MAX_LEN=180`，
   并在报告里说明 100–180 段对 Macrel 是域外。
3. **只报 ρAMP，不报绝对条数**（`amp_group_stats.py` 的既有约定）。
4. **队列不独立**：Cohort2 是 Cohort1 的两个组，Cohort4 是 Cohort3 的两个组，
   Cohort1/2 与 Cohort3/4 样本高度重叠，不能当独立重复验证
   （`METHODS_AMP_metagenome.md` 第 7.4 条）。
5. **这是计算预测，不是实验证据。** 文献中即使 7 工具全阳筛选，
   合成后活性率也只有 70–83%。表述应为"候选"。
6. **UniDL4BioPep 的绝对概率不可解释为后验概率**
   （`METHODS_AMP_metagenome.md` 第 7.2 条：先验错配 + softmax 未校准，
   仅可用于排序）。第 ② 张表就是把这句话落到具体数字上。

---

## 8. 已完成那次共识跑的实测结果（4.43 小时，2026-09-08）

| | Disease_AD | Healthy_NC |
|---|---|---|
| 原始 smORF | 24,813,729 | 23,347,543 |
| 打分（长度+理化过滤后） | 8,136,428（32.8% of raw） | 7,703,149（33.0% of raw） |
| **UniDL4BioPep @0.99** | **5,339,359 = 65.62% of 打分** | **5,033,875 = 65.35% of 打分** |
| **Macrel @0.9** | **2,461 = 0.0302% of 打分** | **2,416 = 0.0314% of 打分** |
| 共识 ≥2 | 2,313 = 0.0093% of raw | 2,256 = 0.0097% of raw |
| **UniDL 淘汰掉的 Macrel 命中** | **6.01%** | **6.62%** |

### 可以直接引用的三条

1. **UniDL4BioPep 在 p ≥ 0.99 下判阳 65% 的打分序列。** ρAMP = 0.2152，
   是文献基准上限 1.65% 的 **13.0 倍**。这不是阈值问题——0.99 已经是
   `REPORT_THRESHOLDS` 里第二高的档位。
2. **UniDL4BioPep 那一票只淘汰掉 6.0–6.6% 的 Macrel 命中。**
   低于 `compare_macrel_vs_unidl.py` 里 10% 的判据，
   所以"双工具共识"在效果上等价于 Macrel 单筛。
3. **组间差异要按效应量读，不能只看 p 值**（`amp_group_stats.py` 报告要点第 3 条）：

   | 工具 | fold_change (AD/NC) | p | 怎么读 |
   |---|---|---|---|
   | AMP (UniDL) | 0.998（−0.20%） | 0.000299 | p 显著纯粹因为分母 2300 万；**−0.2% 无实际意义** |
   | Macrel | 0.958（−4.16%） | 0.138 | 不显著 |
   | consensus2 | 0.965（−3.5%） | 0.224 | 不显著 |
   | KS 检验 AMP | D = 0.0040 | 4.55e-54 | 同样是 p 显著、效应量≈0 |

   **在 Cohort2 上，AD 与 NC 的 AMP 密度差异检测不出来。** 这是一个阴性结果，
   如实报告比硬凑显著性站得住。

### 那次跑的自检给的建议是反的（已修）

日志里 `⚠️ 偏低` 后面跟的是 `建议提高 THRESHOLD 后重跑第 2 步`。
共识率**低于**基准说明判阳太少，提高阈值只会更低。
`run_full_pipeline.sh` 的自检已改成按方向分别给建议。

### 第 4 步为什么跳过

`run_full_pipeline.sh:133` 是 `for c in Cohort1 Cohort3`，而你这次只跑了 Cohort2，
所以 `❌ 未找到任何候选肽`。要补：

```bash
python amp_ad_association.py ~/UniDL4BioPep-main/Predictions_AMP_final \
    --min-tools 2 --cohort Cohort2
```

但**注意**：Cohort2 只有 NC / AD 两个阶段，Cochran-Armitage 趋势检验需要
≥3 个有序阶段，`amp_ad_association.py` 现在会明确提示并跳过。
Cohort2 的组间比较看 `amp_group_stats.py` 的第 ③ 张表就够了。
要做真正的阶段趋势，得跑 Cohort1 或 Cohort3。

---

## 9. 已知问题记录

| 问题 | 处理 |
|---|---|
| `amp_ad_association.py` 在候选量小、家族表为空时崩在 `KeyError: 'CA_p'` | 已修：家族表为空时打印提示并写出带表头的空 TSV，不再崩溃。可下调 `--min-family-count`（默认 5）到 2 或 1 后重跑 |
| 单工具跑被 `amp_ad_association.py` 默认的 `--min-tools 2` 清空 | 必须显式传 `--min-tools 1`（见第 5 节） |
| TensorFlow 报 `Could not load dynamic library 'libcufft.so.10'` / `libcusparse.so.11` → `Skipping registering GPU devices` | CNN 落到 CPU 上跑。**结果正确，只是慢。** `LD_LIBRARY_PATH` 里是 cuda-12.8，而 TF 找的是 CUDA 11 时代的 soname，版本对不上 |
| 日志里 `🖥 设备: cuda` 但 TF 没用上 GPU | 那行是 `torch.cuda.is_available()`，只对 PyTorch/ESM-2 成立，与 TensorFlow 无关 |
| Macrel 命中率偏低（共识跑 `MACREL_TH=0.9` 时约 0.029%，低于文献基准下限 0.1%） | 0.9 是精确率优先。单工具全量回到官方默认 0.5，再用 summary 的多档阈值表回看 0.9/0.95 |
| `run_full_pipeline.sh` 自检在判阳率【偏低】时建议"提高 THRESHOLD" | 已修：方向说反了。现在按偏高/偏低分别给建议，并说明本流程有理化预筛 + 分母是原始 smORF，口径比文献更严，偏低不一定代表出错 |
| `amp_ad_association.py` 对只有 2 个阶段的队列照样跑 Cochran-Armitage 趋势检验 | 已修：<3 个有序阶段时明确提示并跳过，指向 `amp_group_stats.py` 的两比例 z 检验。Cohort2 只有 NC/AD，属于这种情况 |
| `git fetch` 报 `Repository not found` / `Authentication failed for 'https://ghproxy.net/https://github.com/mqgg5630-cyber/UniDL4BioPep.git/'` | 你本机有个名为 `arena` 的 remote 指向一个**不存在的仓库** `mqgg5630-cyber/UniDL4BioPep`（本项目的仓库是 `shaohuawen03-cyber/UniDL4BioPep`），而且还套了 `ghproxy.net` 代理。先 `git remote -v` 和 `git config --get-regexp 'url\..*insteadof'` 查一下，把 remote 换成 `https://github.com/shaohuawen03-cyber/UniDL4BioPep.git` |
