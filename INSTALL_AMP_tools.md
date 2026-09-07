# 工具安装清单

你已有 `env_macrel`。以下按流程顺序列出，**建议分环境安装**（马跃模型依赖旧版 Keras，
与 `bio_pep` 的新版 TF 冲突）。

## 0. 先确认 Macrel

```bash
conda activate env_macrel
macrel --version          # 建议 >= 1.5；1.6.0 已于 2025-11 发布
conda activate bio_pep
```

`run_amp_sorf_cohorts.py` 通过 `--macrel-bin` 调用外部二进制，
所以 Macrel 待在自己的环境里即可，只要给出绝对路径：

```bash
which macrel   # 在 env_macrel 中执行，记下路径
# 例如 /home/wsh/miniconda3/envs/env_macrel/bin/macrel
```

---

## 1. smORF 真实性（第一步，优先做）

### 路径 B：AntiFam —— 蛋白序列（推荐，你的输入就是蛋白）

```bash
conda activate bio_pep
conda install -c bioconda hmmer -y

mkdir -p ~/db/antifam && cd ~/db/antifam
wget https://ftp.ebi.ac.uk/pub/databases/Pfam/AntiFam/current/Antifam.tar.gz
tar xzf Antifam.tar.gz
hmmpress AntiFam.hmm
```

### 路径 A：SmORFinder —— 仅当你有核酸 contigs

```bash
conda create -n smorfinder python=3.8 -y
conda activate smorfinder
pip install smorfinder
smorf              # 首次运行会自动下载模型数据
```

> **注意**：SmORFinder 只吃**核酸 contigs**，它是 Prodigal 之上的过滤层。
> 若 `comparable_sorf_grouped_catalog/*.fa` 是氨基酸序列，请用路径 B。

---

## 2. 家族聚类与比对

```bash
conda activate bio_pep
conda install -c bioconda cd-hit diamond -y
```

---

## 3. Ma et al. 2022 模型（ATT / LSTM）

**必须单独环境**——依赖 `keras.engine.topology`，仅旧版 Keras 可用。

```bash
conda create -n ma_amp python=3.7 -y
conda activate ma_amp
pip install "tensorflow==1.15" "keras==2.2.4" numpy pandas h5py==2.10.0

git clone https://github.com/mayuefine/c_AMPs-prediction.git ~/tools/c_AMPs-prediction
# 权重在 Models/ 目录：10att.h5 / 10lstm.h5
ls ~/tools/c_AMPs-prediction/Models/
```

运行时需要能 `import Attention`（仓库里的 `Attention.py`），脚本会自动把
`--ma-att-h5` 所在目录加入 `sys.path`，把 `Attention.py` 和 `10att.h5` 放一起即可。

---

## 4. amPEPpy（可选第 N 票）

```bash
conda activate bio_pep
pip install git+https://github.com/tlawrence3/amPEPpy.git
# 需自行 train 出模型后用 --ampep-model 指定
```

---

# 推荐执行顺序

```bash
# ── 第 1 步：smORF 真实性（先做，能砍掉大量假阳性，省后续算力）
python smorf_authenticity.py antifam \
    --input comparable_sorf_grouped_catalog/Cohort3_Full476_5Stage/Cohort3_AD.fa \
    --antifam-db ~/db/antifam/AntiFam.hmm \
    --output clean/Cohort3_AD.fa --threads 16

# ── 第 2 步：AMP 共识预测（Macrel + UniDL4BioPep [+ Ma ATT/LSTM]）
python run_amp_sorf_cohorts.py \
    --catalog-dir clean_catalog \
    --output-dir ~/Predictions_AMP_run1 \
    --macrel-bin /home/wsh/miniconda3/envs/env_macrel/bin/macrel \
    --min-tools 2

# ── 第 3 步：组间统计
python amp_group_stats.py ~/Predictions_AMP_run1/

# ── 第 4 步：AD 关联（家族级）
python amp_ad_association.py ~/Predictions_AMP_run1/ --cohort Cohort3
```

**关于马跃模型**：因需独立环境，若不便与主流程同进程运行，可分两步——
先在 `ma_amp` 环境单独对候选肽打分，再把结果并入。主流程也支持直接传入：

```bash
python run_amp_sorf_cohorts.py \
    --ma-models ATT LSTM \
    --ma-att-h5 ~/tools/c_AMPs-prediction/Models/10att.h5 \
    --ma-lstm-h5 ~/tools/c_AMPs-prediction/Models/10lstm.h5
```
