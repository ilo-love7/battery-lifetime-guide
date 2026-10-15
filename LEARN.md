# How I built it：从零复现 Nature Energy 2019 电池寿命预测，并把误差做到 5.9%

> **English summary:** I rebuilt the MIT–Stanford battery cycle-life study (Severson et al., *Nature Energy* 2019) from scratch in Python: loading 124 cells, engineering 6 early-cycle features, and running the official `test_ind` split (n = 43). My first run failed at **33.6%** median error; three rounds of debugging (label pollution, a pandas `iterrows` write-back trap) brought it to **5.9% (95% bootstrap CI 5.1–8.9)** vs. the paper's 9.1% point estimate. I then went beyond the paper: a NASA label audit (**3/34** labels valid), zero-shot transfer analysis (**×75–125** error inflation), a four-experiment domain-gap decomposition, and Kaplan–Meier survival analysis for censored cells. Everything—including every bug—is documented in 13 Chinese documents linked below.

![项目概览](项目宣传图.png)

## 这个项目做了什么

用电池**前 100 圈**的放电数据预测它的总循环寿命。复现论文基线之后，我把重点放在三件论文没做的事上：**审计数据集、解释误差从哪来、让删失数据说话**。

| 关键结果 | 数字 |
|---|---|
| 官方口径中位误差（弹性网络，六特征） | **5.9%** [95% CI 5.1~8.9]，论文点估计 9.1% |
| 随机森林对照 | 8.2% [4.6~12.2] |
| NASA 34 颗标签体检 | 仅 **3/34** 寿命标签成立 |
| 零样本 MIT→NASA 迁移 | 误差膨胀 **×75~125** |
| KM 生存分析 | 11 颗删失电池，中位寿命 **125** 圈 |

## 用到的工具

Python 3.14 · pandas · scikit-learn（弹性网络 / 随机森林）· lifelines（生存分析）· pytest · GitHub Actions CI · matplotlib

## 构建全过程（8 步）

### 第 0 步：先冻结评估口径，再写一行模型代码

读论文、下数据（MIT 124 颗 + NASA 34 颗，放置与下载见 [data/README.md](data/README.md)）。最重要的一件事发生在写代码之前：**锁定官方 `test_ind` 划分（43 颗测试）作为唯一考卷**——后来证明，这是整个项目里回报最高的决定，因为之后的每次"进步"都只能在这一把尺子上度量。

### 第 1 步：不建模，先建立直觉

三个入门实验（[scripts/step1_capacity_fade.py](scripts/step1_capacity_fade.py)、[step2_individual_plots.py](scripts/step2_individual_plots.py)、[step3_dqdv.py](scripts/step3_dqdv.py)）：把每颗电池的容量衰减曲线、单体轨迹、dQ/dV 曲线画出来。这一步不产出模型，但决定了后面所有判断的"手感"。

### 第 2 步：特征工程

从每颗电池前 100 圈数据提取 6 个早期循环特征（提取逻辑见 [scripts/step4_2_mit_features.py](scripts/step4_2_mit_features.py)），配合弹性网络拟合 [scripts/step4_3f_official_split.py](scripts/step4_3f_official_split.py)。

### 第 3 步：首跑翻车——33.6%

第一个"完成品"比论文差 3 倍以上。我没有先调参，而是先回答一个问题：**误差到底从哪来？**

### 第 4 步：三轮排错，33.6% → 5.9%

误差归因链完整记录在 [docs/05_调试实录.md](docs/05_调试实录.md)（17 个编号现场）：

> **33.6% → 32.7% → 31.6% → 毒标签 → 5.9%**

两个最典型的根因：NASA/MIT 续跑电池第 1 圈为 **0 占位值**，污染了 5 颗电池的寿命标签；修复时又踩中 pandas **`iterrows` 副本陷阱**——调试打印全对，下游数据却没被改到。最终修复伴随着一条"合并生效断言"，让这类 bug 永久无法静默回归。

### 第 5 步：给结论装上置信区间

对 43 颗测试误差做 bootstrap，得到 **5.9% [CI 5.1~8.9]**（[results/阶段43f_三口径汇总.csv](results/阶段43f_三口径汇总.csv)）。论文侧没有同口径 CI，所以我只说"点估计接近"，不说"统计显著优于"——这句话后来救了我好几次面试追问。

### 第 6 步：超越复现的四个原创实验（[docs/04_创新篇.md](docs/04_创新篇.md)）

1. **NASA 34 颗逐颗体检**：寿命标签只对 3/34 成立——任何"直接迁移"都在用毒标签；
2. **零样本迁移证据链**：MIT→NASA 误差 ×75~125；按 MIT dev P05–P95 分位，18/18 个电池×特征观测越界；
3. **域间隙四实验拆嫌疑**（[results/5_跨数据集迁移/域间隙拆账_四实验.csv](results/5_跨数据集迁移/域间隙拆账_四实验.csv)）：标签口径排除（E2 ×202）、电压窗口伪影排除**且反转**（E1 ×120→×246，窗口对齐后反而更糟）→ 主犯是**协议级非平稳性**："同样的早期指纹，不同的未来"；
4. **KM 生存分析**（[scripts/step5d_survival_v2.py](scripts/step5d_survival_v2.py)）：把 11 颗删失电池从"被丢弃"变成信息源，中位寿命 125 圈，`lifelines` 交叉验证一致。

### 第 7 步：工程化——让复现可以防回归

生效断言 + 6 项 pytest 测试 + 无需原始数据的 CI（[.github/workflows/ci.yml](.github/workflows/ci.yml)）：clone 即可验证代码逻辑，不必先下载 8.3GB 数据。

### 第 8 步：把"事故现场"写成攻略

13 篇中文文档（[docs/00_怎么用这份攻略.md](docs/00_怎么用这份攻略.md) 是入口）：原理篇每讲一个概念回答"它为什么出现"；调试实录保留每一次误判；还有 FAQ、思考题和 30 问面试卡。网上教程只给正确代码，这个仓库把犯错的过程也给你。

## 你也可以跑一遍

```bash
pip install -r requirements.txt
python scripts/run_project.py --stage quick    # 无数据快速体验
# 完整复现：按 data/README.md 下载 MIT/NASA 数据后：
python scripts/run_project.py --stage paper    # 复现论文主结果
python scripts/run_project.py --stage extensions   # NASA 扩展实验
```

完整说明见 [README「快速开始」](README.md#快速开始三行版)。`full` 模式为自包含教学脚本，会重复解析约 8.3GB MIT 数据。

## 最大的三个坑（省流版）

1. **占位值会变成毒标签**：续跑电池的第 1 圈是 0，不是退化证据——先审计标签，再谈建模；
2. **pandas `iterrows` 副本陷阱**：打印对 ≠ 生效，用断言守住"修改必须落盘"；
3. **跨数据集不能裸迁**：先做逐颗标签体检（本例 3/34），再用分位对照证明"是不是分布外"。

## 学到的三件事

- **误差归因 > 调参**：33.6%→5.9% 的每一步都是"排除一个嫌疑"，不是"换一个模型"；
- **评估口径先冻结**：尺子不统一，一切"进步"都不可信；
- **诚实是性价比最高的策略**：点估计接近就说"接近"，不越界到"统计显著优于"——经得起追问的表述才拿得出手。

## 资源

- 论文：Severson et al., *"Data-driven prediction of battery cycle life before capacity degradation"*, Nature Energy 2019, DOI [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8)
- 数据集下载与放置：[data/README.md](data/README.md)
- 完整调试叙事：[docs/05_调试实录.md](docs/05_调试实录.md)
- 数据集审计方法论（可套用到任何公开数据集）：[docs/06_数据集审计方法论.md](docs/06_数据集审计方法论.md)
