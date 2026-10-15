# 从零复现 Severson 2019：一份带"事故现场"的电池寿命预测学习攻略

**Data-driven prediction of battery cycle life — a reproduction & failure-annotated learning guide**

> 用前 100 圈的数据预测电池还能活多少圈——Severson et al., *Nature Energy* 2019。
> 本仓库复现了这篇论文，并把复现路上的每一个 bug、每一次误判、每一条被数据打脸的预期，全部原样保留。

![结果](https://img.shields.io/badge/EN6%20median%20error-5.9%25%20%5BCI%205.1~8.9%5D-brightgreen) ![论文](https://img.shields.io/badge/paper-9.1%25-blue)

![从零复现 Severson 2019：项目概览](项目宣传图.png)

## English abstract

> **Reproduction of Severson et al., *Nature Energy* 2019** (battery cycle-life prediction from the first 100 cycles) — on the official `test_ind` split (n = 43), an elastic net with 6 early-cycle features reaches **5.9% median percentage error (95% bootstrap CI 5.1–8.9)** vs. the paper's 9.1% point estimate, with the full error-attribution chain (**33.6% → 5.9%**) documented step by step.
>
> **Original extensions:** NASA dataset label audit (**3/34** cycle-life labels validated; **18/18** transfer observations out-of-range); zero-shot MIT→NASA transfer with **×75–125** error inflation; domain-gap decomposition in four matched experiments (voltage-window artifact ruled out — and reversed); Kaplan–Meier survival analysis (median **125** cycles, cross-validated with `lifelines`).
>
> A guide-style repo: **13 Chinese documents** (theory → reproduction → debugging log → interview cards), runnable scripts, tests & CI. *Main text is in Chinese.*

---

## 成绩单（先看结果）

| 模型 | 官方口径中位误差 | 均值 | batch3 外推 | 整批外推 |
|---|---|---|---|---|
| 弹性网络（六特征） | **5.9%** [CI 5.1~8.9] | 36.5%* | 4.9% | 28.7% |
| 随机森林 | 8.2% [4.6~12.2] | 13.9% | 9.4% | 32.4% |
| 论文原报告 | 9.1% | — | 未报告 | 未声称 |

\* 均值被少数夭折电池拉高；剔除寿命<500圈后均值 27.0%，预测截断敏感性结果 24.1%，见 docs/03。论文侧没有同口径 CI，故只称点估计接近，不宣称统计一致或优于。

超越复现的原创部分：

- **数据集审计**：NASA 34 颗电池逐颗体检，证明寿命标签只对 **3/34** 成立——任何"直接迁移"都在用毒标签；
- **零样本迁移证据链**：MIT→NASA 迁移 EN 失控 ×75~125；按 MIT dev P05–P95 计算，NASA 三颗的 18/18 个电池×特征观测越界，明细见 `results/5_跨数据集迁移/MIT特征分位对照.csv`；
- **域间隙拆账**：四实验拆嫌疑——标签口径已排除（E2 ×202），窗口嫌疑已排除且反转（v2.1 重跑：真 3.6–3.8V 切片 E1 ×120→×246，匹配窗口反而更糟）→ 主犯=**协议级非平稳性**——"同样的早期指纹，不同的未来"；
- **KM 生存分析**：把 11 颗删失电池从"被丢弃"变成信息源，中位寿命 125 圈。

## 这份攻略的四种打开方式

| 你是谁 | 路线 | 入口 |
|---|---|---|
| **想学会这套方法** | 按阶段 1→5 跟做，每个阶段有直觉建立 + 动手练习 | [docs/00_怎么用这份攻略.md](docs/00_怎么用这份攻略.md) |
| **只想复现结果** | 环境 → 数据 → 跑 scripts/ 定稿脚本 | [docs/01_环境与数据_快速上手.md](docs/01_环境与数据_快速上手.md) |
| **从复现做个人求职作品** | 完成官方复现 → 设计自己的问题/数据 → 冻结评估口径 → 如实记录贡献 | [docs/11_实习求职包装指南.md](docs/11_实习求职包装指南.md) |
| **（HR/技术面试官）评估作者** | README → 03 复现篇 → 04 创新篇 → 10 问答卡 | 从本页往下读即可 |

## 仓库地图

```
├── README.md                  ← 你在这里
├── CONTRIBUTING.md            ← 数据保护、实验口径与 AI/引用规范
├── .github/workflows/ci.yml   ← 无原始数据的自动测试
├── 项目一页报告.pdf            ★ 投递附件：成绩单+归因链+主图（单页）
├── docs/
│   ├── 00_怎么用这份攻略.md         学习路线图
│   ├── 01_环境与数据_快速上手.md    环境、数据下载、已知坑表
│   ├── 02_原理篇.md                每个概念为什么出现（含思考题）
│   ├── 03_复现篇.md                从模拟数据到 5.9% 的完整叙事
│   ├── 04_创新篇.md                审计/迁移/拆账/生存分析
│   ├── 05_调试实录.md              ★ 17 个编号现场（本攻略最独特部分）
│   ├── 06_数据集审计方法论.md      任何公开数据集套用前的 5 项检查
│   ├── 07_概念地图.md              一张图串起全部原理
│   ├── 08_常见错误FAQ.md           症状→根因速查表
│   ├── 09_读者练习.md              每阶段 2~3 道动手题
│   ├── 10_面试问答卡.md            30 问
│   ├── 11_实习求职包装指南.md      ★ 怎么完成项目/简历模板/面试话术/诚信边界
│   ├── 12_代码优化提案.md          优化实验菜单（约束/优先级/判读标准）
│   ├── buggy_注释版/               ★ 带事故注释的 bug 版代码（勿运行，教学用）
│   └── archive/                    原始工作日志、完整错误清单
├── scripts/                   定稿脚本 step1~5d；step6 为独立优化实验
├── figures/                   各阶段图库（目录名=阶段号）
├── results/                   全部数据表（CSV），含 optimization_v1 实验结果
└── data/README.md             数据集下载与放置说明
```

## 记录与复核材料

所有审查结论、修复时间线、NASA 34 颗全量复跑和发布边界统一见[项目审查与迭代记录](docs/项目审查与迭代记录.md)。三轮审查报告、任务书和前期工作记录的原始快照归档于 `docs/archive/审查原始记录/`；详细优化实验数字和执行过程见[代码优化提案执行记录](docs/工作记录_代码优化提案_2026-10-14.md)。

## 快速开始（三行版）

1. 装环境：Python 3.14，`python -m pip install -r requirements.txt`；无数据快速体验：`python scripts/run_project.py --stage quick`。
2. 完整复现前按 `data/README.md` 下载 MIT/NASA 数据；默认放在 `data/mit`、`data/nasa`。数据另存他处时复制 `config.example.toml` 为被忽略的 `config.toml`，再用 `--show-paths` 检查。
3. `python scripts/run_project.py --stage paper` 复现论文；`extensions` 跑 NASA 扩展，`optimization` 跑 step6，`full` 按全阶段顺序执行。`full` 为自包含教学脚本，会重复解析约 8.3GB MIT 数据；只想复现论文时用 `paper`。生成物位于 `results/generated/`，仓库内 `results/` 是已核验快照。

## 作者的话

这个项目里最有价值的不是 5.9%，而是那条误差归因链：**33.6% → 32.7% → 31.6% → 毒标签 → 5.9%**。每一步"错误→排除→再定位"都记录在 [docs/05_调试实录.md](docs/05_调试实录.md)——网上教程只给你正确的代码，这里把犯错的过程也给你。

> ilo · 新能源/材料方向在读本科生
> 双 AI 交叉审计工作流（独立管道互验 + 逐位一致性检查）

## License & 数据致谢

- 代码：MIT License
- Severson, K.A. et al. *Data-driven prediction of battery cycle life before capacity degradation.* Nat. Energy 4, 383–391 (2019)
- 数据：MIT-Stanford 数据集（原论文官方）与 NASA PCoE 电池数据集，版权归原机构所有
