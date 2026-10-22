<div align="center">

# Reproducing Severson 2019 from Scratch

**A battery cycle-life prediction guide with the crash scenes left in**

![EN6 median error](https://img.shields.io/badge/EN6%20median%20error-5.9%25%20%5BCI%205.1~8.9%5D-brightgreen)
![paper](https://img.shields.io/badge/paper-9.1%25-blue)
![docs](https://img.shields.io/badge/docs-13%20Chinese%20documents-orange)
![license](https://img.shields.io/badge/license-MIT-green)

<p align="center">
  <b>English</b> · <a href="./README.zh-CN.md">简体中文</a>
</p>

</div>

---

![Project overview](项目宣传图.png)

> Predict how many cycles a battery has left from its first 100 cycles — Severson et al., *Nature Energy* 2019.
> This repository reproduces that paper, and keeps every bug, every misjudgement and every expectation that the data knocked down exactly where it happened.

**This is a guide-style repository. The main text — 13 documents covering theory, reproduction, a debugging log and interview cards — is written in Chinese.** This page summarises the results and points to the entry points; the Chinese version is [README.zh-CN.md](./README.zh-CN.md).

## Key results

On the official `test_ind` split (n = 43), an elastic net on 6 early-cycle features reaches **5.9% median percentage error** (95% bootstrap CI 5.1–8.9), against the paper's 9.1% point estimate. The full error-attribution chain (**33.6% → 5.9%**) is documented step by step.

| Model | Official-split median error | Mean | batch3 extrapolation | Whole-batch extrapolation |
|---|---|---|---|---|
| Elastic net (6 features) | **5.9%** [CI 5.1–8.9] | 36.5%* | 4.9% | 28.7% |
| Random forest | 8.2% [4.6–12.2] | 13.9% | 9.4% | 32.4% |
| Paper's reported value | 9.1% | — | not reported | not claimed |

\* The mean is dragged up by a few short-lived cells; excluding cells below 500 cycles gives 27.0%, and the prediction-truncation sensitivity run gives 24.1% — see `docs/03`. The paper reports no CI on the same protocol, so this is described as a point estimate in the same range, **not** as statistical agreement or superiority.

### Original work beyond the reproduction

- **Dataset audit** — all 34 NASA cells examined one by one; the cycle-life label holds for only **3/34**. Any "direct transfer" is training on poisoned labels.
- **Zero-shot transfer evidence chain** — MIT→NASA elastic-net transfer blows up by **×75–125**; measured against the MIT dev P05–P95 range, **18/18** cell × feature observations on the three NASA cells fall outside it. Details in `results/5_跨数据集迁移/MIT特征分位对照.csv`.
- **Domain-gap accounting** — four matched experiments rule out the suspects: label definition is excluded (E2 ×202), the voltage-window hypothesis is excluded *and reversed* (v2.1 rerun: a true 3.6–3.8 V slice goes E1 ×120 → ×246, so matching the window makes it worse). The main culprit is **protocol-level non-stationarity** — "the same early fingerprint, a different future".
- **Kaplan–Meier survival analysis** — the 11 censored cells turn from "discarded" into information: median lifetime **125 cycles**.

## Four ways to use this guide

| If you are… | Route | Entry point |
|---|---|---|
| **Here to learn the method** | Follow stages 1→5, each with intuition building and hands-on exercises | [`docs/00_怎么用这份攻略.md`](docs/00_怎么用这份攻略.md) |
| **Here to reproduce the numbers** | Environment → data → run the frozen scripts in `scripts/` | [`docs/01_环境与数据_快速上手.md`](docs/01_环境与数据_快速上手.md) |
| **Building a job-application portfolio piece from it** | Complete the official reproduction → design your own question/data → freeze the evaluation protocol → record contributions honestly | [`docs/11_实习求职包装指南.md`](docs/11_实习求职包装指南.md) |
| **An interviewer evaluating the author** | README → reproduction doc → original-work doc → Q&A cards | Just keep reading this page |

*All document filenames are in Chinese.*

## Repository map

```
├── README.md                  ← you are here
├── README.zh-CN.md            ← Chinese version
├── CONTRIBUTING.md            ← data protection, experiment protocol, AI/citation rules
├── .github/workflows/ci.yml   ← automatic tests that need no raw data
├── 项目一页报告.pdf            ★ one-page attachment: scorecard + attribution chain + main figure
├── docs/
│   ├── 00_怎么用这份攻略.md          learning roadmap
│   ├── 01_环境与数据_快速上手.md     environment, data download, known pitfalls
│   ├── 02_原理篇.md                why each concept exists (with exercises)
│   ├── 03_复现篇.md                the full narrative from synthetic data to 5.9%
│   ├── 04_创新篇.md                audit / transfer / accounting / survival analysis
│   ├── 05_调试实录.md              ★ 17 numbered incidents — the most distinctive part
│   ├── 06_数据集审计方法论.md       five checks before adopting any public dataset
│   ├── 07_概念地图.md              one diagram linking every concept
│   ├── 08_常见错误FAQ.md           symptom → root cause lookup
│   ├── 09_读者练习.md              2–3 hands-on tasks per stage
│   ├── 10_面试问答卡.md            30 interview questions
│   ├── 11_实习求职包装指南.md      ★ portfolio packaging, resume templates, honesty boundaries
│   ├── 12_代码优化提案.md          optimisation experiment menu
│   ├── buggy_注释版/               ★ annotated buggy code (teaching only, do not run)
│   └── archive/                    raw work logs, full error list
├── scripts/                   frozen scripts step1–5d; step6 is a separate optimisation experiment
├── figures/                   figure library, directory names = stage numbers
├── results/                   all data tables (CSV), incl. the optimization_v1 experiment
└── data/README.md             dataset download and placement instructions
```

## Quick start (three lines)

1. **Environment** — Python 3.14, then `python -m pip install -r requirements.txt`. To try it without any data: `python scripts/run_project.py --stage quick`.
2. **Data** — before a full reproduction, download the MIT/NASA datasets following `data/README.md`; the default location is `data/mit` and `data/nasa`. If the data lives elsewhere, copy `config.example.toml` to the git-ignored `config.toml` and verify with `--show-paths`.
3. **Run** — `python scripts/run_project.py --stage paper` reproduces the paper; `extensions` runs the NASA extensions, `optimization` runs step6, `full` walks every stage in order. `full` is a self-contained teaching script and re-parses roughly 8.3 GB of MIT data, so use `paper` if you only want the reproduction. Generated artifacts go to `results/generated/`; the `results/` committed to the repo is a verified snapshot.

## A note from the author

The most valuable thing here is not the 5.9%. It is the error-attribution chain: **33.6% → 32.7% → 31.6% → poisoned labels → 5.9%**. Every step of "wrong → rule out → relocate" is recorded in [`docs/05_调试实录.md`](docs/05_调试实录.md). Tutorials online only hand you the code that worked; this one hands you the mistakes too.

> ilo · undergraduate in new energy / materials
> dual-AI cross-audit workflow (independent pipelines cross-checked, bit-level consistency verification)

## Review and verification material

All review conclusions, the fix timeline, the full 34-cell NASA rerun and the release boundaries are collected in [`docs/项目审查与迭代记录.md`](docs/项目审查与迭代记录.md). Raw snapshots of three review rounds, task briefs and earlier work records are archived under `docs/archive/审查原始记录/`; the detailed optimisation numbers and execution log are in [`docs/工作记录_代码优化提案_2026-10-14.md`](docs/工作记录_代码优化提案_2026-10-14.md).

## Sibling repository: a second independent reproduction

The same data (NASA PCoE + MIT-Stanford) and the same Severson 2019 paper are reproduced independently in **[battery-ecm-rul](https://github.com/ilo-love7/battery-ecm-rul)**. The two are deliberately kept separate and complement each other: this repository is a **learning path** (step1→6, debugging log, interview cards), that one is a **self-consistent evidence chain** (ECM parameter identification, PyBaMM physical simulation, third-party identification audit, internal-resistance ground-truth ruler). The full division of labour and protocol comparison is in [RELATED.md](RELATED.md).

**Cross-verification results between the two reproductions were updated on 2026-10-18** (controlled experiments; see [RELATED.md](RELATED.md) and [`docs/兄弟仓库互证与受控实验_2026-10-18.md`](docs/兄弟仓库互证与受控实验_2026-10-18.md)):

- **Alignment of the four chains now rests on direct cell-by-cell verification**, not on "RF 8.2% vs 8.12% agree" — the 43 main-test cells match exactly, and cycle-life labels differ by 1 cycle on only 3 continuation cells. A same-protocol control (the other pipeline's 6-feature RF at 9.38%) shows RF is also off by about 1.2 pp in the same direction as EN. The earlier "perfect agreement" was a coincidence between different feature sets, not evidence for the four chains.
- **The old attribution for the 2.4 pp ElasticNet gap was falsified by the other side's experiments and relocated**: `l1_ratio` accounts for only 0.14 pp (fixing it at 0.5 on the same feature snapshot gives 6.18%; grid search gives 6.04%). The gap is mainly a **numerical definition difference** in the two pipelines' 6 paper features — switching the other side's old config to a different feature source jumps from 6.18% to 8.33%, with a completely different error structure. This repository's 5.9% and the other side's archived 8.33% **still cannot be cross-verified, only compared**.
- **Significance discipline agrees with the other side's supplementary tests**: the n = 43 bootstrap CI is 5–9 pp wide, and the other side's supplementary tests show the GBM (6.20%) vs EN (7.77%) gap on the main test set is not significant (Wilcoxon p = 0.93) — model rankings at this sample size are mostly descriptive. This repository's existing discipline ("the point estimate is close to the paper's 9.1%; no claim of statistical agreement or superiority") is corroborated by that.

Three conclusions the other repository can cite directly from this one: only 3/34 NASA labels hold; zero-shot transfer inflates error by ×75–125; domain-gap accounting locates the main culprit in protocol-level non-stationarity.

## License and data credits

- Code: MIT License
- Severson, K.A. et al. *Data-driven prediction of battery cycle life before capacity degradation.* Nat. Energy 4, 383–391 (2019)
- Data: the MIT-Stanford dataset (official, from the original paper) and the NASA PCoE battery dataset; copyright remains with the original institutions
