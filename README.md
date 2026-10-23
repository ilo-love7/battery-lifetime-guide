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

## What you can do with this guide

Six concrete jobs, each with a runnable entry point in this repository (CPU only; the tutorial route needs no datasets at all):

| Job | Entry point |
|---|---|
| **Try the whole pipeline with zero data** (toy random-forest demo; the "floor effect" on a synthetic 12-cell set) | `python scripts/run_project.py --stage quick` · `scripts/toy_random_forest.py` |
| **Reproduce the paper on the official `test_ind` split** (6-feature elastic net → 5.9% median error) | `python scripts/run_project.py --stage paper` · `scripts/step4_3f_official_split.py` |
| **Audit a public battery dataset before trusting its labels** (all 34 NASA cells examined; the label holds for only 3/34) | `python scripts/run_project.py --stage extensions` · `scripts/step5a_parse_nasa_v4.py` |
| **Measure cross-dataset transfer and account for the domain gap** (zero-shot MIT→NASA blows up ×75–125; the culprit is protocol-level non-stationarity) | `scripts/step5b_zero_shot_v2.py` · `scripts/step5c_decompose_v2.py` |
| **Turn censored cells into information with Kaplan–Meier survival analysis** | `scripts/step5d_survival_v2.py` · `scripts/step5d_validate_km.py` |
| **Run and extend the step-6 optimisation experiments** (main / transfer / survival) | `python scripts/run_project.py --stage optimization` · `scripts/step6a_main_optimization_v1.py`, `scripts/step6b_transfer_optimization_v1.py`, `scripts/step6c_survival_optimization_v1.py` |

Typical uses: (1) a **worked reference implementation** of Severson 2019 to diff your own reproduction against;
(2) a **dataset-audit checklist** ([`docs/06_数据集审计方法论.md`](docs/06_数据集审计方法论.md)) to apply before adopting any public battery dataset;
(3) **teaching material** for a battery-data course — 13 Chinese documents with hands-on exercises at every stage;
(4) a **portfolio / interview-preparation route** with 30 Q&A cards ([`docs/10_面试问答卡.md`](docs/10_面试问答卡.md)).

Everything needed to smoke-test the repository is offline: `python -m pytest tests -q` runs **11 tests** with no datasets,
`python scripts/run_project.py --stage quick` runs the data-free tutorial demo, and
`python scripts/run_project.py --stage full --dry-run` validates all 16 script steps of the full route without touching data.
The reproduction routes (`paper`, `extensions`, `optimization`) need the MIT-Stanford and NASA PCoE datasets placed under
`data/mit` and `data/nasa` as described in [`data/README.md`](data/README.md); the `full` route re-parses roughly 8.3 GB of
MIT data, so use `paper` if you only want the reproduction. CI (workflow `tests`) runs on Linux (`ubuntu-latest`) with Python 3.14 only.

## Checking whether two protocols are comparable

Every number on this page rests on one question: **are the two groups of cells even comparable?** Rather than leaving that to prose, `scripts/protocol_drift.py` makes it a reusable diagnostic — a two-sample Kolmogorov–Smirnov drift test plus a quantile-alignment table — and every verdict it prints carries an explicit statement of what the result does **not** prove.

| Diagnostic | Entry point |
|---|---|
| Two-sample drift test (KS statistic, p-value, median shift) | `protocol_drift.ks_drift` |
| Quantile-by-quantile alignment of a target against a reference | `protocol_drift.quantile_alignment` |
| Is each observation inside the reference P05–P95 band? | `protocol_drift.band_audit`, `protocol_drift.feature_band_audit` |
| Every pairwise comparison across a set of protocols | `protocol_drift.ks_matrix` |
| Re-derive this page's headline drift numbers from committed CSVs | `python scripts/protocol_drift.py --evidence` |

```python
from protocol_drift import ks_drift, quantile_alignment

drift = ks_drift(protocol_a_feature, protocol_b_feature)   # KS test + verdict
print(drift.statistic, drift.p_value, drift.significant)
print(drift.verdict_en)                                    # says what this cannot prove

table = quantile_alignment(reference, target)              # columns: quantile, reference, target, delta
```

It is **offline by construction**: the module reads only CSV artifacts committed under `results/`, so the diagnostics run without the multi-GB MIT/NASA downloads. `--demo` prints a known-answer control — synthetic same-distribution pairs must stay under the significance threshold (swept over 400 seeded trials; empirical flag rate 0.028 against α = 0.05), while a deliberate +1 σ shift is caught at D = 0.425, p = 1.4 × 10⁻¹⁶. The suite pins both directions (`python -m pytest tests -q`, **11 tests**).

Reproducing the numbers on this page from committed artifacts (`python scripts/protocol_drift.py --evidence`):

| Claim on this page | Committed artifact | Value reproduced offline |
|---|---|---|
| zero-shot MIT→NASA EN blows up **×75–125** | `results/5_跨数据集迁移/零样本预测案例.csv` | EN **×74.8–125.3** (the unrounded form of ×75–125), RF ×2.25–3.14 |
| matching the voltage window makes it **worse** (E1 **×120 → ×246**) | `results/5_跨数据集迁移/域间隙拆账_四实验.csv` | EN geometric-mean ratio **×120.42 (E0) → ×246.11 (E1)** |
| NASA observations fall outside the MIT dev P05–P95 band | `results/mit_features.csv` (official 84-cell dev pool) + `.../MIT特征分位对照.csv` | **18/18** cell × feature observations outside the band |
| cross-batch drift is measured, not assumed | `results/mit_features.csv` | dev vs batch3: **6/6** features significant (KS) |

**Scope boundary — what this tool cannot show:**

- A significant KS result says two samples are unlikely to share one continuous distribution. It does **not** prove causation, does **not** identify which protocol knob moved, and does **not** license a numeric transfer or a pooled model.
- A non-significant result is **not** evidence that two protocols are identical. With the sample sizes here (n = 3 NASA cells; n = 84 vs 44 MIT cells) power is low, and "no detectable shift" is not "equivalent".
- The figures above are reproduced from **committed CSV artifacts**, not by re-running the pipelines. Regenerating the pipelines from scratch needs the raw datasets: download MIT-Stanford and NASA PCoE following `data/README.md` into `data/mit` and `data/nasa`, then run `python scripts/run_project.py --stage extensions`. Without those downloads the pipeline routes cannot run — the diagnostics can, because they consume only `results/`.

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

## Related repositories

This repository is part of a set of work on batteries and transport modelling by the same author:

| Repository | What it covers |
|---|---|
| [battery-cathode-ml](https://github.com/ilo-love7/battery-cathode-ml) | Cathode material screening: band gap, formation energy, voltage, capacity + uncertainty and multi-objective decisions |
| [battery-lifetime-guide](https://github.com/ilo-love7/battery-lifetime-guide) | From-scratch reproduction of Severson et al. 2019 (cycle-life prediction), crash scenes included |
| [battery-soh-tsfm-benchmark](https://github.com/ilo-love7/battery-soh-tsfm-benchmark) | Do zero-shot time-series foundation models beat linear extrapolation? Three-dataset SOH benchmark |
| [battery-ecm-rul](https://github.com/ilo-love7/battery-ecm-rul) | Thevenin ECM parameter identification + early cycle-life prediction + PyBaMM simulation |
| [drift-diffusion-pn-junction](https://github.com/ilo-love7/drift-diffusion-pn-junction) | Self-contained 1D drift–diffusion–Poisson solver with three-layer verification |
| [car-following](https://github.com/ilo-love7/car-following) | Car-following traffic modelling toolkit (7 models, stability, calibration, lane changing, EV energy) |
| [project-packer](https://github.com/ilo-love7/project-packer) | Turning real project work into interview-defensible experience (fact dossier + question tree) |
