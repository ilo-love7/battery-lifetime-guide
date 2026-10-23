# -*- coding: utf-8 -*-
"""Protocol-drift diagnostics: are two groups of cells comparable at all?

This module turns this repository's central finding -- **protocol-level
non-stationarity**, "the same early fingerprint, a different future" -- into a
reusable diagnostic instead of a sentence in a document.

What it provides
----------------
* ``ks_drift``                      two-sample Kolmogorov-Smirnov drift test + verdict
* ``quantile_alignment``            quantile table for two distributions (fixed schema)
* ``band_audit`` / ``feature_band_audit``  is each observation inside a reference P05-P95 band?
* ``ks_matrix``                     every pairwise KS comparison across a set of protocols
* ``reproduce_committed_evidence``  re-derives this README's headline numbers from the CSV
  artifacts committed under ``results/`` -- no raw datasets required

What it deliberately does NOT do
--------------------------------
It does not prove causation, does not name the protocol knob that moved, and does
not license a numeric transfer or a pooled model between two protocols. A
non-significant result is **not** evidence of equivalence: with small samples the
test has low power. Every verdict string carries that disclaimer, and the test
suite pins it so a later edit cannot quietly drop the discipline.

Offline by construction: importing or running this module needs only
numpy / pandas / scipy and the committed artifacts. It never touches
``data/mit`` or ``data/nasa``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from project_config import enable_utf8_stdout

# 本脚本不导入 project_config 的其它内容，但需要那道 UTF-8 stdout 守卫：
# `--evidence` / 矩阵输出会打印中文结论，Windows 控制台默认 cp1252 会崩。
enable_utf8_stdout()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DRIFT_DIR = RESULTS_DIR / "5_跨数据集迁移"

FEATURES: tuple[str, ...] = (
    "log_var",
    "log_min",
    "log_mean",
    "slope",
    "intercept",
    "discrepancy",
)
#: Official MIT exclusions (docs/03): batch1/batch2 cells the paper dropped.
OFFICIAL_EXCLUDE: dict[str, set[int]] = {
    "batch1": {8, 10, 12, 13, 22},
    "batch2": {7, 8, 9, 15, 16},
}
#: Official dev pool size after merging continuation cells and applying the exclusions.
OFFICIAL_DEV_SIZE = 84
NASA_CELLS: tuple[str, ...] = ("B0005", "B0006", "B0018")

DEFAULT_ALPHA = 0.05
DEFAULT_QUANTILES: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)
BAND_LOW, BAND_HIGH = 5.0, 95.0

QUANTILE_TABLE_COLUMNS: tuple[str, ...] = (
    "quantile",
    "reference",
    "target",
    "delta",
)
BAND_TABLE_COLUMNS: tuple[str, ...] = (
    "label",
    "value",
    "percentile",
    "band_low",
    "band_high",
    "outside",
)
FEATURE_BAND_TABLE_COLUMNS: tuple[str, ...] = ("feature",) + BAND_TABLE_COLUMNS
KS_TABLE_COLUMNS: tuple[str, ...] = (
    "group_a",
    "group_b",
    "n_a",
    "n_b",
    "ks_statistic",
    "p_value",
    "significant",
)
FEATURE_KS_COLUMNS: tuple[str, ...] = (
    "feature",
    "n_a",
    "n_b",
    "ks_statistic",
    "p_value",
    "significant",
)

#: The epistemic discipline. Appended verbatim to every verdict in both languages.
DISCIPLINE_EN = (
    "A significant result does NOT prove causation, does NOT identify which "
    "protocol knob moved, and does NOT license a numeric transfer or a pooled "
    "model. A non-significant result is NOT evidence that the two protocols are "
    "identical: with small samples the test has low power, so 'no detectable "
    "shift' is not 'equivalent'."
)
DISCIPLINE_ZH = (
    "显著不代表因果，也不能指出是哪个协议旋钮动了，更不能据此做数值迁移或合并建模；"
    "不显著也不等于两条协议一致——小样本下检验功效很低，"
    "「没检出漂移」不是「等价」。"
)


def _say(text: str) -> None:
    """Print without dying on a legacy console codepage."""
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover - depends on host console
        print(text.encode("ascii", "replace").decode("ascii"))


def _as_float_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} 必须是一维样本")
    if array.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 含非有限值")
    return array


def _verdict(significant: bool) -> tuple[str, str]:
    if significant:
        en = (
            "drift detected: the two samples are unlikely to share one continuous "
            "distribution. Treat them as NOT directly comparable -- align the protocol "
            "or recalibrate before transferring or pooling. " + DISCIPLINE_EN
        )
        zh = (
            "检出漂移：两份样本不太可能来自同一条连续分布，应视为不可直接比较——"
            "迁移或合并前必须先对齐协议或重新标定。" + DISCIPLINE_ZH
        )
    else:
        en = (
            "no drift detected at this sample size. Treat the comparison as "
            "unresolved, not as evidence that the protocols agree. " + DISCIPLINE_EN
        )
        zh = (
            "当前样本量下未检出漂移，只能算「未决」，不能当作两条协议一致的证据。"
            + DISCIPLINE_ZH
        )
    return en, zh


@dataclass(frozen=True)
class KsDrift:
    """Result of a two-sample KS drift test, with an explicit scope statement."""

    n_a: int
    n_b: int
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    median_a: float
    median_b: float
    median_shift: float
    median_ratio: float
    verdict_en: str
    verdict_zh: str

    def as_dict(self) -> dict:
        return asdict(self)


def ks_drift(
    sample_a: Sequence[float] | np.ndarray,
    sample_b: Sequence[float] | np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> KsDrift:
    """Two-sample Kolmogorov-Smirnov test: distribution drift between two protocols.

    ``significant`` uses the strict ``p < alpha`` rule, so a p-value exactly on the
    threshold is reported as not significant (conservative).
    """
    a = _as_float_array(sample_a, "sample_a")
    b = _as_float_array(sample_b, "sample_b")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 必须落在 (0, 1) 内")
    result = ks_2samp(a, b)
    median_a = float(np.median(a))
    median_b = float(np.median(b))
    shift = median_b - median_a
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = float(median_b / median_a) if median_a != 0 else float("nan")
    significant = bool(result.pvalue < alpha)
    verdict_en, verdict_zh = _verdict(significant)
    return KsDrift(
        n_a=int(a.size),
        n_b=int(b.size),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alpha=float(alpha),
        significant=significant,
        median_a=median_a,
        median_b=median_b,
        median_shift=shift,
        median_ratio=ratio,
        verdict_en=verdict_en,
        verdict_zh=verdict_zh,
    )


def quantile_alignment(
    reference: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Quantile-by-quantile alignment of a target distribution against a reference.

    Returns a frame with the fixed schema ``QUANTILE_TABLE_COLUMNS``; every row is
    one quantile, ``delta = target - reference``.
    """
    ref = _as_float_array(reference, "reference")
    tgt = _as_float_array(target, "target")
    qs = np.asarray(list(quantiles), dtype=float)
    if qs.ndim != 1 or qs.size == 0:
        raise ValueError("quantiles 必须是非空一维序列")
    if np.any((qs < 0) | (qs > 100)):
        raise ValueError("quantiles 必须落在 [0, 100] 内")
    ref_q = np.percentile(ref, qs)
    tgt_q = np.percentile(tgt, qs)
    # Deliberately NO ratio column: these features cross zero (log-variance, slope,
    # discrepancy), and a quantile ratio across zero is silently meaningless.
    # ``delta`` is the only safe summary here.
    return pd.DataFrame(
        {
            "quantile": qs,
            "reference": ref_q,
            "target": tgt_q,
            "delta": tgt_q - ref_q,
        },
        columns=list(QUANTILE_TABLE_COLUMNS),
    )


def percentile_of_score(reference: Sequence[float] | np.ndarray, value: float) -> float:
    """Empirical percentile of ``value`` inside ``reference``.

    Mirrors ``scripts/step5b_zero_shot_v2.py`` so the numbers line up with the
    committed ``MIT特征分位对照.csv``: 100 means "above every reference value",
    0 means "at or below the smallest".
    """
    arr = np.sort(_as_float_array(reference, "reference"))
    return float(np.searchsorted(arr, float(value)) / arr.size * 100.0)


def band_audit(
    reference: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    labels: Sequence[str] | None = None,
    *,
    low: float = BAND_LOW,
    high: float = BAND_HIGH,
) -> pd.DataFrame:
    """Is each observation inside the reference ``[low, high]`` percentile band?

    Returns a frame with the fixed schema ``BAND_TABLE_COLUMNS``. ``outside`` is
    strict: a value exactly on the band edge counts as inside.
    """
    ref = _as_float_array(reference, "reference")
    vals = _as_float_array(values, "values")
    if labels is None:
        labels = [f"obs{index}" for index in range(vals.size)]
    label_list = [str(label) for label in labels]
    if len(label_list) != vals.size:
        raise ValueError("labels 与 values 长度不一致")
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("要求 0 <= low < high <= 100")
    rows = []
    for label, value in zip(label_list, vals):
        pct = percentile_of_score(ref, float(value))
        rows.append(
            {
                "label": label,
                "value": float(value),
                "percentile": pct,
                "band_low": float(low),
                "band_high": float(high),
                "outside": bool(pct < low or pct > high),
            }
        )
    return pd.DataFrame(rows, columns=list(BAND_TABLE_COLUMNS))


def feature_band_audit(
    reference_features: pd.DataFrame,
    target_table: pd.DataFrame,
    *,
    features: Sequence[str] = FEATURES,
    cells: Sequence[str] = NASA_CELLS,
    low: float = BAND_LOW,
    high: float = BAND_HIGH,
) -> pd.DataFrame:
    """Per-feature percentile audit: every ``cell x feature`` value against its own band.

    ``reference_features`` needs one column per feature (MIT dev pool);
    ``target_table`` needs ``<cell>_value`` columns keyed by a ``特征`` column, i.e.
    the committed ``MIT特征分位对照.csv`` schema. Schema:
    ``FEATURE_BAND_TABLE_COLUMNS``.
    """
    frames = []
    for feature in features:
        reference = reference_features[feature].to_numpy(dtype=float)
        values = [
            float(target_table.loc[target_table["特征"] == feature, f"{cell}_value"].iloc[0])
            for cell in cells
        ]
        frame = band_audit(reference, values, labels=list(cells), low=low, high=high)
        frame.insert(0, "feature", feature)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[list(FEATURE_BAND_TABLE_COLUMNS)]


def ks_matrix(
    groups: Mapping[str, Sequence[float] | np.ndarray],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Every unordered pairwise KS comparison across named protocol groups.

    Schema: ``KS_TABLE_COLUMNS``. A missing pair never appears; ``n_a``/``n_b``
    record the sample sizes so a reader can judge the test's power.
    """
    names = list(groups)
    if len(names) < 2:
        raise ValueError("ks_matrix 至少需要两个分组")
    rows = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            drift = ks_drift(groups[name_a], groups[name_b], alpha=alpha)
            rows.append(
                {
                    "group_a": name_a,
                    "group_b": name_b,
                    "n_a": drift.n_a,
                    "n_b": drift.n_b,
                    "ks_statistic": drift.statistic,
                    "p_value": drift.p_value,
                    "significant": drift.significant,
                }
            )
    return pd.DataFrame(rows, columns=list(KS_TABLE_COLUMNS))


def cross_feature_ks(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    *,
    features: Sequence[str] = FEATURES,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """KS drift per feature between two cell tables (e.g. official dev vs batch3)."""
    rows = []
    for feature in features:
        drift = ks_drift(
            group_a[feature].to_numpy(dtype=float),
            group_b[feature].to_numpy(dtype=float),
            alpha=alpha,
        )
        rows.append(
            {
                "feature": feature,
                "n_a": drift.n_a,
                "n_b": drift.n_b,
                "ks_statistic": drift.statistic,
                "p_value": drift.p_value,
                "significant": drift.significant,
            }
        )
    return pd.DataFrame(rows, columns=list(FEATURE_KS_COLUMNS))


def geometric_mean_ratio_by_experiment(
    decomposition: pd.DataFrame, *, model: str = "EN"
) -> dict[str, float]:
    """Geometric-mean prediction/truth ratio per experiment in the decomposition table."""
    result: dict[str, float] = {}
    for experiment, group in decomposition.groupby("实验", sort=False):
        ratios = group.loc[group["模型"] == model, "倍率误差"].to_numpy(dtype=float)
        if ratios.size:
            result[str(experiment)] = float(np.exp(np.mean(np.log(ratios))))
    return result


# --------------------------------------------------------------------------------------
# Committed-artifact loaders (all offline; ``results/`` ships with the repository)
# --------------------------------------------------------------------------------------
def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少已提交产物 {path}；本工具只读 results/ 下的 CSV，"
            "不需要下载 MIT/NASA 原始数据集"
        )
    return path


def load_mit_features(root: Path | str | None = None) -> pd.DataFrame:
    base = Path(root) if root else PROJECT_ROOT
    return pd.read_csv(_require(base / "results" / "mit_features.csv"), encoding="utf-8-sig")


def load_zero_shot_cases(root: Path | str | None = None) -> pd.DataFrame:
    base = Path(root) if root else PROJECT_ROOT
    return pd.read_csv(
        _require(base / "results" / "5_跨数据集迁移" / "零样本预测案例.csv"),
        encoding="utf-8-sig",
    )


def load_decomposition(root: Path | str | None = None) -> pd.DataFrame:
    base = Path(root) if root else PROJECT_ROOT
    return pd.read_csv(
        _require(base / "results" / "5_跨数据集迁移" / "域间隙拆账_四实验.csv"),
        encoding="utf-8-sig",
    )


def load_committed_quantile_table(root: Path | str | None = None) -> pd.DataFrame:
    base = Path(root) if root else PROJECT_ROOT
    return pd.read_csv(
        _require(base / "results" / "5_跨数据集迁移" / "MIT特征分位对照.csv"),
        encoding="utf-8-sig",
    )


def official_dev_pool(features: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the official 84-cell MIT dev pool from the committed feature table.

    batch1 + batch2 minus the official exclusions; batch3 is excluded entirely
    (it is the whole-batch extrapolation target, not part of the dev pool).
    """
    mask = (
        (features["batch"] == "batch1")
        & ~features["cell_idx"].isin(OFFICIAL_EXCLUDE["batch1"])
    ) | (
        (features["batch"] == "batch2")
        & ~features["cell_idx"].isin(OFFICIAL_EXCLUDE["batch2"])
    )
    dev = features.loc[mask].copy()
    if len(dev) != OFFICIAL_DEV_SIZE:
        raise AssertionError(f"官方 dev 池应为 {OFFICIAL_DEV_SIZE} 颗，实际 {len(dev)}")
    return dev


def reproduce_committed_evidence(root: Path | str | None = None) -> dict:
    """Re-derive this README's headline drift numbers from committed CSV artifacts.

    Reproduced here (no raw datasets):

    * zero-shot MIT->NASA elastic-net error multiplier range, and the RF range;
    * ``dev`` P05/P50/P95 recomputed from the committed feature table and checked
      against the committed quantile table;
    * how many NASA ``cell x feature`` observations fall outside that band;
    * the E0 -> E1 elastic-net geometric-mean blow-up ("matching the window makes
      it worse");
    * real cross-batch KS drift: official dev pool vs the batch3 target.

    It does **not** re-run the training pipelines. Regenerating those from scratch
    needs the raw MIT-Stanford and NASA PCoE downloads (see ``data/README.md``).
    """
    base = Path(root) if root else PROJECT_ROOT
    features = load_mit_features(base)
    dev = official_dev_pool(features)
    batch3 = features.loc[features["batch"] == "batch3"]
    if batch3.empty:
        raise AssertionError("committed feature table 缺少 batch3（整批外推靶集）")

    cases = load_zero_shot_cases(base)
    en_ratios = cases["EN倍率"].to_numpy(dtype=float)
    rf_ratios = cases["RF倍率"].to_numpy(dtype=float)

    quantile_table = load_committed_quantile_table(base)
    audit = feature_band_audit(dev, quantile_table)

    reference_agreement: dict[str, bool] = {}
    for feature in FEATURES:
        committed = [
            float(quantile_table.loc[quantile_table["特征"] == feature, column].iloc[0])
            for column in ("MIT_P05", "MIT_P50", "MIT_P95")
        ]
        recomputed = [
            float(np.percentile(dev[feature].to_numpy(dtype=float), q)) for q in (5, 50, 95)
        ]
        reference_agreement[feature] = all(
            abs(got - want) <= 1e-9 * max(1.0, abs(want))
            for got, want in zip(recomputed, committed)
        )

    decomposition = load_decomposition(base)
    geometric = geometric_mean_ratio_by_experiment(decomposition, model="EN")

    batch_ks = cross_feature_ks(dev, batch3)

    return {
        "mit_dev_n": int(len(dev)),
        "mit_batch3_n": int(len(batch3)),
        "zero_shot_en_ratio_min": float(en_ratios.min()),
        "zero_shot_en_ratio_max": float(en_ratios.max()),
        "zero_shot_rf_ratio_min": float(rf_ratios.min()),
        "zero_shot_rf_ratio_max": float(rf_ratios.max()),
        "quantile_reference_recomputed_matches_committed": reference_agreement,
        "reference_band_low": BAND_LOW,
        "reference_band_high": BAND_HIGH,
        "out_of_band_observations": int(audit["outside"].sum()),
        "total_observations": int(len(audit)),
        "en_geometric_mean_ratio_by_experiment": geometric,
        "dev_vs_batch3_ks": batch_ks.to_dict(orient="records"),
        "dev_vs_batch3_all_significant": bool(batch_ks["significant"].all()),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _run_demo(alpha: float) -> None:
    """Known-answer control: a null pair AND a shifted pair, plus calibration.

    The null pair alone can only ever be one coin flip; reporting one lucky seed
    would be cherry-picking. So the same-distribution case is also swept over many
    seeded trials and the empirical false-positive rate is printed next to alpha,
    and a deliberately under-powered case is shown to stay non-significant.
    """
    _say("== synthetic known-answer control (seeded, offline) ==")
    rng = np.random.default_rng(42)
    base = rng.normal(0.0, 1.0, size=200)
    same = rng.normal(0.0, 1.0, size=200)
    shifted = rng.normal(1.0, 1.0, size=200)

    for title, other in (("same distribution", same), ("deliberate +1 sigma shift", shifted)):
        drift = ks_drift(base, other, alpha=alpha)
        _say(
            f"  base vs {title:24s} n=200/200  D={drift.statistic:.4f}  "
            f"p={drift.p_value:.3e}  significant={drift.significant}"
        )
    _say(f"    verdict (shifted): {ks_drift(base, shifted, alpha=alpha).verdict_en}")

    trials = 400
    sweep = np.random.default_rng(20261023)
    false_positives = sum(
        ks_drift(sweep.normal(0.0, 1.0, 200), sweep.normal(0.0, 1.0, 200), alpha=alpha).significant
        for _ in range(trials)
    )
    _say(
        f"  null calibration         {false_positives}/{trials} same-distribution trials "
        f"flagged -> rate {false_positives / trials:.3f} (alpha={alpha})"
    )

    small = np.random.default_rng(1234)
    weak = ks_drift(small.normal(0.0, 1.0, 8), small.normal(0.2, 1.0, 8), alpha=alpha)
    _say(
        f"  low-power case           n=8/8, +0.2 sigma: D={weak.statistic:.4f} "
        f"p={weak.p_value:.3f}  significant={weak.significant}  "
        "(no drift detected != equivalent)"
    )

    table = quantile_alignment(base, shifted)
    _say("  quantile alignment (reference=base, target=shifted):")
    _say("    " + table.to_string(index=False).replace("\n", "\n    "))


def _run_batch_drift(alpha: float) -> None:
    features = load_mit_features()
    dev = official_dev_pool(features)
    batch3 = features.loc[features["batch"] == "batch3"]
    _say("== real committed data: official dev pool vs batch3 (whole-batch target) ==")
    _say(f"  n_dev={len(dev)}  n_batch3={len(batch3)}  alpha={alpha}")
    table = cross_feature_ks(dev, batch3, alpha=alpha)
    _say("  " + table.to_string(index=False).replace("\n", "\n  "))


def _run_evidence(alpha: float) -> None:
    evidence = reproduce_committed_evidence()
    _say("== committed-artifact evidence (no raw datasets) ==")
    _say(
        f"  MIT dev pool            : {evidence['mit_dev_n']} cells "
        f"(batch3 target n={evidence['mit_batch3_n']})"
    )
    _say(
        f"  zero-shot EN ratio      : x{evidence['zero_shot_en_ratio_min']:.1f}"
        f"~{evidence['zero_shot_en_ratio_max']:.1f}"
    )
    _say(
        f"  zero-shot RF ratio      : x{evidence['zero_shot_rf_ratio_min']:.2f}"
        f"~{evidence['zero_shot_rf_ratio_max']:.2f}"
    )
    _say(
        f"  outside MIT P05-P95 band: {evidence['out_of_band_observations']}"
        f"/{evidence['total_observations']} cell x feature observations"
    )
    agreement = evidence["quantile_reference_recomputed_matches_committed"]
    _say(f"  recomputed P05/P50/P95 == committed table: {all(agreement.values())}")
    for experiment, value in evidence["en_geometric_mean_ratio_by_experiment"].items():
        _say(f"  EN geometric-mean ratio : {experiment} -> x{value:.2f}")
    _say(
        "  dev vs batch3 KS all significant: "
        f"{evidence['dev_vs_batch3_all_significant']}"
    )
    _say("")
    _say(f"  {DISCIPLINE_EN}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="synthetic known-answer control")
    parser.add_argument("--batch-drift", action="store_true", help="committed dev vs batch3 KS")
    parser.add_argument("--evidence", action="store_true", help="reproduce README headline numbers")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="significance level")
    parser.add_argument("--json", action="store_true", help="emit --evidence as JSON")
    args = parser.parse_args(argv)

    if args.json:
        _say(json.dumps(reproduce_committed_evidence(), ensure_ascii=False, indent=2))
        return 0

    run_all = not (args.demo or args.batch_drift or args.evidence)
    if args.demo or run_all:
        _run_demo(args.alpha)
    if args.batch_drift or run_all:
        _say("")
        _run_batch_drift(args.alpha)
    if args.evidence or run_all:
        _say("")
        _run_evidence(args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())