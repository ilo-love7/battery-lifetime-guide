"""Unlabeled target recalibration and three-cell few-shot transfer experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from optimization_common import (
    FEATURES,
    bootstrap_median_ci,
    load_official_split,
    paired_median_difference_ci,
    percentage_error,
    seed_everything,
)
from project_config import PATHS


SEED = 42
VD = np.linspace(3.5, 3.9, 1000)
NASA_CELLS = ["B0005", "B0006", "B0018"]
NASA_LIVES = np.array([125.0, 109.0, 97.0])
CV = KFold(n_splits=4, shuffle=True, random_state=SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=PATHS.cache_dir / "mit_parse_cache.pkl")
    parser.add_argument("--nasa-dir", type=Path, default=PATHS.nasa_dir)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PATHS.output_dir / "optimization_v1" / "transfer",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def cycles_container(cell_struct):
    names = getattr(getattr(cell_struct, "dtype", None), "names", None) or ()
    matches = [name for name in names if "cycle" in name.lower()]
    if not matches:
        raise ValueError(f"NASA 结构体缺少 cycle 字段: {names}")
    cycles = cell_struct[matches[0]]
    while cycles.ndim >= 2 and cycles.shape[0] == 1 and cycles.size == 1:
        cycles = cycles[0, 0]
    return cycles


def nasa_features(path: Path) -> dict[str, float]:
    data = sio.loadmat(path)
    top = data[[name for name in data if not name.startswith("__")][0]]
    cycles = cycles_container(top)
    n_cycles = cycles.shape[1] if cycles.ndim == 2 else len(cycles)
    capacities = []
    q_curves = {}
    discharge_order = 0
    for index in range(n_cycles):
        cycle = cycles[0, index] if cycles.ndim == 2 else cycles[index]
        try:
            if "discharge" not in str(np.squeeze(cycle["type"])).lower():
                continue
            discharge_order += 1
            item = cycle["data"][0, 0]
            names = item.dtype.names or ()
            if "Capacity" in names:
                capacity = float(np.ravel(item["Capacity"])[0])
                if np.isfinite(capacity) and 0.05 < capacity < 2.3:
                    capacities.append(capacity)
            if discharge_order in (10, 100):
                voltage = np.ravel(item["Voltage_measured"]).astype(float)
                current = np.ravel(item["Current_measured"]).astype(float)
                time = np.ravel(item["Time"]).astype(float)
                charge = -np.concatenate([
                    [0.0], np.cumsum((current[1:] + current[:-1]) / 2 * np.diff(time))
                ]) / 3600.0
                mask = (voltage >= VD[0]) & (voltage <= VD[-1])
                if mask.sum() > 1:
                    q_curves[discharge_order] = np.interp(
                        VD, voltage[mask][::-1], charge[mask][::-1]
                    )
        except (IndexError, KeyError, TypeError, ValueError):
            continue

    capacity = np.asarray(capacities, dtype=float)
    if len(capacity) < 100 or 10 not in q_curves or 100 not in q_curves:
        raise ValueError(f"{path.stem} 缺少首 100 次放电或 Q(V) 曲线")
    delta = q_curves[100] - q_curves[10]
    slope, intercept = np.polyfit(np.arange(2, 101), capacity[1:100], 1)
    return {
        "log_var": float(np.log10(max(np.var(delta), 1e-12))),
        "log_min": float(np.log10(max(-float(delta.min()), 1e-12))),
        "log_mean": float(np.log10(max(abs(float(delta.mean())), 1e-12))),
        "slope": float(slope),
        "intercept": float(intercept),
        "discrepancy": float(np.polyval((slope, intercept), 100) - capacity[99]),
    }


def _matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, 1e-8) ** power) @ vectors.T


def coral_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    source_scale = source.std(axis=0, ddof=1)
    source_scale[source_scale < 1e-12] = 1.0
    source_z = (source - source_mean) / source_scale
    target_z = (target - source_mean) / source_scale
    target_mean = target_z.mean(axis=0)
    source_cov = np.cov(source_z, rowvar=False)
    target_centered = target_z - target_mean
    target_cov = np.atleast_2d(np.cov(target_centered, rowvar=False))
    source_cov = np.atleast_2d(source_cov)
    source_cov += np.eye(source.shape[1]) * 1e-3
    target_cov += np.eye(source.shape[1]) * 1e-3
    aligned = target_centered @ _matrix_power(target_cov, -0.5) @ _matrix_power(source_cov, 0.5)
    return (aligned + source_z.mean(axis=0)) * source_scale + source_mean


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    development, _, _ = load_official_split(args.cache)
    assert len(development) == 84
    source = development[FEATURES].to_numpy(dtype=float)
    source_y = np.log10(development["cycle_life"].to_numpy(dtype=float))
    target_rows = [nasa_features(args.nasa_dir / f"{cell}.mat") for cell in NASA_CELLS]
    target = pd.DataFrame(target_rows)[FEATURES].to_numpy(dtype=float)
    target_actual = NASA_LIVES.copy()
    assert target.shape == (3, len(FEATURES))
    assert list(pd.DataFrame(target_rows).columns) == FEATURES

    enet = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=CV,
                     random_state=SEED, max_iter=100_000),
    ).fit(source, source_y)
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=3, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1,
    ).fit(source, source_y)

    predictions: dict[str, np.ndarray] = {
        "EN_zero_shot": 10**enet.predict(target),
        "RF_zero_shot": 10**rf.predict(target),
        "EN_zscore_recalibration": 10**enet.predict(
            ((target - target.mean(axis=0)) /
             np.where(target.std(axis=0, ddof=1) > 1e-12,
                      target.std(axis=0, ddof=1), 1.0))
            * source.std(axis=0, ddof=1) + source.mean(axis=0)
        ),
        "EN_CORAL": 10**enet.predict(coral_transform(source, target)),
    }
    predictions["RF_zscore_recalibration"] = 10**rf.predict(
        ((target - target.mean(axis=0)) /
         np.where(target.std(axis=0, ddof=1) > 1e-12,
                  target.std(axis=0, ddof=1), 1.0))
        * source.std(axis=0, ddof=1) + source.mean(axis=0)
    )
    predictions["RF_CORAL"] = 10**rf.predict(coral_transform(source, target))

    # Leave one NASA cell out: two target labels calibrate the scalar source prediction.
    for source_name, model in [("EN", enet), ("RF", rf)]:
        source_prediction = model.predict(target)
        fewshot = np.zeros(3, dtype=float)
        fold_scores = []
        for hold_idx in range(3):
            fit_idx = np.array([idx for idx in range(3) if idx != hold_idx])
            calibrator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            calibrator.fit(source_prediction[fit_idx, None], np.log10(target_actual[fit_idx]))
            fewshot[hold_idx] = calibrator.predict(source_prediction[hold_idx : hold_idx + 1, None])[0]
            fold_scores.append(
                float(percentage_error(10**fewshot[hold_idx : hold_idx + 1],
                                       target_actual[hold_idx : hold_idx + 1])[0])
            )
        predictions[f"{source_name}_fewshot_LOOCV"] = 10**fewshot

    source_p05, source_p95 = np.percentile(source, [5, 95], axis=0)
    outside_count = int(np.sum((target < source_p05) | (target > source_p95)))
    rows = []
    baseline_errors = percentage_error(predictions["EN_zero_shot"], target_actual)
    for name, predicted in predictions.items():
        errors = percentage_error(predicted, target_actual)
        ci = bootstrap_median_ci(errors, n_boot=args.bootstrap, seed=SEED)
        baseline = percentage_error(predictions["EN_zero_shot"], target_actual)
        delta_ci = paired_median_difference_ci(
            errors, baseline, n_boot=args.bootstrap, seed=SEED
        )
        rows.append({
            "method": name,
            "median_error_pct": float(np.median(errors)),
            "median_ci95_low": ci[0],
            "median_ci95_high": ci[1],
            "mean_error_pct": float(np.mean(errors)),
            "delta_vs_EN_zero_shot_pp": float(np.median(errors) - np.median(baseline_errors)),
            "paired_delta_ci95_low": delta_ci[0],
            "paired_delta_ci95_high": delta_ci[1],
            "features_outside_source_P05_P95": outside_count,
        })
    comparison = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.out_dir / "transfer_comparison.csv", index=False, encoding="utf-8-sig")
    detail = []
    for method, predicted in predictions.items():
        for index, cell in enumerate(NASA_CELLS):
            detail.append({
                "cell": cell,
                "method": method,
                "actual_life": target_actual[index],
                "predicted_life": predicted[index],
                "error_pct": percentage_error(predicted[index:index + 1],
                                               target_actual[index:index + 1])[0],
            })
    pd.DataFrame(detail).to_csv(
        args.out_dir / "transfer_cell_predictions.csv", index=False, encoding="utf-8-sig"
    )
    lines = [
        "# 跨域优化实验 B1-B2",
        "",
        "B1 仅使用三颗目标电池的无标签特征做 z-score/CORAL 重标定；这属于 transductive covariate adaptation，未使用其寿命标签。",
        "B2 每折用另外两颗 NASA 电池拟合一维 Ridge 校准器，预测留出的一颗；n=3，仅作探索，不宣称可泛化。",
        f"NASA 3 颗的 6 个特征中有 {outside_count}/18 个落在 MIT dev 的 P05-P95 之外。",
        "bootstrap 对 3 颗案例重采样，区间非常离散，只作描述；不据此改变已冻结的 MIT 官方测试结论。",
        "",
        "| 方法 | 中位误差 | 95% bootstrap CI | 相对 EN 零样本差值 | 配对差 CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.median_error_pct:.1f}% | "
            f"[{row.median_ci95_low:.1f}, {row.median_ci95_high:.1f}] | "
            f"{row.delta_vs_EN_zero_shot_pp:+.1f}pp | "
            f"[{row.paired_delta_ci95_low:+.1f}, {row.paired_delta_ci95_high:+.1f}] |"
        )
    lines.extend(["", "逐颗预测见 `transfer_cell_predictions.csv`。"])
    (args.out_dir / "conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"B1-B2 结果与结论已写入: {args.out_dir}")


if __name__ == "__main__":
    main()