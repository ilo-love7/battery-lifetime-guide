"""Weibull AFT and Nelson-Aalen comparison for audited NASA censoring data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from project_config import PATHS


SEED = 42
BOOTSTRAP = 10_000
EVENT_CLASS = "标准"
CENSOR_CLASS = "删失(未到EOL)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    # The curated snapshot excludes the B0033 boundary cell; raw step5a output does not.
    audit_snapshot = root / "results" / "5_跨数据集迁移" / "NASA34颗体检分类.csv"
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=audit_snapshot,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PATHS.output_dir / "optimization_v1" / "survival",
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    return parser.parse_args()


def fit_weibull(time: np.ndarray, event: np.ndarray) -> tuple[float, float] | None:
    """Profile-likelihood MLE of Weibull shape and scale with right censoring."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    event_count = int(event.sum())
    if event_count == 0 or np.any(time <= 0):
        return None
    log_time = np.log(time)
    event_log_time = float(log_time[event].sum())

    def negative_profile_log_likelihood(log_shape: float) -> float:
        shape = float(np.exp(log_shape))
        log_sum_power = float(logsumexp(shape * log_time))
        log_scale = (log_sum_power - np.log(event_count)) / shape
        log_likelihood = (
            event_count * log_shape
            - event_count * shape * log_scale
            + (shape - 1.0) * event_log_time
            - event_count
        )
        return -log_likelihood

    result = minimize_scalar(
        negative_profile_log_likelihood,
        bounds=(-3.0, 2.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        return None
    shape = float(np.exp(result.x))
    log_scale = (float(logsumexp(shape * log_time)) - np.log(event_count)) / shape
    return shape, float(np.exp(log_scale))


def km_median(time: np.ndarray, event: np.ndarray) -> float:
    order = np.argsort(time, kind="stable")
    times = np.asarray(time, dtype=float)[order]
    events = np.asarray(event, dtype=int)[order]
    survival = 1.0
    index = 0
    while index < len(times):
        current = times[index]
        same = times == current
        at_risk = int(np.sum(times >= current))
        at_time = np.flatnonzero(same)
        event_count = int(events[at_time].sum())
        if event_count:
            survival *= 1.0 - event_count / at_risk
            if survival <= 0.5:
                return float(current)
        index = int(at_time[-1]) + 1
    return float("inf")


def bootstrap_medians(
    time: np.ndarray,
    event: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(time), size=(count, len(time)))
    km_values = np.empty(count, dtype=float)
    weibull_values = np.full(count, np.inf, dtype=float)
    successful_weibull = 0
    for index, sample in enumerate(indices):
        sample_time = time[sample]
        sample_event = event[sample]
        km_values[index] = km_median(sample_time, sample_event)
        fitted = fit_weibull(sample_time, sample_event)
        if fitted is not None:
            shape, scale = fitted
            weibull_values[index] = scale * np.log(2.0) ** (1.0 / shape)
            successful_weibull += 1
    return km_values, weibull_values, successful_weibull


def finite_or_infinite_ci(values: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=float))
    lower_index = max(0, int(np.ceil(0.025 * len(ordered))) - 1)
    upper_index = min(len(ordered) - 1, int(np.ceil(0.975 * len(ordered))) - 1)
    return float(ordered[lower_index]), float(ordered[upper_index])


def read_audit_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))
    header = rows[0]
    normalized = []
    for row in rows[1:]:
        if len(row) == len(header) + 1 and row[0] == "B0033":
            row = [row[0], f"{row[1]},{row[2]}", *row[3:]]
        if len(row) != len(header):
            raise ValueError(f"审计 CSV 行列数异常: {row[0] if row else '<empty>'}")
        normalized.append(row)
    return pd.DataFrame(normalized, columns=header)


def main() -> None:
    args = parse_args()
    if not args.audit_csv.is_file():
        raise FileNotFoundError(f"NASA 分类结果不存在: {args.audit_csv}")
    audit = read_audit_csv(args.audit_csv)
    assert len(audit) == 34, f"NASA 分类表应为 34 颗，实际 {len(audit)}"
    included = audit.loc[audit["类别"].isin([EVENT_CLASS, CENSOR_CLASS])].copy()
    included["event"] = (included["类别"] == EVENT_CLASS).astype(int)
    included["time"] = np.where(
        included["event"].eq(1), included["寿命"], included["n_放电"]
    ).astype(float)
    included = included.sort_values(["time", "event", "cell"], kind="stable").reset_index(drop=True)
    time = included["time"].to_numpy(dtype=float)
    event = included["event"].to_numpy(dtype=int)
    assert len(included) == 14 and int(event.sum()) == 3 and int((1 - event).sum()) == 11

    fitted = fit_weibull(time, event)
    if fitted is None:
        raise RuntimeError("Weibull AFT MLE 不可估，需保留 KM 结果并报告不可拟合")
    shape, scale = fitted
    weibull_median = scale * np.log(2.0) ** (1.0 / shape)

    km_values, weibull_values, successful = bootstrap_medians(
        time, event, args.bootstrap, SEED
    )
    km_ci = finite_or_infinite_ci(km_values)
    weibull_ci = finite_or_infinite_ci(weibull_values)

    # Nelson-Aalen cumulative hazard, with all tied events/censoring handled together.
    survival = 1.0
    cumulative_hazard = 0.0
    table = []
    unique_times = np.unique(time)
    for current in unique_times:
        at_risk = int(np.sum(time >= current))
        mask = time == current
        event_count = int(event[mask].sum())
        censor_count = int(mask.sum() - event_count)
        if event_count:
            survival *= 1.0 - event_count / at_risk
            cumulative_hazard += event_count / at_risk
        table.append({
            "time": current,
            "at_risk": at_risk,
            "events": event_count,
            "censored": censor_count,
            "KM_survival": survival,
            "Nelson_Aalen_cumulative_hazard": cumulative_hazard,
            "Weibull_survival": float(np.exp(-((current / scale) ** shape))),
        })

    censored = included.loc[included["event"] == 0].copy()
    censor_time = censored["time"].to_numpy(dtype=float)
    conditional_base = (censor_time / scale) ** shape
    prediction_table = censored[["cell", "time"]].rename(columns={"time": "censor_time"}).copy()
    prediction_table["weibull_conditional_median_life"] = scale * (
        conditional_base + np.log(2.0)
    ) ** (1.0 / shape)
    prediction_table["weibull_conditional_p10_life"] = scale * (
        conditional_base - np.log(0.9)
    ) ** (1.0 / shape)
    prediction_table["weibull_conditional_p90_life"] = scale * (
        conditional_base - np.log(0.1)
    ) ** (1.0 / shape)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(args.out_dir / "survival_curve_comparison.csv", index=False)
    prediction_table.to_csv(args.out_dir / "weibull_censored_predictions.csv", index=False)
    summary = pd.DataFrame([{
        "events": int(event.sum()),
        "censored": int((1 - event).sum()),
        "weibull_shape": shape,
        "weibull_scale": scale,
        "weibull_median": weibull_median,
        "weibull_median_boot_ci95_low": weibull_ci[0],
        "weibull_median_boot_ci95_high": weibull_ci[1],
        "weibull_bootstrap_success_rate": successful / args.bootstrap,
        "km_median": km_median(time, event),
        "km_median_boot_ci95_low": km_ci[0],
        "km_median_boot_ci95_high": km_ci[1],
        "km_bootstrap_finite_rate": float(np.isfinite(km_values).mean()),
        "bootstrap_resamples": args.bootstrap,
    }])
    summary.to_csv(args.out_dir / "weibull_aft_summary.csv", index=False)

    high_unbounded = not np.isfinite(weibull_ci[1]) or not np.isfinite(km_ci[1])
    lines = [
        "# 生存分析升级 B3",
        "",
        f"入组为 3 个事件 + 11 个右删失（共 {len(included)} 颗），其余异常/边界电池按既有审计规则排除。",
        f"Weibull AFT MLE：shape={shape:.4g}，scale={scale:.1f} 圈，中位数={weibull_median:.1f} 圈。",
        f"KM 中位数={km_median(time, event):.1f} 圈；Weibull 与 KM 的 cell bootstrap 均运行 {args.bootstrap:,} 次。",
        f"Weibull 中位数 bootstrap 95% CI=[{weibull_ci[0]:.1f}, {weibull_ci[1]:.1f}]；KM=[{km_ci[0]:.1f}, {km_ci[1]:.1f}]。",
        f"Weibull bootstrap 可拟合比例={successful / args.bootstrap:.1%}；KM 中位数有限比例={np.isfinite(km_values).mean():.1%}。",
        "",
        "11 颗删失电池的 Weibull 区间是基于总体 AFT 分布、并条件于寿命超过各自删失时间的模型化预测区间；它不是对 11 颗各自真实寿命的直接观测。",
        "若 bootstrap 上界无穷或大量重采样不可估，说明 3 个事件不足以支持窄区间；Weibull 的形式假设不能替代信息量。",
        "Nelson-Aalen 累积风险与 KM/Weibull 曲线表见 `survival_curve_comparison.csv`。",
    ]
    (args.out_dir / "conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"删失电池预测行数: {len(prediction_table)}；结果已写入: {args.out_dir}")


if __name__ == "__main__":
    main()