"""Shared, leakage-aware helpers for the optimization experiments."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["log_var", "log_min", "log_mean", "slope", "intercept", "discrepancy"]
MERGE_PAIRS = [(0, 7), (1, 8), (2, 9), (3, 15), (4, 16)]
MERGED_LIVES = [1851, 2159, 2236, 1433, 1708]
OFFICIAL_EXCLUDE = {
    "batch1": {8, 10, 12, 13, 22},
    "batch2": {7, 8, 9, 15, 16},
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def robust_mit_life(capacity: np.ndarray, eol: float = 0.88, sustain: int = 3) -> int:
    capacity = np.asarray(capacity, dtype=float)
    valid = np.flatnonzero(capacity > 0.95)
    start = int(valid[0]) if len(valid) else 0
    run = 0
    for index, below in enumerate(capacity[start:] < eol):
        run = run + 1 if below else 0
        if run >= sustain:
            return start + index - sustain + 2
    return len(capacity)


def _valid_cache_row(row: pd.Series) -> bool:
    return (
        np.isfinite(row["cycle_life"])
        and row["cycle_life"] > 0
        and np.isfinite(row["q_c100"])
        and 0.4 < row["q_c100"] < 1.6
        and bool(row["qd10_ok"])
        and bool(row["qd100_ok"])
    )


def _features_from_row(row: pd.Series) -> dict[str, float]:
    qd10, qd100 = row["_qd10"], row["_qd100"]
    capacity = np.asarray(row["_q_dis"], dtype=float)
    if qd10 is None or qd100 is None or len(qd10) != len(qd100) or len(capacity) < 100:
        raise ValueError(f"缺少官方六特征输入: {row['batch']}#{row['cell_idx']}")
    delta = np.asarray(qd100, dtype=float) - np.asarray(qd10, dtype=float)
    slope, intercept = np.polyfit(np.arange(2, 101), capacity[1:100], 1)
    return {
        "log_var": float(np.log10(max(np.var(delta), 1e-12))),
        "log_min": float(np.log10(max(-delta.min(), 1e-12))),
        "log_mean": float(np.log10(max(abs(delta.mean()), 1e-12))),
        "slope": float(slope),
        "intercept": float(intercept),
        "discrepancy": float(np.polyval((slope, intercept), 100) - capacity[99]),
    }


def load_official_split(cache_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild the official 84-cell development pool and fixed 41/43 split."""
    raw = pd.read_pickle(cache_path)
    usable = raw.loc[raw.apply(_valid_cache_row, axis=1)].copy()
    records = []
    for _, row in usable.iterrows():
        records.append({
            "batch": row["batch"],
            "cell_idx": int(row["cell_idx"]),
            "cycle_life": float(row["cycle_life"]),
            "_q_dis": np.asarray(row["_q_dis"], dtype=float),
            **_features_from_row(row),
        })
    features = pd.DataFrame(records)
    row_for = {(row.batch, int(row.cell_idx)): idx for idx, row in features.iterrows()}

    for receiver, donor in MERGE_PAIRS:
        receiver_idx = row_for[("batch1", receiver)]
        donor_idx = row_for[("batch2", donor)]
        receiver_capacity = features.at[receiver_idx, "_q_dis"]
        donor_capacity = features.at[donor_idx, "_q_dis"]
        merged = robust_mit_life(np.concatenate([receiver_capacity, donor_capacity]))
        features.at[receiver_idx, "cycle_life"] = merged

    actual_merged = [
        int(features.at[row_for[("batch1", receiver)], "cycle_life"])
        for receiver, _ in MERGE_PAIRS
    ]
    assert actual_merged == MERGED_LIVES, f"续跑标签断言失败: {actual_merged} != {MERGED_LIVES}"

    development = features.loc[
        ((features["batch"] == "batch1")
         & ~features["cell_idx"].isin(OFFICIAL_EXCLUDE["batch1"]))
        | ((features["batch"] == "batch2")
           & ~features["cell_idx"].isin(OFFICIAL_EXCLUDE["batch2"]))
    ].copy()
    development = pd.concat([
        development.loc[development["batch"] == "batch1"].sort_values("cell_idx"),
        development.loc[development["batch"] == "batch2"].sort_values("cell_idx"),
    ], ignore_index=True)
    assert len(development) == 84, f"官方开发池应为 84 颗，实际 {len(development)}"
    assert list(development[FEATURES].columns) == FEATURES
    assert development[FEATURES].notna().all().all()

    test_positions = sorted(set(range(0, len(development), 2)) | {len(development) - 1})
    test = development.iloc[test_positions].copy().reset_index(drop=True)
    train = development.drop(development.index[test_positions]).copy().reset_index(drop=True)
    assert len(train) == 41 and len(test) == 43, (len(train), len(test))
    assert set(zip(train.batch, train.cell_idx)).isdisjoint(set(zip(test.batch, test.cell_idx)))
    assert list(train[FEATURES].columns) == FEATURES
    assert list(test[FEATURES].columns) == FEATURES
    return development, train, test


def percentage_error(prediction: np.ndarray, actual: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return np.abs(prediction - actual) / actual * 100.0


def bootstrap_median_ci(errors: np.ndarray, *, seed: int = 42, n_boot: int = 10_000) -> tuple[float, float]:
    errors = np.asarray(errors, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(errors), size=(n_boot, len(errors)))
    medians = np.median(errors[indices], axis=1)
    return tuple(np.percentile(medians, [2.5, 97.5]).astype(float))


def paired_median_difference_ci(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int = 42,
    n_boot: int = 10_000,
) -> tuple[float, float]:
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.shape != baseline.shape:
        raise ValueError("配对 bootstrap 要求候选和基线逐颗误差一一对应")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(candidate), size=(n_boot, len(candidate)))
    differences = np.median(candidate[indices], axis=1) - np.median(baseline[indices], axis=1)
    return tuple(np.percentile(differences, [2.5, 97.5]).astype(float))


def interpret_difference(delta: float, ci_low: float, ci_high: float) -> str:
    if ci_high < 0 and delta <= -2.0:
        return "优于"
    if ci_low > 0 and delta >= 2.0:
        return "劣于"
    return "一致（改善不足 2pp 或 CI 未排除零）"