"""Training-only optimization experiments for the official MIT split.

This experiment runner evaluates the A1-A4 proposal items. The official 43-cell
test split is kept untouched until every choice has been made with training CV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, ElasticNetCV, HuberRegressor, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from optimization_common import (
    FEATURES,
    bootstrap_median_ci,
    interpret_difference,
    load_official_split,
    paired_median_difference_ci,
    percentage_error,
    seed_everything,
)
from project_config import PATHS


MIT_FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
}
SEED = 42
CV = KFold(n_splits=4, shuffle=True, random_state=SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=PATHS.cache_dir / "mit_parse_cache.pkl")
    parser.add_argument("--mit-dir", type=Path, default=PATHS.mit_dir)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PATHS.output_dir / "optimization_v1" / "main",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def load_qd50(mit_dir: Path, development: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    """Read only cycle-50 Qdlin arrays needed by the additional early-life features."""
    result = {}
    for batch, filename in MIT_FILES.items():
        batch_rows = development.loc[development["batch"] == batch, "cell_idx"].astype(int).tolist()
        with h5py.File(mit_dir / filename, "r") as handle:
            cycles = handle["batch"]["cycles"][()].ravel()
            for cell_idx in batch_rows:
                cell_cycles = handle[cycles[cell_idx]]
                qd_refs = cell_cycles["Qdlin"][()].ravel()
                if len(qd_refs) < 50:
                    raise ValueError(f"{batch}#{cell_idx} 缺少第 50 圈 Qdlin")
                result[(batch, cell_idx)] = np.ravel(np.asarray(handle[qd_refs[49]], dtype=float))
    return result


def _log_positive(value: float) -> float:
    return float(np.log10(max(float(value), 1e-12)))


def _delta_stats(delta: np.ndarray, prefix: str) -> dict[str, float]:
    delta = np.asarray(delta, dtype=float)
    centered = delta - np.mean(delta)
    std = float(np.std(delta))
    skew = float(np.mean(centered**3) / std**3) if std > 1e-12 else 0.0
    return {
        f"{prefix}_log_var": _log_positive(np.var(delta)),
        f"{prefix}_log_min": _log_positive(max(-float(delta.min()), 1e-12)),
        f"{prefix}_log_mean": _log_positive(abs(float(delta.mean()))),
        f"{prefix}_log_max_abs": _log_positive(np.max(np.abs(delta))),
        f"{prefix}_log_median_abs": _log_positive(np.median(np.abs(delta))),
        f"{prefix}_skew": skew,
    }


def build_candidate_features(
    development: pd.DataFrame,
    cache: pd.DataFrame,
    qd50: dict[tuple[str, int], np.ndarray],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    cache_for = {
        (row["batch"], int(row["cell_idx"])): row
        for _, row in cache.iterrows()
    }
    rows = []
    qv_columns = []
    capacity_columns = []
    for _, row in development.iterrows():
        key = (row["batch"], int(row["cell_idx"]))
        cached = cache_for[key]
        qd10 = np.asarray(cached["_qd10"], dtype=float)
        qd50_array = qd50[key]
        qd100 = np.asarray(cached["_qd100"], dtype=float)
        if not (len(qd10) == len(qd50_array) == len(qd100)):
            raise ValueError(f"Qdlin 网格不一致: {key}")

        record = {
            "batch": row["batch"],
            "cell_idx": int(row["cell_idx"]),
            **{feature: float(row[feature]) for feature in FEATURES},
        }
        for label, left, right in (
            ("d10_50", qd10, qd50_array),
            ("d50_100", qd50_array, qd100),
            ("d10_100", qd10, qd100),
        ):
            stats = _delta_stats(right - left, label)
            record.update(stats)
            qv_columns.extend(column for column in stats if column not in qv_columns)

        capacity = np.asarray(row["_q_dis"], dtype=float)[1:100]
        blocks = np.array_split(capacity, 5)
        means = [float(block.mean()) for block in blocks]
        for index, block in enumerate(blocks, start=1):
            name = f"cap_block{index}_log_var"
            record[name] = _log_positive(np.var(block))
            if name not in capacity_columns:
                capacity_columns.append(name)
        for index in range(4):
            name = f"cap_block_slope{index + 1}"
            record[name] = (means[index + 1] - means[index]) / 20.0
            if name not in capacity_columns:
                capacity_columns.append(name)
        rows.append(record)

    extended = pd.DataFrame(rows)
    sets = {
        "baseline_6": FEATURES,
        "delta_q_18": qv_columns,
        "capacity_blocks_9": capacity_columns,
        "combined_33": qv_columns + capacity_columns + FEATURES,
    }
    assert len(sets["delta_q_18"]) == 18, len(sets["delta_q_18"])
    assert len(sets["capacity_blocks_9"]) == 9, len(sets["capacity_blocks_9"])
    return extended, sets


def tune_enet(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    alphas = np.logspace(-4, 1, 50)
    ratios = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    best = (float("inf"), None, None)
    for alpha in alphas:
        for ratio in ratios:
            fold_losses = []
            for train_idx, valid_idx in CV.split(X):
                model = make_pipeline(
                    StandardScaler(),
                    ElasticNet(alpha=float(alpha), l1_ratio=ratio, max_iter=500_000),
                )
                model.fit(X[train_idx], y[train_idx])
                fold_losses.append(np.mean(np.abs(model.predict(X[valid_idx]) - y[valid_idx])))
            score = float(np.mean(fold_losses))
            if score < best[0]:
                best = (score, float(alpha), float(ratio))
    return best[1], best[2], best[0]


def tune_rf(X: np.ndarray, y: np.ndarray) -> tuple[int, int | None, float]:
    best = (float("inf"), None, None)
    for leaf in [2, 3, 5]:
        for depth in [3, 5, None]:
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=SEED,
                n_jobs=-1,
            )
            score = -cross_val_score(
                model, X, y, cv=CV, scoring="neg_mean_absolute_error", n_jobs=1
            ).mean()
            if score < best[0]:
                best = (float(score), leaf, depth)
    return int(best[1]), best[2], best[0]


def make_enet(alpha: float, ratio: float):
    return make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=500_000),
    )


def make_rf(leaf: int, depth: int | None):
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=depth,
        min_samples_leaf=leaf,
        random_state=SEED,
        n_jobs=-1,
    )


def make_stacking_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    *,
    X_train_base: np.ndarray,
    X_predict_base: np.ndarray,
) -> np.ndarray:
    """Fit a Ridge meta-model only on out-of-fold base predictions."""
    inner = KFold(n_splits=4, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y_train), 2), dtype=float)
    for fit_idx, hold_idx in inner.split(X_train):
        enet = make_enet(0.01, 0.5)
        rf = make_rf(2, 3)
        enet.fit(X_train[fit_idx], y_train[fit_idx])
        rf.fit(X_train[fit_idx], y_train[fit_idx])
        oof[hold_idx, 0] = enet.predict(X_train[hold_idx])
        oof[hold_idx, 1] = rf.predict(X_train[hold_idx])
    meta_train = np.column_stack([oof, X_train_base])
    enet_full, rf_full = make_enet(0.01, 0.5), make_rf(2, 3)
    enet_full.fit(X_train, y_train)
    rf_full.fit(X_train, y_train)
    meta_predict = np.column_stack([
        enet_full.predict(X_predict), rf_full.predict(X_predict), X_predict_base
    ])
    meta = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    meta.fit(meta_train, y_train)
    return meta.predict(meta_predict)


def cross_validated_median_error(
    estimator_factory,
    X: np.ndarray,
    y: np.ndarray,
    *,
    stacking: bool = False,
    base_X: np.ndarray | None = None,
) -> float:
    errors = []
    for fit_idx, hold_idx in CV.split(X):
        if stacking:
            assert base_X is not None
            prediction = make_stacking_predictions(
                X[fit_idx], y[fit_idx], X[hold_idx],
                X_train_base=base_X[fit_idx], X_predict_base=base_X[hold_idx],
            )
        else:
            model = estimator_factory()
            model.fit(X[fit_idx], y[fit_idx])
            prediction = model.predict(X[hold_idx])
        errors.extend(percentage_error(10**prediction, 10**y[hold_idx]))
    return float(np.median(errors))


def save_outputs(
    out_dir: Path,
    cv_rows: list[dict],
    predictions: dict[str, np.ndarray],
    test: pd.DataFrame,
    quantiles: dict[str, np.ndarray],
    bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    actual = test["cycle_life"].to_numpy(dtype=float)
    baseline_errors = percentage_error(10**predictions["baseline_EN6"], actual)
    baseline_ci = bootstrap_median_ci(baseline_errors, n_boot=bootstrap, seed=SEED)
    model_rows = []
    prediction_rows = []
    for name, predicted_log in predictions.items():
        predicted = 10**predicted_log
        errors = percentage_error(predicted, actual)
        ci = bootstrap_median_ci(errors, n_boot=bootstrap, seed=SEED)
        delta = float(np.median(errors) - np.median(baseline_errors))
        delta_ci = paired_median_difference_ci(
            errors, baseline_errors, n_boot=bootstrap, seed=SEED
        )
        model_rows.append({
            "model": name,
            "median_error_pct": float(np.median(errors)),
            "median_ci95_low": ci[0],
            "median_ci95_high": ci[1],
            "mean_error_pct": float(np.mean(errors)),
            "delta_vs_EN6_pp": delta,
            "paired_delta_ci95_low": delta_ci[0],
            "paired_delta_ci95_high": delta_ci[1],
            "judgment": interpret_difference(delta, *delta_ci),
        })
        for index, row in enumerate(test.itertuples(index=False)):
            prediction_rows.append({
                "batch": row.batch,
                "cell_idx": int(row.cell_idx),
                "actual_life": actual[index],
                "model": name,
                "predicted_life": predicted[index],
                "error_pct": errors[index],
            })

    comparison = pd.DataFrame(model_rows)
    cv_results = pd.DataFrame(cv_rows)
    comparison.to_csv(out_dir / "official_test_comparison.csv", index=False, encoding="utf-8-sig")
    cv_results.to_csv(out_dir / "training_cv_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(
        out_dir / "official_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    if quantiles:
        pd.DataFrame({
            "batch": test["batch"],
            "cell_idx": test["cell_idx"],
            "actual_life": actual,
            **quantiles,
        }).to_csv(out_dir / "quantile_intervals.csv", index=False, encoding="utf-8-sig")
    return comparison, cv_results


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    if not args.cache.is_file():
        raise FileNotFoundError(f"MIT 缓存不存在: {args.cache}")
    if not args.mit_dir.is_dir():
        raise FileNotFoundError(f"MIT 原始数据目录不存在: {args.mit_dir}")

    development, train, test = load_official_split(args.cache)
    raw_cache = pd.read_pickle(args.cache)
    qd50 = load_qd50(args.mit_dir, development)
    extended, feature_sets = build_candidate_features(development, raw_cache, qd50)
    train_keys = list(zip(train.batch, train.cell_idx))
    test_keys = list(zip(test.batch, test.cell_idx))
    extended_index = extended.set_index(["batch", "cell_idx"])
    y_train = np.log10(train["cycle_life"].to_numpy(dtype=float))
    y_test = np.log10(test["cycle_life"].to_numpy(dtype=float))
    cv_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}

    print("A1: ElasticNet 手写网格搜索 (50 alpha x 7 l1_ratio)", flush=True)
    baseline_X_train = train[FEATURES].to_numpy(dtype=float)
    baseline_X_test = test[FEATURES].to_numpy(dtype=float)
    en_alpha, en_ratio, en_cv_mae = tune_enet(baseline_X_train, y_train)
    print(f"  best alpha={en_alpha:.6g}, l1_ratio={en_ratio:g}, CV log-MAE={en_cv_mae:.5f}")

    baseline_en = make_pipeline(
        StandardScaler(),
        ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            cv=CV,
            random_state=SEED,
            max_iter=500_000,
        ),
    )
    baseline_en.fit(baseline_X_train, y_train)
    predictions["baseline_EN6"] = baseline_en.predict(baseline_X_test)

    tuned_en = make_enet(en_alpha, en_ratio)
    tuned_en.fit(baseline_X_train, y_train)
    predictions["A1_tuned_EN6"] = tuned_en.predict(baseline_X_test)
    cv_rows.append({
        "track": "A1",
        "candidate": "manual_grid_EN6",
        "feature_set": "baseline_6",
        "training_cv_median_error_pct": cross_validated_median_error(
            lambda: make_enet(en_alpha, en_ratio), baseline_X_train, y_train
        ),
        "selected_parameters": json.dumps({"alpha": en_alpha, "l1_ratio": en_ratio}),
    })

    print("A2: 构造 18 个 ΔQ 窗口特征和 9 个容量分块特征", flush=True)
    cv_feature_scores = []
    for set_name, columns in feature_sets.items():
        X_train = extended_index.loc[train_keys, columns].to_numpy(dtype=float)
        X_test = extended_index.loc[test_keys, columns].to_numpy(dtype=float)
        alpha, ratio, cv_mae = tune_enet(X_train, y_train)
        en_cv_error = cross_validated_median_error(
            lambda a=alpha, r=ratio: make_enet(a, r), X_train, y_train
        )
        rf_leaf, rf_depth, rf_cv_mae = tune_rf(X_train, y_train)
        rf_cv_error = cross_validated_median_error(
            lambda leaf=rf_leaf, depth=rf_depth: make_rf(leaf, depth), X_train, y_train
        )
        cv_feature_scores.extend([
            {"track": "A2", "candidate": "ElasticNet", "feature_set": set_name,
             "training_cv_median_error_pct": en_cv_error,
             "selected_parameters": json.dumps({"alpha": alpha, "l1_ratio": ratio})},
            {"track": "A2", "candidate": "RandomForest", "feature_set": set_name,
             "training_cv_median_error_pct": rf_cv_error,
             "selected_parameters": json.dumps({"min_samples_leaf": rf_leaf,
                                                  "max_depth": rf_depth})},
        ])
        if set_name == "baseline_6":
            continue
    cv_rows.extend(cv_feature_scores)

    best_en_row = min(
        (row for row in cv_feature_scores if row["candidate"] == "ElasticNet"),
        key=lambda row: row["training_cv_median_error_pct"],
    )
    best_rf_row = min(
        (row for row in cv_feature_scores if row["candidate"] == "RandomForest"),
        key=lambda row: row["training_cv_median_error_pct"],
    )
    for row, name in [(best_en_row, "A2_selected_EN"), (best_rf_row, "A2_selected_RF")]:
        columns = feature_sets[row["feature_set"]]
        X_train = extended_index.loc[train_keys, columns].to_numpy(dtype=float)
        X_test = extended_index.loc[test_keys, columns].to_numpy(dtype=float)
        params = json.loads(row["selected_parameters"])
        if row["candidate"] == "ElasticNet":
            model = make_enet(params["alpha"], params["l1_ratio"])
        else:
            model = make_rf(params["min_samples_leaf"], params["max_depth"])
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)

    print("A3/A4: 在训练集内比较 HGB、OOF stacking、Huber 与分位数区间", flush=True)
    rf_leaf, rf_depth, _ = tune_rf(baseline_X_train, y_train)
    candidate_factories = {
        "baseline_RF": lambda: make_rf(rf_leaf, rf_depth),
        "A3_HistGradientBoosting": lambda: HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=7,
            l2_regularization=1.0, random_state=SEED,
        ),
        "A4_Huber_log_target": lambda: make_pipeline(
            StandardScaler(), HuberRegressor(epsilon=1.35, max_iter=2_000)
        ),
    }
    predictions["baseline_RF"] = candidate_factories["baseline_RF"]().fit(
        baseline_X_train, y_train
    ).predict(baseline_X_test)
    for name, factory in candidate_factories.items():
        if name == "baseline_RF":
            continue
        model = factory()
        model.fit(baseline_X_train, y_train)
        predictions[name] = model.predict(baseline_X_test)
        cv_rows.append({
            "track": "A3" if name.startswith("A3") else "A4",
            "candidate": name,
            "feature_set": "baseline_6",
            "training_cv_median_error_pct": cross_validated_median_error(
                factory, baseline_X_train, y_train
            ),
            "selected_parameters": "fixed before final test",
        })

    stack_cv_error = cross_validated_median_error(
        None, baseline_X_train, y_train, stacking=True, base_X=baseline_X_train
    )
    stack_prediction = make_stacking_predictions(
        baseline_X_train, y_train, baseline_X_test,
        X_train_base=baseline_X_train, X_predict_base=baseline_X_test,
    )
    predictions["A3_OOF_stacking"] = stack_prediction
    cv_rows.append({
        "track": "A3",
        "candidate": "OOF_EN_RF_Ridge",
        "feature_set": "baseline_6",
        "training_cv_median_error_pct": stack_cv_error,
        "selected_parameters": "base EN/RF fixed; Ridge alpha=1",
    })

    quantile_predictions = {}
    quantile_models = {}
    for quantile in [0.1, 0.5, 0.9]:
        model = GradientBoostingRegressor(
            loss="quantile", alpha=quantile, n_estimators=100,
            max_depth=2, learning_rate=0.03, random_state=SEED,
        )
        model.fit(baseline_X_train, y_train)
        quantile_predictions[quantile] = 10**model.predict(baseline_X_test)
        quantile_models[quantile] = model
    predictions["A3_quantile_GB_median"] = np.log10(quantile_predictions[0.5])
    q10, q50, q90 = (quantile_predictions[q] for q in [0.1, 0.5, 0.9])
    quantiles = {
        "q10_life": q10,
        "q50_life": q50,
        "q90_life": q90,
        "interval_covers_actual": ((test["cycle_life"].to_numpy() >= q10)
                                    & (test["cycle_life"].to_numpy() <= q90)),
    }
    short_life_count = int(train["cycle_life"].between(148, 335).sum())
    cv_rows.append({
        "track": "A4",
        "candidate": "short_life_specialist",
        "feature_set": "baseline_6",
        "training_cv_median_error_pct": np.nan,
        "selected_parameters": (
            f"not fitted: {short_life_count} training examples in 148-335 cycle band; "
            "requires at least 5 for exploratory classification/regression"
        ),
    })

    comparison, cv_results = save_outputs(
        args.out_dir, cv_rows, predictions, test, quantiles, args.bootstrap
    )
    interval_coverage = float(np.mean(quantiles["interval_covers_actual"]))
    comparison.to_csv(args.out_dir / "official_test_comparison.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 主结果优化实验 A1-A4",
        "",
        f"固定官方划分：训练 {len(train)} 颗、最终测试 {len(test)} 颗；续跑标签和六特征列断言均通过。",
        "所有调参与特征组选择仅使用训练集四折 CV；测试集只在本次统一定稿评估中访问一次。",
        f"A1 网格：alpha={en_alpha:.6g}, l1_ratio={en_ratio:g}，CV log-MAE={en_cv_mae:.5f}。",
        f"A4 短命电池专门模型未拟合：官方训练池 148-335 圈仅 {short_life_count} 颗，低于提案探索门槛 5；测试电池未剔除。",
        f"A3 分位数 GB 的 q10-q90 测试覆盖率={interval_coverage:.1%}（n={len(test)}；仅描述性）。",
        "中位误差改善至少 2pp 且配对 bootstrap 95% CI 排除 0 才记为“优于”；否则按提案判为“一致/噪声”。",
        "",
        "| 模型 | 测试中位误差 | 95% bootstrap CI | 相对 EN6 配对差 | 判读 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.median_error_pct:.1f}% | "
            f"[{row.median_ci95_low:.1f}, {row.median_ci95_high:.1f}] | "
            f"{row.delta_vs_EN6_pp:+.1f}pp [{row.paired_delta_ci95_low:+.1f}, "
            f"{row.paired_delta_ci95_high:+.1f}] | {row.judgment} |"
        )
    lines.extend([
        "",
        "CV 选择明细见 `training_cv_selection.csv`；逐颗预测和模型化区间见同目录 CSV。",
        "模型间测试集差异是一次性比较，不据此再调参或改选特征。",
    ])
    (args.out_dir / "conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"A1-A4 结果与结论已写入: {args.out_dir}")


if __name__ == "__main__":
    main()