"""Offline, known-answer tests for ``scripts/protocol_drift.py``.

These tests never touch ``data/mit`` or ``data/nasa``: the synthetic cases carry a
known answer, and the committed-artifact case reads only CSV files shipped under
``results/``.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from protocol_drift import (  # noqa: E402
    BAND_TABLE_COLUMNS,
    DEFAULT_QUANTILES,
    DISCIPLINE_EN,
    DISCIPLINE_ZH,
    QUANTILE_TABLE_COLUMNS,
    band_audit,
    ks_drift,
    quantile_alignment,
    reproduce_committed_evidence,
)


def test_same_distribution_is_not_flagged_and_calibration_holds():
    """Null hypothesis: two draws from one distribution must not be flagged."""
    rng = np.random.default_rng(42)
    base = rng.normal(0.0, 1.0, size=200)
    same = rng.normal(0.0, 1.0, size=200)

    drift = ks_drift(base, same)
    assert drift.significant is False
    assert drift.p_value > 0.05
    assert "no drift detected" in drift.verdict_en
    # The epistemic discipline is embedded verbatim, in both languages.
    assert DISCIPLINE_EN in drift.verdict_en
    assert DISCIPLINE_ZH in drift.verdict_zh

    # One lucky seed proves nothing, so sweep many seeded null trials: the flag rate
    # must sit near alpha -- not 0 (test never fires) and not 1 (always fires).
    sweep = np.random.default_rng(20261023)
    flagged = sum(
        ks_drift(sweep.normal(0.0, 1.0, 200), sweep.normal(0.0, 1.0, 200)).significant
        for _ in range(400)
    )
    assert 0 < flagged <= 30, f"null false-positive rate out of control: {flagged}/400"

    # Low power must not be dressed up as equivalence.
    small = np.random.default_rng(1234)
    weak = ks_drift(small.normal(0.0, 1.0, 8), small.normal(0.2, 1.0, 8))
    assert weak.significant is False
    assert "unresolved" in weak.verdict_en
    assert "未决" in weak.verdict_zh


def test_deliberate_shift_is_detected():
    """A known +1 sigma shift must be flagged, with a large statistic and tiny p."""
    rng = np.random.default_rng(42)
    base = rng.normal(0.0, 1.0, size=200)
    rng.normal(0.0, 1.0, size=200)  # the null draw consumed in the same stream
    shifted = rng.normal(1.0, 1.0, size=200)

    drift = ks_drift(base, shifted)
    assert drift.significant is True
    assert drift.statistic > 0.25
    assert drift.p_value < 1e-6
    assert drift.median_shift > 0.5
    assert "drift detected" in drift.verdict_en
    assert DISCIPLINE_EN in drift.verdict_en


def test_quantile_alignment_schema():
    """The quantile table keeps a fixed schema and an exact delta definition."""
    rng = np.random.default_rng(7)
    reference = rng.normal(0.0, 1.0, size=300)
    target = rng.normal(0.5, 1.0, size=300)

    table = quantile_alignment(reference, target)
    assert list(table.columns) == list(QUANTILE_TABLE_COLUMNS)
    assert len(table) == len(DEFAULT_QUANTILES)
    assert table["quantile"].is_monotonic_increasing
    assert np.allclose(table["delta"], table["target"] - table["reference"])
    assert np.all(table["target"] > table["reference"])  # the known +0.5 shift

    custom = quantile_alignment(reference, target, quantiles=(10.0, 90.0))
    assert list(custom["quantile"]) == [10.0, 90.0]


def test_band_audit_schema_and_band_edges():
    """Percentile band audit: fixed schema, and a value exactly on the edge is inside."""
    reference = np.arange(100, dtype=float)  # 0..99, so P05 == 5 and P95 == 95
    table = band_audit(reference, [5.0, 95.0, 4.0, 96.0], labels=["p05", "p95", "below", "above"])

    assert list(table.columns) == list(BAND_TABLE_COLUMNS)
    assert list(table["label"]) == ["p05", "p95", "below", "above"]
    assert list(table["outside"]) == [False, False, True, True]
    assert list(table["band_low"]) == [5.0] * 4
    assert list(table["band_high"]) == [95.0] * 4


def test_committed_artifacts_reproduce_readme_claims_offline():
    """The README's headline drift numbers are re-derivable from committed CSVs alone."""
    evidence = reproduce_committed_evidence()

    assert evidence["mit_dev_n"] == 84
    assert (evidence["zero_shot_en_ratio_min"], evidence["zero_shot_en_ratio_max"]) == (74.8, 125.3)
    assert (evidence["zero_shot_rf_ratio_min"], evidence["zero_shot_rf_ratio_max"]) == (2.25, 3.14)
    assert evidence["out_of_band_observations"] == 18
    assert evidence["total_observations"] == 18
    assert all(evidence["quantile_reference_recomputed_matches_committed"].values())

    geometric = evidence["en_geometric_mean_ratio_by_experiment"]
    assert 119.5 < geometric["E0 全窗+70%"] < 121.5  # committed: x120.42
    assert 245.0 < geometric["E1 窗口3.6-3.8+70%"] < 247.5  # committed: x246.11
    assert evidence["dev_vs_batch3_all_significant"] is True