import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from optimization_common import robust_mit_life


def test_robust_life_skips_zero_placeholder():
    capacity = np.array([0.0, 1.12, 1.08, 0.87, 0.86, 0.85])
    assert robust_mit_life(capacity) == 4


def test_robust_life_ignores_single_cycle_dip():
    capacity = np.array([1.12, 1.08, 0.84, 1.02, 0.86, 0.85, 0.84])
    assert robust_mit_life(capacity) == 5


def test_robust_life_marks_exactly_three_sustained_low_cycles():
    capacity = np.array([1.12, 0.87, 0.86, 0.85])
    assert robust_mit_life(capacity) == 2