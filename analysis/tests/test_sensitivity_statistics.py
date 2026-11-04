"""Patient-pairing, tied-rank bootstrap and exact EF decomposition checks."""
import numpy as np
import pytest
from scipy.stats import spearmanr

from analysis.sensitivity_statistics import spearman_ci, volume_error_contributions, paired_summary


def test_bootstrap_reranks_duplicates_and_ties():
    x = np.array([1, 1, 3, 4, 8, 9.])
    y = np.array([6, 3, 3, 4, 2, 1.])
    result = spearman_ci(x, y, seed=8, repetitions=250)
    assert result["spearman_rho"] == pytest.approx(spearmanr(x, y).statistic)
    indices = np.random.default_rng(8).integers(0, len(x), size=(250, len(x)))
    explicit = [spearmanr(x[i], y[i]).statistic for i in indices]
    explicit = [r for r in explicit if np.isfinite(r)]
    np.testing.assert_allclose([result["ci95_low"], result["ci95_high"]], np.percentile(explicit, [2.5, 97.5]))


def test_undefined_association_remains_missing():
    result = spearman_ci([1, 1, 1, 1], [1, 2, 3, 4])
    assert result["n"] == 4 and result["spearman_rho"] is None and result["ci95_low"] is None


@pytest.mark.parametrize("d,e,pd,pe", [(100,50,110,55), (100,50,100,60), (100,50,90,60)])
def test_volume_decomposition_is_exact(d, e, pd, pe):
    delta = 100*(1-pe/pd)-100*(1-e/d)
    result = volume_error_contributions(d, e, pd, pe, delta)
    assert sum(result.values()) == pytest.approx(delta)
    assert result["ef_rounding_remainder_pp"] == pytest.approx(0, abs=1e-12)


def test_volume_decomposition_keeps_rounding_residual():
    delta = round(100*(1-60/91))-round(100*(1-50/101))
    result = volume_error_contributions(101, 50, 91, 60, delta)
    assert sum(result.values()) == pytest.approx(delta)
    assert abs(result["ef_rounding_remainder_pp"]) <= 1


def test_repeated_patient_not_treated_as_independent():
    row = {"patient_id": "p", "method": "m", "scenario_id": "s", "reference_ef_group": "le45", "image_quality": "Good"}
    with pytest.raises(ValueError, match="Repeated patient"):
        paired_summary([row, row], {"seed": 523, "bootstrap_repetitions": 2000})
