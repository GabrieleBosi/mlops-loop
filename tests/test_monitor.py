"""Step 9: PSI, checked against numbers computed by hand rather than against itself."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops_loop import config, monitor
from mlops_loop.monitor import MonitorError, psi

# expected shares 0.8 / 0.2, actual shares 0.5 / 0.5
#   PSI = (0.5 - 0.8) * ln(0.5 / 0.8) + (0.5 - 0.2) * ln(0.5 / 0.2)
#       = (-0.3) * (-0.4700036292) + (0.3) * (0.9162907319)
#       =   0.1410010888          +   0.2748872196
#       =   0.4158883083
HAND_COMPUTED = 0.4158883083


def test_psi_categorical_matches_the_hand_computed_value() -> None:
    expected = np.array(list("A" * 8 + "B" * 2))
    actual = np.array(list("A" * 5 + "B" * 5))
    assert psi(expected, actual, categorical=True) == pytest.approx(HAND_COMPUTED, abs=1e-9)


def test_psi_numeric_quantile_bins_match_the_hand_computed_value() -> None:
    """Ten values into two quantile bins puts the split at 4.5, giving 5/5 in the reference.

    The batch is eight rows below the split and two above, so the shares are 0.8 / 0.2
    against 0.5 / 0.5, which is the same arithmetic as the categorical case above.
    """
    expected = np.arange(10)
    actual = np.array([0] * 8 + [9] * 2)
    assert psi(expected, actual, bins=2) == pytest.approx(HAND_COMPUTED, abs=1e-9)


def test_psi_is_zero_for_identical_distributions() -> None:
    values = np.arange(100)
    assert psi(values, values, bins=10) == pytest.approx(0.0, abs=1e-12)
    letters = np.array(list("ABCABCABC"))
    assert psi(letters, letters, categorical=True) == pytest.approx(0.0, abs=1e-12)


def test_psi_is_symmetric_for_categories_and_not_for_numerics() -> None:
    """Categorical bins come from the union; numeric bins come from the reference."""
    left = np.array(list("A" * 8 + "B" * 2))
    right = np.array(list("A" * 5 + "B" * 5))
    assert psi(left, right, categorical=True) == pytest.approx(psi(right, left, categorical=True))

    expected = np.arange(10)
    actual = np.array([0] * 8 + [9] * 2)
    assert psi(expected, actual, bins=2) != pytest.approx(psi(actual, expected, bins=2))


def test_a_category_absent_from_the_reference_is_large_but_finite() -> None:
    expected = np.array(list("A" * 10))
    actual = np.array(list("A" * 5 + "B" * 5))
    value = psi(expected, actual, categorical=True)
    assert np.isfinite(value)
    assert value > 5.0


def test_psi_grows_as_the_batch_moves_away() -> None:
    reference = np.random.default_rng(0).normal(0, 1, 5000)
    near = np.random.default_rng(1).normal(0.1, 1, 5000)
    far = np.random.default_rng(2).normal(2.0, 1, 5000)
    assert psi(reference, near) < psi(reference, far)
    assert psi(reference, near) < 0.10
    assert psi(reference, far) > 0.25


def test_a_constant_reference_column_does_not_divide_by_zero() -> None:
    constant = np.ones(50)
    assert psi(constant, np.ones(50)) == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(psi(constant, np.zeros(50)))


def test_empty_input_stops() -> None:
    with pytest.raises(MonitorError, match="non-empty"):
        psi(np.array([]), np.arange(5))


def test_bands_follow_the_conventional_reading() -> None:
    def band(value: float) -> str:
        return monitor.FeaturePSI("x", value, 0.25, "numeric", 10).band

    assert band(0.05) == "none"
    assert band(0.10) == "moderate"
    assert band(0.25) == "moderate"
    assert band(0.30) == "major"


def test_thresholds_must_name_exactly_the_model_features(splits) -> None:
    """A feature nobody set a threshold for is a feature nobody is watching."""
    psi_cfg = dict(config.drift_config()["psi"])
    psi_cfg["thresholds"] = {"tenure": 0.25}
    with pytest.raises(MonitorError, match="does not match the model's features"):
        monitor.feature_psi(splits.train, splits.batch("fibre-monthly"), psi_cfg)


def test_feature_psi_covers_every_model_input(splits) -> None:
    from mlops_loop.features import FEATURE_COLUMNS

    results = monitor.feature_psi(
        splits.train, splits.batch("fibre-monthly"), config.drift_config()["psi"]
    )
    assert {item.feature for item in results} == set(FEATURE_COLUMNS)
    assert all(np.isfinite(item.psi) for item in results)
    assert results == sorted(results, key=lambda item: -item.psi)


def test_internet_service_is_the_loudest_signal(splits) -> None:
    """Train has no fibre customer and both batches are all fibre, so this must dominate."""
    results = monitor.feature_psi(
        splits.train, splits.batch("fibre-monthly"), config.drift_config()["psi"]
    )
    assert results[0].feature == "InternetService"
    assert results[0].breached


def test_a_batch_against_itself_shows_no_drift(splits) -> None:
    psi_cfg = config.drift_config()["psi"]
    results = monitor.feature_psi(splits.train, splits.train, psi_cfg)
    assert all(not item.breached for item in results)
    assert max(item.psi for item in results) < 0.01
