import json

import numpy as np
import pytest

from src.monitoring.drift import (
    build_reference_distribution,
    compute_psi,
    is_drifted,
    DEFAULT_THRESHOLD,
)


def test_reference_distribution_proportions_sum_to_one():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 500)
    edges, proportions = build_reference_distribution(values, n_bins=10)
    assert abs(sum(proportions) - 1.0) < 1e-9
    assert len(edges) == len(proportions) + 1


def test_reference_distribution_edges_are_json_serializable():
    """
    Regression test for a real bug: the original implementation used
    literal -inf/+inf for the outer bin edges, which np.quantile/tolist()
    happily produces but standard JSON (and Postgres's json column type)
    reject as invalid tokens. Caught by actually trying to insert into
    real Postgres, not by unit-testing build_reference_distribution in
    isolation -- worth keeping this as an explicit regression test since
    it wouldn't have been caught by testing the math alone.
    """
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 500)
    edges, proportions = build_reference_distribution(values, n_bins=10)

    # must not raise -- json.dumps allows Infinity by default (a Python-specific
    # non-standard extension), so this only catches the bug if edges are
    # actually finite, not merely "didn't crash"
    serialized = json.dumps(edges)
    assert "Infinity" not in serialized
    assert all(np.isfinite(e) for e in edges)


def test_reference_distribution_raises_on_too_few_values():
    with pytest.raises(ValueError):
        build_reference_distribution(np.array([1.0, 2.0, 3.0]), n_bins=10)


def test_reference_distribution_raises_on_constant_series():
    with pytest.raises(ValueError):
        build_reference_distribution(np.full(50, 5.0), n_bins=10)


def test_psi_near_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 1000)
    edges, proportions = build_reference_distribution(reference, n_bins=10)

    # current data drawn from the SAME distribution
    current = rng.normal(0, 1, 1000)
    psi = compute_psi(edges, proportions, current)
    assert psi < 0.05  # should be small, though not exactly 0 due to sampling noise


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 1000)
    edges, proportions = build_reference_distribution(reference, n_bins=10)

    # current data shifted well outside the reference's range
    current = rng.normal(5, 1, 1000)
    psi = compute_psi(edges, proportions, current)
    assert psi > DEFAULT_THRESHOLD
    assert is_drifted(psi)


def test_psi_handles_current_values_outside_reference_range():
    """
    With the finite-sentinel outer bounds, values far outside the
    reference's observed range should land in the extreme bins rather
    than raising or being silently dropped.
    """
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 1000)
    edges, proportions = build_reference_distribution(reference, n_bins=10)

    extreme_current = np.array([1e5, -1e5, 1e5, -1e5] * 50)
    psi = compute_psi(edges, proportions, extreme_current)
    assert np.isfinite(psi)
    assert psi > DEFAULT_THRESHOLD  # extreme values should register as heavy drift


def test_compute_psi_raises_on_all_nan_current():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 500)
    edges, proportions = build_reference_distribution(reference, n_bins=10)

    with pytest.raises(ValueError):
        compute_psi(edges, proportions, np.full(10, np.nan))


def test_is_drifted_threshold_boundary():
    assert is_drifted(0.2, threshold=0.2) is True  # >= is drifted, not just >
    assert is_drifted(0.19999, threshold=0.2) is False
