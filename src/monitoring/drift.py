"""
Population Stability Index (PSI) drift detection.

Two-step design matching the storage schema in src/storage/models.py:
1. build_reference_distribution() runs ONCE, at training time, on the
   training set's feature values -- producing bin edges + reference
   proportions that get stored in Postgres (FeatureReferenceDistribution).
2. compute_psi() runs on every scheduled drift check, comparing freshly
   fetched data against those STORED bins -- it never needs the original
   raw training sample again, just the summary it already saved.

Standard PSI interpretation thresholds (not a hard science, but the
common convention): < 0.1 no meaningful shift, 0.1-0.2 moderate shift
worth watching, >= 0.2 significant shift. DEFAULT_THRESHOLD reflects the
0.2 convention.

Quantile (equal-frequency) binning is used for the reference distribution
rather than equal-width binning: it guarantees the reference is
~uniform across bins by construction, which makes the statistic more
sensitive to genuine shifts in the CURRENT data rather than being diluted
by bins that were already unevenly populated in the reference.
"""
from __future__ import annotations

import numpy as np

DEFAULT_N_BINS = 10
DEFAULT_THRESHOLD = 0.2
_EPSILON = 1e-4  # floor to avoid log(0) / division by zero in empty bins
_OUTER_BOUND = 1e6  # finite stand-in for +-inf; see build_reference_distribution docstring


def build_reference_distribution(
    reference_values: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> tuple[list[float], list[float]]:
    """
    Builds quantile-based bin edges from reference_values and returns
    (bin_edges, reference_proportions). The outer edges are replaced with
    a large finite sentinel (+-1e6) rather than literal +-inf, so that
    future current-data values outside the reference's observed range
    still fall into the extreme bins rather than being dropped -- and
    critically, so these edges survive round-tripping through JSON and
    Postgres's json column type, which reject the Infinity/-Infinity
    tokens Python's own json.dumps would otherwise silently allow. 1e6 is
    comfortably larger than any realistic value for the ratio/return
    features this project uses (a 1e6 log return would be an absurd,
    effectively-impossible market move), so it behaves identically to
    +-inf in practice without being a non-standard JSON value.
    """
    reference_values = np.asarray(reference_values, dtype=float)
    reference_values = reference_values[~np.isnan(reference_values)]
    if len(reference_values) < n_bins:
        raise ValueError(
            f"Need at least n_bins={n_bins} reference values, got {len(reference_values)}"
        )

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(reference_values, quantiles)
    edges = np.unique(edges)  # duplicate quantiles collapse for highly repeated values
    if len(edges) < 2:
        raise ValueError("Reference values have no spread -- cannot build bins from a constant series")

    edges[0] = -_OUTER_BOUND
    edges[-1] = _OUTER_BOUND

    counts, _ = np.histogram(reference_values, bins=edges)
    proportions = counts / counts.sum()

    return edges.tolist(), proportions.tolist()


def compute_psi(
    bin_edges: list[float],
    reference_proportions: list[float],
    current_values: np.ndarray,
) -> float:
    """
    Computes PSI comparing current_values against a stored reference
    distribution (bin_edges + reference_proportions from
    build_reference_distribution, loaded back from Postgres).
    """
    current_values = np.asarray(current_values, dtype=float)
    current_values = current_values[~np.isnan(current_values)]
    if len(current_values) == 0:
        raise ValueError("No current values to compute drift against (all NaN or empty)")

    bin_edges = np.asarray(bin_edges, dtype=float)
    reference_proportions = np.asarray(reference_proportions, dtype=float)

    counts, _ = np.histogram(current_values, bins=bin_edges)
    current_proportions = counts / counts.sum()

    # floor both sides so a bin with zero observations on either side
    # doesn't produce log(0) or a division by zero
    ref = np.clip(reference_proportions, _EPSILON, None)
    cur = np.clip(current_proportions, _EPSILON, None)

    psi = np.sum((cur - ref) * np.log(cur / ref))
    return float(psi)


def is_drifted(psi_score: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    return psi_score >= threshold
