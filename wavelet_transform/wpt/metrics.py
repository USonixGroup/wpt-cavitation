"""Metric registry and helpers for computing statistical metrics.

This centralises the mapping between metric attribute names and the
functions in ``wavelet_transform.utils.statistical_features`` and provides
helpers to compute a single metric or a dict of metrics for a given data
array. Keeping this logic here avoids duplication between ``Node`` and the
info functions in the inverse/results modules.
"""
from typing import Iterable, Optional, Dict
import numpy as np
import wavelet_transform.utils.statistical_features as stats


# Central registry mapping metric attribute name -> function name in stats.
# Keep the attribute names identical to the previous per-node properties
# (e.g. "average_energy") so downstream code and DataFrame columns remain
# stable.
METRIC_REGISTRY = {
    "energy": "energy",
    "log_energy": "log_energy",
    "average_energy": "avg_energy",
    "log_average_energy": "log_avg_energy",
    "kurtosis": "kurtosis",
    "crest_factor": "crest_factor",
    "skewness": "skewness",
    "abs_skewness": "abs_skewness",
    "entropy": "entropy",
    "spectral_entropy": "spectral_entropy",
    "level_entropy": "entropy",  # special case: needs total_energy arg
    "spectral_kurtosis": "spectral_kurtosis",
    "corrected_spectral_entropy": "spectral_entropy",  # special case: needs ref_psd arg
    "corrected_spectral_kurtosis": "spectral_kurtosis"  # special case: needs ref_psd arg
}

SPECIAL_METRICS = {  # metrics that need extra reference data
    "level_entropy",
    "corrected_spectral_entropy",
    "corrected_spectral_kurtosis"
}

FULL_LENGTH_METRICS = {  # metrics that should be computed on full-length inverse data
    "corrected_spectral_entropy",
    "corrected_spectral_kurtosis"
}


def display_name(metric_name: str) -> str:
    """Return a human-friendly display name for a metric.

    Replaces underscores with spaces (e.g. "average_energy" -> "average energy").
    """
    return metric_name.replace("_", " ")


def _get_stats_func(func_name: str):
    """Return callable stats function by name or raise AttributeError."""
    if not hasattr(stats, func_name):
        raise AttributeError(f"stats module has no function '{func_name}'")
    func = getattr(stats, func_name)
    if not callable(func):
        raise TypeError(f"stats.{func_name} is not callable")
    return func


def compute_metric_for_data(
    data,
    metric_name: str,
    reference=None,  # total energy or ref_array depending on metric
):
    """Compute a single metric for the provided data.

    Parameters
    ----------
    data: array-like
        The coefficient array to compute the metric on.
    metric_name: str
        The metric key (e.g. "energy", "average_energy"). Must be present in
        ``METRIC_REGISTRY``.
    total_energy: float | None
        Optional total energy to pass to metrics that need it (e.g. level_entropy).

    Returns
    -------
    value
        The computed metric value. Any exceptions raised by the underlying
        stats function are propagated to the caller.
    """
    func_name = METRIC_REGISTRY.get(metric_name)
    if func_name is None:
        raise KeyError(f"metric '{metric_name}' not registered")

    func = _get_stats_func(func_name)
    # Special-case metrics that need extra args
    if metric_name in SPECIAL_METRICS:
        return func(data, reference)
    return func(data)


def compute_metrics_for_data(
    data,
    metric_names: Optional[Iterable[str]] = None,
    reference=None,  # total energy or ref_array depending on metric
) -> Dict[str, Optional[float]]:
    """
    Compute multiple metrics for data and return a dict metric_name->value.

    If metric_names is None, computes all metrics in METRIC_REGISTRY.
    Any metric whose underlying function raises will have value set to None.
    """
    if metric_names is None:
        metric_names = list(METRIC_REGISTRY.keys())

    out = {}
    for name in metric_names:
        try:
            out[name] = compute_metric_for_data(data, name, reference)
        except Exception:
            out[name] = None
    return out

def get_ref(metric_name: str, transform) -> Optional[np.ndarray]:
    """
    Get the reference value needed for certain metrics.

    Parameters
    ----------
    metric_name: str
        The name of the metric.
    transform: Transform
        The transform object to get reference data from.

    Returns
    -------
    reference: float | np.ndarray | None
        The reference value needed for the metric, or None if not needed.
    """
    if metric_name == "level_entropy":
        return transform.total_energy(0)
    elif metric_name in ["corrected_spectral_entropy", "corrected_spectral_kurtosis"]:
        ref_array = transform.ref_psd
        return ref_array
    elif metric_name in SPECIAL_METRICS:
        raise ValueError(f"Metric '{metric_name}' requires a reference value")
    else:
        return None
    
def get_data(full_data, trimmed_data, metric_name: str):
    """
    Get the appropriate data array for certain metrics.

    Parameters
    ----------
    full_data: np.ndarray
        The full inverse transformed data.
    trimmed_data: np.ndarray
        The trimmed inverse transformed data.
    metric_name: str
        The name of the metric.

    Returns
    -------
    data: np.ndarray
        The appropriate data array for the metric.
    """
    if metric_name in FULL_LENGTH_METRICS:  # check for corrected spectral kurtosis/entropy
        return full_data
    else:
        return trimmed_data

