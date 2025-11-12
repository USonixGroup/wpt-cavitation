"""Metric registry and helpers for computing statistical metrics.

This centralises the mapping between metric attribute names and the
functions in ``wavelet_transform.utils.statistical_features`` and provides
helpers to compute a single metric or a dict of metrics for a given data
array. Keeping this logic here avoids duplication between ``Node`` and the
info functions in the inverse/results modules.
"""
from typing import Iterable, Optional, Dict
import inspect
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
    # level_entropy is a special-case that uses stats.entropy with total energy
    "level_entropy": "entropy",
    "spectral_kurtosis": "spectral_kurtosis",
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
    total_energy: Optional[float] = None,
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
    # Special-case level_entropy: pass total_energy as second arg
    if metric_name == "level_entropy":
        return func(data, total_energy)
    return func(data)


def compute_metrics_for_data(
    data,
    metric_names: Optional[Iterable[str]] = None,
    total_energy: Optional[float] = None,
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
            out[name] = compute_metric_for_data(data, name, total_energy)
        except Exception:
            out[name] = None
    return out
