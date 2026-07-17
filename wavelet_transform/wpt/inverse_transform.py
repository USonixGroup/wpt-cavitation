import numpy as np
import pandas as pd
import scipy.signal as sp

from wavelet_transform.wpt.transform_object import Transform
from wavelet_transform.wpt.transform_node import Node
from wavelet_transform.wpt.metrics import (
    METRIC_REGISTRY, SPECIAL_METRICS, compute_metric_for_data, display_name, get_data, get_ref
)

# ---------------- Inverse Transforms -----------------
def inv_dwt(transform_object: Transform):
    """
    Calls the waverec() function from the pywavelet library
    """
    t = transform_object
    nodes = t.dwt_results()
    return sum([inv_node(node) for node in nodes])

def inv_dwpt(transform_object: Transform, level="leaf"):
    """
    Calls the inv_node() function on each node at a given level of the
    transform object, and sums the results.
    Parameters
    ----------
    transform_object: Transform
        the transform object containing the nodes to be reconstructed
    level: int or "leaf"
        the level of the transform to reconstruct. If "leaf", reconstructs
        all leaf nodes.
    Returns
    -------
    output: array
        the signal reconstructed from the provided nodes. This might be longer
        than the original data, in which case the coefficients at the right end
        can be cut.
    """
    if level == "leaf":
        nodes = transform_object.get_leaf_nodes()
    else:
        nodes = transform_object.get_level(level)

    # Reconstruct each node separately
    return sum([inv_node(node) for node in nodes])


# --------------- Reconstruct node -----------------

def inv_node(node: Node):
    """
    Calls the single level idwt function recursively to find the inverse
    transform of a list of packet nodes at a given level.

    Parameters
    ----------
    node: Node
        the dwt node to be reconstructed

    Returns
    -------
    output: array
        the signal reconstructed from the provided nodes. This might be longer
        than the original data, in which case the coefficients at the right end
        can be cut.
    """

    # generate list of coefficients
    coeffs = node.data.copy()
    tr_obj: Transform = node.transform

    filter_bank = Transform.pywt_filter_bank(tr_obj.wavelet)
    rec_filt_approx = filter_bank[2]
    rec_filt_detail = filter_bank[3]

    for ad in reversed(node.path):
        approx = coeffs if ad == "a" else None
        detail = coeffs if ad == "d" else None
        coeffs = idwt(approx, detail, rec_filt_approx, rec_filt_detail)
    assert isinstance(coeffs, np.ndarray), "Node data must be a numpy array"
    return coeffs[: len(tr_obj.data)]


def idwt(approx, detail, rec_filt_approx, rec_filt_detail):
    assert approx is not None or detail is not None, "At least one of approx or detail must be provided"
    rec_approx = 0 if approx is None else _get_rec(approx, rec_filt_approx)
    rec_detail = 0 if detail is None else _get_rec(detail, rec_filt_detail)

    rec_full = rec_approx + rec_detail
    return rec_full


def _get_rec(data, filter_coeffs):
    data_upsampled = upsample(data)
    data_rec = sp.convolve(data_upsampled, filter_coeffs, mode="valid")
    return data_rec


def upsample(signal):
    upsampled_signal = np.zeros(len(signal) * 2 + 1)

    # Fill upsampled signal with the original values
    upsampled_signal[1 : 2 * len(signal) : 2] = signal
    return upsampled_signal


class InverseTransform:
    "Class for performing the inverse transform on a Transform object"

    def __init__(self, transform_object: Transform):
        self.transform_object = transform_object

    def info(
        self,
        level="leaf",
        order="natural",
        freq_threshold=1.,
        keep_ratio=1.,
        *,
        include_node_info=False,
        metrics=None
    ):
        """
        Perform the inverse transform on each node,
        calculate statistical metrics for each.

        Parameters
        ----------
        level: int or str or None
            the level of transform to display information for. If "leaf", displays 
            info for leaf nodes. If None, displays info for max level.
        order: str
            "natural" or "freq" (see Transform.get_level())
        leaf_nodes: bool
            display info about the leaf nodes of the transform
        freq_threshold: float
            between 0-1. Represents the maximum normalised frequency to include
            in the dataframe.

        Returns
        -------
        info: pd.DataFrame
            A dataframe containing statistical measures calculated from the
            reconstruction coefficients at each node.
        """

        # Return data from transform object at level 0
        if level == 0:
            return self.transform_object.info(level=0, order=order)

        # Reconstruct each node separately
        info_data = []

        # Accept integer levels, None (meaning max level), or the string 'leaf'
        if level == "leaf":
            nodes = self.transform_object.get_leaf_nodes(order=order)
        elif level is None or isinstance(level, int):
            nodes = self.transform_object.get_level(level, order=order)
        else:
            raise ValueError("level must be an int, None, or 'leaf'")

        for node in nodes:
            assert isinstance(node, Node)
            # Skip nodes above frequency threshold
            if not node.is_in_threshold(freq_threshold):
                continue

            # Calculate inverse transform
            inverse = inv_node(node)
            inverse_trim = self._trim(inverse, keep_ratio)

            # base/static fields
            row = {
                "level": node.level,
                "node number": node.node_number,
                "norm frequency range": node.norm_frequency_range,
                "centre frequency": np.average(node.numeric_freq_range),
            } if include_node_info else {}

            # populate metric fields using the central helper which encapsulates
            # how metrics map to functions and how special cases are handled.
            if metrics is not None:
                # replace spaces with underscores in metric names
                _metrics = [m.replace(" ", "_") for m in metrics]
            else:
                _metrics = METRIC_REGISTRY.keys()

            for metric in _metrics:
                display = display_name(metric)
                reference = get_ref(metric, self.transform_object)
                data = get_data(inverse, inverse_trim, metric)
                row[display] = compute_metric_for_data(
                    data, metric, reference
                )

            info_data.append(row)
        info = pd.DataFrame(info_data)
        return info

    def _trim(self, data, ratio):
        lim = int((1 - ratio) / 2 * len(data))
        if lim == 0:
            return data
        return data[lim:-lim]
