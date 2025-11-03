from pywt import waverec  # , idwt as pyidwt
import numpy as np
import pandas as pd
import scipy.signal as sp

from wavelet_transform.wpt.transform_object import Transform
import wavelet_transform.utils.statistical_features as stats
from wavelet_transform.wpt.transform_node import Node


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
        level=None,
        order="natural",
        leaf_nodes=False,
        freq_threshold=1,
        keep_ratio=1,
    ):
        """
        Perform the inverse transform on each node,
        calculate statistical metrics for each.

        Parameters
        ----------
        level: int
            the level of transform to display information for
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

        tr_obj = self.transform_object

        # Return data from transform object at level 0
        if level == 0:
            return self.transform_object.info(level=0, order=order)

        # Reconstruct each node separately
        info_data = []
        if not isinstance(leaf_nodes, bool):
            raise ValueError("leaf_nodes argument must be boolean")
        if leaf_nodes is False:
            nodes = self.transform_object.get_level(level, order=order)
        else:
            nodes = self.transform_object.get_leaf_nodes(order, freq_threshold)
        total_energy = tr_obj.total_energy(0)
        for node in nodes:
            assert isinstance(node, Node)
            # Calculate inverse transform
            if node.norm_frequency_range[1] > freq_threshold:
                continue
            inverse = inv_node(node)
            inverse = self._trim(inverse, keep_ratio)
            info_data.append(
                {
                    "level": node.level,
                    "node number": node.node_number,
                    "norm frequency range": node.norm_frequency_range,
                    "frequency range": node.freq_range,
                    "centre frequency": np.average(node.numeric_freq_range),
                    "energy": stats.energy(inverse),
                    "log energy": stats.log_energy(inverse),
                    "average energy": stats.avg_energy(inverse),
                    "log average energy": stats.log_avg_energy(inverse),
                    "kurtosis": stats.kurtosis(inverse),
                    "crest factor": stats.crest_factor(inverse),
                    "skewness": stats.skewness(inverse),
                    "abs skewness": stats.abs_skewness(inverse),
                    "entropy": stats.entropy(inverse),
                    "spectral entropy": stats.spectral_entropy(inverse),
                    "level entropy": stats.entropy(inverse, total_energy),
                }
            )
        info = pd.DataFrame(info_data)
        return info

    def _trim(self, data, ratio):
        lim = int((1 - ratio) / 2 * len(data))
        if lim == 0:
            return data
        return data[lim:-lim]
