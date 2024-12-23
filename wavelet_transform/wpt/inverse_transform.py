from pywt import waverec  # , idwt as pyidwt
import numpy as np
import pandas as pd
import scipy.signal as sp

from wavelet_transform.wpt.transform_object import Transform
import wavelet_transform.utils.statistical_features as stats
from wavelet_transform.wpt.transform_node import Node


def nodes2data(nodes):
    """
    Converts a list of nodes to a list of arrays containing their data.
    """
    return [None if node is None else node.data for node in nodes]


# ---------------- Inverse DWT -----------------


def inv_dwt(transform_object: Transform):
    """
    Calls the waverec() function from the pywavelet library
    """
    t = transform_object
    nodes = t.dwt_results()
    coeffs = nodes2data(nodes)
    return waverec(coeffs, t.wavelet, t.mode)[: len(transform_object.data)]


# def inv_dwt(transform_object: Transform):
#     """
#     Iteratively reconstructs each node by level
#     """
#     # TODO


# --------------- Inverse DWPT -----------------


def inv_dwpt(transform_object: Transform, level=None, **kwargs):
    """
    Calls the single level idwt function recursively to find the inverse
    transform of a list of packet nodes at a given level.

    Parameters
    ----------
    transform_object: Transform
        Transform object containing nodes to be transformed
    level: int
        the level at which nodes will be transformed. max level by default.

    Keyword Arguments
    -----------------
    nodes: list of str or list of int
        List of nodes to reconstruct. Can specify a path e.g. "ad" or the node number (0-indexed)
    nodes_to_leave: list of str or list of int
        List of nodes to leave out of reconstruction.
    frequencies: list
        List of frequenices (int or float) to reconstruct. Only nodes containing these frequencies
        will be reconstructed.
    frequencies_to_leave: list
        List of frequenices (int or float) to leave out of reconstruction. Nodes containing these
        frequencies will not be reconstructed.
    min_energy: float or int
        The minimum energy a node must have to be added to the reconstruction.
    max_energy: float or int
        The maximumm energy a node can have to be added to the reconstruction.

    Returns
    -------
    output: array
        the signal reconstructed from the provided nodes. This might be longer
        than the original data, in which case the coefficients at the right end
        can be cut.
    """
    if level is not None and not isinstance(level, int):
        raise ValueError("level must be an integer")

    # generate list of coefficients
    nodes = packet_nodes_for_rec(transform_object, level, **kwargs)
    coeffs = nodes2data(nodes=nodes)  # change to nodes2fulldata for filtfilt inverse
    assert all(item is None for item in coeffs) is False, "node not found"
    levels = int((np.log2(len(coeffs))))

    filter_bank = Transform.pywt_filter_bank(transform_object.wavelet)
    rec_filt_approx = filter_bank[2]
    rec_filt_detail = filter_bank[3]

    input = coeffs
    for _ in range(levels):
        output = []
        for i in range(int(len(input) / 2)):
            approx = input[2 * i]
            detail = input[2 * i + 1]
            if (approx is None) and (detail is None):  # if neither node is used
                output.append(None)
                continue
            c = idwt(approx, detail, rec_filt_approx, rec_filt_detail)
            # c = pyidwt(approx, detail, transform_object.wavelet, transform_object.mode)
            output.append(c)
        input = output.copy()
    return output[0][: len(transform_object.data)]


def idwt(approx, detail, rec_filt_approx, rec_filt_detail):
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


def packet_nodes_for_rec(transform_object: Transform, level, **kwargs):
    """
    Gather nodes for reconstruction.

    e.g.
    transform_object.packet_nodes_for_rec(keep=['aa','dd'], min_energy=0.1)
    will reconstruct nodes 'aa' and 'dd' as long as they have minimum
    energy of 0.1.

    Parameters
    ----------
    level: int
        the level of transform to start reconstruct from

    Keyword Arguments
    -----------------
    See transform_object.find_nodes()


    Returns
    -------
    node_list: list
        list of nodes meeting requirements

    """
    tr_obj = transform_object
    nodes = tr_obj.get_full_level(level)  # max level by default

    nodes_to_keep = tr_obj.find_nodes(nodes, **kwargs)

    node_list = [node if node in nodes_to_keep else None for node in nodes]

    # check if any nodes are remaining after filtering
    if node_list.count(None) == len(node_list):
        raise ValueError("could not found any nodes matching input arguments")
    return node_list


class InverseTransform:
    "Class for performing the inverse transform on a Transform object"

    def __init__(self, transform_object: Transform):
        self.transform_object = transform_object

    def inv_dwpt(self, levels, **kwargs):
        return inv_dwpt(self.transform_object, levels, **kwargs)

    def inv_dwt(self, levels, **kwargs):
        return inv_dwt(self.transform_object, levels, **kwargs)

    def info(
        self,
        level=None,
        order="natural",
        leaf_nodes=False,
        freq_threshold=1,
        keep_ratio=1,
    ):
        """
        Perform the inverse transform on each node then calculate statistical measures on each.

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
            A dataframe containing statistical measures calculated from the reconstruction
            coefficients at each node.
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
        elif leaf_nodes is True:
            nodes = self.transform_object.get_leaf_nodes(order, freq_threshold)
        total_energy = tr_obj.total_energy(0)
        for node in nodes:
            assert isinstance(node, Node)
            # Calculate inverse transform
            if node.norm_frequency_range[1] > freq_threshold:
                continue
            inverse = inv_dwpt(tr_obj, node.level, nodes=node)
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
