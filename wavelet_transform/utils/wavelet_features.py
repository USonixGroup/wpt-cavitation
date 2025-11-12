import wavelet_transform.wpt.transform as wpt

from typing import Union
import pandas as pd

def extract_wavelet_features(
    data,
    levels=None,
    wavelet="haar",
    mode="symmetric",
    *,
    best_basis=None,
    min_level=0,
    freq_threshold=1.,
    keep_ratio: Union[str, int, float]="full",
    metrics=None,
    include_node_info=False,
):
    """
    Generate training data for clustering using the wavelet packet transform. Statistical
    features are calculated using the array of coefficients at each node.

    Parameters
    ----------
    data: np.ndarray
        the input signal to be processed
    levels: int
        the number of levels used in the wavelet transform
    wavelet: str
        the wavelet used in the wavelet transform (see pywt library)
    mode: str
        the type of padding used in the transform

    Keyword Arguments
    -----------------
    best_basis: str or None
        whether to use the best basis decomposition
    min_level: int
        the minimum level of decomposition for the best basis decomposition
    freq_threshold: float
        between 0 - 1. nodes with normalised frequency range (a, b) where a and b are
        between 0 and 1 will be removed if b > freq_threshold.
    keep_ratio: str, int, float
        the section of data to keep when calculating statistical metrics on nodes.
        see Node class for more details. This sets the 'keep_ratio' attribute of the
        transform object which will be used for all nodes when calculating metrics with
        node.metric().
    metrics: list or None
        list of metric names to include in the dataframe. If None, includes all
        available metrics.
    include_node_info: bool
        whether to include node information (level, node number, frequency range) in the dataframe

    Returns
    -------
    wpt_df: pd.DataFrame
        a dataframe containing training data where each row represents a node with
        several features
    tr_obj: wpt.Transform
        the wavelet packet transform object containing the nodes and coefficients"""
    # DWPT
    tr_obj = wpt.dwpt(
        data,
        levels,
        wavelet,
        mode,
        best_basis=best_basis,
        min_level=min_level,
        freq_threshold=freq_threshold,
    )

    tr_obj.keep_ratio = keep_ratio
    wpt_df = tr_obj.info(
        levels, order="natural", leaf_nodes=True, metrics=metrics, include_node_info=include_node_info
    )

    # Return the transform object if specified
    if not isinstance(tr_obj, wpt.Transform):
        raise ValueError("transform object could not be created")
    if not isinstance(wpt_df, pd.DataFrame):
        raise ValueError("dataframe could not be created")
    return wpt_df, tr_obj