import wavelet_transform.wpt.transform as wpt

def extract_wavelet_features(
    data,
    levels=None,
    wavelet="haar",
    mode="symmetric",
    *,
    best_basis=None,
    min_level=0,
    include_tr_obj=False,
    freq_threshold=1.0,
    keep_ratio="full",
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
    include_tr_obj: bool
        whether to return the transform object used for the transform containing
        info about the nodes.
    keep_ratio: str, int, float
        the section of data to keep when calculating statistical metrics on nodes.
        see Node class for more details.

    Returns
    -------
    wpt_df: pd.DataFrame
        a dataframe containing training data where each row represents a node with
        several features
    tr_obj: optional
        the transform object if include_tr_obj=True
    """
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
    wpt_df = tr_obj.info(levels, order="natural", leaf_nodes=True)

    # Return the transform object if specified
    if include_tr_obj:
        return wpt_df, tr_obj
    else:
        return wpt_df