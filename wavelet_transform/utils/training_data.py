import pandas as pd

import wavelet_transform.wpt.transform as wpt
from wavelet_transform.utils.bubbledynamics import BubbleDynamics as BD


def generate_data(R_init, Rdot_init, frequency, pac, time, tstep, r_measure):
    "Generate data using bubble dynamics."
    # Generate sound wave
    BD_obj = BD(R_init, Rdot_init, frequency, pac, time, tstep)
    BD_obj.solver()

    # Calculate pressure radiated (proportional to measured voltage)
    BD_obj.calculate_p_radiated(r_measure=r_measure)
    data = BD_obj.p_radiated

    return data


def generate_training_data(
    data, levels, wavelet, mode, min_energy=0, include_tr_ob=False
):
    """
    Generate training data for clustering using the wavelet packet transform. Statistical
    features are calculated using the array of coefficients at each node.

    Paremeters
    ----------
    data: np.ndarray
        the input signal to be processed
    levels: int
        the number of levels used in the wavelet transform
    wavelet: str
        the wavelet used in the wavelet transform (see pywt library)
    mode: str
        the type of padding used in the transform
    min_energy: float
        the ratio of the total energy that a node must have to be used as training data
    include_tr_ob: bool
        whether to return the transform object used for the transform containing
        info about the nodes.

    Returns
    -------
    wpt_df: pd.DataFrame
        a dataframe containing training data where each row represents a node with
        several features

    """
    # DWPT
    tr_ob = wpt.dwpt(data, levels, wavelet, mode)

    # calculate energy, kurtosis, and crest factor of reconstructed signals
    wpt_df = pd.DataFrame(
        columns=[
            "node",
            "energy",
            "kurtosis",
            "crest",
            "entropy",
            "skewness",
        ]
    )

    total_energy = sum(node.energy for node in tr_ob.get_level(level=levels))
    for node in tr_ob.get_level(level=levels, order="freq"):
        if node.energy / total_energy < min_energy:
            continue
        # Calculate statistical measures
        counter = node.node_number
        wpt_df.loc[counter, "node"] = node
        wpt_df.loc[counter, "energy"] = node.energy
        wpt_df.loc[counter, "kurtosis"] = node.kurtosis
        wpt_df.loc[counter, "crest"] = node.crest_factor
        wpt_df.loc[counter, "entropy"] = node.entropy
        wpt_df.loc[counter, "skewness"] = node.skewness

    if include_tr_ob is True:
        return wpt_df, tr_ob
    else:
        return wpt_df
