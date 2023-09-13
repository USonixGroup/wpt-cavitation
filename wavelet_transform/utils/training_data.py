import numpy as np
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
    data,
    levels,
    wavelet,
    mode,
    *,
    min_energy=0,
    include_low_energy=0,
    include_tr_ob=False,
    print_nodes_kept=False,
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
    include_low_energy: float
        probability between 0-1. Randomly keep nodes with less than min_energy with probability.
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

    # Create new dataframe which is a subset of tr_ob.info()
    rows_to_add = []

    # Filter out low energy nodes
    total_energy = sum(node.energy for node in tr_ob.get_level(level=levels))
    for _, row in tr_ob.info(level=levels, order="freq").iterrows():
        # Skip if the node does not meet energy requirements.
        energy = row["energy"]
        if (
            energy / total_energy < min_energy
            and np.random.random() > include_low_energy
        ):
            continue

        # Add the entire row to the new DataFrame
        rows_to_add.append(row)

    wpt_df = pd.DataFrame(rows_to_add)

    # Print additional info about how many nodes were retained from the DWPT
    if print_nodes_kept is True:
        print(f"nodes kept: {len(wpt_df['energy'])}/{2**levels}")

    # Return the transform object if specified
    if include_tr_ob is True:
        return wpt_df, tr_ob
    else:
        return wpt_df
