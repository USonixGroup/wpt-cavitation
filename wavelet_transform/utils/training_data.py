import os
import numpy as np
import pandas as pd
import pickle
from scipy.signal import windows, resample

import wavelet_transform.wpt.transform as wpt
from wavelet_transform.utils.bubbledynamics import BubbleDynamics as BD
import itertools
import multiprocessing
from functools import partial

# ------------- Generating Bubble Dynamics Data --------------

# Get path to bubble dynamics folder in docs
FILE_PATH = os.path.abspath(__file__)
RELATIVE_PATH = "../../../docs/bubble_dynamics_data"
FOLDER_PATH = os.path.abspath(os.path.join(FILE_PATH, RELATIVE_PATH))

# Default parameters rate for generating bubble dynamics
SAMPLING_RATE = 1e9
TSTEP = 1.0 / SAMPLING_RATE
R_MEASURE = 500e-6
RDOT_INIT = 0


def generate_training_data_with_wavelet_par(
    input_params,
    sampling_rate,
    std_noise,
    levels,
    wavelet,
    mode,
    min_entropy,
    min_level,
    freq_threshold,
):
    R_init = input_params[0]
    frequency = input_params[1]
    pac = input_params[2]

    data = get_data(R_init, frequency, pac, sampling_rate)
    data = data + np.random.normal(0, std_noise, size=len(data))
    data = data * windows.blackmanharris(len(data), sym=False)
    data = (data - np.mean(data)) / np.std(data)

    # Generate traning data using the wavelet transform
    training_data = generate_training_data(
        data,
        levels,
        wavelet,
        mode,
        min_entropy=min_entropy,
        min_level=min_level,
        freq_threshold=freq_threshold,
    )

    return training_data


def generate_training_data_with_wavelet(
    wavelet,
    sampling_rate,
    levels,
    mode,
    std_noise,
    min_entropy=False,
    min_level=0,
    freq_threshold=None,
):
    """
    Generate training data from a range of signals.
    min_level forces the minimum entropy decomposition to decompose the signal to a specified
    number of samples.
    """

    R_init = [i * 1e-6 for i in range(1, 9, 1)]
    frequency = [i * 1e5 for i in range(5, 20, 1)]
    pac = [-i * 1e3 for i in range(100, 3000, 100)]
    paramlist = list(itertools.product(R_init, frequency, pac))

    pool = multiprocessing.Pool(processes=16)
    results = pool.map(
        partial(
            generate_training_data_with_wavelet_par,
            sampling_rate=sampling_rate,
            std_noise=std_noise,
            levels=levels,
            wavelet=wavelet,
            mode=mode,
            min_entropy=min_entropy,
            min_level=min_level,
            freq_threshold=freq_threshold,
        ),
        paramlist,
    )
    # Concat all dataframes into a single dataframe
    wpt_df = pd.concat(results, ignore_index=True)

    if freq_threshold is not None:
        wpt_df = _ditch_nodes_above_threshold(wpt_df, freq_threshold)

    return wpt_df


def get_data(R_init, frequency, pac, sampling_rate, n_cycles=100, return_time=False):
    """
    Get data from bubble dynamics model or generate it if it doesn't exist.

    Parameters
    ----------
    R_init: float
        the initial radius of the bubble
    frequency: float
        the frequency of the ultrasound field
    pac: float
        the intensity of the ultrasound field
    sampling_rate: float
        the desired sampling rate
    n_cycles: float
        the number of cycles of the signal to generate
    return_time: bool
        set to True to return the time array alongside the signal

    """

    # Get name of file containing data
    identifier = f"R_{R_init}_freq_{frequency}_pac_{pac}.pkl"
    file_path = os.path.join(FOLDER_PATH, identifier)

    # Create folder if it doesn't exist
    os.makedirs(FOLDER_PATH, exist_ok=True)

    # Check if file exists
    if os.path.isfile(file_path):
        # Load data from file
        with open(file_path, "rb") as file:
            data = pickle.load(file)

    else:
        # Generate data
        data = _generate_data_for_storage(R_init, frequency, pac, n_cycles)
        # Save the data for future use
        if data is None:  # Failed data generation
            return None
        with open(file_path, "wb") as file:
            pickle.dump(data, file)

    # Downsample data
    time = None
    if return_time is True:
        tmax = n_cycles / frequency
        tstep = 1 / SAMPLING_RATE
        time = np.arange(0, tmax, tstep)

    return _resample(data, sampling_rate, time)


def _resample(data, sampling_rate, time=None):
    # Calculate sampling rate ratio and downsample the data
    new_length = len(data) * sampling_rate / SAMPLING_RATE
    return resample(data, int(new_length), time)


def _generate_data_for_storage(R_init, frequency, pac, n_cycles=20):
    """
    Generate data using bubble dynamics at a sampling rate of 1e9 Hz
    and resample to a given sampling rate.
    """
    i = 0
    while True:
        # Generate sound wave
        t_max = n_cycles / frequency
        time = np.arange(0.0, t_max, TSTEP)
        data = generate_data(R_init, RDOT_INIT, frequency, pac, time, TSTEP, R_MEASURE)

        # Check if the dynamics model failed
        if not np.isnan(data).any():
            break

        # Repeat and give up after a number of attempts.
        i += 1
        if i > 50:
            print(
                f"Could not generate dynamics data with R_init: {R_init}, frequency:"
                f" {frequency}, pac: {pac}"
            )
            return None

    if np.isnan(np.array(data)).any():
        raise ValueError("nan found in data")

    return data


def generate_data(R_init, Rdot_init, frequency, pac, time, tstep, r_measure):
    "Generate data from bubble dynamics."
    # Generate sound wave
    BD_obj = BD(R_init, Rdot_init, frequency, pac, time, tstep)
    BD_obj.solver()

    # Calculate pressure radiated (proportional to measured voltage)
    BD_obj.calculate_p_radiated(r_measure)
    data = BD_obj.p_radiated

    return data


# ------------ Generate Training Data from Single Sound Wave ------------


def generate_training_data(
    data,
    levels=None,
    wavelet="haar",
    mode="symmetric",
    *,
    min_entropy=False,
    min_level=0,
    include_tr_ob=False,
    freq_threshold=None,
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

    Keyword Arguments
    -----------------
    min_entropy: bool
        whether to use the minimum entropy decomposition
    min_level: int
        the minimum level of decomposition for the minimum entropy decomposition
    freq_threshold: float
        between 0 - 1. nodes with normalised frequency range (a, b) where a and b are
        between 0 and 1 will be removed if b > freq_threshold.
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
    tr_ob = wpt.dwpt(
        data, levels, wavelet, mode, min_entropy=min_entropy, min_level=min_level
    )

    wpt_df = tr_ob.info(levels, order="natural", leaf_nodes=min_entropy)

    if freq_threshold is not None:
        wpt_df = _ditch_nodes_above_threshold(wpt_df, freq_threshold)
    # Return the transform object if specified
    if include_tr_ob is True:
        return wpt_df, tr_ob
    else:
        return wpt_df


def _ditch_nodes_above_threshold(wpt_df, freq_threshold):
    "Remove nodes from the dataframe with noramlisesd upper frequency below threshold"
    if not 0 <= freq_threshold <= 1:
        raise ValueError(
            "freq_threshold must be the normalised with the nyquist frequency"
        )
    # Filter out nodes which have upper bound freq greater than threshold
    node_mask = wpt_df["norm frequency range"].apply(
        lambda x: float(x[1]) <= freq_threshold
    )

    return wpt_df[node_mask]
