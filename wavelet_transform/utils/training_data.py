import os
import numpy as np
import pandas as pd
import pickle
from scipy.signal import resample


import wavelet_transform.wpt.transform as wpt
from wavelet_transform.utils.bubbledynamics import BubbleDynamics as BD


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


def generate_training_data_with_wavelet(
    wavelet,
    sampling_rate,
    levels,
    mode,
    std_noise,
    data_len=None,
    *,
    min_entropy=False,
    min_level=0,
):
    """
    Generate training data from a range of signals.
    min_level forces the minimum entropy decomposition to decompose the signal to a specified
    number of samples.
    """
    wpt_df = None
    for R_init in (i / 1e6 for i in range(4, 11, 2)):
        for frequency in [i * 1e5 for i in range(5, 20, 2)]:
            for pac in [-i * 1e4 for i in range(1, 20, 2)]:
                data = get_data(R_init, frequency, pac, sampling_rate)
                data += np.random.normal(0, std_noise, size=len(data))

                length = len(data) if data_len is None else data_len
                data = data[:length]
                data *= np.blackman(len(data))
                data = (data - np.mean(data)) / np.std(data)

                # Generate traning data using the wavelet transform
                training_data = generate_training_data(
                    data,
                    levels,
                    wavelet,
                    mode,
                    min_entropy=min_entropy,
                    min_level=min_level,
                )

                if wpt_df is None:  # Init dataframe
                    wpt_df = training_data
                else:  # Add data to dataframe
                    wpt_df = pd.concat([wpt_df, training_data], ignore_index=True)

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


# ------------ Generate Training Data from Sound Wave ------------


def generate_training_data(
    data,
    levels=None,
    wavelet="haar",
    mode="symmetric",
    *,
    min_entropy=False,
    min_level=0,
    include_tr_ob=False,
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

    # Return the transform object if specified
    if include_tr_ob is True:
        return wpt_df, tr_ob
    else:
        return wpt_df
