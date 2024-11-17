import os
import numpy as np
import pandas as pd
import pickle
from scipy.signal import resample as sc_resample
import itertools
import multiprocessing
from functools import partial

import wavelet_transform.wpt.transform as wpt
from wavelet_transform.utils.bubbledynamics import BubbleDynamics as BD
from wavelet_transform.utils.preprocessing import preprocess
from wavelet_transform.utils.extensions import sci_form


# ------------- Generating Bubble Dynamics Data --------------

# Get path to bubble dynamics folder in docs
FILE_PATH = os.path.abspath(__file__)
RELATIVE_PATH = "../../../docs/bubble_dynamics_data"
FOLDER_PATH = os.path.abspath(os.path.join(FILE_PATH, RELATIVE_PATH))

# Create folder if it doesn't exist
os.makedirs(FOLDER_PATH, exist_ok=True)

# Default parameters rate for generating bubble dynamics
SAMPLING_RATE = 1e9  # sampling rate for bd data generation
TSTEP = 1.0 / SAMPLING_RATE
R_MEASURE = 500e-6
RDOT_INIT = 0
TIME_DELAY = 0
PHASE_INITIAL = 0
PHASE_SHIFT = 0
NO_HARMONICS = 1

default_parameters = {
    "R_init": None,  # None will be filled in _get_params_from_kwargs
    "Rdot_init": RDOT_INIT,
    "frequency": None,
    "pac": None,
    "time_delay": TIME_DELAY,
    "pulse_width": None,  # default is calculated as duration of the time array
    "phase_initial": PHASE_INITIAL,
    "phase_shift": PHASE_SHIFT,
    "no_harmonics": NO_HARMONICS,
    "time": None,
    "deltat": TSTEP,
}


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
    window=False,
):
    R_init = input_params[0]
    frequency = input_params[1]
    pac = input_params[2]

    data = get_data(R_init, frequency, pac, sampling_rate)
    data += np.random.normal(0, std_noise, len(data))
    data = preprocess(data, window=window)

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
    window=False,
):
    """
    Generate training data from a range of signals.
    min_level forces the minimum entropy decomposition to decompose the signal to a specified
    number of samples.
    """

    R_init = [i * 1e-6 for i in range(1, 9, 2)]
    frequency = [i * 1e5 for i in range(5, 20, 2)]
    pac = [-i * 1e3 for i in range(100, 3000, 500)]
    paramlist = list(itertools.product(R_init, frequency, pac))

    pool = multiprocessing.Pool()
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
            window=window,
        ),
        paramlist,
    )
    # Concat all dataframes into a single dataframe
    wpt_df = pd.concat(results, ignore_index=True)

    return wpt_df


def get_data(
    R_init,
    frequency,
    pac,
    sampling_rate,
    n_cycles=50,
    r_measure=R_MEASURE,
    return_time=False,
    **kwargs,
):
    """
    Read stored from bubble dynamics model or generate and store it if it doesn't exist.

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

    Keyword arguments
    -----------------
    See BD class for more options.

    r_measure: float
        the distance from the bubble that the radiated pressure is measured at

    """
    parameters = _get_params_from_kwargs(R_init, frequency, pac, n_cycles, **kwargs)

    # Get name of file containing data
    identifier = str()

    for i, j in zip(default_parameters, parameters):
        if i in ["time"]:
            continue  # Leave out some parameters from the name of the file
        identifier += f"{i}{sci_form(j)}"

    identifier += ".pkl"

    file_path = os.path.join(FOLDER_PATH, identifier)

    generate_new_data = False

    # Check if file exists
    if os.path.isfile(file_path):
        # Load data from file
        with open(file_path, "rb") as file:
            file_data = pickle.load(file)
            stored_cycles = float(file_data["n_cycles"])
            data = file_data["data"]

        if n_cycles <= stored_cycles:
            time = _get_time_array(n_cycles, frequency)
            data = data[: len(time)]  # trim down existing data
        else:
            generate_new_data = True

    else:
        generate_new_data = True

    if generate_new_data is True:
        # Generate data
        data = _generate_data_for_storage(r_measure, *parameters)
        file_data = {
            "data": data,
            "n_cycles": float(n_cycles),
            "sampling rate": SAMPLING_RATE,
        }
        # Save the data for future use
        if data is None:  # Failed data generation
            return None
        with open(file_path, "wb") as file:
            pickle.dump(file_data, file)

    # Downsample data
    if return_time is True:
        time = _get_time_array(n_cycles, frequency)
    else:
        time = None

    return resample(data, sampling_rate, time)


def resample(data, sampling_rate, time=None):
    # Calculate sampling rate ratio and downsample the data
    new_length = len(data) * sampling_rate / SAMPLING_RATE
    return sc_resample(data, int(new_length), time)


def _generate_data_for_storage(r_measure, *parameters):
    """
    Generate data using bubble dynamics at a sampling rate of 1e9 Hz
    and resample to a given sampling rate. There are several default
    parameters which cannot be changed to generate data for storage.
    """
    i = 0

    while True:
        # Generate sound wave
        data = generate_data(r_measure, *parameters)

        # Check if the dynamics model failed
        if not np.isnan(data).any():
            break

        # Repeat and give up after a number of attempts.
        i += 1
        if i > 50:
            R_init, _, frequency, pac, *_ = parameters
            print(
                f"Could not generate dynamics data with R_init: {R_init}, frequency:"
                f" {frequency}, pac: {pac}"
            )
            return None

    if np.isnan(np.array(data)).any():
        raise ValueError("nan found in data")

    return data


def generate_data(r_measure, *params):
    """
    Generate data from bubble dynamics.
    See BD class for *params
    """
    # Generate sound wave
    BD_obj = BD(*params)
    BD_obj.solver()

    # Calculate pressure radiated (proportional to measured voltage)
    BD_obj.calculate_p_radiated(r_measure)
    data = BD_obj.p_radiated

    return data


def _get_params_from_kwargs(R_init, frequency, pac, n_cycles, **kwargs):
    "Convert input into a list of parameters for the BD object"
    kwargs.setdefault("R_init", R_init)
    kwargs.setdefault("frequency", frequency)
    kwargs.setdefault("pac", pac)

    default_parameters["time"] = _get_time_array(n_cycles, frequency)
    default_parameters["pulse_width"] = (
        n_cycles / frequency
    )  # pulse lasts for the same time as the array

    parameters = [kwargs.get(k, v) for k, v in default_parameters.items()]

    return parameters


def _get_time_array(n_cycles, frequency):
    tmax = n_cycles / frequency
    time = np.arange(0, tmax, TSTEP)
    return time


# ------------ Generate Training Data from Single Sound Wave ------------


def generate_training_data(
    data,
    levels=None,
    wavelet="haar",
    mode="symmetric",
    *,
    min_entropy=False,
    min_level=0,
    include_tr_obj=False,
    freq_threshold=1,
    keep_ratio="full",
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
    include_tr_obj: bool
        whether to return the transform object used for the transform containing
        info about the nodes.
    keep_ratio: str, int, float
        the section of data to keep when caluclating statistical metrics on nodes.
        see Node class for more details.

    Returns
    -------
    wpt_df: pd.DataFrame
        a dataframe containing training data where each row represents a node with
        several features

    """

    # DWPT
    tr_obj = wpt.dwpt(
        data,
        levels,
        wavelet,
        mode,
        min_entropy=min_entropy,
        min_level=min_level,
        freq_threshold=freq_threshold,
    )

    tr_obj.keep_ratio = keep_ratio

    wpt_df = tr_obj.info(levels, order="natural", leaf_nodes=True)

    # Return the transform object if specified
    if include_tr_obj is True:
        return wpt_df, tr_obj
    else:
        return wpt_df


def add_noise(data, signal2noise, display_std_noise=False):
    """
    Add noise to a signal with a given signal to noise ratio.
    Returns the noisy data.
    """
    data_to_noise_power_ratio = 10 ** (signal2noise / 20)
    power_noise = np.var(data) / data_to_noise_power_ratio
    std_noise = np.sqrt(power_noise)
    noise = np.random.normal(0, std_noise, len(data))
    data_noisy = data.copy() + noise

    if display_std_noise is True:
        print("std_noise:", std_noise)
    return data_noisy
