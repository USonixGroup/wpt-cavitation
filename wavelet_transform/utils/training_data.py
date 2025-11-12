import numpy as np
import pandas as pd
import itertools
import multiprocessing
from functools import partial

from wavelet_transform.utils.bubble_data_manager import get_data
from wavelet_transform.utils.wavelet_features import extract_wavelet_features
from wavelet_transform.utils.preprocessing import preprocess

def get_training_data(
    wavelet,
    sampling_rate,
    levels,
    mode,
    std_noise,
    best_basis=None,
    min_level=0,
    freq_threshold=None,
    window=False,
    parameter_ranges=None,
    *,
    include_node_info=True,
    metrics=None
):
    """
    Generate training data from bubble dynamics signals with wavelet features.

    Parameters
    ----------
    wavelet: str
        Wavelet type to use for feature extraction
    sampling_rate: float
        Sampling rate for the bubble dynamics data
    levels: int
        Number of wavelet decomposition levels
    mode: str
        Padding mode for wavelet transform
    std_noise: float
        Standard deviation of noise to add to signals
    best_basis: str or None, optional
        Whether to use best basis decomposition
    min_level: int, optional
        Minimum level for best basis decomposition
    freq_threshold: float, optional
        Frequency threshold for node filtering
    window: bool, optional
        Whether to apply windowing to the signals
    parameter_ranges: dict, optional
        Custom ranges for bubble parameters. If None, uses default ranges.
        Expected format: {
            'R_init': list of initial radii in meters,
            'frequency': list of frequencies in Hz,
            'pac': list of acoustic pressures in Pa
        }

    Returns
    -------
    pd.DataFrame
        DataFrame containing wavelet features for all parameter combinations
    """
    if parameter_ranges is None:
        parameter_ranges = {
            'R_init': [i * 1e-6 for i in range(1, 9, 2)],  # 1-9 microns
            'frequency': [i * 1e5 for i in range(5, 20, 2)],  # 0.5-2 MHz
            'pac': [-i * 1e3 for i in range(100, 3000, 500)]  # -100 to -3000 kPa
        }

    paramlist = list(itertools.product(
        parameter_ranges['R_init'],
        parameter_ranges['frequency'],
        parameter_ranges['pac']
    ))

    # Process parameter combinations in parallel
    with multiprocessing.Pool() as pool:
        results = pool.map(
            partial(
                _process_single_parameter_set,
                sampling_rate=sampling_rate,
                std_noise=std_noise,
                levels=levels,
                wavelet=wavelet,
                mode=mode,
                best_basis=best_basis,
                min_level=min_level,
                freq_threshold=freq_threshold,
                window=window,
                include_node_info=include_node_info,
                metrics=metrics,
            ),
            paramlist,
        )

    # Extract only DataFrames if any tuples are present
    dfs = [r[0] if isinstance(r, tuple) else r for r in results]
    return pd.concat(dfs, ignore_index=True)

def _process_single_parameter_set(
    input_params,
    sampling_rate,
    std_noise,
    levels,
    wavelet,
    mode,
    best_basis,
    min_level,
    freq_threshold,
    window=False,
    *,
    include_node_info=True,
    metrics=None,
):
    """
    Process a single set of bubble parameters to generate training data.

    Parameters
    ----------
    input_params: tuple
        (R_init, frequency, pac) values
    Other parameters are the same as in generate_bubble_training_data
    """
    R_init, frequency, pac = input_params

    # Get bubble dynamics data
    data = get_data(R_init, frequency, pac, sampling_rate)
    if isinstance(data, tuple):
        data = data[0]  # Extract data if time was also returned
    if data is None:
        raise ValueError("get_data returned None")

    # Preprocess the data
    data = np.asarray(data)
    data += np.random.normal(0, std_noise, len(data))
    data = preprocess(data, window=window)

    # Extract wavelet features
    features, tr = extract_wavelet_features(
        data,
        levels,
        wavelet,
        mode,
        best_basis=best_basis,
        min_level=min_level,
        freq_threshold=freq_threshold,
        include_node_info=include_node_info,
        metrics=metrics
    )

    if isinstance(features, tuple):
        raise ValueError("features should be a dataframe")

    # Add parameter information to features
    features['R_init'] = R_init
    features['frequency'] = frequency
    features['pac'] = pac

    return features