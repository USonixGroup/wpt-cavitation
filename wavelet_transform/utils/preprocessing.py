import numpy as np
import scipy.signal as sig


def preprocess(data, window=False):
    """
    Normalise the data by subtracting its mean and dividing through by its std.
    Use blackman harris window function if window=True. Returns processed data.

    Parameters
    ----------
    data: np.ndarray
        array of data to preprocess
    window: bool
        whether to apply a blackmanharris window to the data

    Returns
    -------
    data: np.ndarray
        preprocessed data
    """
    d = data.copy()
    if window is True:
        win_func = sig.windows.blackmanharris(len(d), sym=False)  # type: ignore
        d = (d - np.mean(d)) / np.std(d)
        data_windowed = d * win_func
        data_windowed_centred = (data_windowed - np.mean(data_windowed))
        return data_windowed_centred
    else:
        data = (data - np.mean(data)) / np.std(data)
        return data

def add_noise(data, signal2noise):
    """
    Add noise to a signal with a given signal to noise ratio.
    Returns the noisy data.
    """
    data_to_noise_power_ratio = 10 ** (signal2noise / 10)
    power_noise = np.var(data) / data_to_noise_power_ratio
    std_noise = np.sqrt(power_noise)
    
    noise = np.random.normal(0, std_noise, len(data))
    data_noisy = data.copy() + noise

    return data_noisy