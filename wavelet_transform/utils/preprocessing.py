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
        win_func = sig.windows.blackmanharris(len(d), sym=False)
        data_windowed = d * win_func  # / np.std(win_func)
        data_windowed = (data_windowed - np.mean(data_windowed)) / np.std(data_windowed)
        return data_windowed
    else:
        data = (data - np.mean(data)) / np.std(data)
        return data


def unwindow(data):
    win_func = sig.windows.blackmanharris(len(data), sym=False)
    return data.copy() / win_func
