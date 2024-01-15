import numpy as np
import scipy.signal as sig


def preprocess(data, window=False):
    if window is True:
        win_func = sig.windows.blackmanharris(len(data), sym=False)
        data_windowed = data * win_func  # / np.std(win_func)
        data_windowed = (data_windowed - np.mean(data_windowed)) / np.std(data_windowed)
        return data_windowed
    else:
        data = (data - np.mean(data)) / np.std(data)
        return data
