import numpy as np
import scipy.signal as sig


def preprocess(data, window=False):
    data = (data - np.average(data)) / np.std(data)
    if window is True:
        win_func = sig.windows.blackmanharris(len(data), sym=False)
        data_windowed = data * win_func  # / np.std(win_func)
        # data_windowed = (data_windowed - np.average(data_windowed)) / np.std(data_windowed)
        return data_windowed
    return data
