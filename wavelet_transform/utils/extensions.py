import numpy as np

from wavelet_transform.test.synthetic_data import find_time_indices


def sci_form(number, dp=3):
    "returns a string of the scientific form of a number"
    e_power = 0
    if number == 0:
        return f"{number}"
    while number >= 10:
        number /= 10
        e_power += 1
    while number < 1:
        number *= 10
        e_power -= 1
    return f"{round(number,dp)}e{e_power}"


def to_list(x, *, convert_to_float=None):
    "handling keyword arguments"
    if x is None:
        return None
    x = [x] if not isinstance(x, list) else x
    if convert_to_float is True:
        x = [float(i) if isinstance(i, int) else i for i in x]
    return x


def fft(array, sampling_rate=None, fundamental_freq=None):
    """
    Calculate the fft of the array. If sampling rate is passed, the function will
    return a matching array of frequency. If the fundamental frequency is also passed,
    the frequency array will be normalised.
    """
    signal_length = len(array)

    # Calculate FFT with numpy, take the first half of the signal and normalise
    fft = np.fft.fft(array)[0 : int(signal_length / 2)] / (signal_length / 2)

    if sampling_rate is not None:
        freq = np.fft.fftfreq(signal_length, d=1 / sampling_rate)[
            0 : int(signal_length / 2)
        ]
        if fundamental_freq is not None:
            freq /= fundamental_freq
        return fft, freq

    return fft


def plot_log10(array):
    return [10 * np.log10(i) if i > 0 else 0 for i in array]


def trim_data(data, t, t_lims, return_time=False):
    start, end = find_time_indices(t, *t_lims)
    data = data[start:end]

    if return_time is True:
        t = t[start:end]
        return data, t

    return data
