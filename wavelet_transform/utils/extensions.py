import numpy as np


def sci_form(number, dp=3):
    """
    Returns a string of the scientific form of a number
    """
    return f"{number:.{dp}e}"


def eng_form_plot(number, power, dp=3):
    """
    Returns converts the number to scientific form A*10*(power)
    and returns A.
    """
    assert isinstance(power, int)
    sci_form_number = sci_form(number, dp)
    n, p = sci_form_number.split("e")
    n, p = float(n), float(p)

    inc = power - p
    n /= 10 ** inc
    return f"{n:.{dp}f}"

def auto_eng_form(number, dp=3):
    """
    Returns a string of the engineering form of a number
    """
    if number == 0:
        return f"{0:.{dp}f}e0"

    sci_form_number = sci_form(number, dp)
    n, p = sci_form_number.split("e")
    n, p = float(n), float(p)

    eng_p = int(p) - (int(p) % 3)
    inc = eng_p - p
    n /= 10 ** inc
    return f"{n:.{dp}f}e{eng_p}"


def fft(array, sampling_rate=None, fundamental_freq=None):
    """
    Calculate the fft of the array. If sampling rate is passed, the function will
    return a matching array of frequency. If the fundamental frequency is also passed,
    the frequency array will be normalised.
    """
    n = len(array)

    # Calculate FFT with numpy, take the first half of the FFT of the signal
    #  i.e., positive freqs, and normalise by setting norm=forward
    fft = np.fft.fft(array, norm="forward")[0 : int(n / 2)]

    if sampling_rate is not None:
        freq = np.fft.fftfreq(n, d=1 / sampling_rate)[
            0 : int(n / 2)
        ]
        if fundamental_freq is not None:
            freq /= fundamental_freq
        return fft, freq

    return fft


def psd(array, sampling_rate=None, fundamental_freq=None):
    """
    Calculate the power spectral density of a signal. If sampling rate is passed, the
    function will return a matching array of frequency. If the fundamental frequency is
    also passed, the frequency array will be normalised.

    Parameters
    ----------
    array: array_like
        an array containing the data to be transformed
    sampling_rate: float or None
        the sampling rate of the signal. If None, no frequency array will be returned.
    fundamental_freq: float or None
        the fundamental frequency of the signal. If None, no normalisation will be performed.
    """
    if isinstance(fundamental_freq, (float, int)) and sampling_rate is None:
        raise ValueError(
            "fundamental_freq cannot be passed without sampling_rate"
        )

    n = len(array)

    # Calculate FFT with numpy, take the first half of the FFT of the signal,
    # i.e., positive freqs, and normalise by setting norm=forward
    fft = np.fft.fft(array, norm=None)[0 : int(n / 2)]
    psd = np.abs(fft) ** 2 / n
    psd[1:-1] *= 2  # account for the fact that we are only taking the positive frequencies

    if sampling_rate is not None:
        freq = np.fft.fftfreq(n, d=1 / sampling_rate)[
            0 : int(n / 2)
        ]
        if isinstance(fundamental_freq, (float, int)):
            freq /= fundamental_freq
        elif fundamental_freq is not None:
            raise ValueError(
                "fundamental_freq must be a float or int if passed"
            )
        return psd, freq

    return psd


def to_dB(array):
    """
    converting amplitude to dB

    Parameters
    ----------
        array (numpy): absolute of a signal, e.g. np.abs(fft_output).

    Returns
    -------
        list: amplitude in dB
    """
    return [10 * np.log10(i) if i > 0 else 0 for i in array]


def trim_data(data, t, t_lims, return_time=False):
    start, end = find_time_indices(t, *t_lims)
    data = data[start:end]

    if return_time is True:
        t = t[start:end]
        return data, t

    return data

def find_time_indices(t, delay, end):
    delay_index = find_delay_index(t, delay)
    end_index = find_end_index(t, end)
    return delay_index, end_index


def find_delay_index(t, delay):
    "finds the index of the start of the delayed signal"
    for delay_index in range(len(t)):
        if t[delay_index] > delay:
            return delay_index
    raise ValueError("index not found")


def find_end_index(t, end):
    "finds the index of the end of the cut off signal"
    end_index = len(t) - 1
    for _ in range(len(t)):
        if t[end_index] < end:
            return end_index
        end_index -= 1
    raise ValueError("index not found")
