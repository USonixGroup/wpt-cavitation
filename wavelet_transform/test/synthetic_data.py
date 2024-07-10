import numpy as np


def sin(amplitude, frequency, phi, t):
    """
    returns a sine wave with specified parameters

    Parameters
    ----------
    amplitude: float
        the amplitude of the sine wave
    frequency: float
        the frequency of the sine wave
    phi: float
        the phase angle of the sine wave
    t: arraylike

    Returns
    -------
    sin_wave: array
        a sine wave represented as an array of amplitudes
    """
    omega = 2 * np.pi * frequency
    sin_wave = amplitude * np.sin(omega * t - phi)
    return sin_wave


def sound_wave(
    amplitudes, frequencies, phases, duration, sampling_rate, delay=None, end=None
):
    """
    Returns a sound wave with made up of sine waves with frequencies given as a
    list. delay and end can be specified to start the wave early and cut it of
    before the specified duration.

    Parameters
    ----------
    amplitudes: list or float
        a list of amplitudes of sin waves to produce the signal.
    frequencies: list or float
        a list of frequencies of sine waves to produce the signal
    phases: list or float
        a list of phases of sine waves to produce the signal
    duration: float
        the time duration of the signal.
    sampling_rate: float
        sampling_rate of the sensor in Hz
    delay: float
        a time delay for the signal.
    end: float
        time that the signal ends at.

    Returns
    -------
    wave: array
        tuple containing an array of the sound wave
    t: array
        the array of time to plot the wave against
    """

    t = np.arange(0, duration, 1 / sampling_rate)
    wave = np.zeros(int(len(t)))

    # initialise parameters

    if isinstance(frequencies, (int, float)):
        frequencies = [frequencies]  # compatibility with a single frequency
    if isinstance(phases, (int, float)):
        phases = [phases for _ in range(len(frequencies))]
    if isinstance(amplitudes, (int, float)):
        amplitudes = [amplitudes for _ in range(len(frequencies))]

    for amplitude, frequency, phi in zip(amplitudes, frequencies, phases):
        # generate sine wave
        sin_wave = sin(amplitude, frequency, phi, t)
        # add sine wave to total wave
        wave = wave + sin_wave

    # add delay and end
    if delay is None:
        delay = 0
    if end is None:
        end = duration
    delay_index, end_index = find_time_indices(t, delay, end)

    wave[:delay_index] = 0
    wave[end_index:] = 0

    return wave, t


def time_array(duration=None, sampling_rate=None, time=None):
    "creates an array with the same length as the synthetic data"
    if duration is not None and sampling_rate is not None:
        t = np.arange(0, duration, 1 / sampling_rate)
    elif time:
        t = time
    else:
        raise ValueError(
            """
            provide the sampling rate and duration of the signal or an array\
            of time
            """
        )
    return t


def noise(standard_deviation, duration=None, sampling_rate=None, time=None):
    """
    Creates and array of noise with the same dimensions as the synthetic data.
    Can be specified using either an array of time to fit noise into or can
    accept the sampling rate and duration used to generate the signal.
    """
    t = time_array(duration, sampling_rate, time)
    noise = np.random.normal(0, standard_deviation, len(t))
    return noise


def transients(
    amplitudes, times, durations, duration=None, sampling_rate=None, time=None
):
    """
    create synthetic transients using narrow window functions. compiles them
    into an array with the same length as the synthetic data

    Parameters
    ----------
    amplitudes: list or float
        a list of amplitudes for the transients
    times: list or float
        a list of times that the transients start at
    durations:
        a list of the durations of the transients
    duration: float
        duration of the transient, in seconds
    sampling_rate: float
        sampling rate of the signal. used alongside duration to defined the
        length of the returned array
    time: array
        array of time to fit the transients to. can be called instead of
        duration and sampling_rate

    Returns
    -------
    transient_array: array
        array containing synthetic transients
    """

    t = time_array(duration, sampling_rate, time)

    if isinstance(amplitudes, (float, int)):
        amplitudes = list(amplitudes)
        times = list(times)
        durations = list(durations)

    transient_array = np.zeros(len(t))
    warned = False
    for amp, time, dur in zip(amplitudes, times, durations):
        i = find_delay_index(t, time)
        j = find_delay_index(t, dur)
        if j < 10 and warned is True:
            print(
                f"""
                  Userwarning: duration of transient is {j} coefficients long
                  """
            )
            warned = True
        transient_array[i : i + j] += amp * np.hanning(j)

    return transient_array


def find_time_indices(t, delay, end):
    delay_index = find_delay_index(t, delay)
    end_index = find_end_index(t, end)
    return delay_index, end_index


def find_delay_index(t, delay):
    "finds the index of the start of the delayed signal"
    for delay_index in range(len(t)):
        if t[delay_index] > delay:
            return delay_index


def find_end_index(t, end):
    "finds the index of the end of the cut off signal"
    end_index = len(t) - 1
    for _ in range(len(t)):
        if t[end_index] < end:
            return end_index
        end_index -= 1
