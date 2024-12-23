import numpy as np
import scipy.stats as sc

from wavelet_transform.utils.extensions import psd


def crest_factor(array):
    rms = np.sqrt(sum(array**2) / len(array))
    return max(abs(array)) / rms


def energy(array):
    return np.sum(array**2)


def log_energy(array):
    return 10 * np.log10(energy(array))


def avg_energy(array):
    return energy(array) / len(array)


def log_avg_energy(array):
    return 10 * np.log10(avg_energy(array))


def kurtosis(array):
    return sc.kurtosis(array, fisher=False)


def skewness(array):
    return sc.skew(array)


def abs_skewness(array):
    return np.abs(skewness(array))


def entropy(array, total_energy=None):
    "Calculate the wavelet energy entropy within a node of coefficients"
    if not isinstance(array, np.ndarray):
        raise ValueError("input array must be np.ndarray")
    energy = array**2
    if total_energy is None:
        total_energy = np.sum(energy)
        if total_energy == 0:
            return 0
    norm_energy = energy / total_energy
    entropy = -sum(0 if i == 0 else i * np.log(i) for i in norm_energy)
    return entropy


def spectral_entropy(array):
    # Calculate the power spectrum
    half_length_array = int(len(array) / 2)
    power_spectrum = psd(array)  # np.abs(fft(array)) ** 2
    norm_spectrum = power_spectrum / np.sum(power_spectrum)

    spectral_entropy = -sum(0 if i == 0 else i * np.log2(i) for i in norm_spectrum)
    spectral_entropy /= np.log2(half_length_array)
    return spectral_entropy
