import numpy as np
import scipy.stats as sc


def crest_factor(array):
    rms = np.sqrt(1 / len(array) * sum(array**2))
    return max(array) / rms


def energy(array):
    return sum(array**2)


def log_energy(array):
    return np.log10(energy(array))


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
        total_energy = sum(energy)
        if total_energy == 0:
            return 0
    norm_energy = energy / total_energy
    entropy = -sum(0 if i == 0 else i * np.log(i) for i in norm_energy)
    return entropy


def spectral_entropy(array):
    # Calculate the power spectrum
    power_spectrum = np.abs(np.fft.fft(array)) ** 2
    normalized_spectrum = power_spectrum / np.sum(power_spectrum)

    spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum))
    return spectral_entropy
