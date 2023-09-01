import numpy as np
import scipy.stats as sc


def crest_factor(array):
    rms = np.sqrt(1 / len(array)) * sum(array**2)
    return 10 * np.log10(max(array**2) / rms)


def energy(array):
    return np.var(array)


def kurtosis(array):
    return sc.kurtosis(array, fisher=False)


def skewness(array):
    return sc.skew(array)


def entropy(array):
    "Calculate the wavelet energy entropy within a node of coefficients"
    if not isinstance(array, np.ndarray):
        raise ValueError("input array must be np.ndarray")
    energy = array**2
    total_energy = sum(energy)
    norm_energy = energy / total_energy
    entropy = -sum(0 if i == 0 else i * np.log(i) for i in norm_energy)
    return entropy
