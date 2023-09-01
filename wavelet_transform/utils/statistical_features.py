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
    "Calculate the wavelet energy entropy of an np.array of coefficients."
    if not isinstance(array, np.ndarray):
        raise ValueError("array must be an ndarray")
    energy_array = array**2
    total = sum(energy_array)  # total energy
    norm_array = energy_array / total
    entropy = -sum(
        0 if i == 0 else i * np.log(i) for i in norm_array
    )  # 0log0 is 0 by convention
    return entropy
