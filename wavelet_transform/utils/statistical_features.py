import numpy as np
import scipy.stats as sc


def calc_crest_factor(array):
    rms = np.sqrt(1 / len(array)) * sum(array**2)
    return 10 * np.log10(max(array**2) / rms)


def calc_energy(array):
    return np.var(array)


def calc_kurtosis(array):
    return sc.kurtosis(array, fisher=False)


def calc_skewness(array):
    return sc.skew(array)


def calc_entropy(energy, total_energy):
    energy_relative = energy / total_energy
    entropy_node = -energy_relative * np.log2(energy_relative)
    return entropy_node
