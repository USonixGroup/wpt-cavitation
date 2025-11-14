import numpy as np
import scipy.stats as sc

from wavelet_transform.utils.extensions import psd

# When adding metrics, add them to METRIC_REGISTRY in metrics.py
# This will ensure that nodes can access them via the metric("metric_name") function

def crest_factor(array):
    rms = np.sqrt(sum(array ** 2) / len(array))
    return max(abs(array)) / rms


def energy(array):
    return np.sum(array ** 2)


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
    if len(array) == 0:
        raise ValueError("empty array")
    energy = array ** 2
    if total_energy is None:
        total_energy = np.sum(energy)
        if total_energy == 0:
            return 0
    norm_energy = energy / total_energy
    entropy = -sum(0 if i == 0 else i * np.log(i) for i in norm_energy)
    return entropy

# import plotly.graph_objects as go
def spectral_entropy(array, ref_psd=None):
    """
    Calculate the spectral entropy of the input array.
    If ref_psd is provided, the power spectrum is limited to the minimum
    of the power spectra of the two arrays.
    """
    # Calculate the power spectrum
    power_spectrum = psd(array)  # np.abs(fft(array)[<half_length_array>]) ** 2 / (len(array) / 2)
    if ref_psd is not None:
        try:
            power_spectrum = np.minimum(power_spectrum, ref_psd)
        except ValueError:
            raise ValueError("shape mismatch between power_spectrum and ref_psd. corrected_spectral_entropy does not work for wavelet coefficients")
    norm_spectrum = power_spectrum / np.sum(power_spectrum)  # type: ignore

    spectral_entropy = -sum(0 if i == 0 else i * np.log2(i) for i in norm_spectrum)
    spectral_entropy /= np.log2(len(norm_spectrum))
    return spectral_entropy

def spectral_kurtosis(array, ref_psd=None):
    # Calculate the power spectrum
    power_spectrum = psd(array)  # np.abs(fft(array)[<half_length_array>]) ** 2 / (len(array) / 2)
    if ref_psd is not None:
        try:
            power_spectrum = np.minimum(power_spectrum, ref_psd)
        except ValueError:
            raise ValueError("shape mismatch between power_spectrum and ref_psd. corrected_spectral_kurtosis does not work for wavelet coefficients")
    norm_spectrum = power_spectrum / np.sum(power_spectrum)  # type: ignore

    spectral_kurtosis = kurtosis(norm_spectrum)
    return spectral_kurtosis