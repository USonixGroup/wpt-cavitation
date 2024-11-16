from wavelet_transform.wpt.node_base import NodeBase

from wavelet_transform.utils.extensions import sci_form
from wavelet_transform.utils.statistical_features import (
    energy,
    log_energy,
    avg_energy,
    log_avg_energy,
    kurtosis,
    skewness,
    abs_skewness,
    crest_factor,
    entropy,
    spectral_entropy,
)


class Node(NodeBase):
    "Extension on NodeBase which calculates statistical metrics based on the node data."

    def __init__(self, parent, a_d, transform_object):
        self.transform = transform_object
        super().__init__(parent, a_d, transform_object)

    # ------ Trim length by defined length -----
    @property
    def data_keep(self):
        """
        The data that is retained for calculating statistical metrics.
        This is either based on the
        """
        keep_ratio = self.transform.keep_ratio

        if keep_ratio in (None, "full"):
            return self.data
        elif isinstance(keep_ratio, str):
            lim = self.get_trim_len(self.get_filter_len(), self.level, keep_ratio)
        else:  # Keep_ratio is float or int
            leave_ratio = 1 - keep_ratio
            lim = int(leave_ratio * self.data_len / 2)

        if lim == 0:
            return self.data
        data = self.data[lim:-lim]
        return data

    def get_filter_len(self):
        return len(self.transform.filter_bank[0])

    def get_trim_len(self, filter_len, level, mode="valid"):
        """
        Get length to trim off both ends of a downsammpled full convolution to convert it to a
        downsampled valid or same convolution.
        """
        # Calculate addition length to trim due to level
        scale_factor = 2 * (1 - 2**-level)

        # Calculate trim length based on mode
        if mode == "valid":
            trim_val = filter_len - 1
        elif mode == "same":
            trim_val = (filter_len - 1) // 2
        else:
            raise ValueError("""mode must be in ["valid", "same"]""")
        return int(trim_val * scale_factor / 2)  # Downsampled data is trimmed less

    @property
    def data_len(self):
        if not hasattr(self, "_data_len"):
            self._data_len = len(self.data)
        return self._data_len

    # ----- Node info -----
    @property
    def node_number(self):
        "Converts path into node number then stores it for subsequent calls"
        if not hasattr(self, "_node_number"):
            path = self.path
            node_no = 0
            bit = 1
            for i in reversed(path):
                if i == "d":
                    node_no += bit
                bit *= 2
            self._node_number = node_no
        return self._node_number

    @property
    def energy(self):
        if not hasattr(self, "_energy"):
            self._energy = energy(self.data_keep)
        return self._energy

    @property
    def log_energy(self):
        if not hasattr(self, "_log_energy"):
            self._log_energy = log_energy(self.data_keep)
        return self._log_energy

    @property
    def average_energy(self):
        if not hasattr(self, "_average_energy"):
            self._average_energy = avg_energy(self.data_keep)
        return self._average_energy

    @property
    def log_average_energy(self):
        if not hasattr(self, "_log_average_energy"):
            self._log_average_energy = log_avg_energy(self.data_keep)
        return self._log_average_energy

    @property
    def kurtosis(self):
        if not hasattr(self, "_kurtosis"):
            self._kurtosis = kurtosis(self.data_keep)
        return self._kurtosis

    @property
    def crest_factor(self):
        if not hasattr(self, "_crest_factor"):
            self._crest_factor = crest_factor(self.data_keep)
        return self._crest_factor

    @property
    def skewness(self):
        if not hasattr(self, "_skewness"):
            self._skewness = skewness(self.data_keep)
        return self._skewness

    @property
    def abs_skewness(self):
        if not hasattr(self, "_abs_skewness"):
            self._abs_skewness = abs_skewness(self.data_keep)
        return self._abs_skewness

    @property
    def entropy(self):
        "Calculate the entropy of the node (packet) normalised with its (packet) energy."
        if not hasattr(self, "_entropy"):
            self._entropy = entropy(self.data_keep)
        return self._entropy

    @property
    def spectral_entropy(self):
        if not hasattr(self, "_spectral_entropy"):
            self._spectral_entropy = spectral_entropy(self.data_keep)
        return self._spectral_entropy

    @property
    def level_entropy(self):
        """
        Calculate the entropy of the node where each energy component is normalised with the
        total energy of the transform
        """
        total_energy = self.transform.total_energy(0)
        if not hasattr(self, "_total_entropy"):
            self._level_entropy = entropy(self.data_keep, total_energy)
        return self._level_entropy

    @property
    def freq_range(self):
        """
        Returns the frequency range if possible, otherwise returns the
        normalised frequency range
        """
        range_to_use = self.frequency_range or self.norm_frequency_range
        return tuple(sci_form(bound) for bound in range_to_use)

    @property
    def numeric_freq_range(self):
        """
        Returns the frequency range or normalised frequency range as floats.
        """
        range_to_use = self.frequency_range or self.norm_frequency_range
        return tuple(i for i in range_to_use)

    def is_freq_calculated(self):
        return self.frequency_range is not None


class BaseNode(Node):
    """
    The base node (0, 0) for the transform. Contains data from the original sound wave

    Parameters
    ----------
    data: array
        An array containing the data from the original sound wave.
    """

    def __init__(self, data, transform_object):
        self.parent = None
        self.data = data
        self.path = ""
        self.level = 0
        self.norm_frequency_range = (0, 1)
        self.frequency_range = None
        self.transform = transform_object
