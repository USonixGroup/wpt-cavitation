from wavelet_transform.wpt.node_base import NodeBase

from wavelet_transform.utils.extensions import sci_form
from wavelet_transform.utils.statistical_features import (
    energy,
    log_energy,
    kurtosis,
    skewness,
    abs_skewness,
    crest_factor,
    entropy,
    spectral_entropy,
)


class Node(NodeBase):
    "Extension on NodeBase which"
    # --------------- Node Info ----------------

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
            self._energy = energy(self.data)
        return self._energy

    @property
    def log_energy(self):
        if not hasattr(self, "_log_energy"):
            self._log_energy = log_energy(self.data)
        return self._log_energy

    @property
    def kurtosis(self):
        if not hasattr(self, "_kurtosis"):
            self._kurtosis = kurtosis(self.data)
        return self._kurtosis

    @property
    def crest_factor(self):
        if not hasattr(self, "_crest_factor"):
            self._crest_factor = crest_factor(self.data)
        return self._crest_factor

    @property
    def skewness(self):
        if not hasattr(self, "_skewness"):
            self._skewness = skewness(self.data)
        return self._skewness

    @property
    def abs_skewness(self):
        if not hasattr(self, "_abs_skewness"):
            self._abs_skewness = abs_skewness(self.data)
        return self._abs_skewness

    @property
    def entropy(self):
        if not hasattr(self, "_entropy"):
            self._entropy = entropy(self.data)
        return self._entropy

    @property
    def spectral_entropy(self):
        if not hasattr(self, "_spectral_entropy"):
            self._spectral_entropy = spectral_entropy(self.data)
        return self._spectral_entropy

    @property
    def level_entropy(self):
        """
        Calculate the entropy of the node where each energy component is normalised with the
        total energy of the level.
        """
        if not hasattr(self, "_total_entropy"):
            level = len(self.path)
            total_energy = self.transform_object.total_energy(level)
            self._total_entropy = entropy(self.data, total_energy)
        return self._total_entropy

    @property
    def freq_range(self):
        """
        returns the frequency range if possible, otherwise returns the
        normalised frequency range
        """
        range_to_use = self.frequency_range or self.norm_frequency_range
        return tuple(sci_form(bound) for bound in range_to_use)


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
        self.transform_object = transform_object
