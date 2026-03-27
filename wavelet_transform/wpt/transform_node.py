from wavelet_transform.wpt.node_base import NodeBase

from wavelet_transform.utils.extensions import sci_form
from wavelet_transform.wpt.metrics import METRIC_REGISTRY, compute_metric_for_data, get_data, get_ref

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
        scale_factor = 2 * (1 - 2 ** -level)

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
    
    # ----- Statistical metrics -----
    @property
    def metrics(self):
        """
        Compute a dictionary of all registered metrics for this node.

        The registry (`METRIC_REGISTRY`) maps a metric key (node attribute name)
        to a function name in `wavelet_transform.utils.statistical_features`.
        This property computes each metric by calling the corresponding function
        with `self.data_keep`. The special metric `level_entropy` requires the
        total energy of the transform and is handled specially.
        """
        data = self.data_keep
        metrics = {}
        total_energy = None
        # compute once for efficiency when level_entropy is requested
        if "level_entropy" in METRIC_REGISTRY:
            total_energy = self.transform.total_energy(0)
        for metric_name in METRIC_REGISTRY.keys():
            metrics[metric_name] = compute_metric_for_data(data, metric_name, total_energy)

        return metrics
    

    def metric(self, metric_name):
        """
        Get a single metric by name.

        Parameters
        ----------
        metric_name: str
            The name of the metric to retrieve.
        """
        metric_name = metric_name.replace(" ", "_")
        reference = get_ref(metric_name, self.transform)
        metric = compute_metric_for_data(self.data_keep, metric_name, reference)
        if not isinstance(metric, (int,float)):
            raise ValueError(f"metric '{metric_name}' did not return a numeric value")
        return metric
    
    # ----- Frequency range properties -----

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
    
    def is_in_threshold(self, freq_threshold):
        """
        Check if the node is within the frequency threshold.

        Parameters
        ----------
        freq_threshold: float
            The frequency threshold to check against.
        """
        return self.norm_frequency_range[1] <= freq_threshold
    
    def can_be_in_threshold(self, freq_threshold):
        """
        Check if the node can have children within the frequency threshold.

        Parameters
        ----------
        freq_threshold: float
            The frequency threshold to check against.
        """
        return self.norm_frequency_range[0] < freq_threshold


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
        self.norm_frequency_range = (0., 1.)
        self.frequency_range = None
        self.transform = transform_object
