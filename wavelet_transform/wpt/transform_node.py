from pywt import pad as pywtpad
import scipy.signal as sp

from wavelet_transform.utils.extensions import sci_form
from wavelet_transform.utils.statistical_features import (
    calc_energy,
    calc_kurtosis,
    calc_skewness,
    calc_crest_factor,
)


class Node:
    def __init__(self, parent, a_d, transform_object):
        """
        Stores the information for each set of coefficients in the transform
        Each node has an assigned parent and path e.g. node 'aaa' is the third
        level approximation coefficients with parent node 'aa'.

        Parameters
        ----------
        parent: object
            the the node's parent node i.e. the node that was filtered to
            obtain the data in this one.
        a_d: string
            whether a high pass or low pass filter was used to obtain the
            data for this node. 'a' for low pass, 'd' for high pass
        transform_object: object
            an object 'Transform()' representing the transform the node is
            part of
        """
        # set parent, path, and transform object
        self.transform_object = transform_object
        self.parent = parent
        self.path = f"{parent.path}{a_d}"  # eg 'aaa' for 3rd level approx
        self.a_d = a_d
        self.level = parent.level + 1
        # calculate coefficients
        filter_bank = self.transform_object.filter_bank
        filter_coeffs = filter_bank[0] if a_d == "a" else filter_bank[1]
        self.data = self.filter(filter_coeffs, self.transform_object.mode)

        # calculate additional info about the node
        self.norm_frequency_range = self.calc_norm_freq()
        self.frequency_range = None

    def filter(self, filter_coeffs, mode):
        "performs the convolution between an adapted signal and the filter"
        parent_data = self.parent.data.copy()

        # calculate signal extension length
        length = len(filter_coeffs) - 2
        l2 = length
        if len(parent_data) % 2:  # for odd length data, add coef on the right
            l2 += 1
        # signal extension
        data = pywtpad(parent_data, (length, l2), mode)
        # perform convolution
        data = sp.fftconvolve(data, filter_coeffs, "valid")
        # downsample
        data = data[::2]
        return data

    def calc_norm_freq(self):
        """
        calculates the frequency range of the node as an upper and lower bound
        between 0-1. for example, node 'aa' would have a normalised frequency
        range of 0-0.25

        Parameters
        ----------
        None

        Returns
        -------
        lower_bound,upper_bound: tuple
            tuple containing the normalised upper and lower bounds of the
            node's frequency range
        """
        parent = self.parent

        lower_bound, upper_bound = parent.norm_frequency_range

        hi_or_lo = 0  # even if the node takes the lower half of frequecy range

        if self.path[:-1].count("d") % 2:  # if there is an odd number of d's
            hi_or_lo += 1

        if self.a_d == "d":
            hi_or_lo += 1

        diff = upper_bound - lower_bound
        if hi_or_lo % 2 == 0:
            upper_bound -= diff / 2
        else:
            lower_bound += diff / 2
        return lower_bound, upper_bound

    def calc_freq(self, sampling_rate):
        "converts the normalised frequency range to an actual freuqency range"
        nyquist_frequency = sampling_rate / 2
        self.sampling_rate = sampling_rate
        frequency_range = tuple(
            [bound * nyquist_frequency for bound in self.norm_frequency_range]
        )
        return frequency_range

    def match_ids(self, id):
        """
        determines whether the node contains the specified frequency in its
        frequency range or has the specified path

        Paremeters
        ----------
        id: float or string or list of floats or strings
            can be a frequency or path e.g. 'aa'. can also be a list of ids to
            match. Frequencies must be floats. integers are interpreted as node numbers.

        Returns
        -------
        match: boolean
            whether the node matches the id

        Raises
        ------
        AttributeError: error
            raised when trying to match by frequencies and the sampling rate
            has not been supplied
        """
        if not isinstance(id, list):
            id = [id]

        for i in id:
            if self.match_id(i):
                return True
        else:
            return False

    def match_id(self, id):
        "determines whether the node matches a given id"
        if isinstance(id, str):
            return id == self.path
        elif isinstance(id, (float)):
            if self.frequency_range is None:
                error = """
                call transform_object.calc_frequencies() or svd_object .calc_frequencies() before finding node by frequency
                """
                raise AttributeError(error)
            freq_range = self.frequency_range
            return freq_range[0] <= id <= freq_range[1]
        elif isinstance(id, int):
            return id == self.node_number
        elif isinstance(id, Node):
            return self == id

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
            self._energy = calc_energy(self.data)
        return self._energy

    @property
    def kurtosis(self):
        if not hasattr(self, "_kurtosis"):
            self._kurtosis = calc_kurtosis(self.data)
        return self._kurtosis

    @property
    def crest_factor(self):
        if not hasattr(self, "_crest_factor"):
            self._crest_factor = calc_crest_factor(self.data)
        return self._crest_factor

    @property
    def skewness(self):
        if not hasattr(self, "_skewness"):
            self._skewness = calc_skewness(self.data)
        return self._skewness

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

    def __init__(self, data):
        self.parent = None
        self.data = data
        self.path = ""
        self.level = 0
        self.norm_frequency_range = (0, 1)
