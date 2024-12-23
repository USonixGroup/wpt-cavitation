from typing import TYPE_CHECKING

from pywt import pad as pywtpad
import scipy.signal as sp

if TYPE_CHECKING:
    from wavelet_transform.wpt.transform_object import Transform


class NodeBase:
    """
    Base class for nodes, contains the implementations for convolution, calculating frequency
    ranges, and identifying nodes.
    """

    def __init__(self, parent, a_d, transform_object: "Transform"):
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
        # Set parent, path, and transform object
        self.parent = parent
        self.path = f"{parent.path}{a_d}"  # eg 'aaa' for 3rd level approx
        self.a_d = a_d
        self.level = parent.level + 1

        # Calculate coefficients
        filter_bank = transform_object.filter_bank
        filter_coeffs = filter_bank[0] if a_d == "a" else filter_bank[1]
        self.data = self.filter(
            self.parent.data.copy(), filter_coeffs, transform_object.mode
        )

        # Calculate additional info about the node
        self.norm_frequency_range = self.calc_norm_freq()
        self.frequency_range = None

    def filter(self, data, filter_coeffs, mode):
        # Filter1 and filter2 do the same thing for even length filters
        return self.filter2(data, filter_coeffs, mode)

    def filter1(self, parent_data, filter_coeffs, mode):
        """
        Performs the convolution between an adapted signal and the filter
        using the method described by pywavelets.
        """
        # Calculate signal extension length
        length = len(filter_coeffs) - 2
        l2 = length
        if len(parent_data) % 2:  # for odd length data, add coef on the right
            l2 += 1
        # Signal extension
        data_pad = pywtpad(parent_data, (length, l2), mode)
        # Perform convolution
        data_conv = sp.convolve(data_pad, filter_coeffs, "valid")

        # Downsample
        data_out = data_conv[::2]
        return data_out

    def filter2(self, parent_data, filter_coeffs, mode):
        """
        Performs the convolution between an adapted signal and the filter
        using the method described by matlab.
        """
        # Calculate signal extension length for full convolution
        length = len(filter_coeffs) - 1
        # Signal extension
        data_pad = pywtpad(parent_data, (length, length), mode)
        # Perform full convolution
        data_conv = sp.convolve(data_pad, filter_coeffs, "valid")

        # Downsample
        lim = 2 * (len(data_conv) // 2)
        data_out = data_conv[1:lim:2]
        return data_out

    # --------------- Frequency Calculation ---------------

    def calc_norm_freq(self):
        """
        Calculates the frequency range of the node as an upper and lower bound
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

    def set_frequency_range(self, sampling_rate):
        "Converts the normalised frequency range to an actual freuqency range"
        nyquist_frequency = sampling_rate / 2
        self.sampling_rate = sampling_rate
        frequency_range = tuple(
            [bound * nyquist_frequency for bound in self.norm_frequency_range]
        )
        self.frequency_range = frequency_range

    # -------------- Finding Nodes ----------------

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
        elif isinstance(id, self.__class__):
            return self == id
