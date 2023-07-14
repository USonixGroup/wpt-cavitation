from pywt import Wavelet as pywtWavelet
from pywt import pad as pywtpad
import numpy as np
import scipy.signal as sp
from ..utils.extensions import sci_form

from scipy.stats import kurtosis


class Transform:
    """
    An object created for each transform. Stores all of the relevant
    information about the transform

    Parameters
    ----------
    data: array
        the sound wave to be transformed
    wavelet: string
        the chosen wavelet for the transform
    mode: string
        the chosen type of padding, see pywt.pad for options

    """

    def __init__(self, data, wavelet, mode):
        self.wavelet = wavelet
        self.mode = mode
        self.data = data
        self.base_node = BaseNode(data)  # initialise base node
        self.levels = {0: [self.base_node]}  # initialise list of all nodes
        self.filter_bank = self.pywt_filter_bank(self.wavelet)

    def get_level(self, level=None, order="natural"):
        """
        returns a list of nodes for the chosen level. returns max level by
        default

        Parameters
        ----------
        level: int
            the level of transform to extract nodes from
        order: str
            the order in which to list the nodes. see get_level in pywt.

        Returns
        -------
        nodes: list
            list of nodes in selected level
        """
        if level is None:
            level = self.max_level()
        if order == "natural":
            return self.levels[level]
        elif order == "freq":
            nodes = self.levels[level].copy()
            nodes.sort(key=lambda x: x.norm_frequency_range[0])
            return nodes

    def add_level(self):
        "increments the number of levels by 1"
        self.levels[len(self.levels)] = []

    def max_level(self):
        "returns the last level of transform"
        return list(self.levels.keys())[-1]

    def add_node(self, node):
        "appends a node to node list"
        # initialise the level if necessary
        if len(self.levels) == node.level:
            self.add_level()
        self.levels[node.level].append(node)

    def create_node(self, parent, a_d):
        "creates a node object, adds it to list of nodes"
        # initialise node object
        node = Node(parent, a_d, self)
        self.add_node(node)

    def pywt_filter_bank(self, wavelet="haar"):
        "obtains the filter coefficients from the pywt filter bank function"
        w = pywtWavelet(wavelet)
        return w.filter_bank

    def calc_frequencies(self, sampling_rate=None):
        "set the sampling rate to give information about nyquist frequency"
        if sampling_rate is None:
            return
        self.sampling_rate = sampling_rate
        for level in self.levels:
            for node in self.levels[level]:
                node.frequency_range = node.calc_freq(self.sampling_rate)

    def calc_total_energy(self, level=None):
        "calculates the total energy of the transform at a given level"
        total_energy = 0
        for node in self.get_level(level):
            total_energy += np.sum(node.data**2)
        return total_energy

    def show_all_energy(self):
        "prints the energy for each node"
        for level in self.levels:
            print(f"level {level} nodes:")
            for node in self.levels[level]:
                energy = sci_form(node.energy)
                try:
                    lf = sci_form(node.frequency_range[0])
                    uf = sci_form(node.frequency_range[1])
                    print(f"node: {node.path} [{lf}-{uf} Hz] energy: {energy}")
                except AttributeError:  # if the frequency is not calculated
                    lf = sci_form(node.norm_frequency_range[0])
                    uf = sci_form(node.norm_frequency_range[1])
                    print(
                        f"""
                        node: {node.path} [({lf}-{uf})*nyquist frequency Hz]
                        \nenergy: {energy}
                        """
                    )
                print()

    def list_energy(self, level=None, order="natural"):
        """
        creates a list of all energies at a specified level, max level
        by default
        """
        nodes = self.get_level(level, order)
        energies = []
        for node in nodes:
            energies.append(node.energy)
        return energies

    def get_node(self, id, nodes):
        "finds a node for a given path or frequency from a list of nodes"
        for node in nodes:
            if node.match_id(id):
                return node
        # error message of node is not found
        if isinstance(id, str):  # search for node with path
            raise ValueError(f"could not find node with specified path {id}")
        else:  # search for node with frequency content
            raise ValueError(
                f"""
                could not find node containing specified frequency:\
                 {sci_form(id)} Hz
                """
            )

    def find_nodes(self, nodes, ids=None, min_energy=0, max_energy=np.inf):
        """
        list nodes with specified frequency and/or path

        Parameters
        ----------
        nodes: list
            a list containing nodes to search
        ids: list
            list of frequecies or paths to match with nodes
        min_energy: float
            specify a minimum energy for the nodes
        max_energy: float
            specify a maximum energy for the nodes

        Returns
        -------
        node_list: list
            list containing node objects with specified ids or minimum energy
        """
        node_list = []
        if ids:  # if the ids are specified
            for id in ids:
                node = self.get_node(id, nodes)
                if (node.energy > min_energy) & (node.energy < max_energy):
                    node_list.append(node)
        else:
            for node in nodes:
                if (node.energy > min_energy) & (node.energy < max_energy):
                    node_list.append(node)
        return node_list

    def packet_nodes_for_rec(
        self, level=None, *, keep=None, leave=None, min_energy=0, max_energy=np.inf
    ):
        """
        gather nodes for reconstruction with more than min_energy. Must be
        called before inverse transform. keep and leave cannot be used in
        tandem. min_energy and max_energy will filter out kept nodes that
        do not meet requirements. If you want to keep all nodes, instead use
        nodes_for_rec = transform_object.get_level().

        e.g.
        transform_object.packet_nodes_for_rec(keep=['aa','dd'],min_energy=0.1)
        will reconstruct nodes 'aa' and 'dd' as long as they have minimum
        energy of 0.1.

        Parameters
        ----------
        level: int
            the level of transform to start reconstruct from
        keep: list
            list of ids (path or frequency content) to reconstruct
        leave: list
            list of ids (path of frequency content) to leave out of
            reconstruction
        min_energy: float
            only add nodes with specified minimum energy
        max_energy: float
            only append nodes with specified maximum energy

        Returns
        -------
        node_list: list
            list of nodes meeting requirements

        Raises
        ------
        SyntaxError:
            when specifying more than one of keep and leave.
        """
        # only one of keep, leave
        if keep and leave:
            raise SyntaxError(
                "only specify one of keep, leave, min_energy, and max_energy"
            )

        if (not isinstance(keep, list)) & (keep is not None):
            raise ValueError("keep must be a list")
        if (not isinstance(leave, list)) & (leave is not None):
            raise ValueError("leave must be a list")

        node_list = []
        nodes = self.get_level(level)  # max level by default
        if keep:
            for node in nodes:
                # print('testing id',id)
                if node.match_ids(keep):
                    node_list.append(node)
                else:
                    node_list.append(
                        None
                    )  # append None if node does not meet requirements
        elif leave:
            for node in nodes:
                if node.match_ids(id):
                    node_list.append(None)
                else:
                    node_list.append(node)
        else:
            node_list = self.get_level(level)

        for node in node_list:  # filter node list by energy range
            if node is None:
                continue
            if node.energy is None:
                if min_energy > 0 and max_energy:
                    print(
                        """
                        could not sort by energy, call function\
                         transform_object.calc_frequencies()
                        """
                    )
                break
            if not ((node.energy >= min_energy) & (node.energy <= max_energy)):
                node_list[node_list.index(node)] = None
        for node in node_list:
            if node is not None:
                break
        else:
            print("could not find any nodes matching input arguments")
        return node_list

    def nodes2data(self, nodes=None, *, synth_data=False):
        """
        converts a list of nodes to a list of arrays containing their data.
        use before inverse transform to convert output from
        packet_nodes_for_rec() or get_level()

        Parameters
        ----------
        nodes: list
            list of node objects
        synth_data: bool
            replaces all node data with synthetic data

        Returns
        -------
        output: list
            list of arrays, replacing the respective node in the input list
            with its data
        """
        output = []
        if nodes:
            for node in nodes:
                if node:
                    if not synth_data:
                        output.append(node.data)
                    else:
                        output.append(synth_data)
                else:
                    output.append(None)
        else:
            for level in self.levels:
                for node in self.levels[level]:
                    output.append(node.data)
        return output

    def dwt_results(self):
        """
        extracts the nodes containing dwt results from a dwt or dwpt transform
        object

        Parameters
        ----------
        None

        Returns
        -------
        results: list
            list of nodes containing the results for the dwt
        """
        results = []
        # append all of the relevant detail nodes
        for level in range(1, len(self.levels)):
            nodes = self.get_level(level)
            detail_node = nodes[1]
            results.insert(0, detail_node)
        # append final approx node
        approx_node = nodes[0]
        results.insert(0, approx_node)
        return results


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
        if a_d == "a":
            filter_coeffs = filter_bank[0]
        else:
            filter_coeffs = filter_bank[1]
        self.data = self.filter(filter_coeffs, self.transform_object.mode)

        # calculate energy, frequency, and kurtosis
        self.energy = self.calc_energy()
        self.kurtosis = self.calc_kurtosis()
        self.crest_factor = self.calc_crest_factor()
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

    def calc_energy(self):
        "calculates the energy in the node"
        data = self.data
        return 1 / len(data) * np.sum(data**2)

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

        lower_bound = parent.norm_frequency_range[0]
        upper_bound = parent.norm_frequency_range[1]

        hi_or_lo = 0  # even if the node takes the lower half of frequecy range

        if self.path[:-1].count("d") % 2:  # if there is an odd number of d's
            hi_or_lo += 1

        if self.a_d == "d":
            hi_or_lo += 1

        diff = upper_bound - lower_bound
        if not hi_or_lo % 2:
            upper_bound -= diff / 2
        else:
            lower_bound += diff / 2
        return lower_bound, upper_bound

    def calc_kurtosis(self):
        "caclulates the kurtosis of the data in the node"
        return kurtosis(self.data)

    def calc_crest_factor(self):
        "calculates the crest factor of the node data in dB"
        rms = np.sqrt(1 / len(self.data) * sum(self.data**2))
        return 10 * np.log10(max(self.data**2) / rms)

    def calc_freq(self, sampling_rate):
        "converts the normalised frequency range to an actual freuqency range"
        nyquist_frequency = sampling_rate / 2
        lower_bound = self.norm_frequency_range[0] * nyquist_frequency
        upper_bound = self.norm_frequency_range[1] * nyquist_frequency
        return lower_bound, upper_bound

    def match_ids(self, id):
        """
        determines whether the node contains the specified frequency in its
        frequency range or has the specified path

        Paremeters
        ----------
        id: float or string or list of floats or strings
            can be a frequency or path e.g. 'aa'. can also be a list of ids to
            match

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
        if isinstance(id, list):
            for i in id:
                if self.match_id(i):
                    return True
            else:
                return False
        else:
            return self.match_id(id)

    def match_id(self, id):
        "determines whether the node matches a given id"
        if isinstance(id, str):
            return id == self.path
        elif isinstance(id, (float, int)):
            if self.frequency_range is None:
                error = """
                call transform_object.calc_frequencies() or svd_object\
                .calc_frequencies() before finding node by frequency
                """
                raise AttributeError(error)
            freq_range = self.frequency_range
            return (id >= freq_range[0]) & (id <= freq_range[1])

    def node_number(self):
        "converts path into node number to compare with matlab"
        path = self.path
        node_no = 0
        bit = 1
        for i in reversed(path):
            if i == "d":
                node_no += bit
            bit *= 2
        return node_no


class BaseNode(Node):
    """
    the base node for the transform. contains data from the original sound wave

    Parameters
    ----------
    data: array
        an array containing the data from the original sound wave
    """

    def __init__(self, data):
        self.parent = None
        self.data = data
        self.path = ""
        self.level = 0
        self.norm_frequency_range = (0, 1)
        self.energy = self.calc_energy()
