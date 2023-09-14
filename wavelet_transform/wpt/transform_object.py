from pywt import Wavelet as pywtWavelet

from wavelet_transform.wpt.transform_node import BaseNode, Node
from wavelet_transform.wpt.transform_results import TransformResults


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
        self.results = TransformResults(self)
        self.wavelet = wavelet
        self.mode = mode
        self.data = data
        self.base_node = BaseNode(data, self)  # initialise base node
        self.levels = {0: [self.base_node]}  # initialise list of all nodes
        self.filter_bank = self.pywt_filter_bank(self.wavelet)
        self.sampling_rate = (
            None  # used for calculating the frequency range of each node
        )

    # -------------------- Functions used for Transform ---------------

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
        level = self.max_level() if level is None else level
        if order == "natural":
            return self.levels.get(level, [])
        elif order == "freq":
            nodes = self.levels[level].copy()
            nodes.sort(key=lambda x: x.norm_frequency_range[0])
            return nodes

    def get_leaf_nodes(self):
        "Get the leaf nodes in the tree. Ordered by frequency."
        leaf_nodes = [
            node
            for level in self.levels
            for node in self.get_level(level)  # gather nodes with no children
            if 2 * node.node_number not in self.node_numbers(level + 1)
        ]
        # Sort nodes in order of frequency.
        leaf_nodes.sort(key=lambda x: x.norm_frequency_range[0])
        return leaf_nodes

    def node_numbers(self, level=None):
        "Generator for node numbers"
        return (node.node_number for node in self.get_level(level))

    def create_node(self, parent, a_d, add_node=True):
        """
        Creates a node object, adds it to list of nodes if add_node is true.
        add_node allows for further logic to be implemented in the transform
        for adding nodes (e.g. for choosing best basis).
        """
        # Initialise node object
        node = Node(parent, a_d, self)
        if add_node is True:
            self.add_node(node)
        return node

    def add_node(self, node: Node):
        "appends a node to node list"
        self.levels.setdefault(node.level, list()).append(node)

    def pywt_filter_bank(self, wavelet):
        "obtains the filter coefficients from the pywt filter bank function"
        w = pywtWavelet(wavelet)
        return w.filter_bank

    def max_level(self):
        "returns the last level of transform"
        return list(self.levels.keys())[-1]

    # ------------------ Generate Additional Data ----------------

    def calc_frequencies(self, sampling_rate):
        "set the sampling rate to give information about nyquist frequency"
        self.sampling_rate = sampling_rate
        for level in self.levels:
            for node in self.levels[level]:
                node.frequency_range = node.calc_freq(self.sampling_rate)

    def total_energy(self, level=None):
        "Return the total energy of all nodes in a level. Max level by default."
        if not hasattr(self, "_total_energy"):
            self._total_energy = sum(node.energy for node in self.get_level(level))
        return self._total_energy

    # ---------------- Generate Results ----------------

    def find_nodes(self, node_list, **kwargs):
        return self.results.find_nodes(node_list, **kwargs)

    def dwt_results(self):
        return self.results.dwt_results()

    def info(self, level=None, order="natural", leaf_nodes=False):
        return self.results.info(level, order, leaf_nodes)
