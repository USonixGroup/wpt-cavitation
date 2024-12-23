from pywt import Wavelet as pywtWavelet

from wavelet_transform.wpt.transform_node import BaseNode, Node
from wavelet_transform.wpt.transform_results import TransformResults
from wavelet_transform.wpt.tree_graph import TreeGraph


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
        self.sampling_rate = None  # for calculating the frequency range of each node
        self.tree = TreeGraph(self)

        # Retained data for calculating statistical features
        self.keep_ratio = "full"

    # -------------------- Parameters --------------------

    @property
    def parameters(self):
        if not hasattr(self, "_parameters"):
            self._parameters = {
                "wavelet": self.wavelet,
                "mode": self.mode,
                "data": self.data,
                "min_level": self.min_level,
                "min_entropy": self.min_entropy,
            }
        return self._parameters

    @property
    def min_level(self):
        "The minimum level of transform"
        if not hasattr(self, "_min_level"):
            return None
        else:
            return self._min_level

    @min_level.setter
    def min_level(self, value):
        assert isinstance(value, int)
        self._min_level = value

    @property
    def min_entropy(self):
        "Whether or not the minimum entropy transform has been used"
        if not hasattr(self, "_min_entropy"):
            return False
        else:
            return self._min_entropy

    @min_entropy.setter
    def min_entropy(self, value):
        assert isinstance(value, bool)
        self._min_entropy = value

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
        if order not in ["natural", "freq"]:
            raise ValueError("order must be 'nutaral' or 'freq")
        level = self.max_level() if level is None else level
        if order == "natural":
            return self.levels.get(level, [])
        elif order == "freq":
            nodes = self.levels[level].copy()
            nodes.sort(key=lambda x: x.norm_frequency_range[0])
            return nodes

    def get_full_level(self, level=None):
        """
        Fill out missing nodes in the level with None. Used for the inverse transform
        with a minimum entropy decomposition.
        """
        if level is None:
            level = self.max_level()
        nodes = self.get_level(level, order="natural")  # order must be natural
        full_level_nodes = [None for _ in range(2**level)]
        for node in nodes:
            full_level_nodes[node.node_number] = node
        return full_level_nodes

    def get_leaf_nodes(self, order="natural", freq_threshold=None):
        """
        Get the leaf nodes in the tree. Ordered by frequency.
        Frequency threshold is the normalised frequency threshold.
        """
        leaf_nodes = [
            node
            for level in self.levels
            for node in self.get_level(level)  # gather nodes with no children
            if (
                2 * node.node_number not in self.node_numbers(level + 1)
                and 2 * node.node_number + 1 not in self.node_numbers(level + 1)
            )
        ]
        if freq_threshold is not None:
            if freq_threshold > 1 or freq_threshold < 0:
                raise ValueError(
                    "frequency threshold must be normalised with the nyquist frequency"
                )
            leaf_nodes = [
                i for i in leaf_nodes if i.norm_frequency_range[1] <= freq_threshold
            ]
        if order not in ["freq", "natural"]:
            raise ValueError("order must be 'freq' or 'natural'")
        # Sort nodes in order of frequency.
        if order == "freq":
            leaf_nodes.sort(key=lambda x: x.norm_frequency_range[0])
        return leaf_nodes

    def node_numbers(self, level=None):
        "Generator for node numbers"
        return (node.node_number for node in self.get_level(level))

    def create_node(self, parent, a_d):
        """
        Creates a node object, adds it to list of nodes if add_node is true.
        add_node allows for further logic to be implemented in the transform
        for adding nodes (e.g. for choosing best basis).
        """
        # Initialise node object
        node = Node(parent, a_d, self)
        return node

    def add_node(self, node: Node):
        "appends a node to node list"
        self.levels.setdefault(node.level, list()).append(node)

    @classmethod
    def pywt_filter_bank(cls, wavelet):
        "obtains the filter coefficients from the pywt filter bank function"
        w = pywtWavelet(wavelet)
        return w.filter_bank

    def max_level(self):
        "returns the last level of transform"
        return list(self.levels.keys())[-1]

    # ------------------ Generate Additional Data ----------------

    def calc_frequencies(self, sampling_rate):
        "Set the sampling rate to give information about nyquist frequency"
        self.sampling_rate = sampling_rate
        for level in self.levels:
            for node in self.levels[level]:
                node.set_frequency_range(self.sampling_rate)

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

    @property
    def keep_ratio(self):
        """
        Determines the length of the signal used for calculating statistical metrics.

        str in ["full", "valid", "same"]: trim data length to match valid or samse convolution modes

        full uses the full length of data and is set by default.

        int or float: number between 0-1 determining the ratio of the original data retained
        centered around the original data.
        """
        return self._keep_ratio

    @keep_ratio.setter
    def keep_ratio(self, value):
        if not isinstance(value, (str, float, int)):
            raise ValueError("keep_ratio must be str, float, or int")
        if isinstance(value, str) and value not in ("full", "valid", "same"):
            raise ValueError("keep_ratio must be 'full', 'valid', or 'same'")
        if isinstance(value, (int, float)):
            if value > 1 or value < 0:
                raise ValueError("keep_ratio must be between 0-1")
        self._keep_ratio = value

    # ---------------- Plot Tree Diagram ---------------

    def plot_tree(
        self,
        node_size=1500,
        node_colour="skyblue",
        freq_threshold=None,
        graph_legend="Graph legend",
        fig_name=None,
        **kwargs
    ):
        """
        Plot the tree diagram for the wavelet transform.

        Parameters
        ----------
        node_size: int or str
            use an integer to set a global size for all nodes.
            use a string with the same name as an attribute of the Transform.info()
            dataframe to set size based on the attribute e.g. node_size = "energy"
            will make higher energy nodes larger.
        node_colour: str or list
            use a string to set a global colour name.
            use a list of labels (one for each leaf node) to colour the leaf nodes by label
        freq_threshold: float
            the frequency threshold for leaf nodes to be labeled. Nodes with upper bound
            of frequency range above this threshold will not be labeled

        keyword arguments
        -----------------
        sampling_rate: float
            supply the sampling rate to calculate frequencies for nodes. Will plot
            normalised frequencies otherwise.
        order: str
            "natural" or "freq" to order the tree in natural or freuqency order
        """
        self.tree.plot_tree(
            node_size, node_colour, freq_threshold, graph_legend, fig_name, **kwargs
        )
