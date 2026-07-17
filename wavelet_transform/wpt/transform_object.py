from pywt import Wavelet as pywtWavelet  # type: ignore
from typing import Optional, Union

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
        self.levels: dict[int, list[BaseNode | Node]] = {0: [self.base_node]}  # initialise list of all nodes
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
                "best_basis": self.best_basis,
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
    def best_basis(self):
        "Whether or not the best basis transform has been used"
        if not hasattr(self, "_best_basis"):
            return False
        else:
            return self._best_basis

    @best_basis.setter
    def best_basis(self, value):
        assert isinstance(value, str)
        self._best_basis = value

    # -------------------- Get Nodes ---------------

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
        nodes: list of Node
            list of nodes in selected level
        """
        if not isinstance(order, str):
            raise ValueError("order must be 'natural' or 'freq'")
        level = self.max_level() if level is None else level
        if order == "natural":
            return self.levels.get(level, [])
        elif order == "freq":
            nodes = self.levels[level].copy()
            nodes.sort(key=lambda x: x.norm_frequency_range[0])
            return nodes
        else:
            raise ValueError("order must be 'natural' or 'freq'")

    def get_leaf_nodes(self, order="natural", freq_threshold=None):
        """
        Get the leaf nodes in the tree. Ordered by node number or frequency.
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
                i for i in leaf_nodes if i.is_in_threshold(freq_threshold)
            ]
        if order not in ["freq", "natural"]:
            raise ValueError("order must be 'freq' or 'natural'")
        # Sort nodes in order of frequency.
        if order == "freq":
            leaf_nodes.sort(key=lambda x: x.norm_frequency_range[0])
        return leaf_nodes
    
    def get_node(self, level, id):
        """
        Finds and returns a node matching the given id.

        Parameters
        ----------
        level: int or "leaf"
            the level of the transform
        id: str, int, float, or Node
            the identifier for the node. Can be the path (str), node number (int),
            frequency (float), or the node object itself.
        """
        if level == "leaf":
            nodes = self.get_leaf_nodes()
        else:
            nodes = self.get_level(level)
        for node in nodes:
            if node.match_id(id):
                return node
        raise ValueError("No node found matching the given id")

    def node_numbers(self, level=None):
        "Generator for node numbers"
        return (node.node_number for node in self.get_level(level))

    # -------------------- Creating and Adding Nodes ----------------

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
        "Appends a node to node list."
        self.levels.setdefault(node.level, list()).append(node)

    @classmethod
    def pywt_filter_bank(cls, wavelet):
        "Obtains the filter coefficients from the pywt filter bank function."
        w = pywtWavelet(wavelet)
        return w.filter_bank

    def max_level(self):
        "Returns the maximum level of transform as an integer."
        return list(self.levels.keys())[-1]

    # ------------------ Generate Additional Data ----------------

    def calc_frequencies(self, sampling_rate):
        "Set the sampling rate to give information about nyquist frequency"
        self.sampling_rate = sampling_rate
        for level in self.levels:
            for node in self.levels[level]:
                node.set_frequency_range(self.sampling_rate)

    def total_energy(self, level=None):
        """
        Return the total energy of all nodes in a level. Max level by default.
        Adds metric to cache for future use.
        """
        if level is None:
            level = self.max_level()
        if not hasattr(self, "_total_energy"):  # initialise cache
            # annotate cache as possibly holding floats or None so type checkers accept assignments
            self._total_energy: dict[int, Optional[float]] = (
                {i: None for i in self.levels.keys()}
            )
        if self._total_energy.get(level) is not None:  # check cache
            return self._total_energy[level]
        else:
            self._total_energy[level] = sum(
                float(node.metric("energy")) for node in self.get_level(level)
            )
        return self._total_energy[level]
    
    # ---------------- Find Nodes by Frequency ----------------
    
    def get_node_at_frequency(self, frequency: float, level="leaf"):
        """
        Finds a node in a transform at a given level that contains the specified
        frequency in its frequency range.

        Parameters
        ----------
        frequency: float
            the frequency to calculate the reconstruction range for
        level: int or "leaf"
            the level of the transform

        Returns
        -------
        Node
            the node that contains the specified frequency
        """

        if level == "leaf":
            # If level is "leaf", find the leaf node containing the frequency
            nodes = self.get_leaf_nodes()
        elif isinstance(level, int):
            if level <= 0:
                raise ValueError("Level must be positive integer")
            if level > self.max_level():
                raise ValueError("Level exceeds maximum level of transform")
            # If level is an integer, find the node at that level containing the frequency
            nodes = self.get_level(level)
        else:
            raise ValueError("Level must be a positive integer or 'leaf'")

        for node in nodes:
            if node.match_id(frequency):
                return node
        raise ValueError("No node found containing the specified frequency")

    # ---------------- Generate Results ----------------

    def dwt_results(self):
        """
        Retrieves the discrete wavelet transform (DWT) results from the transform.

        Returns
        -------
        results: list of Node
            a list of nodes related to the dwt
        """
        return self.results.dwt_results()

    def info(
            self, 
            level:Optional[Union[int, str, None]]="leaf", 
            order="natural", 
            keep_ratio:Union[str, float, int]=1.,
            *,
            include_node_info=False,
            metrics=None,
            freq_threshold=1.,
        ):
        """
        Returns a dataframe with information about each node in a specified level
        of the transform.

        Parameters
        ----------
        level: int or "leaf" or None
            the level of transform to display information for. If `level` is
            "leaf", displays info for leaf nodes. If None, displays info for max level.
        order: str
            "natural" or "freq" (see self.get_level())
        keep_ratio: float
            between 0-1. Represents the ratio of coefficients kept in each node.
            does not affect the keep_ratio of the transform object itself.

        metrics: list or None
            list of metric names to include in the dataframe. If None, includes all
            available metrics.

        Returns
        -------
        info: pd.DataFrame
            A dataframe containing statistical measures calculated from the coefficients
            at each node.
        """
        return self.results.info(
            level=level, order=order, keep_ratio=keep_ratio,
            metrics=metrics, include_node_info=include_node_info,
            freq_threshold=freq_threshold
        )

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

    @property
    def ref_psd(self):
        """
        Calculate and return the reference power spectral density (PSD) array
        for the transform data.

        Returns
        -------
        ref_array: np.ndarray
            The reference PSD array.
        """
        return self.results.ref_psd

    # ---------------- Plot Tree Diagram ---------------

    def plot_tree(
        self,
        node_size="default",
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
        order: str
            "natural" or "freq" to order the tree in natural or freuqency order
        show_frequency: bool
            whether to show the frequency ranges of each node on the plot
        use_plotly: bool
            whether to use plotly to generate the plot
        """
        self.tree.plot_tree(
            node_size, node_colour, freq_threshold, graph_legend, fig_name, **kwargs
        )