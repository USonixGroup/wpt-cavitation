from pywt import Wavelet as pywtWavelet

import numpy as np
import pandas as pd

from wavelet_transform.utils.extensions import to_list
from wavelet_transform.wpt.transform_node import BaseNode, Node


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
        if level is None:
            level = self.max_level()
        if order == "natural":
            return self.levels[level]
        elif order == "freq":
            nodes = self.levels[level].copy()
            nodes.sort(key=lambda x: x.norm_frequency_range[0])
            return nodes

    def create_node(self, parent, a_d):
        "creates a node object, adds it to list of nodes"
        # initialise node object
        node = Node(parent, a_d, self)
        self.add_node(node)

    def add_node(self, node):
        "appends a node to node list"
        self.levels.setdefault(node.level, list()).append(node)

    def pywt_filter_bank(self, wavelet="haar"):
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

    def calc_total_energy(self, level=None):
        "calculates the total energy of the transform at a given level"
        total_energy = 0
        for node in self.get_level(level):
            total_energy += np.sum(node.data**2)
        return total_energy

    def info(self, level=None, order="natural"):
        """
        Returns a dataframe with information about each node in a specified level
        of the transform.

        Parameters
        ----------
        level: int
            the level of transform to display information for
        order: str
            "natural" or "freq" (see self.get_level())

        Returns
        -------
        info: pd.DataFrame
            A dataframe containing statistical measures calculated from the coefficients
            at each node.
        """
        if level is None:
            level = self.max_level()

        info_data = []
        for node in self.get_level(level, order=order):
            info_data.append(
                {
                    "node number": node.node_number,
                    "energy": node.energy,
                    "log energy": node.log_energy,
                    "kurtosis": node.kurtosis,
                    "crest factor": node.crest_factor,
                    "skewness": node.skewness,
                    "frequency range": node.freq_range,
                    "entropy": node.entropy,
                    "spectral entropy": node.spectral_entropy,
                }
            )

        info = pd.DataFrame(info_data)
        return info

    # ---------------- Functions Used for Inverse Transform -----------------

    def find_nodes(self, node_list, **kwargs):
        """
        list nodes with specified frequency and/or path from a give list of nodes

        Parameters
        ----------
        node_list: list
            list of Node objects to seasrch though.

        Keyword Arguments
        -----------------
        nodes: list of str or list of int
            List of nodes to reconstruct. Can specify a path e.g. "ad" or the node number (0-indexed)
        nodes_to_leave: list of str or list of int
            List of nodes to leave out of reconstruction.
        frequencies: list
            List of frequenices (int or float) to reconstruct. Only nodes containing these frequencies
            will be reconstructed.
        frequencies_to_leave: list
            List of frequenices (int or float) to leave out of reconstruction. Nodes containing these
            frequencies will not be reconstructed.
        min_energy: float or int
            The minimum energy a node must have to be added to the reconstruction.
        max_energy: float or int
            The maximumm energy a node can have to be added to the reconstruction.

        Returns
        -------
        node_list: list
            list containing node objects with specified ids or minimum energy
        """

        # Gather kwargs
        nodes = to_list(kwargs.get("nodes", None))
        nodes_to_leave = to_list(kwargs.get("nodes_to_leave", None))

        # Check types of
        for _list in [nodes, nodes_to_leave]:
            if _list is None:
                continue
            if any((isinstance(i, float) for i in _list)):
                raise ValueError(
                    """nodes are defined by strings or integers. Use kwargs "frequencies" or "frequencies_to_leave" to sort using frequencies """
                )
        # Convert frequencies to floats for compatibility with Node.match_id()
        frequencies = to_list(kwargs.get("frequencies", None), convert_to_float=True)
        frequencies_to_leave = to_list(
            kwargs.get("frequencies_to_leave", None), convert_to_float=True
        )
        min_energy = kwargs.get("min_energy", None)
        max_energy = kwargs.get("max_energy", None)

        # Concatenate node lists
        keep = []
        for i in [nodes, frequencies]:
            if i is not None:
                keep += i

        leave = []
        for i in [nodes_to_leave, frequencies_to_leave]:
            if i is not None:
                leave += i

        # Only one of keep, leave
        if keep and leave:
            raise SyntaxError("Cannot specify nodes to keep and nodes to leave")

        if keep:
            node_list = [node for node in node_list if node.match_ids(keep)]
        elif leave:
            node_list = [node for node in node_list if node.match_ids(leave)]

        # Filter nodes by energy
        filtered_node_list = [
            node
            for node in node_list
            if (min_energy is None or min_energy <= node.energy)
            and (max_energy is None or node.energy <= max_energy)
        ]
        return filtered_node_list

    def packet_nodes_for_rec(self, level, **kwargs):
        """
        Gather nodes for reconstruction with more than min_energy. Must be
        called before inverse transform. keep and leave cannot be used in
        tandem.

        e.g.
        transform_object.packet_nodes_for_rec(keep=['aa','dd'], min_energy=0.1)
        will reconstruct nodes 'aa' and 'dd' as long as they have minimum
        energy of 0.1.

        Parameters
        ----------
        level: int
            the level of transform to start reconstruct from

        Keyword Arguments
        -----------------
        See self.find_nodes()


        Returns
        -------
        node_list: list
            list of nodes meeting requirements

        Raises
        ------
        SyntaxError:
            when specifying more than one of keep and leave.
        """
        nodes = self.get_level(level)  # max level by default

        nodes_to_keep = self.find_nodes(nodes, **kwargs)
        node_list = []
        node_list = [node if node in nodes_to_keep else None for node in nodes]

        # check if any nodes are remaining after filtering
        if node_list.count(None) == len(node_list):
            raise ValueError("could not found any nodes matching input arguments")
        return node_list

    def nodes2data(self, nodes):
        """
        Converts a list of nodes to a list of arrays containing their data.

        Parameters
        ----------
        nodes: list
            list of node objects

        Returns
        -------
        output: list
            list of arrays, replacing the respective node in the input list
            with its data
        """

        output = [node.data if node is not None else None for node in nodes]

        return output

    def dwt_results(self):
        """
        Extracts the nodes containing dwt results from a dwt or dwpt transform
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
