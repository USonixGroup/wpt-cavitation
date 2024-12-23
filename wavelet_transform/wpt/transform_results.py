from typing import TYPE_CHECKING
import pandas as pd
from numpy import average

from wavelet_transform.utils.extensions import to_list
from wavelet_transform.wpt.transform_node import Node

if TYPE_CHECKING:
    from wavelet_transform.wpt.transform_object import Transform


class TransformResults:
    "Class for processing the results for the Transform."

    def __init__(self, transform_obj: "Transform"):
        self.tr_obj = transform_obj

    # ---------------- Functions Used for Inverse Transform -----------------

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
        # Append all of the relevant detail nodes
        for level in range(1, len(self.tr_obj.levels)):
            nodes = self.tr_obj.get_level(level)
            detail_node = nodes[1]
            results.insert(0, detail_node)
        # Append final approx node
        approx_node = nodes[0]
        results.insert(0, approx_node)
        return results

    def info(self, level=None, order="natural", leaf_nodes=False):
        """
        Returns a dataframe with information about each node in a specified level
        of the transform.

        Parameters
        ----------
        level: int
            the level of transform to display information for
        order: str
            "natural" or "freq" (see self.get_level())
        leaf_nodes: bool
            display info about the leaf nodes of the transform.

        Returns
        -------
        info: pd.DataFrame
            A dataframe containing statistical measures calculated from the coefficients
            at each node.
        """
        max_level = self.tr_obj.max_level()
        if isinstance(level, int) and level > max_level and leaf_nodes is False:
            raise ValueError(
                f"transform has no level {level}. max level is: {max_level}"
            )
        if level is None:
            level = self.tr_obj.max_level()

        info_data = []
        if not isinstance(leaf_nodes, bool):
            raise ValueError("leaf_nodes argument must be boolean")
        if leaf_nodes is False:
            nodes = self.tr_obj.get_level(level, order=order)
        elif leaf_nodes is True:
            nodes = self.tr_obj.get_leaf_nodes(order=order)
        short_node_len = len(nodes[0].data_keep)
        for node in nodes:
            assert isinstance(node, Node)
            node_len = len(node.data_keep)
            if node_len < short_node_len:
                short_node_len = node_len
            info_data.append(
                {
                    "level": node.level,
                    "node number": node.node_number,
                    "norm frequency range": node.norm_frequency_range,
                    "frequency range": node.freq_range,
                    "centre frequency": average(node.numeric_freq_range),
                    "energy": node.energy,
                    "log energy": node.log_energy,
                    "average energy": node.average_energy,
                    "log average energy": node.log_average_energy,
                    "kurtosis": node.kurtosis,
                    "crest factor": node.crest_factor,
                    "skewness": node.skewness,
                    "abs skewness": node.abs_skewness,
                    "entropy": node.entropy,
                    "spectral entropy": node.spectral_entropy,
                    "level entropy": node.level_entropy,
                }
            )
        if short_node_len < 500:
            print(f"WARNING: short node data. Shortest node data is ({short_node_len})")
        info = pd.DataFrame(info_data)
        return info

    def find_nodes(self, node_list, **kwargs):
        """
        list nodes with specified frequency and/or path from a give list of nodes

        Parameters
        ----------
        node_list: list
            list of Node objects to seasrch though.

        Keyword Arguments
        -----------------
        nodes: list of: str, int, and/or node objects
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
        nodes = to_list(kwargs.get("nodes"))
        nodes_to_leave = to_list(kwargs.get("nodes_to_leave"))

        # Check types of node ids
        for _list in [nodes, nodes_to_leave]:
            if _list is None:
                continue
            if any(isinstance(i, float) for i in _list):
                raise ValueError(
                    """Nodes are defined by strings, integers, or node objects. Use kwargs "frequencies" or "frequencies_to_leave" to sort using frequencies """
                )
        # Convert frequencies to floats for compatibility with Node.match_id()
        frequencies = to_list(kwargs.get("frequencies"), convert_to_float=True)
        frequencies_to_leave = to_list(
            kwargs.get("frequencies_to_leave"), convert_to_float=True
        )
        min_energy = kwargs.get("min_energy")
        max_energy = kwargs.get("max_energy")

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
            node_list = [node for node in node_list if node and node.match_ids(keep)]
        elif leave:
            node_list = [
                node for node in node_list if node and not node.match_ids(leave)
            ]

        # Filter nodes by energy
        filtered_node_list = [
            node
            for node in node_list
            if (min_energy is None or min_energy <= node.energy)
            and (max_energy is None or node.energy <= max_energy)
        ]
        return filtered_node_list
