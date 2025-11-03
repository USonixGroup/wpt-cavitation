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
        nodes = []  # avoid unbound error
        # Append all of the relevant detail nodes
        for level in range(1, self.tr_obj.max_level()+1):
            if 1 not in [i for i in self.tr_obj.node_numbers(level)]:
                raise ValueError("call dwt_results on full transform")
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
        if not isinstance(leaf_nodes, bool):
            raise ValueError("leaf_nodes must be bool")
        
        if level is None:
            level = self.tr_obj.max_level()

        info_data = []
        if not isinstance(leaf_nodes, bool):
            raise ValueError("leaf_nodes argument must be boolean")
        if leaf_nodes is False:
            nodes = self.tr_obj.get_level(level, order=order)
        else:
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
