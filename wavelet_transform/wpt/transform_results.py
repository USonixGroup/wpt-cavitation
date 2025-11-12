from typing import TYPE_CHECKING
import pandas as pd
from numpy import average

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

    def info(
            self, 
            level=None, 
            order="natural", 
            leaf_nodes=False, 
            keep_ratio=1.,
            *,
            include_node_info=False,
            metrics=None
        ):
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

        old_keep_ratio = self.tr_obj.keep_ratio  # store old keep ratio
        self.tr_obj.keep_ratio = keep_ratio  # temporarily set keep ratio
        for node in nodes:
            assert isinstance(node, Node)
            node_len = len(node.data_keep)
            if node_len < short_node_len:
                short_node_len = node_len

            # start with static fields
            row = {
                "level": node.level,
                "node number": node.node_number,
                "norm frequency range": node.norm_frequency_range,
                "centre frequency": average(node.numeric_freq_range),
            } if include_node_info else {}

            # populate metrics from the Node.metrics property
            if metrics is not None:
                # replace spaces with underscores in metric names
                metrics = [m.replace(" ", "_") for m in metrics]
                metrics_dict = {}
                for metric_name in metrics:
                    metrics_dict[metric_name] = node.metric(metric_name)
            else:
                metrics_dict = getattr(node, "metrics")

            for attr_name, value in metrics_dict.items():
                display_name = attr_name.replace("_", " ")
                row[display_name] = value

            info_data.append(row)
        if short_node_len < 500:
            print(f"WARNING: short node data. Shortest node data is ({short_node_len})")
        info = pd.DataFrame(info_data)
        self.tr_obj.keep_ratio = old_keep_ratio  # restore old keep ratio
        return info
