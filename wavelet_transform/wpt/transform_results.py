from typing import TYPE_CHECKING, Optional, Union
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
        level: int, None, or "leaf"
            the level of transform to display information for. If `level` is
            `None`, uses the transform's max level. If `level` is the string
            `'leaf'`, information for leaf nodes is returned.
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
        max_level = self.tr_obj.max_level()

        # Accept integer levels, None (meaning max level), or the string 'leaf'
        if level == "leaf":
            nodes = self.tr_obj.get_leaf_nodes(order=order)
        else:
            # treat None as max level
            if level is None:
                level_to_use = self.tr_obj.max_level()
            else:
                if not isinstance(level, int):
                    raise ValueError("level must be an int, None, or 'leaf'")
                level_to_use = level
            if level_to_use > max_level:
                raise ValueError(
                    f"transform has no level {level_to_use}. max level is: {max_level}"
                )
            nodes = self.tr_obj.get_level(level_to_use, order=order)
        info_data = []
        short_node_len = len(nodes[0].data_keep)

        old_keep_ratio = self.tr_obj.keep_ratio  # store old keep ratio
        self.tr_obj.keep_ratio = keep_ratio  # temporarily set keep ratio
        for node in nodes:
            if not node.is_in_threshold(freq_threshold):
                continue
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
    
    @property
    def ref_psd(self):
        """Return the reference PSD for spectral metrics."""
        from wavelet_transform.utils.extensions import psd
        if not hasattr(self, "_ref_psd"):
            data = self.tr_obj.get_node(0, 0).data
            self._ref_psd = psd(data)
        return self._ref_psd