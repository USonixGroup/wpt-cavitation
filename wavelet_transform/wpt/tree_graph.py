from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from networkx.drawing.nx_agraph import graphviz_layout, write_dot
import networkx as nx

from wavelet_transform.utils.extensions import sci_form

if TYPE_CHECKING:
    from wavelet_transform.wpt.transform_object import Transform


class TreeGraph:
    colour_dict = {
        -1: "skyblue",
        0: "orange",
        1: "brown",
        2: "green",
        3: "purple",
        4: "cyan",
        5: "red",
        6: "yellow",
        7: "magenta",
        8: "pink",
        9: "lime",
        10: "navy",
    }  # Map label number to colour

    def __init__(self, tr_obj: "Transform"):
        self.tr_obj = tr_obj  # Transform object

    def plot_tree(
        self,
        node_size=1500,
        node_colour="skyblue",
        freq_threshold=None,
        graph_legend="Graph legend",
        fig_name=None,
        **kwargs,
    ):
        "Plot the tree diagram for the wavelet packet transform."

        if not isinstance(node_colour, (str, list)):
            raise ValueError("node_colour must be str or list")

        order = kwargs.get("order", "natural")
        sampling_rate = kwargs.get("sampling_rate")
        show_frequency = kwargs.get("show_frequency", False)

        self.graph_nodes, self.graph_edges = self._get_nodes_and_edges(order)
        self._generate_plot(
            self.graph_nodes,
            self.graph_edges,
            node_size,
            node_colour,
            freq_threshold,
            graph_legend,
            fig_name,
            sampling_rate,
            show_frequency,
        )

    # ---------------- Getting Nodes & Edges ----------------

    def _get_nodes_and_edges(self, order):
        "Get nodes and edges."
        graph_nodes = []
        graph_edges = []

        for i in range(self.tr_obj.max_level(), -1, -1):
            for node in reversed(self.tr_obj.get_level(i, order=order)):
                node_number = node.node_number
                graph_nodes.append(((i, node_number), {"node": node}))
                graph_edges.append(((i, node_number), (i - 1, node_number // 2)))

        del graph_edges[-1]
        graph_nodes.reverse()
        graph_edges.reverse()
        return graph_nodes, graph_edges

    # ------------------ Plotting -------------------

    def _generate_plot(
        self,
        graph_nodes,
        graph_edges,
        node_size,
        node_colour,
        freq_threshold,
        graph_legend,
        fig_name,
        sampling_rate=None,
        show_frequency=False,
    ):
        # Make networkx graph object
        G = nx.DiGraph()
        G.add_nodes_from(graph_nodes)
        G.add_edges_from(graph_edges)
        node_attributes = nx.get_node_attributes(G, "node")

        # Get colours for labels
        if isinstance(node_colour, (list)):
            n_clusters = max(node_colour) + 1
            node_colour = self._get_colours(G, node_colour, freq_threshold)

        # Calculate node sizes
        if isinstance(node_size, str):
            node_size = np.array(
                list(
                    getattr(node_attributes[i], node_size.replace(" ", "_")) for i in G
                )
            )
            node_size = (node_size - np.mean(node_size)) / np.std(node_size) + 1.5
            node_size *= 1e4 / np.max(node_size)

        # Plot figure
        fig, ax = plt.subplots(1, 1, figsize=(40, 20))
        pos = graphviz_layout(G, prog="dot")
        pos = {node: (x, -y) for node, (x, y) in pos.items()}  # Negate y-coordinates
        nx.draw(
            G,
            pos=pos,
            ax=ax,
            arrowsize=5,
            verticalalignment="top",
            font_size=35,
            font_color="red",
            with_labels=True,
            node_size=node_size,
            node_color=node_colour,
            node_shape="s",
            alpha=0.8,
            linewidths=4,
            width=4,
            edge_color="green",
            style="solid",
        )

        # Display frequency range of each node
        sampling_rate = 1 if sampling_rate is None else sampling_rate
        labels = {
            node: tuple(
                sci_form(i * sampling_rate, 1)
                for i in getattr(node_attributes[node], "norm_frequency_range")
            )
            for node in G
        }

        if show_frequency is False:
            labels = {}

        offset_pos = {node: (x, y - 20) for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(
            G,
            offset_pos,
            labels=labels,
            font_size=20,
            font_color="black",
            verticalalignment="top",
        )

        # Add legend for clusters
        if isinstance(node_colour, list):
            # Create legend labels
            legend_labels = dict(
                (i, f"Cluster {i}") for i in range(len(node_colour) + 1)
            )
            legend_labels[-1] = "Unclassified"

            colour_dict = {
                i: self.colour_dict[i]
                for i in sorted(self.colour_dict.keys())[: n_clusters + 1]
            }

            # Get relevant clusters
            legend_patches = [
                mpatches.Patch(color=color, label=legend_labels[label])
                for label, color in colour_dict.items()
            ]

            # Add legend to the plot
            plt.legend(
                handles=legend_patches,
                loc="best",
                title=graph_legend,
                title_fontsize=40,
                fontsize=40,
            )

        self.Graph = G
        self.Graph_pydot = nx.nx_pydot.to_pydot(G)
        if fig_name:
            plt.savefig(fig_name + ".png", format="PNG")
            plt.savefig(fig_name + ".pdf", format="PDF")
            write_dot(G, fig_name + ".dot")

    def _get_colours(self, G, node_colour, freq_threshold):
        labels = []
        leaf_nodes = self.tr_obj.get_leaf_nodes(freq_threshold=freq_threshold)

        i = 0
        nodes = [node for node in G]
        node_attributes = nx.get_node_attributes(G, "node")

        for node in nodes:
            if node_attributes[node] in leaf_nodes:
                labels.append(node_colour[i])
                i += 1
            else:
                labels.append(-1)
        node_colour = [self.colour_dict[i] for i in labels]
        return node_colour
