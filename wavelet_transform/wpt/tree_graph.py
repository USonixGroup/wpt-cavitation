from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx

if TYPE_CHECKING:
    from wavelet_transform.wpt.transform_object import Transform


class TreeGraph:
    def __init__(self, tr_obj: "Transform"):
        self.tr_obj = tr_obj  # Transform object
        self.colour_dict = {
            -1: "skyblue",
            0: "orange",
            1: "blue",
            2: "green",
            3: "purple",
        }  # Map label number to colour

    def plot_tree(self, node_size=1500, node_colour="skyblue"):
        "Plot the tree diagram for the wavelet packet transform."
        self.graph_nodes, self.graph_edges = self._get_nodes_and_edges()
        self._generate_plot(self.graph_nodes, self.graph_edges, node_size, node_colour)

    # ---------------- Getting Nodes & Edges ----------------

    def _get_nodes_and_edges(self):
        "Get nodes and edges."
        graph_nodes = []
        graph_edges = []

        for i in range(self.tr_obj.max_level(), -1, -1):
            for node in reversed(self.tr_obj.get_level(i, order="natural")):
                node_number = node.node_number
                graph_nodes.append(((i, node_number), {"node": node}))
                graph_edges.append(((i, node_number), (i - 1, node_number // 2)))

        del graph_edges[-1]
        graph_nodes.reverse()
        graph_edges.reverse()
        return graph_nodes, graph_edges

    # ------------------ Plotting -------------------

    def _generate_plot(self, graph_nodes, graph_edges, node_size, node_colour):
        # Make networkx graph object
        G = nx.DiGraph()
        G.add_nodes_from(graph_nodes)
        G.add_edges_from(graph_edges)
        node_attributes = nx.get_node_attributes(G, "node")

        # Get colours for labels
        if isinstance(node_colour, (list)):
            n_clusters = max(node_colour) + 1
            node_colour = self._get_colours(G, node_colour)

        # Calculate node sizes
        if isinstance(node_size, str):
            node_size = [
                getattr(node_attributes[i], node_size.replace(" ", "_")) for i in G
            ]
            node_size = [1e4 / max(node_size) * i for i in node_size]

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
            font_size=40,
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
        labels = {
            node: f"""{getattr(node_attributes[node], "freq_range")}""" for node in G
        }

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
            legend_labels = {
                -1: "Unclassified",
                0: "Cluster 0",
                1: "Cluster 1",
                2: "Cluster 2",
                3: "Cluster 3",
            }
            colour_dict = {
                i: self.colour_dict[i]
                for i in list(self.colour_dict.keys())[: n_clusters + 1]
            }

            # Get relevant clusters
            legend_patches = [
                mpatches.Patch(color=color, label=legend_labels[label])
                for label, color in colour_dict.items()
            ]

            # Add legend to the plot
            plt.legend(
                handles=legend_patches,
                loc="upper right",
                title="Legend",
                title_fontsize=30,
                fontsize=30,
            )

        plt.show()

    def _get_colours(self, G, node_colour):
        labels = []
        leaf_nodes = self.tr_obj.get_leaf_nodes()
        i = 0
        for node in G:
            if nx.get_node_attributes(G, "node")[node] in leaf_nodes:
                labels.append(node_colour[i])
                i += 1
            else:
                labels.append(-1)
        node_colour = [self.colour_dict[i] for i in labels]
        return node_colour
