from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from networkx.drawing.nx_agraph import graphviz_layout, write_dot
import networkx as nx

import plotly.graph_objects as go
from wavelet_transform.utils.extensions import auto_eng_form, sci_form

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
        node_size="default",
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
        show_frequency = kwargs.get("show_frequency", False)
        use_plotly = kwargs.get("use_plotly", False)

        self.graph_nodes, self.graph_edges = self._get_nodes_and_edges(order)

        if use_plotly:
            if node_size == "default":
                node_size = 20
            self._generate_plot_plotly(
                self.graph_nodes,
                self.graph_edges,
                node_size,
                node_colour,
                freq_threshold,
                graph_legend,
                fig_name,
                show_frequency,
            )
        else:
            if node_size == "default":
                node_size = 1500
            self._generate_plot(
                self.graph_nodes,
                self.graph_edges,
                node_size,
                node_colour,
                freq_threshold,
                graph_legend,
                fig_name,
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
                    getattr(node_attributes[i], "metric")(node_size.replace(" ", "_")) for i in G
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
            width=5,
            edge_color="green",
            style="solid",
        )

        # Display frequency range of each node
        labels = {
            node: tuple(sci_form(i, 1) for i in getattr(node_attributes[node], "numeric_freq_range")) for node in G
        }

        if show_frequency is False:
            labels = {}

        offset_pos = {node: (x, y) for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(
            G,
            offset_pos,
            labels=labels,
            font_size=20,
            font_color="black",
            verticalalignment="bottom",
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
                for i in sorted(self.colour_dict.keys())[: n_clusters + 1]  # type:ignore
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

    def _generate_plot_plotly(
        self,
        graph_nodes,
        graph_edges,
        node_size,
        node_colour,
        freq_threshold,
        graph_legend,
        fig_name,
        show_frequency=False,
    ):
        # Build directed graph
        G = nx.DiGraph()
        G.add_nodes_from(graph_nodes)
        G.add_edges_from(graph_edges)
        node_attributes = nx.get_node_attributes(G, "node")

        # Node colours
        if isinstance(node_colour, list):
            node_colour = self._get_colours(G, node_colour, freq_threshold)

        # Node sizes
        big_node_size = 50
        node_size_vals = None
        if isinstance(node_size, str):
            node_size_vals = np.array(
                [getattr(node_attributes[i], "metric")(node_size.replace(" ", "_")) for i in G]
            )
            node_size_vals = (node_size_vals - np.mean(node_size_vals)) / np.std(node_size_vals) + 1.5
            node_size_vals *= big_node_size / np.max(node_size_vals)

        # Node positions using graphviz layout
        pos = graphviz_layout(G, prog="dot")
        pos = {n: (x, -y) for n, (x, y) in pos.items()}

        # Normalise pos
        x_values = [x for x, y in pos.values()]
        y_values = [y for x, y in pos.values()]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        levels = self.tr_obj.max_level()
        pos = {n: (
            (x - x_min) / (x_max - x_min) * levels,
            (y - y_min) / (y_max - y_min) * levels
        ) for n, (x, y) in pos.items()}

        # Reverse y-axis
        pos = {n: (x, levels - y) for n, (x, y) in pos.items()}

        # Edge trace
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(50,150,50,0.5)'),
            hoverinfo='none',
            mode='lines'
        )

        # Node trace
        node_x, node_y, hover_text = [], [], []
        for i, node in enumerate(G.nodes()):
            text = ""
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_obj = node_attributes.get(node)

            # Add freqeuncy info to hover text
            freq_range = ""
            if show_frequency:
                low, high = getattr(node_obj, "numeric_freq_range")
                freq_range = f"<br>Freq: {auto_eng_form(low, 2)}–{auto_eng_form(high, 2)}"

            text += f"Node {node}{freq_range}"

            # Add metric info to hover text
            if isinstance(node_size, str):
                text += (
                    f"<br>node size ({node_size}): "
                    f"{
                        auto_eng_form(
                            getattr(node_obj, 'metric')(node_size.replace(' ', '_')), 2
                        )
                    }"
                )

            hover_text.append(text)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=node_size if node_size_vals is None else node_size_vals,
                color=node_colour,
                line=dict(width=2, color='darkgreen'),
                opacity=0.9,
            ),
            hovertext=hover_text
        )

        # Build the figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=graph_legend,
                title_x=0.5,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=40),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            )
        )

        # Reverse y-axis
        fig.update_yaxes(autorange="reversed")
        # Hide x labels
        fig.update_xaxes(showticklabels=False)

        # Save to HTML or show
        if fig_name:
            fig.write_html(fig_name + ".html")
        else:
            fig.show()

        # Store for later use if needed
        self.Graph = G
        self.Graph_plotly = fig

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
