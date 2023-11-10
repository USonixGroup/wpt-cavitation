from wavelet_transform.utils.extensions import sci_form, fft
import wavelet_transform.wpt.inverse_transform as inv
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from wavelet_transform.wpt.tree_graph import TreeGraph

FS = 15  # fontsize


def full_plot(nodes, data, t_plot, sampling_rate, title="", frequencies=None):
    """
    formatted charts for plotting the results from a dwt or dpwt. Combines
    relevant plots including those for the sound wave, frequency content of
    the sonud wave (using synthetic data package), and plots of the data at
    each node.

    Parameters
    ----------
    nodes: list
        list of node objects
    data: array_like
        an array of the data from the sound wave
    t_plot: array_like
        the time array to plot the data against
    nyquist_frequency: float
        1/2*sampling rate
    frequencies: list
        list of floats containing the frequencies of the synthetic sound wave
        for the historgram
    """
    # plot data
    fig_1, ax_1 = plt.subplots()
    plot_wave(data, t_plot, ax_1, title)

    # plot transform coefficients all at once
    if frequencies:  # plot a histogram of frequency content in data as well
        fig_2, ax_2 = plt.subplots(2)
        plot_all_final_nodes(nodes, ax_2[0], title)
        plot_frequencies(frequencies, ax_2[1], title)
    else:
        fig_2, ax_2 = plt.subplots()
        plot_all_final_nodes(nodes, ax_2, title)

    # plot each node separately.
    rows = round(len(nodes) / 2)
    fig_3, ax_3 = plt.subplots(rows, 2)
    fig_3.set_size_inches(12, 3 * rows)
    plot_nodes(nodes, ax_3, sampling_rate, rows, columns=2)

    fig_1.tight_layout()
    fig_2.tight_layout()
    fig_3.tight_layout()
    plt.show()


def plot_nodes(nodes, ax, sampling_rate, rows=None, columns=1):
    """
    separately plots the data from a provided list of nodes.

    Parameters
    ----------
    nodes: list
        list of nodes to plot the data for
    ax: axis
        ax from fig,ax = plt.subplots()
    nyquist_frequency:  float
        1/2*sampling rate
    rows: int
        the number of rows to create
    columns: int
        the number of columns to create
    """
    if not rows:  # default number of rows
        rows = round(len(nodes))
    section_counter = 1

    nyquist_frequency = sampling_rate / 2
    for row in range(rows):
        for column in range(columns):
            # plot coefficients
            try:
                node = nodes[section_counter - 1]
            except IndexError:
                break
            if rows == 1:
                axis = ax[column]
            elif columns == 1:
                axis = ax[row]
            else:
                axis = ax[row, column]
            axis.plot(node.data)

            # set labels
            axis.set_xlabel("coefficient number", fontsize=FS)
            axis.set_ylabel("magnitude", fontsize=FS)
            axis.grid(True)

            lower_frequency = node.norm_frequency_range[0] * nyquist_frequency
            upper_frequency = node.norm_frequency_range[1] * nyquist_frequency

            lf = sci_form(lower_frequency)
            uf = sci_form(upper_frequency)

            n = len(nodes)

            title = f"""
            Coefficients, {lf}-{uf} Hz
            energy: {sci_form(node.energy)} [section {section_counter} of {n}]
            """
            axis.set_title(title, fontsize=FS)
            section_counter += 1


def plot_all_final_nodes(nodes, ax, title=""):
    "concatenate the data from all of the nodes and plot"
    output = np.array([])
    for node in nodes:
        output = np.concatenate((output, node.data))
    ax.plot(output)
    ax.set_title(f"Coefficients {title}", fontsize=FS)
    ax.set_xlabel("Coefficient Number", fontsize=FS)
    ax.set_ylabel("Coefficient", fontsize=FS)
    ax.grid(True)


def plot_wave(data, t_plot, ax, title=""):
    "plots the data"
    ax.plot(t_plot, data)
    ax.set_xlabel("Time [s]", fontsize=FS)
    ax.set_ylabel("Amplitude [mV]", fontsize=FS)
    ax.set_title(f"Data {title}", fontsize=FS)
    ax.grid(True)


def plot_frequencies(frequencies, ax, title=""):
    "creates a histogram of the frequencies present in the synthetic data"
    if frequencies:
        ax.hist(frequencies)
        ax.set_title(f"Frequency Distribution {title}", fontsize=FS)
        ax.set_xlabel("Frequency [Hz]", fontsize=FS)
        ax.set_ylabel("Frequency of frequency", fontsize=FS)
        ax.grid(True)


def plot_energy_vs_freq(transform_obj, sampling_rate=None, ax=None, title=""):
    "plot the energy vs frequency for each packet in the last transform level"
    if sampling_rate:
        transform_obj.calc_frequencies(sampling_rate)
    nodes = transform_obj.get_level(order="freq")
    energy = transform_obj.info(order="freq")["energy"]
    frequencies = []
    for node in nodes:
        freq_range = node.frequency_range
        frequencies.append(1 / 2 * (freq_range[0] + freq_range[1]))
    if ax:
        ax.plot(frequencies, energy)
        ax.set_title(f"Energy vs Subband Frequency {title}", fontsize=FS)
        ax.set_xlabel("Frequencies [Hz]", fontsize=FS)
        ax.set_ylabel("Energy (arbitrary)", fontsize=FS)
    else:
        plt.plot(frequencies, energy)
        plt.title(f"Energy vs Subband Frequency {title}", fontsize=FS)
        plt.xlabel("Frequencies [Hz]", fontsize=FS)
        plt.ylabel("Energy (arbitrary)", fontsize=FS)


def heisenberg(transform_object, t_end=1):
    """
    calls imshow to create a heisenberg plot of the wavelet packet

    Parameters
    ----------
    transform_object: object
        contains data from the transform to be plotted
    t_end: float
        the time the signal ends at
    x_tick_space: int
        formats the x axis such that the ticks are more spread out
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 9)

    nodes = transform_object.get_leaf_nodes(order="freq")
    data = np.zeros((len(nodes), len(nodes[0].data)))
    for i, j in enumerate(range(len(nodes) - 1, -1, -1)):  # maybe reverse
        node = nodes[j]
        data[i, :] = np.abs(node.data)  # plot absolute values of wavelet coefficients

    graph = ax.imshow(np.log10(data), cmap="turbo", aspect="auto")
    ax.set_ylabel("Frequency of Subband [Hz]", fontsize=20)
    ax.set_xlabel("Time [s]", fontsize=20)
    ax.set_title("DWPT of Signal", fontsize=25)

    frequencies = [node.frequency_range for node in nodes]
    if any((freq is None for freq in frequencies)):
        raise ValueError(
            "Check that the frequencies have been calculated ->"
            " transform_obj.calc_frequencies()"
        )
    y_labels = [
        (sci_form(freq[0], 1), sci_form(freq[1], 1)) for freq in reversed(frequencies)
    ]
    y_space = max(len(frequencies) // 16, 1)
    sparse_y_labels = [
        label if not id % y_space else "" for id, label in enumerate(y_labels)
    ]
    x_labels = np.linspace(0, t_end, len(node.data))
    x_labels = [sci_form(x, 1) for x in x_labels]
    x_space = int(len(x_labels) / 10)
    sparse_x_labels = [
        label if id % x_space == 0 else "" for id, label in enumerate(x_labels)
    ]
    all_x_ticks = [i for i in range(len(x_labels))]
    all_y_ticks = [i for i in range(len(y_labels))]
    ax.set_xticks(all_x_ticks, sparse_x_labels, fontsize=15)
    ax.set_yticks(all_y_ticks, sparse_y_labels, fontsize=15)

    fig.tight_layout()
    cbar = fig.colorbar(graph)
    cbar.set_label(label="Coefficient amplitude", size=20)

    plt.show()


# ---------------- Plot Reconstructed Data & FFTs ---------------


def get_reconstructions_by_group(tr_ob, labels, n_clusters, freq_threshold):
    signal_length = len(tr_ob.data)
    # List reconstructions by group
    reconstructions = [np.zeros(signal_length) for _ in range(n_clusters)]

    # Sum all reconstructions belonging to the same group
    for node, assignment in zip(
        tr_ob.get_leaf_nodes(freq_threshold=freq_threshold), labels
    ):  # natural order
        rec_signal = inv.inv_dwpt(tr_ob, level=node.level, nodes=node)
        i = assignment
        reconstructions[i] = reconstructions[i] + rec_signal

    return reconstructions


def plot_reconstructions_by_group(
    transform_object, labels, no_clusters, t_plot=None, freq_threshold=None
):
    """
    Reconstruct each node from the level of transform used to train the clustering algorithm
    and plot the summation of those belonging to the same cluster.

    Assumes that the labels are arranged in frequency order by default. Always uses leaf nodes.

    Parameters
    ----------
    transform_object: Transform
        transform object from this library
    labels: list
        list of labels from training. Must be a power of 2
    no_clusters: int
        the number of clusters used. If set to None, the number of clusters will be
        derived from the number of unique elements in the list of labels.
    t_plot: array-like, optional
        time array for plotting
    freq_threshold: float, optional
        frequency threshold for reconstruction

    Returns
    -------
    fig: plotly.graph_objects.Figure
        The Plotly figure object.
    """
    if no_clusters is None:
        no_clusters = len(set(labels))

    reconstructions = get_reconstructions_by_group(
        transform_object, labels, no_clusters, freq_threshold
    )

    fig = go.Figure()

    if t_plot is None:
        xvals = len(transform_object.data)
    else:
        xvals = t_plot

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=transform_object.data,
            mode="lines",
            name="Original signal",
        )
    )

    colours = TreeGraph.colour_dict  # Get colours for each cluster from tree graph

    # Iterate through clusters
    for i in range(no_clusters):
        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=reconstructions[i],
                mode="lines",
                name=f"Cluster {i}",
                line=dict(color=colours[i]),
            )
        )

    fig.update_layout(
        title="Reconstructed nodes by group",
        xaxis_title="Time [s]" if t_plot is not None else None,
        yaxis_title="Amplitude",
        legend=dict(title="Clusters"),
    )

    fig.show()

    return fig


def plot_fft_by_group(
    transform_object,
    labels,
    no_clusters,
    sampling_rate,
    clusters=None,
    xlim=None,
    fundamental_freq=None,
    freq_threshold=None,
):
    """
    Plot FFTs of reconstructed signals and original signal

    Paramters (incomplete)
    ---------
    clusters: list of int
        a list of clusters to plot.
    """
    if isinstance(clusters, int):
        clusters = [clusters]

    if no_clusters is None:
        no_clusters = len(set(labels))

    reconstructions = get_reconstructions_by_group(
        transform_object, labels, no_clusters, freq_threshold
    )

    fig = go.Figure()

    # FFT of original signal
    fft_original, freq = fft(transform_object.data, sampling_rate, fundamental_freq)

    fig.add_trace(
        go.Scatter(
            x=freq,
            y=20 * np.log10(np.abs(fft_original)),
            mode="lines",
            name="Original Signal",
            line=dict(color="blue"),
        )
    )

    # Plot FFTs
    if clusters is None:
        clusters = range(no_clusters)

    colours = TreeGraph.colour_dict  # Get colours for each cluster from tree graph

    for i in clusters:
        fft_rec = fft(reconstructions[i])
        fig.add_trace(
            go.Scatter(
                x=freq,
                y=20 * np.log10(np.abs(fft_rec)),
                mode="lines",
                name=f"Cluster {i}",
                line=dict(color=colours[i]),
            )
        )

    fig.update_layout(
        title="FFT Comparison",
        xaxis_title="Frequency [Hz]" if fundamental_freq is None else "f / f_0 [DL]",
        yaxis_title="Amplitude [dB]",
        legend=dict(title="Clusters"),
        xaxis=dict(range=[0, xlim] if xlim is not None else None),
    )

    fig.show()

    return fig
