from wavelet_transform.utils.extensions import sci_form, psd, to_dB, eng_form_plot
import wavelet_transform.wpt.inverse_transform as inv
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from itertools import cycle
from wavelet_transform.wpt.tree_graph import TreeGraph

fsize = 24  # font size
tick_size = 24  # tick size
cl_label_fsize = 20  # clusters labels font size
margine_value = 0  # zero margine for plotly plots
margine_top_value = 50


def plot_nodes(nodes, ax, sampling_rate, rows=None, columns=1):
    """
    Separately plots the data from a provided list of nodes.

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
            axis.set_xlabel("coefficient number", fontsize=fsize)
            axis.set_ylabel("magnitude", fontsize=fsize)
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
            axis.set_title(title, fontsize=fsize)
            section_counter += 1


def plot_all_final_nodes(nodes, ax, title=""):
    "Concatenate the data from all of the nodes and plot"
    output = np.array([])
    for node in nodes:
        output = np.concatenate((output, node.data))
    ax.plot(output)
    ax.set_title(f"Coefficients {title}", fontsize=fsize)
    ax.set_xlabel("Coefficient Number", fontsize=fsize)
    ax.set_ylabel("Coefficient", fontsize=fsize)
    ax.grid(True)


def plot_wave(data, t_plot, ax, title=""):
    "Plots the data"
    ax.plot(t_plot, data)
    ax.set_xlabel("Time [s]", fontsize=fsize)
    ax.set_ylabel("Magnitude [mV]", fontsize=fsize)
    ax.set_title(f"Data {title}", fontsize=fsize)
    ax.grid(True)


def plot_frequencies(frequencies, ax, title=""):
    "Creates a histogram of the frequencies present in the synthetic data"
    if frequencies:
        ax.hist(frequencies)
        ax.set_title(f"Frequency Distribution {title}", fontsize=fsize)
        ax.set_xlabel("Frequency [Hz]", fontsize=fsize)
        ax.set_ylabel("Frequency of frequency", fontsize=fsize)
        ax.grid(True)


def plot_energy_vs_freq(transform_obj, sampling_rate=None, ax=None, title=""):
    "Plot the energy vs frequency for each packet in the last transform level"
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
        ax.set_title(f"Energy vs Subband Frequency {title}", fontsize=fsize)
        ax.set_xlabel("Frequencies [Hz]", fontsize=fsize)
        ax.set_ylabel("Energy (arbitrary)", fontsize=fsize)
    else:
        plt.plot(frequencies, energy)
        plt.title(f"Energy vs Subband Frequency {title}", fontsize=fsize)
        plt.xlabel("Frequencies [Hz]", fontsize=fsize)
        plt.ylabel("Energy (arbitrary)", fontsize=fsize)


def heisenberg(
    transform_object,
    t_end=1,
    max_freq=1,
    padding="interpolate",
    use_log_scale=True,
    title="DWPT of signal",
    vmin=None,
    vmax=None,
    color_map="seismic",
):
    """
    Calls imshow to create a heisenberg plot of the wavelet packet leaf nodes.

    Parameters
    ----------
    transform_object: object
        contains data from the transform to be plotted
    t_end: float
        the time the signal ends at
    max_freq: float
        the maximum frequency to plot (normalised by nyquist)
    padding: str
        can be set to "interpolate" or "repeat" to interpolate time series data
        or just repeat it when dealing with packets of different levels.
    use_log_scale: bool
        set to True to use log10 scale for plot.
    title_arg: str
        figure title
    vmax: float
        the maximum value displayed in the colorbar
    vmin: float
        the minimum value displayed in the colorbar
    """
    if not isinstance(use_log_scale, bool):
        raise ValueError("use_log_scale must be bool")

    if padding not in ["interpolate", "repeat"]:
        raise ValueError("padding must be 'interpolate' or 'repeat'")

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 9)

    nodes = transform_object.get_leaf_nodes(order="freq", freq_threshold=max_freq)

    if any((not node.is_freq_calculated() for node in nodes)):
        raise ValueError(
            "Check that the frequencies have been calculated ->"
            " transform_obj.calc_frequencies()"
        )

    n_nodes = len(nodes)
    node_lengths = (len(nodes[i].data) for i in range(n_nodes))
    len_data = max(node_lengths)

    # Find which nodes to plot in each row
    i = 0
    max_level = transform_object.max_level()
    nodes_per_row = []
    for node in nodes:
        for _ in range(int(2 ** (max_level - node.level))):
            nodes_per_row.append(node)
            i += 1

    pad_func = None
    if padding == "interpolate":
        pad_func = interpolate_data
    elif padding == "repeat":
        pad_func = repeat_data
    # Add node data to a matrix
    data = np.zeros((len(nodes_per_row), len_data))
    for i, node in enumerate(reversed(nodes_per_row)):
        node_data = node.data
        padded_data = pad_func(node_data, len_data)
        data[i, :] = np.abs(padded_data)  # plot abs values of wavelet coefficients

    # Plot graph
    frequencies = [1 / 2 * sum(node.numeric_freq_range) for node in nodes_per_row]
    graph = ax.imshow(
        data if use_log_scale is False else np.log10(data),
        cmap=color_map,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_ylabel("$f_c$ [MHz]", fontsize=fsize)
    ax.set_xlabel("t [$\mathrm{\mu s}$]", fontsize=fsize)  # noqa
    ax.set_title(title, fontsize=fsize)
    # ax.set_yscale("log")

    # Generate axis tick labels

    y_labels = [
        eng_form_plot(freq, 6, 1)  # (sci_form(freq[0], 1), sci_form(freq[1], 1))
        for freq in reversed(frequencies)
    ]
    y_space = max(len(frequencies) // 16, 1)
    sparse_y_labels = [
        label if not id % y_space else "" for id, label in enumerate(y_labels)
    ]
    sparse_y_labels = _unrepeat(sparse_y_labels)
    x_labels = np.linspace(0, t_end, len_data)
    x_labels = [eng_form_plot(x, -6, 1) for x in x_labels]
    x_space = max(int(len(x_labels) / 10), 1)
    sparse_x_labels = [
        label if id % x_space == 0 else "" for id, label in enumerate(x_labels)
    ]
    sparse_x_labels = sparse_x_labels
    all_x_ticks = [i for i in range(len(x_labels))]
    all_y_ticks = [i for i in range(len(y_labels))]

    # Set ticks
    ax.set_xticks(all_x_ticks, sparse_x_labels, fontsize=tick_size)
    ax.set_yticks(all_y_ticks, sparse_y_labels, fontsize=tick_size)

    # Formatting
    cbar_label = (
        "Magnitude of DWPT coeffs [dB]"
        if use_log_scale is True
        else "Magnitude of DWPT coeffs"
    )
    fig.tight_layout()
    cbar = fig.colorbar(graph, pad=0.01)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.set_label(label=cbar_label, size=fsize)

    plt.show()
    return fig


def _unrepeat(array):
    "Remove repeated axis tick labels"
    x = ""
    indices = []
    count = 1
    for i in range(len(array)):
        y = array[i]
        if y == "" and i != len(array) - 1:
            continue
        if y == x:
            count += 1
            indices.append(i)
        else:
            if count > 1:
                index = int(sum(indices) / len(indices))
                for n in range(indices[0], indices[-1] + 1):
                    if n != index:
                        array[n] = ""
                    else:
                        array[n] = x
            count = 1
            indices = [i]
            x = y
    return array


def repeat_data(data, target_length):
    """
    Pad the data for nodes containing too few elements. Fill in remaining
    elements evenly, starting from the beginning.

    """
    len_data = len(data)

    # Find length to pad
    if target_length <= len_data:
        return data
    repetitions = target_length // len_data
    remaining_elements = target_length % len_data

    # Pad data evenly
    repetitions_per_element = repetitions + (np.arange(len_data) < remaining_elements)
    padded_data = np.repeat(data, repetitions_per_element)

    return padded_data


def interpolate_data(data, length):
    "Interpolate data to desired length"
    original_time = np.arange(len(data))
    new_time = np.linspace(0, len(data) - 1, length)

    interp_func = interp1d(original_time, data, kind="linear")  # 'linear' interpolation
    interpolated_signal = interp_func(new_time)

    return interpolated_signal


# ---------------- Plot Reconstructed Data & PSDs ---------------


def get_label(label_no, label_map):
    if isinstance(label_map, dict):
        return label_map.get(label_no)

    elif label_map is None:
        return f"Cluster {label_no}"

    else:
        raise ValueError("label_map must be a dictionary of label_no: label")


def get_reconstructions_by_group(tr_obj, labels, freq_threshold=1):
    signal_length = len(tr_obj.data)
    n_clusters = len(set(labels))
    # List reconstructions by group
    reconstructions = [np.zeros(signal_length) for _ in range(n_clusters)]

    # Sum all reconstructions belonging to the same group
    for node, assignment in zip(
        tr_obj.get_leaf_nodes(freq_threshold=freq_threshold), labels
    ):  # natural order
        rec_signal = inv.inv_dwpt(tr_obj, level=node.level, nodes=node)
        i = assignment  # shorten name for readability
        reconstructions[i] = reconstructions[i] + rec_signal

    return reconstructions


def plot_reconstructions_by_group(
    transform_object,
    labels,
    t_plot=None,
    freq_threshold=None,
    label_map=None,
    title="Reconstructed nodes by group",
    fig_height=400,
    fig_width=1000,
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
    t_plot: dict, array-like, float, int
        time array for plotting. if this is a float or int, it will be assumed to represent the
        sampling rate in sample/s. if it is an array, it is assumed it is time array in s. if it is a dict, the key of the dict is used for the label and the value MUST be an array for the axis.
    freq_threshold: float, optional
        frequency threshold for reconstruction
    label_map: dict int: str
        relabel cluster numbers to a string

    Returns
    -------
    fig: plotly.graph_objects.Figure
        The Plotly figure object.
    """
    n_clusters = len(set(labels))

    reconstructions = get_reconstructions_by_group(
        transform_object, labels, freq_threshold=1
    )

    colours = TreeGraph.colour_dict  # Get colours for each cluster from tree graph
    lines = ["solid", "dashdot", "solid", "dot", "dash", "longdash", "longdashdot"]
    linestyles = cycle(lines)
    line_width = [3] * n_clusters + [1]

    fig = go.Figure()

    data_len = len(transform_object.data)
    if t_plot is None:
        xvals = np.arange(data_len)
        x_label = ["x"]
    elif isinstance(t_plot, (np.ndarray)):
        data_len = len(t_plot)
        xvals = t_plot.copy()
        xvals *= 10**6  # get time in microseconds
        x_label = "t [&mu;s]"
    elif isinstance(t_plot, (float, int)):
        xvals = np.arange(0, data_len / t_plot, 1 / t_plot)
        xvals *= 10**6  # get time in microseconds
        x_label = "t [&mu;s]"
    elif isinstance(t_plot, dict):
        x_label = list(t_plot.keys())[0]
        xvals = list(t_plot.values())[0]
        data_len = len(xvals)
    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=transform_object.data[0:data_len],
            mode="lines",
            name="Original signal",
            line=dict(color="blue", dash="solid", width=line_width[-1]),
        )
    )

    # Iterate through clusters
    for i in range(n_clusters):
        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=reconstructions[i][0:data_len],
                mode="lines",
                name=get_label(i, label_map),
                line=dict(color=colours[i], dash=next(linestyles), width=line_width[i]),
            )
        )

    fig.update_layout(
        width=fig_width,
        height=fig_height,
        font_size=fsize,
        title=title,
        xaxis=dict(title=x_label, tickfont=dict(size=tick_size)),
        yaxis=dict(title="Magnitude", tickfont=dict(size=tick_size)),
        legend=dict(title="", traceorder="normal", font=dict(size=fsize)),
        plot_bgcolor="white",
        margin=go.layout.Margin(
            l=margine_value,  # left margin
            r=margine_value,  # right margin
            b=margine_value,  # bottom margin
            t=margine_top_value,  # top margin
        ),
    )

    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )
    fig.show()

    return fig


def plot_fft_by_group(
    transform_object,
    labels,
    sampling_rate,
    clusters=None,
    xlims=None,
    ylims=None,
    fundamental_freq=None,
    freq_threshold=None,
    label_map=None,
    title="PSD comparison",
    fig_height=400,
    fig_width=1000,
    plot_full_reconstruction=False,
):
    """
    Plot PSDs of reconstructed signals and original signal.

    Paramters (incomplete)
    ---------
    clusters: list of int
        a list of clusters to plot.
    """
    if isinstance(clusters, int):
        clusters = [clusters]

    n_clusters = len(set(labels))

    reconstructions = get_reconstructions_by_group(
        transform_object, labels, freq_threshold=1
    )

    colours = TreeGraph.colour_dict  # Get colours for each cluster from tree graph
    lines = ["dot", "dashdot", "solid", "dot", "dash", "longdash", "longdashdot"]
    linestyles = cycle(lines)
    line_width = [3] * n_clusters + [1]

    fig = go.Figure()

    # PSD of original signal
    if not isinstance(sampling_rate, (float, int)):
        raise ValueError("sampling rate must be provided as a float or int")
    psd_original, freq = psd(transform_object.data, sampling_rate, fundamental_freq)

    fig.add_trace(
        go.Scatter(
            x=freq,
            y=10 * np.log10(psd_original),
            mode="lines",
            name="Original Signal",
            line=dict(color="blue", dash="solid", width=line_width[-1]),
        )
    )

    # Plot PSDs
    if clusters is None:
        clusters = range(n_clusters)

    for i in clusters:
        psd_rec = psd(reconstructions[i])
        fig.add_trace(
            go.Scatter(
                x=freq,
                y=10 * np.log10(psd_rec),
                mode="lines",
                name=get_label(i, label_map),
                line=dict(color=colours[i], dash=next(linestyles), width=line_width[i]),
            )
        )

    if plot_full_reconstruction is True:
        fig.add_trace(
            go.Scatter(
                x=freq,
                y=10 * np.log10(psd(sum(reconstructions))),
                mode="lines",
                name="full reconstruction",
                line=dict(dash="dot", color="blue", width=line_width[-1]),
            )
        )

    fig.update_layout(
        width=fig_width,
        height=fig_height,
        font_size=fsize,
        title=title,
        xaxis=dict(
            title=(
                "Frequency [Hz]"
                if fundamental_freq is None
                else "f / f<sub>0</sub> [DL]"
            ),
            tickfont=dict(size=tick_size),
            range=xlims if xlims is not None else None,
        ),
        yaxis=dict(title="Magnitude [dB]", tickfont=dict(size=tick_size), range=ylims),
        legend=dict(title="", traceorder="normal", font=dict(size=fsize)),
        plot_bgcolor="white",
        margin=go.layout.Margin(
            l=margine_value,  # left margin
            r=margine_value,  # right margin
            b=margine_value,  # bottom margin
            t=margine_top_value,  # top margin
        ),
    )

    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )
    fig.show()

    return fig


def get_reconstructions_by_threshold(tr_obj, feature, threshold, freq_threshold=1):
    """
    Thresholds nodes according to a given feature and returns the total
    reconstructions of nodes above and below the threshold.

    Parameters
    ----------
    tr_obj: Transform
        object containing nodes
    feature: str or func
        str - the chosen feature calculated on the wavelet coefficients (see Node class for options)
        func - calculate the feature on the reconstructed signal
    threshold: float
        the numerical threshold to sort the nodes by
    freq_threshold: float
        ignore all nodes above a given frequency threshold

    Returns
    -------
    reconstructions: list of np.array
        list containing 2 np.arrays. These are the reconstructed signals above
        and below the threshold respectively
    """
    if not isinstance(feature, str) and not callable(feature):
        raise ValueError("feature must be a string or a function")
    signal_length = len(tr_obj.data)
    # List reconstructions
    reconstructions = [np.zeros(signal_length) for _ in range(2)]

    # Sum all reconstructions belonging to the same group
    for node in tr_obj.get_leaf_nodes(freq_threshold=freq_threshold):  # natural order
        rec_signal = inv.inv_dwpt(tr_obj, level=node.level, nodes=node)
        _feature = (
            getattr(node, feature) if isinstance(feature, str) else feature(rec_signal)
        )
        i = 0 if _feature > threshold else 1  # shorten name for readability
        reconstructions[i] = reconstructions[i] + rec_signal

    return reconstructions


# ---------------------- Clustering Results ----------------------


def plot_clustering(
    data_train,
    centers,
    labels,
    metrics_to_plot,
    title="Clustering results",
    label_map=None,
    levels=None,
):
    colours = list(TreeGraph.colour_dict.values())[1:]

    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig = plt.figure(figsize=(1300 * px, 800 * px))
    ax = fig.add_subplot(projection="3d")

    # Get level markers
    if levels is not None:
        if not isinstance(levels, list):
            raise ValueError("levels must be a list")
        if len(levels) != data_train.shape[0]:
            raise ValueError("Number of levels doesn't match number of nodes")
        unique_levels = sorted(set(levels))
        level_markers = _get_level_markers()
        if len(unique_levels) > len(level_markers):
            raise ValueError(
                f"too many unique levels ({len(unique_levels)}), \
                max supported ({len(level_markers)})"
            )
        max_level = max(unique_levels)
        level_markers = {max_level - k: v for k, v in level_markers.items()}
    else:
        level_markers = {0: "o"}
        levels = [0 for _ in range(data_train.shape[0])]
        unique_levels = [0]

    # Make scatter plots
    legend_elements = []
    for center in range(centers.shape[0]):
        for level in unique_levels:  # iterate  through level markers
            mask = (labels == center) & (np.array(levels) == level)
            scatter = ax.scatter(
                data_train.loc[mask, metrics_to_plot[0]],
                data_train.loc[mask, metrics_to_plot[1]],
                data_train.loc[mask, metrics_to_plot[2]],
                c=colours[center],
                marker=level_markers[level],
                label=f"{level}",
            )
            if center == 0:
                legend_elements.append(scatter)

        ax.scatter(
            centers.loc[center, metrics_to_plot[0]],
            centers.loc[center, metrics_to_plot[1]],
            centers.loc[center, metrics_to_plot[2]],
            c="red",
            marker="x",
            s=100,
        )

        ax.text(
            centers.loc[center, metrics_to_plot[0]],
            centers.loc[center, metrics_to_plot[1]],
            centers.loc[center, metrics_to_plot[2]],
            get_label(center, label_map),
            color="black",
            fontsize=cl_label_fsize,
        )

        ax.set_xlabel(metrics_to_plot[0].capitalize(), fontsize=fsize)
        ax.set_ylabel(metrics_to_plot[1].capitalize(), fontsize=fsize)
        ax.set_zlabel(metrics_to_plot[2].capitalize(), fontsize=fsize)
        ax.set_title(title, fontsize=fsize)

    # a2s = lambda xarray: ["%.2f" % x for x in xarray]
    # x_ticks = np.linspace(min(centers[metrics_to_plot[0]]), 10*max(centers[metrics_to_plot[0]]),10)
    # y_ticks = np.linspace(min(centers[metrics_to_plot[1]]), 10*max(centers[metrics_to_plot[1]]),10)
    # z_ticks = np.linspace(min(centers[metrics_to_plot[2]]), max(centers[metrics_to_plot[2]]),10)
    # ax.set_xticks(x_ticks,a2s(x_ticks),fontsize=tick_size)
    # ax.set_yticks(y_ticks,a2s(y_ticks),fontsize=tick_size)
    # ax.set_zticks(z_ticks,a2s(z_ticks),fontsize=tick_size)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")
    ax.set_box_aspect(None, zoom=0.92)
    if not all(i == 0 for i in levels):
        ax.legend(handles=legend_elements, title="Node level")
    fig.tight_layout()
    plt.show()

    return fig


def _get_level_markers():
    level_markers = {
        0: "o",
        1: "+",
        2: "^",
        3: "D",
        4: "s",
        5: "*",
    }
    return level_markers


def plot_centers(classifying_metrics, centers, label_map=None, xlabel_user=None):
    "Plot the clustering centres for one signal"
    n_clusters = centers.shape[0]
    colours = list(TreeGraph.colour_dict.values())[1:]

    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig, ax = plt.subplots(1, len(classifying_metrics), figsize=(1500 * px, 500 * px))
    for i, metric in enumerate(classifying_metrics):
        yvals = centers[metric]
        assert label_map is None or len(yvals) == len(
            label_map
        ), "each cluster must be labelled"
        ylabel = "Magnitude [DL]"
        if "energy" in metric:
            ylabel = "Magnitude [$\mathrm{unit}^2$]"
        if ("log" in metric) and ("energy" in metric):
            ylabel = "Magnitude [dB]"
        if "entropy" in metric:
            yvals = yvals
            ylabel = "Magnitude [DL]"

        if isinstance(label_map, dict) and xlabel_user is None:
            xlabel = "signal component"
            xvals = [v for v in dict(sorted(label_map.items())).values()]
            label_angle = 45
        elif isinstance(label_map, dict) and isinstance(xlabel_user, str):
            xlabel = xlabel_user
            xvals = [v for v in dict(sorted(label_map.items())).values()]
            label_angle = 0
        else:
            xlabel = "Cluster number"
            xvals = [i for i in range(n_clusters)]
            label_angle = 0

        ax[i].bar(xvals, yvals, color=colours)
        ax[i].set_title(f"{metric.capitalize()}", fontsize=fsize)
        ax[i].set_xlabel(xlabel, fontsize=fsize)
        ax[i].set_ylabel(ylabel, fontsize=fsize)
        ax[i].tick_params(axis="x", rotation=label_angle)
        ax[i].tick_params(labelsize=tick_size)
    plt.tight_layout()
    plt.show()

    return fig


def plot_multiple_centers(
    classifying_metrics,
    centers,
    label_maps,
    exp_labels=None,
    bar_width=0.2,
    error_bars=None,
):
    """
    Plot the results for different sets of 'n' experiments. Input parameters are parallel
    lists of length 'n' containing information about each experiment.

    Parameters
    ----------
    classifying_metrics: list of list
        A list containing n lists. Each list contains the classifying metrics used in the
        experiment.
    centers: list of array
        Each array contains the clustering centers from each experiment.
    label_maps: list of dict
        Dictionaries map cluster number to feature (fundamental, broadband noise, etc.).
    exp_labels: list of str
        A list of str names for each experiment.
    bar_width: int
        Width of the bars in the bar chart.
    error_bars: list of array
        Arrays contain error bar information for each experiment.
    """
    if any(not isinstance(i, list) for i in [classifying_metrics, centers, label_maps]):
        raise ValueError(
            "all inputs must be lists of lists, arrays, dictionaries respectively"
        )

    # Get all unique metrics and post-hoc feature labels from each experiment
    all_metrics = [i for metrics in classifying_metrics for i in metrics]
    unique_metrics = list(set(all_metrics))

    try:
        all_labels = [i for label_map in label_maps for i in label_map.values()]
    except AttributeError:
        raise ValueError("label map must be defined")
    unique_labels = list(set(all_labels))

    # Plot each metric on a separate graph
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig, ax = plt.subplots(
        len(unique_metrics), figsize=(1000 * px, 300 * px * len(unique_metrics))
    )
    for i, metric in enumerate(unique_metrics):
        handles = []
        labels = []
        for j, center in enumerate(centers):
            # Get centre data corresponding to the classifying metric
            try:
                metric_index = classifying_metrics[j].index(metric)
            except ValueError:  # metric was not used in this algorithm
                continue
            center_data = center[:, metric_index]
            error_data = error_bars[j]
            error_data = error_data[:, metric_index]

            # Reorder center data by feature labels
            label2index = {v: k for k, v in label_maps[j].items()}

            idx = [label2index.get(label, None) for label in unique_labels]
            ordered_center_data = [
                0 if index is None else center_data[index] for index in idx
            ]
            yerr = (
                [0 if index is None else error_data[index] for index in idx]
                if error_bars is not None
                else None
            )

            if "energy" in metric and "log" not in metric:
                ordered_center_data = to_dB(ordered_center_data)

            # Offset for bars
            offset = bar_width * j
            x = np.arange(len(unique_labels)) + offset
            exp_label = str(exp_labels[j]) if exp_labels is not None else str(j)

            # Plot bars with offset
            bar = ax[i].bar(
                x, ordered_center_data, width=bar_width, label=exp_label, yerr=yerr
            )
            handles.append(bar)
            labels.append(exp_label)

        # Formatting
        ylabel = (
            "Magnitude [dB]" if "energy" in metric and "log" not in metric else "Value"
        )

        n_experiments = len(classifying_metrics)
        ax[i].set_xticks(
            np.arange(len(unique_labels)) + bar_width * (n_experiments / 2 - 0.5)
        )
        ax[i].set_xticklabels([i.capitalize() for i in unique_labels])
        # ax[i].set_xlabel("Feature")
        ax[i].set_ylabel(ylabel)
        ax[i].set_title(f"{metric.capitalize()}")

        # Create legend for centers
        ax[i].legend(handles, labels)
    fig.tight_layout()
    plt.show()

    return fig


def get_center_metrics(classifying_metrics, centers, label_maps):
    """
    Get the mean and standard deviation for a list of centres. Returns centres and error
    bars for the plot_multiple_centres function above.

    Parameters
    ----------
    Parameters are parallel lists containing data from different experiments.
    classifying_metrics: list of list
        list of lists containing strings correspending to the classifying metrics used
        in the experiment.
    centers: list of array
        list of numpy arrays containing the centers from clustering
    label_maps: list of dict
        list of dictionaries mapping cluster number to label
    """

    # Get unique metrics
    def get_unique_metrics(metrics):
        unique_metrics = list(set(i for metrics in metrics for i in metrics))
        # Check that each input uses the same metrics
        sorted_metrics = sorted(unique_metrics)
        if any(sorted(i) != sorted_metrics for i in metrics):
            raise ValueError("The same metrics must be used for each experiment.")
        return unique_metrics

    unique_metrics = get_unique_metrics(classifying_metrics)

    # Get unique labels
    unique_labels = list(set(i for label_map in label_maps for i in label_map.values()))

    def map_indices(target, source):
        return [source.index(i) for i in target]

    def get_indices(target, mapping):
        reverse_map = {v: k for k, v in mapping.items()}
        return [reverse_map.get(i) for i in target if reverse_map.get(i) is not None]

    # Find the average and std of the centres corresponding to each label and metric.
    centres_sorted = []
    for centre, label_map, metrics in zip(centers, label_maps, classifying_metrics):
        label_indices = get_indices(unique_labels, label_map)
        metric_indices = map_indices(unique_metrics, metrics)
        centres_for_label = centre[label_indices]
        centres_sorted.append(centres_for_label[:, metric_indices])

    def apply_func_to_arrays(array_list, f, max_len):
        array_contents = [
            f([c[i] for c in array_list if len(c) > i], axis=0) for i in range(max_len)
        ]
        return np.array(array_contents)

    centres_avg = apply_func_to_arrays(centres_sorted, np.mean, len(unique_labels))
    centres_std = apply_func_to_arrays(centres_sorted, np.std, len(unique_labels))

    label_map = {k: v for k, v in enumerate(unique_labels)}

    return centres_avg, centres_std, unique_metrics, label_map


def relabel(labels, centers, metric, ascending=False):
    """
    Relabels cluster numbers based on a given metric.

    Parameters
    ----------
    labels: arraylike
        the corresponding labels to each entry in the training data
    centers: pd.df
        the centres from clustering
    metric: string
        the name of the metric in centres used to order them
    ascending: bool
        whether to sort by ascending

    Returns
    -------
    ordered_centers: pd.df
        clustering centres ordered by given metric
    ordered_labels: list
        labels ordered by centre metric
    """

    ordered_centers = centers.sort_values(by=metric, ascending=ascending)
    center_mapping = {k: v for v, k in enumerate(ordered_centers.index.tolist())}
    ordered_labels = np.array([center_mapping[i] for i in labels])
    ordered_centers = ordered_centers.reset_index()
    return ordered_centers, ordered_labels
