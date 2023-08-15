import matplotlib.pyplot as plt
import numpy as np


def plot_pywp_test(wp_object, levels, plot=True):
    """
    testing the transform from the python library

    Parameters
    ----------
    wp_object: object
        the object that comes from the function WaveletPacket()
    levels: int
        the level of transform to be plotted
    plot: boolean
        whether to plot or just return concatenated data

    Returns
    -------
    coeffs: array
        the data that was plotted,a concatenation of each of data from all nodes
        in the specified level
    """
    nodes = [node.path for node in wp_object.get_level(levels)]

    # concatenante coefficients
    coeffs = []
    for node in nodes:
        node_data = wp_object[node].data
        coeffs = np.concatenate((coeffs, node_data))

    # plot coefficients
    if plot:
        plt.plot(coeffs)
        plt.grid(True)
        plt.show()

    return coeffs


def plot_test(nodes):
    """
    testing the wavelet transform in this library

    Parameters
    ----------
    nodes: list
        a list of node objects containing data to plot

    Returns
    -------
    output: array
        an array of the plotted data, a concatenation of the data from the
        provided nodes
    """

    # concatenate coefficients
    output = np.array([])
    for node in nodes:
        output = np.concatenate((output, node.data))
    # plot coefficients
    plt.plot(output)
    plt.grid(True)
    plt.show()

    return output
