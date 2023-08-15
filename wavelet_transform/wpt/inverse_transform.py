from pywt import waverec, idwt
from numpy import log2


def inv_dwt(transform_object):
    """
    calls the waverec() function from the pywavelet library
    """
    t = transform_object
    nodes = t.dwt_results()
    coeffs = t.nodes2data(nodes)
    return waverec(coeffs, t.wavelet, t.mode)[: len(transform_object.data)]


def inv_dwpt(transform_object, level=None, **kwargs):
    """
    calls the single level idwt function from the pywt library recursively
    to find the inverse transform from a list of packet nodes at a given level.

    Parameters
    ----------
    transform_object: Transform
        Transform object containing nodes to be transformed
    level: int
        the level at which nodes will be transformed. max level by default.

    Keyword Arguments
    -----------------
    nodes: list of str or list of int
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
    output: array
        the signal reconstructed from the provided nodes. This might be longer
        than the original data, in which case the coefficients at the right end
        can be cut.
    """
    if level is not None and not isinstance(level, int):
        raise ValueError("level must be an integer")

    # generate list of coefficients
    nodes = transform_object.packet_nodes_for_rec(level, **kwargs)
    coeffs = transform_object.nodes2data(nodes)
    levels = int((log2(len(coeffs))))

    input = coeffs
    for _ in range(levels):
        output = []
        for i in range(int(len(input) / 2)):
            approx = input[2 * i]
            detail = input[2 * i + 1]
            if (approx is None) and (
                detail is None
            ):  # if neither node is used, move on
                output.append(None)
                continue
            c = idwt(approx, detail, transform_object.wavelet, transform_object.mode)
            output.append(c)
        input = output.copy()
    return output[0][: len(transform_object.data)]
