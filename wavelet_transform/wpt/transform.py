from .transform_object import Transform
import numpy as np


def dwpt(
    data,
    max_level=None,
    wavelet="haar",
    mode="symmetric",
    *,
    min_entropy=False,
    min_level=0,
    freq_threshold=np.inf,
):
    """
    Packet transform. Returns a transform object containing node objects
    which returns the coefficients at every level. Automatically performs a max-level
    transform

    Parameters
    ----------
    data: array_like
        an array containing the data to be transformed
    max_level: int
        the maximum number of levels of transform to perform.
        set to None to perform max level decomposition.
    wavelet: string
        the wavelet to use in the transform. see pywavelets docs for options
    mode: string
        the extension mode for the signal. see numpy for options
    min_entropy: bool
        set to True to perform the minimum entropy decomposition
    min_level: int
        force the minimum entropy decomposition to decompose the signal to a certain level.
        this is 0 by default.

    Returns
    -------
    transform: an object containing the results from the transform
    """

    filter_len = len(Transform.pywt_filter_bank(wavelet)[0])
    if max_level is None:
        max_level = calc_max_level(data, filter_len)
    else:
        max_level_check(data, max_level, filter_len)

    if isinstance(data, list):
        data = np.array(data)

    # Create approximamtion and detail nodes for each node in the last level
    if min_entropy is False:
        transform = _dwpt(data, max_level, wavelet, mode, freq_threshold)

    # Minimum entropy decomposition
    elif min_entropy is True:
        transform = _min_entropy_dwpt(
            data, max_level, wavelet, mode, min_level, freq_threshold
        )

    else:
        raise ValueError("min_entropy: bool")

    assert transform.get_level(
        1
    ), "transform has no decompositions, try setting a minimum level"

    transform.min_level = min_level
    return transform


def _dwpt(data, max_level, wavelet, mode, freq_threshold):
    transform = Transform(data, wavelet, mode)  # create wavelet object
    create_node = transform.create_node
    for level in range(max_level):
        for node in transform.get_level(level):
            # Create nodes
            node1 = create_node(node, "a")
            node2 = create_node(node, "d")
            if node1.norm_frequency_range[0] < freq_threshold:
                transform.add_node(node1)
            if node2.norm_frequency_range[0] < freq_threshold:
                transform.add_node(node2)
    return transform


def _min_entropy_dwpt(data, max_level, wavelet, mode, min_level, freq_threshold):
    if not isinstance(min_level, int):
        raise ValueError("min_level must be an integer")

    transform = Transform(data, wavelet, mode)
    transform.min_entropy = True
    for level in range(max_level):
        for node in transform.get_level(level):
            # If a node has 0 entropy, it does not need to be further decomposed
            if node.entropy == 0:
                continue
            # Create nodes from parent node
            node1 = transform.create_node(node, "a")
            node2 = transform.create_node(node, "d")
            # Check if the entropy has decreased
            entropy = node1.level_entropy + node2.level_entropy

            # Perform decomposition with a given minimum level.
            if (node.level_entropy > entropy) or (level in range(min_level)):
                # Add nodes if they have lower combined entropy than their parent
                if node1.norm_frequency_range[0] < freq_threshold:
                    transform.add_node(node1)
                if node2.norm_frequency_range[0] < freq_threshold:
                    transform.add_node(node2)

    return transform


def dwt(data, max_level, wavelet="haar", mode="symmetric"):
    """
    discrete wavelet transform

    Parameters
    ----------
    data: array_like
        an array containing the data to be transformed
    max_level: int
        the number of levels of transform to perform
    wavelet: string
        the wavelet to use in the transform. see pywavelets docs for options
    mode: string
        the extension mode for the signal. see numpy for options

    Returns
    -------
    transform: an object containing the results from the transform
    """
    filter_coeffs = Transform.pywt_filter_bank(wavelet)
    max_level_check(data, max_level, len(filter_coeffs))

    if isinstance(data, list):
        data = np.array(data)

    transform = Transform(data, wavelet, mode)

    for level in range(max_level):
        # get approx node in the last level
        node = transform.get_level(level)[0]
        # store in output array for next level
        transform.create_node(node, "a")
        transform.create_node(node, "d")

    return transform


def max_level_check(data, level, filter_len=None):
    """
    checks if there are too many levels

    Parameters
    ----------
    data: array or int
        an array of the data or just its length
    levels: int
        the number of levels of transform

    Raises
    ------
    ValueError:
        when the number of levels is too high
    """
    max_level = calc_max_level(data)
    rec_max_level = calc_max_level(data, filter_len)
    error_message = f"""
    too many levels. length of data is length of data is: {len(data)}
    the maximum level is: {max_level}
    """
    if level > max_level:
        raise ValueError(error_message)
    if level > rec_max_level:
        print(f"WARNING: recommended max level is {rec_max_level}")


def calc_max_level(data, filter_len=None):
    "Get the maximum decomposition level for a 1-dimensional array of data"

    # Check that there are fewer nodes at max level than data points
    if filter_len is None:
        max_level = 0
        sections = 1
        while sections < len(data):
            max_level += 1
            sections *= 2
        return max_level
    # Check that max level node oontains more coeffs than filter
    else:
        max_level = 0
        len_ = len(data)
        while len_ > filter_len:
            len_ = int(len_ / 2)
            max_level += 1
        return max_level
