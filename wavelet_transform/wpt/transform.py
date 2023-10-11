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
    transform_object: an object containing the results from the transform
    """

    if max_level is None:
        max_level = calc_max_level(data)
    else:
        max_level_check(data, max_level)

    if isinstance(data, list):
        data = np.array(data)

    # Create approximamtion and detail nodes for each node in the last level
    if min_entropy is False:
        transform_object = _dwpt(data, max_level, wavelet, mode)

    # Minimum entropy decomposition
    elif min_entropy is True:
        transform_object = _min_entropy_dwpt(data, max_level, wavelet, mode, min_level)

    else:
        raise ValueError("min_entropy: bool")

    assert transform_object.get_level(
        1
    ), "transform has no decompositions, try setting a minimum level"
    return transform_object


def _dwpt(data, max_level, wavelet, mode):
    transform_object = Transform(data, wavelet, mode)  # create wavelet object

    for level in range(max_level):
        for node in transform_object.get_level(level):
            # Create nodes
            transform_object.create_node(node, "a")
            transform_object.create_node(node, "d")
    return transform_object


def _min_entropy_dwpt(data, max_level, wavelet, mode, min_level):
    if not isinstance(min_level, int):
        raise ValueError("min_level must be an integer")

    transform_object = Transform(data, wavelet, mode)
    for level in range(max_level):
        for node in transform_object.get_level(level):
            # If a node has 0 entropy, it does not need to be further decomposed
            if node.entropy == 0:
                continue
            # Create nodes from parent node
            node1 = transform_object.create_node(node, "a", add_node=False)
            node2 = transform_object.create_node(node, "d", add_node=False)
            # Check if the entropy has decreased
            entropy = node1.level_entropy + node2.level_entropy

            # Perform decomposition with a given minimum level.
            if (node.level_entropy > entropy) or (level in range(min_level)):
                # Add nodes if they have lower combined entropy than their parent
                transform_object.add_node(node1)
                transform_object.add_node(node2)

    return transform_object


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
    transform_object: an object containing the results from the transform
    """
    max_level_check(data, max_level)

    if isinstance(data, list):
        data = np.array(data)

    transform_object = Transform(data, wavelet, mode)

    for level in range(max_level):
        # get approx node in the last level
        node = transform_object.get_level(level)[0]
        # store in output array for next level
        transform_object.create_node(node, "a")
        transform_object.create_node(node, "d")

    return transform_object


def max_level_check(data, level):
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
    error_message = f"""
    too many levels. length of data is length of data is: {len(data)}
    the maximum level is: {max_level}
    """
    if level > max_level:
        raise ValueError(error_message)


def calc_max_level(data):
    "finds the maximum level a 1-dimensional array of data"
    max_level = 0
    sections = 1
    while sections < len(data):
        max_level += 1
        sections *= 2
    return max_level
