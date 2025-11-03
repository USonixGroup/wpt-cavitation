from .transform_object import Transform
import numpy as np

# ----- Discrete Wavelet Transform -----
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
    max_level_check(data, max_level)

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

# ----- Discrete Wavelet Packet Transform -----

def dwpt(
    data,
    max_level=None,
    wavelet="haar",
    mode="symmetric",
    *,
    best_basis=None,
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
    best_basis: str or None
        perform best basis decomposition "min_entropy" or "energy". If False,
        performs full decomposition.
    min_level: int
        force the best basis decomposition to decompose the signal to a
        certain level. This is 0 by default.
    freq_threshold: float
        the nodes with norm_freq_range above this threshold will not be kept

    Returns
    -------
    transform: an object containing the results from the transform
    """

    if max_level is None:
        max_level = calc_max_level(data)
    else:
        max_level_check(data, max_level)

    if isinstance(data, list):
        data = np.array(data)

    # Create approximamtion and detail nodes for each node in the last level
    if best_basis is None:
        transform = _dwpt(data, max_level, wavelet, mode, freq_threshold)

    # Best basis decomposition
    elif isinstance(best_basis, str):
        transform = _best_basis_dwpt(
            data, max_level, wavelet, mode, min_level, freq_threshold, best_basis
        )

    else:
        raise ValueError("best_basis must be None or a string")

    assert transform.get_level(1), "transform has no decompositions"

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


def _best_basis_dwpt(data, max_level, wavelet, mode, min_level, freq_threshold, best_basis="min_entropy"):
    if not isinstance(min_level, int):
        raise ValueError("min_level must be an integer")

    transform = Transform(data, wavelet, mode)
    transform.best_basis = best_basis
    for level in range(max_level):
        for node in transform.get_level(level):
            # Create nodes from parent node
            node1 = transform.create_node(node, "a")
            node2 = transform.create_node(node, "d")

            # Perform decomposition with a given minimum level.
            if (
                level < min_level 
                or _split_condition(node, node1, node2, transform, best_basis)
            ):
                # Add nodes if they have lower combined entropy than their parent
                if node1.norm_frequency_range[0] < freq_threshold:
                    transform.add_node(node1)
                if node2.norm_frequency_range[0] < freq_threshold:
                    transform.add_node(node2)

    return transform

def _split_condition(node, node1, node2, tr_obj:Transform, condition):
    """Choose condition to call splitting function"""
    if condition == "min_entropy":
        return _min_entropy_condition(node, node1, node2)
    elif condition == "energy":
        return _energy_condition(node, node1, node2, tr_obj.get_level(0)[0])


def _min_entropy_condition(node, node1, node2):
    """Split condition based on shannon entropy"""
    # If a node has 0 entropy, it does not need to be further decomposed
    if node.level_entropy <= 1e-14:  # small threshold for numerical errors
        return False
    
    # Split on lower shannon entropy
    if (node1.level_entropy + node2.level_entropy) < node.level_entropy:
        return True
    return False

def _energy_condition(node, node1, node2, base_node):
    """Split condition based on energy distribution"""
    # Don't split on low energy
    if node.energy < base_node.energy * 0.01:
        return False
    
    # Split on unbalanced energy
    energy_threshold = 1.5
    if (node1.average_energy / node2.average_energy > energy_threshold
        or node2.average_energy / node1.average_energy > energy_threshold
    ):
        return True

    return False


# ----- Utility functions -----


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
    rec_max_level = calc_max_level(data)
    error_message = f"""
    too many levels. length of data is length of data is: {len(data)}
    the maximum level is: {max_level}
    """
    if level > max_level:
        raise ValueError(error_message)
    if level > rec_max_level:
        print(f"WARNING: recommended max level is {rec_max_level}")


def calc_max_level(data):
    "Get the maximum decomposition level for a 1-dimensional array of data"
    return int(np.log2(len(data)))
