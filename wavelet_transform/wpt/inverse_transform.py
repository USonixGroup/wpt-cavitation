from pywt import waverec, idwt
from numpy import log2


def inv_dwt(coeffs, wavelet="haar", mode="symmetric"):
    """
    calls the waverec() function from the pywavelet library
    coeffs are from transform_object.nodes2data(transform_object.dwt_results())
    """
    return waverec(coeffs, wavelet, mode)


def inv_dwpt(coeffs, wavelet="haar", mode="symmetric"):
    """
    calls the single level idwt function from the pywt library recursively
    to find the inverse transform from a list of packet nodes at a given level.

    Parameters
    ----------
    coeffs: list
        list of arrays, each array containing data from a node in a level
        of the packet transform. these nodes must be given in order of
        frequency range lowest to highest. Examples are found in the docs
        folder under inverse testing.
    wavelet: str
        the type of wavelet used in the transform
    mode: str
        the padding used in the transform

    Returns
    -------
    output: array
        the signal reconstructed from the provided nodes. This might be longer
        than the original data, in which case the coefficients at the right end
        can be cut.
    """
    input = coeffs
    levels = int((log2(len(coeffs))))
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
            c = idwt(approx, detail, wavelet, mode)
            output.append(c)
        input = output.copy()
    return output[0]
