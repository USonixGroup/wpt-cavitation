import numpy as np
from ..wpt import transform as wpt
from scipy import stats
from scipy.signal import blackmanharris


class SVD:
    """
    class which tracks the transform objects used to create the windowed data

    Parameters
    ----------
    data: array
        the input sound wave to be windowed
    windows: int
        the number of windows to create
    levels: int
        the number of levels of transform
    overlap: float
        the overlap between the windows on a range of 0-1
    wavelet: string
        the wavelet used for the transform
    sampling_rate: float
        the sampling rate used to calculate the frequencies ranges of each subband.
        Not necessary for most functions of the SVD class to be carried out.
    mode: string
        the mode for padding used in the wavelet transforms. See pywt library for options.
    norm: string
        the method for normalising the energy matrix before SVD
    """

    def __init__(
        self,
        data,
        windows,
        levels,
        overlap=0,
        wavelet="haar",
        sampling_rate=None,
        mode="symmetric",
        norm="kernel_peak",
    ):
        # track input parameters
        self.data = data
        self.windows = windows
        self.levels = levels
        self.input_overlap = overlap
        self.wavelet = wavelet
        self.sampling_rate = sampling_rate
        self.mode = mode
        self.norm = norm
        self.win_parameter_check()

        # generate results
        self.windowed_data, self.overlap, self.offset = self.multi_window()
        self.energy_matrix, self.transform_objects = self.form_energy_matrix()
        self.svd_matrices, self.norm_energy_matrix = self.SVD()
        self.stability_marker = self.calc_stability_marker()

    def multi_window(self):
        """
        create overlapping blackman harris windows on the data

        Parameters
        ----------
        data: array
            the data to be windowed
        windows: int
            the number of windows

        Returns
        -------
        windowed_data: array
            a matrix of the windowed data, 1 window per row
        actual_overlap: float
            the actual overlap is greater than specified
        """
        # input parameters
        data = self.data
        windows = self.windows

        # create window
        window_size, offset = self.calc_win_size()
        window = blackmanharris(window_size)

        # calculate actual overlap
        actual_overlap = 1 - offset / window_size

        # window the data and store in a matrix
        start = 0
        end = start + window_size
        windowed_data = np.zeros((windows, window_size))
        for i in range(windows):
            data_segment = data[start:end]  # segment data according to window size
            windowed_data[i, :] = data_segment * window  # store each window in a row
            start += offset
            end = start + window_size

        return windowed_data, actual_overlap, offset

    def calc_win_size(self):
        """
        calculates the window length for the specified parameters. Always rounds down,
        giving a  greater window overlap than specified. this can be mitigated by using
        larger windows by increasing the length of the data or reducing the number of
        windows. This algorithm optimises for using more data instead of obtaining an
        accurate overlap.

        Parameters
        ----------
        self: SVD object

        Returns
        -------
        win_size: int
            the length of the window array
        offest: int
            the distance between neighbouring windows
        """
        # input parameters
        data_len = len(self.data)
        windows = self.windows
        overlap = self.input_overlap

        # calculate window size
        win_size = int(
            data_len / ((windows - 1) * (1 - overlap) + 1)
        )  # calculate window size, round down
        offset = int((1 - overlap) * win_size)  # calculate and round offset
        return win_size, offset

    def win_parameter_check(self):
        "raises errors if input parameters are wrong"
        if self.input_overlap >= 1 or self.input_overlap < 0:
            raise ValueError("overlap must be in range 0-1")
        if len(self.data) < self.windows:
            error = "too many windows, reduce level of transform, provide more data, or decrease number of windows"
            raise ValueError(error)
        if not isinstance(self.windows, int):
            raise ValueError("number of windows must be an integer")
        elif self.windows < 2:
            raise ValueError("use at least 2 windows")

    def form_energy_matrix(self):
        """
        creates a matrix of energy from a matrix of windowed data

        Parameters
        ----------
        windowed_data: np.array
            output matrix from multi_window(), rows represent a window
        levels: int
            the level of transform
        wavelet: string
            the wavelet to use in the transform
        sampling_rate: int
            used to calculate the frequency ranges for each transform object

        Returns
        -------
        energy_matrix: np.array
            matrix with rows and columns representing energy in nodes and windows respectively
        transform_objects: list
            a list of the transforms used for each window
        """
        # input parameters
        levels = self.levels
        windowed_data = self.windowed_data
        wavelet = self.wavelet
        sampling_rate = self.sampling_rate

        # initialise arrays
        transform_objects = []  # initialise list of transforms
        energy_matrix = np.zeros(
            (windowed_data.shape[0], 2**levels)
        )  # initialise energy matrix
        for i in range(windowed_data.shape[0]):
            # transform
            window = windowed_data[i, :]
            transform_object = wpt.dwpt(window, levels, wavelet, self.mode)
            transform_object.calc_frequencies(sampling_rate)

            # add data to each matrix
            transform_objects.append(transform_object)

            energies = list(
                transform_object.info()["energy"]
            )  # create list of energies
            energy_matrix[i, :] = energies  # store energies for each window in columns

        return energy_matrix, transform_objects

    def calc_frequencies(self, sampling_rate):
        "calculate frequency ranges for each node in each transform"
        for transform in self.transform_objects:
            transform.calc_frequencies(sampling_rate)

    def SVD(self):
        "perform singular value decomposition on the matrix of energies"
        energy_matrix = self.energy_matrix
        norm_energy_matrix = np.zeros(energy_matrix.shape)
        # normalise energy matrix
        for column in range(energy_matrix.shape[1]):  # for each subband
            energy_col = energy_matrix[
                :, column
            ]  # extract list of energy in one subband
            data = np.array(energy_col)
            if self.norm == "kernel_peak":  # method for normalisation
                kde = stats.gaussian_kde(data)  # equivalent of ksdensity in matlab
                x = np.linspace(data.min(), data.max(), 100)
                p = kde(
                    x
                )  # probability density estmate evaluated at 100 points (given by x)
                kernel_peak = x[p.argmax()]
                norm_array = energy_col / kernel_peak
            elif self.norm == "mean_centre":
                mean = np.average(data)
                norm_array = energy_col / mean
            elif self.norm == "mean":
                mean = np.average(data)
                norm_array = energy_col - mean
            elif (self.norm is None) or (self.norm == "none"):
                norm_energy_matrix = energy_matrix.copy()
                break
            norm_energy_matrix[:, column] = norm_array
        return np.linalg.svd(norm_energy_matrix), norm_energy_matrix

    def calc_stability_marker(self):
        """
        Computes the stability marker using the first eigenvector in the first
        SVD matrix.
        """
        eigenvector = self.svd_matrices[2][0, :]  # first row of last matrix
        stability_markers = np.sqrt(
            1 / abs(eigenvector)
        )  # initialise stability markers
        # stability_markers = -abs(eigenvector)

        return stability_markers
