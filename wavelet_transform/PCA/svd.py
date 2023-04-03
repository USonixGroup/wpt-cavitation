import numpy as np
from ..wpt import transform as wpt
from scipy import stats

class SVD():
    '''
    class which tracks the transform objects used to create the windowed data

    Parameters
    ----------
    data: array
        the input sound wave to be windowed
    windows: int
        the number of windows to create
    levels: int
        the number of levels of transform
    wavelet: string
        the wavelet used for the transform
    overlap: float
        the overlap between the windows on a range of 0-1
    '''
    def __init__(self,data,windows,levels,overlap = 0,wavelet = 'haar',sampling_rate = None):
        # track input parameters
        self.data = data
        self.windows = windows
        self.levels = levels
        self.input_overlap = overlap
        self.wavelet = wavelet
        self.sampling_rate = sampling_rate
        self.win_parameter_check()

        # generate results
        self.windowed_data, self.overlap, self.offset = self.multi_window()
        self.energy_matrix, self.transform_objects = self.form_energy_matrix()
        self.svd_matrices, self.norm_energy_matrix = self.SVD()
        self.stability_marker = self.calc_stability_marker()

    def multi_window(self):
        '''
        create overlapping blackman windows on the data

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
        '''
        # input parameters
        data = self.data
        windows = self.windows

        # create window
        window_size,offset = self.calc_win_size()
        window = np.blackman(window_size)

        # calculate actual overlap
        actual_overlap = 1-offset/window_size

        # window the data and store in a matrix
        start = 0
        end = start + window_size
        windowed_data = np.zeros((windows,window_size))
        for i in range(windows):
            data_segment = data[start:end] # segment data according to window size
            windowed_data[i,:] = data_segment*window # store each window in a row
            start += offset
            end = start + window_size

        return windowed_data,actual_overlap,offset

    def calc_win_size(self):
        '''
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
            the distance between the start of neighbouring windows
        '''
        # input parameters
        data_len = len(self.data)
        windows = self.windows
        overlap = self.input_overlap

        # calculate window size
        win_size = int(data_len/((windows-1)*(1-overlap)+1)) # calculate window size, round down
        offset = int((1-overlap)*win_size) # calculate and round offset
        if offset == 0:
            offset = 1 # offset must be at least 1
        return win_size,offset

    def win_parameter_check(self):
        'raises errors if input parameters are wrong'
        if self.input_overlap >= 1 or self.input_overlap < 0:
            raise ValueError('overlap must be in range 0-1')
        if len(self.data) < self.windows:
            raise ValueError('too many windows')
        
    def form_energy_matrix(self):
        '''
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
        '''
        # input parameters
        levels = self.levels
        windowed_data = self.windowed_data
        wavelet = self.wavelet
        sampling_rate = self.sampling_rate

        # initialise arrays
        transform_objects = [] # initialise list of transforms
        energy_matrix = np.zeros((2**levels,windowed_data.shape[0])) # initialise energy matrix
        for i in range(windowed_data.shape[0]):
            # transform
            window = windowed_data[i,:]
            transform_object = wpt.dwpt(window,levels,wavelet)
            transform_object.calc_frequencies(sampling_rate)

            # add data to each matrix
            transform_objects.append(transform_object)

            energies = transform_object.list_energy() # create list of energies
            energy_matrix[:,i] = energies # store energies for each window in columns

        return energy_matrix,transform_objects
    
    def calc_frequencies(self,sampling_rate):
        'calculate frequency ranges for each node in each transform'
        for transform in self.transform_objects:
            transform.calc_frequencies(sampling_rate)

    def SVD(self):
        'perform singular value decomposition on the matrix of energies'
        energy_matrix = self.energy_matrix
        norm_energy_matrix = np.zeros(energy_matrix.shape)
        # normalise energy matrix
        for row in range(energy_matrix.shape[0]):
            energy_row = energy_matrix[row,:]

            data = np.array(energy_row) # equivalent of ksdensity in matlab
            kde = stats.gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 100)
            p = kde(x)
            kernel_peak = x[p.argmax()]
            norm_array = []
            for i in energy_row:
                norm_array.append(i/kernel_peak)
            norm_energy_matrix[row,:] = norm_array
        return np.linalg.svd(self.energy_matrix),norm_energy_matrix
    
    def calc_stability_marker(self):
        '''
        Computes the stability marker using the first eigenvector in the  first
        SVD matrix.
        '''
        eigenvector = self.svd_matrices[0][:,0] # first column of first matrix
        stability_markers = [] # initialise stability markers
        for i in eigenvector:
            stability_markers.append(np.sqrt(1/abs(i)))
        print('eigenvector',eigenvector)
        print('stability markers',stability_markers)
        return stability_markers