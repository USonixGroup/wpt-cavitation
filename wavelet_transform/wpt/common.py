import pywt
import numpy as np
from .extensions import sci_form, pywt_filter_bank

from scipy.stats import kurtosis
class Transform():
    '''
    Created for each transform. Stores all of the relevant information about
    the transform

    Parameters
    ----------
    data: array
        the sound wave to be transformed
    wavelet: string
        the chosen wavelet for the transform
    mode: string
        the chosen type of padding
    '''
    def __init__(self, data, wavelet, mode):
        self.wavelet = wavelet
        self.mode = mode
        self.data = data
        self.base_node = BaseNode(data)
        self.levels = {0:[self.base_node]}

    def get_level(self,level):
        'returns a list of nodes for the chosen level'
        return self.levels[level]

    def add_node(self,node):
        'appends a node to node list'
        # initialise the level if necessary
        if len(self.levels) == node.level:
            self.add_level()
        self.levels[node.level].append(node)

    def add_level(self):
        'increments the number of levels by 1'
        self.levels[len(self.levels)] = []

    def create_node(self, parent, a_d):
        'creates a node object, adds it to list of nodes'
        # initialise node object
        node = Node(parent,a_d,self)
        self.add_node(node)

    def calc_frequencies(self,sampling_rate):
        'set the sampling rate to give information about nyquist frequency'
        self.sampling_rate = sampling_rate
        for level in self.levels:
            for node in self.levels[level]:
                node.frequency_range = node.freq_calc(self.sampling_rate)
    
    def show_all_energy(self):
        'prints the energy for each node'
        for level in self.levels:
            print(f'level {level} nodes:')
            for node in self.levels[level]:
                energy = sci_form(node.energy)
                try:
                    lf = sci_form(node.frequency_range[0])
                    uf = sci_form(node.frequency_range[1])   
                    print(f'node: {node.path} [{lf}-{uf} Hz]\nenergy: {energy}')
                except AttributeError: # if the frequency is not yet calculated
                    lf = sci_form(node.norm_frequency_range[0])
                    uf = sci_form(node.norm_frequency_range[1])
                    print(f'node: {node.path} [({lf}-{uf})*nyquist frequency Hz]\nenergy: {energy}')

                print()
    
    def get_node(self,id,nodes):
        'finds a node for a given path or frequency from a list of nodes'
        for node in nodes:
            if node.match_id(id):
                return node
        # error message of node is not found
        if isinstance(id,str): # search for node with path
            print(f'could not find node with specified path {id}')
            return None
        else: # search for node with frequency content
            print(f'could not find node containing specified frequency: {sci_form(id)} Hz')

    def find_nodes(self,nodes,ids,min_energy=0):
        'list nodes with specified frequency and/or path'
        node_list = []
        for id in ids:
            node = self.get_node(id,nodes)
            if node.energy > min_energy:
                node_list.append(node)
        return node_list
    
    def packet_nodes_for_rec(self,level,*,keep=None,leave=None,min_energy=0):
        '''
        gather nodes for reconstruction with more than min_energy.
        if ids is specified, only uses nodes specified in ids, regardless of energy.
        keep, leave, and min_energy cannot be used in tandem

        parameters
        ----------
        level: int
            the level of transform to start reconstruct from
        keep: list
            list of ids (path or frequency content) to reconstruct
        leave: list
            list of ids (path of frequency content) to leave out of reconstruction
        min_energy: float

        '''
        counter = 0
        for arg in [keep,leave,min_energy]:
            if arg:
                counter += 1
        if counter > 1:
            print('only specify one of leave, keep, and min_energy')
            return None
        node_list = []
        nodes = self.get_level(level)
        if keep:
            for node in nodes:
                for id in keep:
                    if node.match_id(id):
                        node_list.append(node)
                        continue
                else:
                    node_list.append(None)
        elif leave:
            for node in nodes:
                for id in leave:
                    if node.match_id(id):
                        node_list.append(None)
                        continue
                else:
                    node_list.append(node)
        else:
            for node in nodes:
                if node.energy > min_energy:
                    node_list.append(node)  
        return node_list

    def dwt_results(self):
        'extracts the nodes containing dwt results from a dwt or dwpt transform object'
        results = []
        # append all of the relevant detail nodes
        for level in range(1,len(self.levels)):
            nodes = self.get_level(level)
            detail_node = nodes[1]
            results.insert(0,detail_node)
        # append final approx node
        approx_node = nodes[0]
        results.insert(0,approx_node)
        return results
    
    def nodes2data(self,nodes = None):
        'returns a list of arrays. use before inverse transform'
        output = []
        if nodes:
            for node in nodes:
                if node:
                    output.append(node.data)
                else:
                    output.append(None)
        else:
            for level in self.levels:
                for node in self.levels[level]:
                    output.append(node.data)
        return output

    
class Node():
    def __init__(self, parent, a_d, transform_object):
        '''
        Stores the information for each set of coefficients in the transform
        Each node has an assigned parent and path e.g. node aaa is the third 
        level approximation coefficients with parent node aa.

        Parameters
        ----------
        parent: object
            the the node's parent node i.e. the node that was filtered to 
            obtain the data in this one.
        a_d: string
            whether a high pass or low pass filter was used to obtain the 
            data for this node. 'a' for low pass, 'd' for high pass
        level: int
            the level of transform the node is in
        transform_object: Transform()
            an object representing the transform the node is part of
        '''
        # set parent, path, and transform object
        self.transform_object = transform_object
        self.parent = parent
        self.path = f'{parent.path}{a_d}' # eg 'aaa' for 3rd level approx
        self.a_d = a_d
        self.level = parent.level+1
        # calculate coefficients
        filter_bank = pywt_filter_bank(self.transform_object.wavelet)
        if a_d == 'a':
            filter_coeffs = filter_bank[0]
        else:
            filter_coeffs = filter_bank[1]
        self.data = self.filter(filter_coeffs, self.transform_object.mode)

        # calculate energy, frequency, and kurtosis
        self.energy = self.energy_calc()
        self.kurtosis = self.calc_kurtosis()
        self.norm_frequency_range = self.norm_freq_calc()
        self.frequency_range = None

    def filter(self, filter_coeffs, mode):
        'performs the convolution between an adapted signal and the filter'
        parent_data = self.parent.data.copy()
        
        length = len(filter_coeffs) - 2
        l2 = length
        if len(parent_data) % 2:
                l2 += 1
        # signal extension
        data = pywt.pad(parent_data,(length,l2),mode)
        # perform convolution
        data = np.convolve(data, filter_coeffs, 'valid')
        # downsample
        data = data[::2]
        return data
    
    def energy_calc(self):
        return np.var(self.data)
    
    def norm_freq_calc(self):
        '''
        calculates an upper and lower bound between 
        0-1 to be multiplied with nyquist frequency
        '''
        parent = self.parent
        
        lower_bound = parent.norm_frequency_range[0]
        upper_bound = parent.norm_frequency_range[1]

        diff = upper_bound - lower_bound
        if self.a_d == 'a':
            upper_bound -= diff/2
        elif self.a_d == 'd':
            lower_bound += diff/2
        return lower_bound,upper_bound
    
    def calc_kurtosis(self):
        return kurtosis(self.data)    
    
    def freq_calc(self,sampling_rate):
        'converts the normalised frequency range to an actual freuqency range'
        nyquist_frequency = sampling_rate/2
        lower_bound = self.norm_frequency_range[0]*nyquist_frequency
        upper_bound = self.norm_frequency_range[1]*nyquist_frequency
        return lower_bound,upper_bound 
    
    def match_id(self,id):
        if isinstance(id,str):
            return id == self.path
        else:
            if not self.frequency_range:
                error = 'call transform_object.calc_frequencies() before finding node by frequency'
                raise SyntaxError(error)
            freq_range = self.frequency_range
            return (id >= freq_range[0]) & (id <= freq_range[1])

class BaseNode(Node):
    def __init__(self,data):
        self.parent = None
        self.data = data
        self.path = ''
        self.level = 0
        self.norm_frequency_range = (0,1)
        self.energy = self.energy_calc()
    
