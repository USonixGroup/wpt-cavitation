from .transform_object import *

def dwpt(data, levels, wavelet='haar', mode = 'symmetric'):
    '''
    Packet transform. Returns a transform object containing node objects 
    which returns the coefficients at every level.

    Parameters
    ----------
    data: array_like
        an array containing the data to be transformed
    levels: int
        the number of levels of transform to perform
    wavelet: string
        the wavelet to use in the transform. see pywavelets docs for options
    mode: string
        the extension mode for the signal. see numpy for options

    Returns
    -------
    transform_object: an object containing the results from the transform
    '''
    max_level_check(data,levels)
    
    if isinstance(data,list):
        data = np.array(data)

    transform_object = Transform(data, wavelet, mode) # create wavelet object

    # create approximamtion and detail nodes for each node in the last level
    for level in range(0,levels):
        for node in transform_object.get_level(level):
            # create nodes
            transform_object.create_node(node,'a')
            transform_object.create_node(node,'d')        

    return transform_object

def dwt(data, levels, wavelet='haar', mode = 'symmetric'):
    '''
    discrete wavelet transform

    Parameters
    ----------
    data: array_like
        an array containing the data to be transformed
    levels: int
        the number of levels of transform to perform
    wavelet: string
        the wavelet to use in the transform. see pywavelets docs for options
    mode: string
        the extension mode for the signal. see numpy for options

    Returns
    -------
    transform_object: an object containing the results from the transform
    '''
    max_level_check(data,levels)

    if isinstance(data,list):
        data = np.array(data)
    
    transform_object = Transform(data,wavelet,mode)

    for level in range(1,levels+1):
        # get approx node in the last level
        node = transform_object.get_level(level-1)[0]
        # store in output array for next level
        transform_object.create_node(node,'a')
        transform_object.create_node(node,'d')

    return transform_object

def max_level_check(data, levels):
    '''
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
    '''
    max_level = calc_max_level(data)
    error_message = f'''too many levels. length of data is length of data is: {len(data)}
    the maximum level is: {max_level}
    '''
    if levels > max_level:
        raise ValueError(error_message)
    
def calc_max_level(data):
    'finds the maximum level a 1-dimensional array of data'
    max_level = 0
    sections = 1
    while sections < len(data):
        max_level += 1
        sections *= 2
    return max_level