from .nodes import *
from .extensions import max_level_check

def dwpt(data, levels, wavelet, mode = 'symmetric'):
    '''
    daubechies packet transform. Returns a transform object
    which contains all of the results
    '''
    max_level_check(data,levels)
    
    if isinstance(data,list):
        data = np.array(data)

    transform_object = Transform(data, wavelet, mode)

    # create approximamtion and detail nodes for each node in the last level
    for level in range(1,levels+1):
        for node in transform_object.get_level(level-1):
            # create nodes
            transform_object.create_node(node,'a')
            transform_object.create_node(node,'d')        

    return transform_object

def dwt(data, levels, wavelet, mode = 'symmetric'):
    'discrete wavelet transform'
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