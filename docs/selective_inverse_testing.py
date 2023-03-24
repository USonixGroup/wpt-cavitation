from wavelet_transform.transform import *
from wavelet_transform.inverse_transform import *

if __name__ == '__main__':
    # transform
    levels = 3
    wavelet = 'db2'
    data = [i for i in range(22)]
    object = dwpt(data,levels, wavelet)

    # selective inverse dwpt
    nodes_to_keep = ['aaa','add',1]
    dwpt_result_nodes = object.packet_nodes_for_rec(levels,keep = nodes_to_keep)
    kept_coefficients = object.nodes2data(dwpt_result_nodes)
    inverse = idwpt(kept_coefficients,levels,wavelet)

