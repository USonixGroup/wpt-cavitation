from wavelet_transform.plotting import *
from wavelet_transform.transform import *
from wavelet_transform.inverse_transform import *

if __name__ == '__main__':
    # transform
    levels = 3
    wavelet = 'db2'
    data = [i for i in range(22)]
    object = dwpt(data,levels, wavelet)

    # inverse dwt
    dwt_result_nodes = object.dwt_results()
    dwt_coeffs = object.nodes2data(dwt_result_nodes)
    dwt_inverse = idwt(dwt_coeffs,wavelet)
    print('dwt inverse',dwt_inverse)
    print('error dwt',data-dwt_inverse)

    # inverse dwpt
    dwpt_result_nodes = object.get_level(levels)
    dwpt_coeffs = object.nodes2data(dwpt_result_nodes)
    dwpt_inverse = idwpt(dwpt_coeffs, levels, wavelet)
    print('dwpt inverse',dwpt_inverse)
    print('length', len(dwpt_inverse))
    print('error, packet', np.array(data)-dwpt_inverse[:len(data)])