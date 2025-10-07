import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sd


def larget_lyapunov_exp(
    data,
    sampling_rate,
    avg_period=None,
    dimension=2,
    lag=None,
    tol=1e-10,
    parallel=False,
    plot_flag=False,
):
    """
    Calculate the largest Lyapunov exponent using Rosenstein method, see Step 4 above, and
    Sato method, see Step 4.1 above.
    dimension: int
        the phase space dimension for the dynamic system
    Input Parameters:
        data: numpy array
            time series data, in vector format
        sampling_rate:float
            sampling rate in Hz (samples per second)
        dimension: int
            embedding dimension for constructing the dynamic trajectory (phase space) from the tome series data
        lag: int
            lag or reconstruction delay for creatinng the dynamic trajectory (phase space) from the tome series data
        avg_period: float
            average period of the timer series data, in seconds
        tol: float
            tolerance to deal with small distances to avoid numerical issues
        parallel: bool,
            to use the parallelised algorithm
        plot_flag: bool,
            to display the results
    """

    deltat = 1 / sampling_rate
    if avg_period:
        avg_period = int(np.ceil(avg_period / deltat))

    data_reshaped, M, _, lag, avg_period = timeseries_to_phasespace(
        data, deltat, dimension, lag, avg_period
    )
    data_distance = pairwise_distance(data_reshaped)
    xi_value = np.arange(M) * deltat
    if parallel:
        distance = numba_parallel_NN_distances(data_distance, avg_period)
    else:
        distance = NN_distances(data_distance, avg_period)
    distance[distance <= tol] = 1
    yi_value = np.sum(np.log(distance), axis=0) / M
    lambda_sato_alg = np.zeros((M, M))
    for i in range(M - 1):
        for k in range(1, M - i):
            lambda_sato_alg[i, k] = (1 / k / deltat) * (yi_value[i + k] - yi_value[i])
    print(distance.shape, yi_value.shape, xi_value.shape)
    line_seg_indx = find_linear_segment(xi_value, yi_value, avg_period, threshold=5)
    lle = np.polyfit(xi_value[0:line_seg_indx], yi_value[0:line_seg_indx], 1)
    print("LLE value based on the slope of the fitted line is:", lle[0])
    if plot_flag:
        plt.figure()
        plt.plot(xi_value, yi_value)
        plt.plot(
            xi_value[0:line_seg_indx], np.poly1d(lle)(xi_value[0:line_seg_indx]), "--r"
        )
        plt.show()

    return (
        data_reshaped,
        xi_value,
        yi_value,
        distance,
        lambda_sato_alg[:, 1:],
        lle[0],
        avg_period,
    )


def autocorr(x):
    result = np.correlate(x, x, mode="full")
    return result[result.size // 2 :]


def timeseries_to_phasespace(data, deltat, dimension, lag=None, avg_period=None):
    """
    Conatruct the phase space for a time series using the method of delays. See Step 2 above.
    """
    ac_data = autocorr(data)
    fft_amp = np.fft.fft(data, norm="forward")[0 : int(len(data) / 2)]
    fft_freq = np.fft.fftfreq(len(data), d=deltat)[0 : int(len(data) / 2)]
    N = len(data)

    if lag is None:
        lag = int(np.where(ac_data < (1 - 1 / np.e) * ac_data[0])[0][0])
    if avg_period is None:
        avg_period = int(
            np.ceil(1 / np.average(fft_freq, weights=np.abs(fft_amp)) / deltat)
        )

    print("lag & avg_period are:", lag, avg_period)

    M = N - (dimension - 1) * lag
    data_reshaped = np.zeros((M, dimension))
    for i in range(M):
        data_reshaped[i, :] = data[i : i + (dimension - 1) * lag + 1 : lag]

    data_reshaped = np.asanyarray(data_reshaped)
    return data_reshaped, M, dimension, lag, avg_period


def pairwise_distance(data_reshaped):
    """
    Calculate the distance between all combination of M states of the phase space.
    The self-state distance is set to inf (instead of 0) to avoid division by zero and log of zero erros
    """
    distance = sd.squareform(sd.pdist(data_reshaped, "euclidean"))
    for i in range(len(distance)):
        distance[i, i] = np.inf
    return distance


def find_NN(distance_sqm, j, avg_period):
    index_temp = np.argmin(distance_sqm[j, :])
    if np.abs(index_temp - j) > avg_period:
        index = index_temp
    else:
        distance_sqm[j, index_temp] = np.inf
        index = find_NN(distance_sqm, j, avg_period)
    return index


import numba


@numba.jit(parallel=True, nogil=True)
def numba_parallel_NN_distances(distance_sqm, avg_period):
    """
    Locate the nearest neighbour (NN) of each point on the dynamic trajectory (phase space), see Step 3 above.
    """
    M = distance_sqm.shape[0]
    distance_time = np.ones((M, M))
    distance_sqm_tmp = distance_sqm.copy()
    for j in numba.prange(M):
        index = find_NN(distance_sqm_tmp, j, avg_period)
        distance_time[j, 0] = distance_sqm[j, index]
        for i in range(M):
            if (index + i) < M and (j + i) < M:
                distance_time[j, i] = distance_sqm[j + i, index + i]
    return distance_time


def NN_distances(distance_sqm, avg_period):
    """
    Locate the nearest neighbour (NN) of each point on the dynamic trajectory (phase space), see Step 3 above.
    """
    M = distance_sqm.shape[0]
    distance_time = np.ones((M, M))
    distance_sqm_tmp = distance_sqm.copy()
    for j in range(M):
        index = find_NN(distance_sqm_tmp, j, avg_period)
        distance_time[j, 0] = distance_sqm[j, index]
        for i in range(M):
            if (index + i) < M and (j + i) < M:
                distance_time[j, i] = distance_sqm[j + i, index + i]
    return distance_time


def find_linear_segment(x_value, y_value, avg_period, threshold=5):
    """
    To locate the linear segment of the yi curve using the 2nd derivative of the curve. See the following link for more info:
    https://stackoverflow.com/questions/40226357/second-derivative-in-python-scipy-numpy-pandas
    """
    from scipy.interpolate import UnivariateSpline

    y_spl = UnivariateSpline(x_value, y_value, s=0, k=5)
    y_spl_2d = y_spl.derivative(n=2)
    yspl_2d = y_spl_2d(x_value)
    nw = np.where(np.abs(yspl_2d) < 0.01 * max(yspl_2d))[0]
    if np.any(nw):
        line_indx_tmp = nw[nw > threshold * avg_period]
        if np.any(line_indx_tmp):
            line_indx = line_indx_tmp[0]
        else:
            line_indx = nw[nw > avg_period][0]
    else:
        line_indx = len(y_value)
    return line_indx
