from wavelet_transform.plotting import *
from wavelet_transform.transform import *
from wavelet_transform.inverse_transform import *
from synthetic_data.synthetic_data import *

if __name__ == '__main__':
    amp_sin_waves = 1
    sampling_rate = 30e6

    # test_frequencies = [1e6*(i) for i in range(2,11)]
    test_frequencies = [1e6]

    for frequency in test_frequencies:
        frequencies = [frequency] # change frequency content. = frequency of non-randomised wave

        time_period = 1/frequency
        signal_duration = 10*time_period # full signal duration, signal exists between delay and end
        delay = 2*time_period
        end = 8*time_period

        gaussian_noise_std = 0.3 # set the standard devaition of the noise

        # generate data
        nyquist_frequency = sampling_rate/2

        data, t_plot = sound_wave(
            amp_sin_waves, frequencies, signal_duration, 
            sampling_rate, delay, end
            )

        gaussian_noise = noise(gaussian_noise_std,signal_duration,sampling_rate)
        data += gaussian_noise
        data = window(data)

        # transform
        levels = 3
        db_order = 9
        wavelet = f'db{db_order}'
        transform_object = dwpt(data,levels,wavelet)
        final_nodes = transform_object.get_level(levels)
        transform_object.calc_frequencies(sampling_rate)

        # selective inverse dwpt
        nodes_to_keep = [0,10e6]
        dwpt_result_nodes = transform_object.packet_nodes_for_rec(levels,keep = nodes_to_keep)
        kept_coefficients = transform_object.nodes2data(dwpt_result_nodes)
        inverse = idwpt(kept_coefficients,levels,wavelet)
        inverse = inverse[:len(data)]

        # # plot transform
        # fig1,ax1 = plt.subplots()
        # plot_wave(data,t_plot,ax1)
        # plt.show()

        # plot inverse transform
        title = f'inverse transform with nodes: [not implemented]'
        fig,ax = plt.subplots()
        plot_wave(inverse,t_plot,ax,title)
        plt.show()     

        plot_final_nodes(final_nodes,data,t_plot,frequencies,'',nyquist_frequency)