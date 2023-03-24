from wavelet_transform.plotting import *
from wavelet_transform.transform import *
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
        final_level_nodes = transform_object.get_level(levels)

        # calculate energy
        transform_object.calc_frequencies(sampling_rate)
        transform_object.show_all_energy()

        # set title
        f = sci_form(frequency)
        title = f"- signal frequency = {f} Hz"

        # plot everything
        plot_final_nodes(
            final_level_nodes, data, t_plot, frequencies, 
            title, nyquist_frequency
            )

        # # find nodes with frequencies of interest        
        # frequencies_of_interest = [1e6,1e7,0.5e7]

        # nodes_of_interest = transform_object.find_nodes(final_level_nodes,frequencies_of_interest)

        # # plot data from selected nodes
        # rows = int(len(nodes_of_interest))
        # fig,ax = plt.subplots(rows)
        # plot_nodes(nodes_of_interest,ax,nyquist_frequency,rows,columns = 1)
        # fig.set_size_inches(12,3*rows)

        # # create window
        # a = ScrollableWindow(fig)