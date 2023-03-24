from .extensions import sci_form
import matplotlib.pyplot as plt
import numpy as np
from wavelet_transform.extensions import ScrollableWindow

def plot_final_nodes(
    nodes, data, t_plot, frequencies, title, 
    nyquist_frequency
    ):
    'formatted charts for plotting the results from a dwt or dpwt'
    # plot data
    fig_1, ax_1 = plt.subplots()
    plot_wave(data, t_plot, ax_1, title)

    # plot transform coefficients all at once
    # plot a histogram of frequency content in data
    fig_2, ax_2 = plt.subplots(2)
    plot_all_final_nodes(nodes, ax_2[0], title)
    plot_frequencies(frequencies, ax_2[1], title) 

    # plot each node separately.
    rows = round(len(nodes)/2)
    fig_3, ax_3 = plt.subplots(rows,2)
    fig_3.set_size_inches(12,3*rows)
    plot_nodes(nodes, ax_3, nyquist_frequency, rows)
    u = ScrollableWindow(fig_3)
    plt.show()
    

def plot_nodes(nodes, ax, nyquist_frequency, rows, columns = 2):
    '''
    separately plots the data from a provided list of nodes
    there are always 2 columns
    '''
    section_counter = 1
    for row in range(rows):
        for column in range(columns):
            # plot coefficients
            try:
                node = nodes[section_counter-1]
            except IndexError:
                break
            if rows == 1 :
                axis = ax[column]
            if columns == 1:
                axis = ax[row]
            else:
                axis = ax[row, column]
            axis.plot(node.data)
            
            # set labels
            axis.set_xlabel('coefficient number')
            axis.set_ylabel('magnitude')
            axis.grid(True)

            lower_frequency = node.norm_frequency_range[0]*nyquist_frequency
            upper_frequency = node.norm_frequency_range[1]*nyquist_frequency

            lf = sci_form(lower_frequency)
            uf = sci_form(upper_frequency)

            title = f'''
            Coefficients, {lf}-{uf} Hz
            energy: {sci_form(node.energy)} [section {section_counter} of {len(nodes)}]
            '''
            axis.set_title(title)
            section_counter += 1

def plot_all_final_nodes(nodes, ax, title = ''):
    'concatenate the data from all of the nodes and plot'
    output = np.array([])
    for node in nodes:
        output = np.concatenate((output,node.data))
    ax.plot(output)
    ax.set_title(f"Coefficients {title}")
    ax.set_xlabel("Coefficient Number")
    ax.set_ylabel("Coefficient")
    ax.grid(True)

def plot_wave(data, t_plot, ax, title = ''):
    'plots the data'
    ax.plot(t_plot,data)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [m]")
    ax.set_title(f"Data {title}")
    ax.grid(True)

def plot_frequencies(frequencies, ax, title =''):
    'creates a histogram of the frequencies present in the synthetic data'
    ax.hist(frequencies)
    ax.set_title(f"Frequency Distribution {title}")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Frequency of frequency")
    ax.grid(True)