from ..utils import extensions as ext
import matplotlib.pyplot as plt
import numpy as np

def full_plot(
    nodes, data, t_plot, sampling_rate, 
    title='', frequencies=None 
    ):
    '''
    formatted charts for plotting the results from a dwt or dpwt. Combines relevant plots
    including those for the sound wave, frequency content of the sonud wave (using synthetic data 
    package), and plots of the data at each node. 

    Parameters
    ----------
    nodes: list
        list of node objects
    data: array_like
        an array of the data from the sound wave
    t_plot: array_like
        the time array to plot the data against
    nyquist_frequency: float
        1/2*sampling rate
    frequencies: list
        list of floats containing the frequencies of the synthetic sound wave for the historgram
    '''
    # plot data
    fig_1, ax_1 = plt.subplots()
    plot_wave(data, t_plot, ax_1, title)

    # plot transform coefficients all at once
    if frequencies: # plot a histogram of frequency content in data as well
        fig_2, ax_2 = plt.subplots(2)
        plot_all_final_nodes(nodes, ax_2[0], title)
        plot_frequencies(frequencies, ax_2[1], title) 
    else:
        fig_2, ax_2 = plt.subplots()
        plot_all_final_nodes(nodes, ax_2, title)

    # plot each node separately.
    rows = round(len(nodes)/2)
    fig_3, ax_3 = plt.subplots(rows,2)
    fig_3.set_size_inches(12,3*rows)
    plot_nodes(nodes, ax_3, sampling_rate, rows, columns=2)

    fig_1.tight_layout()
    fig_2.tight_layout()
    fig_3.tight_layout()
    plt.show()
    

def plot_nodes(nodes, ax, sampling_rate, rows=None, columns=1):
    '''
    separately plots the data from a provided list of nodes.

    Parameters
    ----------
    nodes: list
        list of nodes to plot the data for
    ax: axis
        ax from fig,ax = plt.subplots()
    nyquist_frequency:  float
        1/2*sampling rate
    rows: int
        the number of rows to create
    columns: int
        the number of columns to create
    '''
    if not rows: # default number of rows
        rows = round(len(nodes))
    section_counter = 1

    nyquist_frequency = sampling_rate/2
    for row in range(rows):
        for column in range(columns):
            # plot coefficients
            try:
                node = nodes[section_counter-1]
            except IndexError:
                break
            if rows == 1 :
                axis = ax[column]
            elif columns == 1:
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

            lf = ext.sci_form(lower_frequency)
            uf = ext.sci_form(upper_frequency)

            title = f'''
            Coefficients, {lf}-{uf} Hz
            energy: {ext.sci_form(node.energy)} [section {section_counter} of {len(nodes)}]
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
    if frequencies:
        ax.hist(frequencies)
        ax.set_title(f"Frequency Distribution {title}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Frequency of frequency")
        ax.grid(True)