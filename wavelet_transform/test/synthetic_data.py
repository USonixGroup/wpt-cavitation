import numpy as np
from scipy import signal

def sin(A, frequency, phi, t):
    omega = 2*np.pi*frequency
    sin_wave = A*np.sin(omega*t-phi)
    return sin_wave

def sound_wave(A, frequencies, duration, sampling_rate,  delay = 0, end = 0):
    ''' 
    Returns a randomly generated sound wave made up of "n" 
    sine waves. The frequencies present in the wave is tracked 
    with the variable "frequencies" This will be plotted in a 
    histogram. Creates a single sine wave by default.

    Parameters
    ----------
    A_max: float
        the maximum amplitude of each sin wave.
    bandwidth: float
        the maximum frequency of each sin wave.
    duration: float
        the time duration of the signal.
    sampling_rate: float
        sampling_rate of the sensor in hz
    n: int
        number of sine waves to be added to the signal
    band_range: float
        the frequency is selected from the last "band_range"
        (e.g. 1/4) of the selected bandwidth
    delay: float
        a time delay for the signal.
    end: float
        time that the signal ends at.
    rand: boolean
        choose whether to randomise the frequency, phase, 
        and amplitude of the sin wives. Set to False to 
        suppress randomness.
    fundamental_frequency: float
        if this is passed, n harmonics will be added together 
        to form the wave
    '''

    t = np.arange(0,duration,1/sampling_rate)
    wave = np.zeros(int(len(t)))

    # initialise parameters
    phi = 0

    for frequency in frequencies:
        # generate sine wave
        sin_wave = sin(A,frequency,phi,t)
        # add sine wave to total wave
        wave = wave + sin_wave     
        # track the frequency of the sine wave added

    # add delay and end
    delay_index = find_delay_index(t,delay)
    end_index = find_end_index(t,end)

    wave[:delay_index] = 0
    wave[end_index:] = 0
    
    return wave, t

def noise(standard_deviation, duration, sampling_rate):
    t = np.arange(0,duration,1/sampling_rate)
    noise = np.random.normal(0,standard_deviation,len(t))
    return noise

def find_delay_index(t,delay):
    'finds the index of the start of the delayed signal'
    for delay_index in range(len(t)):
        if t[delay_index] > delay:
            return delay_index

def find_end_index(t,end):
    'finds the index of the end of the cut off signal'
    end_index = len(t) - 1
    for _ in range(len(t)):
        if t[end_index] < end:
            return end_index
        end_index -= 1

def window(data,mode = 'hanning'):
    if mode == 'hanning':
        data *= np.hanning(len(data))
    elif mode == 'gaussian':
        data *= signal.windows.gaussian(len(data),1)
    return data*np.hanning(len(data))