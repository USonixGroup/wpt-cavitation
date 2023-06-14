import numpy as np

def sin(A,frequency,phi,t):
    '''
    returns a sine wave with specified parameters

    Parameters
    ----------
    A: float
        the amplitude of the sine wave
    frequency: float
        the frequency of the sine wave
    phi: float
        the phase angle of the sine wave
    t: arraylike
        
    Returns
    -------
    sin_wave: array
        a sine wave represented as an array of amplitudes
    '''
    omega = 2*np.pi*frequency
    sin_wave = A*np.sin(omega*t-phi)
    return sin_wave

def sound_wave(A, frequencies, duration, sampling_rate,  delay = None, end = None):
    ''' 
    Returns a sound wave with made up of sine waves with frequencies given as a 
    list. delay and end can be specified to start the wave early and cut it of 
    before the specified duration.

    Parameters
    ----------
    A_max: float
        the maximum amplitude of each sin wave.
    frequencies: list or float
        a list of frequencies to include in the sound wave or a single frequency
    duration: float
        the time duration of the signal.
    sampling_rate: float
        sampling_rate of the sensor in hz
    delay: float
        a time delay for the signal.
    end: float
        time that the signal ends at.

    Returns
    -------
    wave: array
        tuple containing an array of cthe sound wave 
    t: array
        the array of time to plot the wave against
    '''

    t = np.arange(0,duration,1/sampling_rate)
    wave = np.zeros(int(len(t)))

    # initialise parameters
    phi = 0

    if isinstance(frequencies,(int,float)): 
        frequencies = [frequency] # compatibility with a single frequency

    for frequency in frequencies:
        # generate sine wave
        sin_wave = sin(A,frequency,phi,t)
        # add sine wave to total wave
        wave = wave + sin_wave     

    # add delay and end
    if end is None:
        end = duration
    if delay is None:
        delay = 0
    delay_index = find_delay_index(t,delay)
    end_index = find_end_index(t,end)

    wave[:delay_index] = 0
    wave[end_index:] = 0
    
    return wave, t

def noise(standard_deviation, duration=None, sampling_rate=None, time=None):
    '''
    creates and array of noise with the same dimensions as the synthetic data
    can be specified using either an array of time to fit noise into or can 
    '''
    if duration is not None and sampling_rate is not None:
        t = np.arange(0,duration,1/sampling_rate)
    elif time:
        t = time
    else:
        raise ValueError('provide the sampling rate and duration of the signal or an array of time')
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