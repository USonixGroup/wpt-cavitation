import pandas as pd
import numpy as np
import scipy.io as sio


def add_0s(number, i):
    "Add 0s to an integer to make an i-digit long string"
    num = str(number)
    for _ in range(i - len(str(number))):
        num = "0" + num
    return num


def get_data(dataset):
    if dataset in ["5-10", "5-34", "6-12"]:
        return dataset1(dataset)  # get data from 1st dataset

    elif (
        isinstance(dataset, tuple)
        and dataset[0] in [80, 120, 200, 280]
        and dataset[1] in [1, 5, 10]
    ):
        return dataset2(dataset)  # get data from 2nd dataset

    else:
        raise ValueError("invalid dataset named")


def dataset1(dataset):
    # Get parameters for extracting signal from input dataset
    datasets1 = {
        "5-10": (50, 3, 20, 2e6, 2500),
        "5-34": (0, 2, 20, 2e6, 2150),
        "6-12": (1, 2, 22, 7.37e6, 0),
    }
    sampling_rate = 50e6
    i, n, n_cycles, freq, i_lo = datasets1[dataset]
    i_hi = int(i_lo + n_cycles / freq * sampling_rate)

    # Get raw data
    num = add_0s(i, n)

    if dataset == "5-10":
        # i = [50, 100], n = 3, (i = 50, i_lo = [2500::5000])
        file = (
            r"C:\Users\maxau\OneDrive\Codes\WPT-Cavitation\docs\PCD data for paper\TestID5-10W per cm2 intensity\C1test00"
            + num
            + ".txt"
        )

    elif dataset == "5-34":
        # i = [00 - 21], n = 2, (i = 00, i_lo = [2150::5000])
        file = (
            r"C:\Users\maxau\OneDrive\Codes\WPT-Cavitation\docs\PCD data for paper\TestID5-34W per cm2 intensity\C2test016"
            + num
            + ".txt"
        )

    elif dataset == "6-12":
        # i = [01 - 20], n = 2, (i = 01, i_lo = [0::5000])
        file = (
            r"C:\Users\maxau\OneDrive\Codes\WPT-Cavitation\docs\PCD data for paper\TestID6-12W per cm2 intensity\C1test000"
            + num
            + ".txt"
        )
    dfb = pd.read_csv(file, sep=",", header=None)

    data = dfb.iloc[:, -1]
    time = dfb.iloc[:, -2]

    time -= time[0]

    return np.array(data[i_lo:i_hi]), np.array(time[i_lo:i_hi])


def dataset2(dataset):
    voltage, frequency = dataset
    file = (
        r"C:\Users\maxau\OneDrive\Codes\WPT-Cavitation\docs\Data used for the IOP paper 2014\Test2_HIFU_NoVial_"
        + str(voltage)
        + r"mVpp_Modf.mat"
    )
    data = sio.loadmat(file)
    signal = data["c" + str(frequency) + "hat"]  # Get c1hat, c5hat, c10hat
    time = np.arange(len(signal)) / data["fs"]  # Generate time array with sampling rate

    return signal.squeeze(), time.squeeze()
