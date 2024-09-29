import numpy as np


def add_0s(number, i):
    "Add 0s to an integer to make an i-digit long string"
    num = str(number)
    for _ in range(i - len(str(number))):
        num = "0" + num
    return num


def get_data(voltage, exp_no, duty_cycle, acd=True):
    num = add_0s(exp_no, 5)
    pe = "PE_" if acd is True else ""
    file = f"../ACD-sept 2024/C1{pe}{voltage}mVpp_1p1MHz_5p5kHzPRF_{duty_cycle}perDC{num}.dat"

    with open(file, "r") as file:
        lines = np.loadtxt(file)

    acd_data = lines[:, 1]
    time = lines[:, 0]
    time -= time[0]

    return acd_data, time


def to_float(array):
    return [float(i) for i in array]
