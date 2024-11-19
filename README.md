# WPT-Cavitation

A python library for wavelet transforms and machine learning for classifying cavitation using data captured by hydrophones. 

This library was developed as part of an ongoing project associated with an upcoming paper:

Acoustic cavitation detection and classification using wavelet packet transform and
K-Means clustering

## Dependencies

WPT-Cavitation is dependent on numpy and scipy for the wavelet transform. Plotting requires matplotlib and plotly.

## Features

This library implements its own discrete wavelet packet transform (DWPT) and wavelet transform (DWT) based off results from the pywt library. The implementation is focused around extracting key data from the nodes obtained from the DWPT and creating graphs to visualise features such as energy spread and kurtosis across the subbands. Unsupervised machine learning models have been implemented and explored to classify packets and break down the signal into components representing inertial and stable cavitation.

## Wavelet Packet Transform

Discrete wavelet and discrete wavelet packet transforms have been implemented and validated as well as their inverses. Packets can be reconstructed separately using the idwt and idwpt functions.

## Visualisation

The library provides several options for visualising the results from the wavelet transform and outputs from the machine learning algorithms.

Heisenberg plots display the time-frequency characteristics of the signal, useful for identifying stable and transisent features. Tree plots help visualise the structure of the wavelet packet decomposition, providing information about the distribution of statistical features across the subbands.

## Get help

## Citation
