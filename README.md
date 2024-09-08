# WPT-Cavitation

This code was developed as part of an ongoing project. It aims to understand the characteristics of acoustic cavitation using data measured with hydrophones.

## Features
This library implements its own discrete wavelet packet transform (DWPT) and wavelet transform (DWT) based off results from the pywt library. It implements further code aimed to extract key data from the nodes obtained from the DWPT and create graphs to visualise features such as energy spread and kurtosis across the subbands. Unsupervised machine learning models have been implemented and explored to classify packets to break down the signal into components representing inertial and stable cavitation.

## Wavelet Packet Transform
Discrete wavelet and discrete wavelet packet transforms have been implemented and validated as well as their inverses. Packets can be reconstructed separately using the idwt and idwpt functions.

## Visualisation
Heisenberg and tree plots are available and fully integrated with the transforms.
