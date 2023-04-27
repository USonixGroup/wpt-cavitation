# WPT-Cavitation

This code was developed as part of an undergraduate project. It aims to understand the characteristics of acoustic cavitation using data measured with hydrophones. This has been developed directly from the method proposed by Teh M. in their thesis titled 'Mult-resolution analysis of acoustic emission signals using the packet transform' September 2014.

## Features
This library implements its own discrete wavelet packet transform (DWPT) and wavelet transform (DWT) based off results from the pywt library. It implements further code aimed to extract key data from the nodes obtained from the DWPT and create graphs to visualise the energy spread across the subbands, the stability, and entropy of each node. The measure of stability and energy is critical in separating out the harmonics, noise, and broadband noise that represents a mixture of stable and inertial cavitation.
