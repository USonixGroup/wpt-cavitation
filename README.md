# WPT-Cavitation

A python library for wavelet transforms and machine learning for 1D time-seriers data. 

The main features are:

- a 1D implementation of the discrete wavelet packet transform (DWPT) and wavelet transform (DWT)
- inverse transforms which can reconstructed selected nodes individually
- minimum entropy decomposition
- decomposition for nodes under a selected frequency threshold
- statistical features of node data
- bubble dynamics simulations with the Gilmmore-Akulichev model
- visualisation with tree diagrams and spectrograms
- classification of nodes with clustering algorithms

The wavelet transform has been tested against results from the PyWavelets library and matlab although the implementation only relies on filter coefficients from PyWavelets.

This library was developed as part of an ongoing project associated with an upcoming paper:

Acoustic cavitation detection and classification using wavelet packet transform and
K-Means clustering

This library is partially funded by the Focused Ultrasound Foundation.

## Examples

Bubble dynamics simulation:
![bdsimulation](bdsimulation.jpg)

Heisenberg plot of wavelet packet transform:
![heisenberg](heisenberg.png)

Tree plot of wavelet transform with node size corresponding to the energy in the node:
![treeplot](treeplot.png)

Reconstructed data:
![reconstruction](reconstruction.jpg)

## Installation

This library can be used by cloning and pulling from main.

WPT-Cavitation is dependent on NumPy, SciPy, and PyWavelets for the wavelet transform and scikit-learn for the k-means algorithm. Plotting requires matplotlib, plotly, and pygraphviz.

## Get help

Contact on GitHub or at max.au-yeung20@ucl.ac.uk.

## Citation

This library can be cited as ...