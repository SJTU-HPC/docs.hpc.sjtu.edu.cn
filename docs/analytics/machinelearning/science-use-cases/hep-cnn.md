# Deep Networks for HEP

This page provides example code, datasets and recipes for running HEP
Physics analyses using deep neural networks on Cori. The current
scripts were those used for the CNN classification and timing studies
reported at [this ACAT paper](https://arxiv.org/abs/1711.03573).

## Datasets

These contain simulated data with an ATLAS-like detector. Data is
available from
http://portal.nersc.gov/project/mpccc/wbhimji/RPVSusyData/ . A README
is provided in the directory.

## Convolutional Neural Network for Classification

This provides a network for classification (RPVSusy signal vs QCD
background) on 3-channel (calorimeter + track) whole-detector images
as described in [the ACAT paper](https://arxiv.org/abs/1711.03573). It
implements 3 convolution+pooling units with rectified linear unit
(ReLU) activation functions. These layers output into two fully
connected layers, with cross-entropy as the loss function and the ADAM
optimizer.

## Code 

The Keras code to implement the CNN used in the paper is available at
https://github.com/eracah/atlas_dl/tree/micky . This single script is
fairly self explanatory and easily run at NERSC.

Code for preselection of data as well as for Lasgne/Theano
implementations is in the main branch of that repository.

Code for shallow classifiers compared in the paper is available at
https://github.com/sparticlesteve/acat2017-rpvdl-shallow.

!!! note
	A more recent tensorflow implementation is used in the NERSC
	science [benchmarks](../benchmarks.md#cnn)
