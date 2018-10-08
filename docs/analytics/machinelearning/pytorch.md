# PyTorch

[PyTorch](https://pytorch.org/) is a high-productivity Deep Learning framework
based on dynamic computation graphs and automatic differentiation.
It is designed to be as close to native Python as possible for maximum
flexibility and expressivity.

## Availability on Cori

PyTorch can be picked up from the Anaconda python installations (e.g. via
`module load python`) or from dedicated modules with MPI enabled. You can
see which version are available with `module avail pytorch-mpi`.

## Multi-node training

PyTorch makes it fairly easy to get up and running with multi-node training
via its included _distributed_ package. Refer to the distributed tutorial for
details: https://pytorch.org/tutorials/intermediate/dist_tuto.html

Note the above tutorial doesn't actually document our currently recommended
approach of using DistributedDataParallelCPU. See examples below.

## Examples

We're putting together a coherent set of example problems, datasets, models,
and training code in this repository:
https://github.com/NERSC/pytorch-examples

This repository can serve as a template for your research projects with a
flexibly organized design for layout and code structure. The `template` branch
contains the core layout without all of the examples so you can build your
code on top of that minimal, fully functional setup. The code provided should
minimize your own boiler plate and let you get up and running in a distributed
fashion on Cori as quickly and seamlessly as possible.

The examples include:

* A simple hello-world example
* HEP-CNN classifier
* ResNet50 CIFAR10 image classification
* HEP-GAN for generation of RPV SUSY images.

The repository will also be used to benchmark our system for single and
multi-node training.
