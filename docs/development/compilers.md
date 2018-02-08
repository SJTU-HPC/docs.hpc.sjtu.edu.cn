# Compilers

Cray provides a convenient set of wrapper commands that should be used in almost all cases for compiling and linking parallel programs. Invoking the wrappers will automatically link codes with MPI libraries and other Cray system software. All MPI and Cray system include directories are also transparently imported. In addition the wrappers append the compiler's target processor arguments for the Cori compute node processors.

!!!note
	The intention is that programs are compiled on the login nodes and executed on the compute nodes. Because the compute nodes and login nodes have different operating systems, binaries created for compute nodes may not run on the login node. The wrappers mentioned above guarantee that codes compiled using the wrappers are prepared for running on the compute nodes.

!!!warning
	On Cori there are two types of compute nodes: Haswell and KNL.  While binaries built for Haswell do run on KNL (not vice versa), it is necessary to build for KNL explicitly in order to exploit the new AVX512 architecture. Please see below for more information on how to compile for KNL compute nodes.

## Basic Example

The Cray compiler wrappers take the place of other common wrappers such `mpif90`, `mpicc`, `mpiicc` and `mpic++`. By default the appropriate MPI libraries are included and if the `cray-libsci` module is loaded then BLAS and LAPACK will also be included if necessary.

### Fortran

```shell
ftn -o example.x example.f90
```

### C

```shell
cc -o example.x example.c
```

### C++

```shell
CC -o example.x example.cpp
```


## Makefile

In the ideal case one simply has to replace any variables or compiler commands with the Cray compiler wrappers.

## configure

Often specifying the compiler wrappers is enough for a configure step to succeed.

```shell
	./configure CC=CC cc=cc FC=ftn
```

Reading the output of `./configure --help` is often very instructive when determining what other options are appropriate. (e.g. `--enable-mpi`)
