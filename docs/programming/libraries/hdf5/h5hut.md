## Description and Overview
H5hut (HDF5 Utility Toolkit) is a veneer API for HDF5: H5hut files are also
valid HDF5 files and are compatible with other HDF5-based interfaces and tools.
For example, the h5dump tool that comes standard with HDF5 can export H5hut
files to ASCII or XML for additional portability. H5hut also includes tools to
convert H5hut data to the Visualization ToolKit (VTK) format and to generate
scripts for the GNUplot data plotting tool.

Main modules:

* H5Part: Variable-length 1D particle arrays.
* H5Block: Rectilinear 3D scalar and vector fields.
* H5Fed: Adaptively refined tetrahedral and triangle meshes.

## How to Use H5Hut

```
% module load cray-hdf5
% module load h5hut
% cc 
% ftn  
```
```
% module load cray-hdf5-parallel 
% module load h5hut-parallel
% cc 
% ftn  
```

[Documentation](https://gitlab.psi.ch/H5hut/src/wikis/home)

## Availability

* Cori: 1.99.13, 2.0.0rc2((Default))

