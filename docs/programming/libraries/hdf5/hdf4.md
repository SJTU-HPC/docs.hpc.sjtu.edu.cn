## Description and Overview

Hierarchical Data Format is a set of file formats, libraries, and tools for
storing and managing large scientific datasets. Originally developed at the
National Center for Supercomputing Applications, it is currently supported by
the non-profit HDF Group and is now typically called 'HDF4' to distinguish it
from the later 'HDF5' package.

HDF files written with older versions can be read with the current version.
HDF is also forward compatible, at least in regards to the data. Metadata, such
as attributes, may not be readable by previous releases, but the data will be.
For more information on backward and forward compatibility for HDF see the
[HDF data compatibility table](https://support.hdfgroup.org/products/hdf4/HDF-FAQ.html#compat).


## How to User HDF4

Load the appropriate modulefile and then you can use the HDF C or Fortran
wrappers:

```
module load hdf
h4cc ... 
h4fc ... 
```

## Documentation
* [HDF web site](https://portal.hdfgroup.org/display/HDF4/HDF4) 
* [How is HDF5 different than HDF4](https://portal.hdfgroup.org/display/knowledge/How+is+HDF5+different+than+HDF4)

## Availability at NERSC

* Cori: hdf/4.2.11
* Edison: hdf/4.2.8

