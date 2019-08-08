# Data Formats

I/O continues to be one of the main bottlenecks for scientific
applications at scale.  We support two software packages that many application
developers use to manage I/O of heterogeneous types of binary
application data used on many different platforms.  HDF5 and NETCDF
are both implemented on top of MPI‐IO and have gained popularity as
alternatives to basic POSIX I/O.  HDF5 is a machine-independent and
self-documenting file format. Each HDF5 file "looks" like a directory
tree, with subdirectories, and leaf nodes that contain the actual
data. This means that data can be found in a file by referring to its
name, rather than its location in the file.  NetCDF is a file format
and support library developed at the National Center for Atmospheric
Research (NCAR).  Like HDF5, NetCDF is self-describing and portable
across platforms.

* [HDF5](../programming/libraries/hdf5/index.md)
* [NetCDF](../programming/libraries/netcdf/index.md)
