# Data Formats

I/O continues to be one of the main bottlenecks for scientific applications.  Here are two software packages that many application developers use to manage input/output of heterogeneous types of binary application data used on many different platforms.  HDF5 and NETCDF are both implemented on top of MPI‐IO and have gained popularity as alternatives to  basic POSIX API.  HDF5 is a machine-independent and self-documenting file format. Each HDF5 file "looks" like a directory tree, with subdirectories, and leaf nodes that contain the actual data. This means that data can be found in a file by referring to its name, rather than its location in the file.  NetCDF is a file format and support library developed at the National Center for Atmospheric Research (NCAR).  Like HDF5, NetCDF is self-describing and portable across platforms.

## HDF5

Hierarchical Data Format version 5 (HDF5) is a set of file formats, libraries, and tools for storing and managing large scientific datasets. Originally developed at the National Center for Supercomputing Applications, it is currently supported by the non-profit HDF Group.

HDF5 is different product from previous versions of software named HDF, representing a complete redesign of the format and library.  It also includes improved support for parallel I/O. The HDF5 file format is not compatible with HDF 4.x versions. You can use the 'h5toh4' and 'h4toh5' converters that are available on all NERSC machines. 

## netCDF

NetCDF (Network Common Data Form) is a set of software libraries and machine-independent data formats that support the creation, access, and sharing of array-oriented scientific data.  This includes the libnetcdf.a library as well as the NetCDF Operators (NCO), Climate Data Operators (CDO), NCCMP, and NCVIEW packages.
