## Description and Overview
NetCDF (Network Common Data Form) is a set of software libraries and
machine-independent data formats that support the creation, access, and sharing
of array-oriented scientific data.  This includes the NetCDF library as
well as the NetCDF Operators (NCO), Climate Data Operators (CDO), NCCMP, and
NCVIEW packages.

Files written with previous versions can be read or written with the current
version.

## Using NetCDF on Cray System
NetCDF libraries on the Cori are provided by Cray. To see the available Cray
installations and versions use the following command:

```
module avail cray-netcdf
```
Loading the Cray NetCDF module files will also load a cray-hdf5 module file. 
No compile or link options are required as long as you use the Cray compiler wrappers. Below is an example for using the Cray compiler on Cori:

```
% module swap PrgEnv-intel PrgEnv-cray  (# or similar if needed to swap to another compiler)
% module load cray-netcdf 
% ftn  ... 
% cc   ... 
% CC   ... 
```

## NetCDF Operators (NCO)

The NetCDF Operators (NCO) are a suite of file operators that facilitate
manipulation and analysis of self-describing data stored in the NetCDF or
HDF4/5 formats.

To access the NetCDF operators, just load the nco module file with the 'module
load nco' command. This command will automatically load the NetCDF module file.

## NCCMP

The NCCMP tool compares two NetCDF files bitwise or with a user-defined
tolerance (absolute or relative percentage).

To use NCCMP, just load the nccmp module file with the 'module load nccmp' command. This command will automatically load the NetCDF module file.

## Documentation
The NetCDF software was developed at the
[Unidata Program Center](http://www.unidata.ucar.edu/) in Boulder, Colorado.

## Availability at NERSC

* Serial: cray-netcdf/4.4.1.6, cray-netcdf/4.6.1.3(default)
* Parallel: cray-netcdf-hdf5parallel/4.4.1.1.6, cray-netcdf-hdf5parallel/4.6.1.3(default)
