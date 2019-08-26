# HDF5 -  Hierarchical·Data·Format·version·5

Hierarchical Data Format version 5 (HDF5) is a set of file formats,
libraries, and tools for storing and managing large scientific
datasets. Originally developed at the National Center for
Supercomputing Applications, it is currently supported by the
non-profit [HDF Group](https://www.hdfgroup.org/).

HDF5 is different product from previous versions of software named
HDF, representing a complete redesign of the format and library.  It
also includes improved support for parallel I/O. The HDF5 file format
is not compatible with HDF 4.x versions.

!!! note
	`h5toh4` and `h4toh5` converters are available on all NERSC
	machines.

## Using HDF5

Cray provides native HDF5 libraries. Use the command `module avail
cray-hdf5` to see the available Cray versions.

### Serial HDF5 on Cori

```bash
cori$ module load cray-hdf5
```

### Parallel HDF5 on Cori

```bash
cori$ module load cray-hdf5-parallel
```

## Further Information

* [HDF5 Performance Tuning](../../../performance/io/library/index.md)
  Best practices for tuning HDF5 applications' I/O performance
* [ExaHDF5](https://sdm.lbl.gov/exahdf5/) Research and Development
  funded by ECP
* [The HDF Group](https://www.hdfgroup.org/) Documments and Support
  from official HDF group

!!! tip
	For questions about HDF5 on any NERSC systems, please
	contact [NERSC Consulting](https://help.nersc.gov).
