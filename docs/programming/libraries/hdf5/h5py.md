## Description and Overview
The [h5py](http://www.h5py.org/) package is a Pythonic interface to the [HDF5](https://www.hdfgroup.org/) library.

H5py provides an easy-to-use high level interface, which allows an application
to store huge amounts of numerical data, and easily manipulate that data from
NumPy.  H5py uses straightforward Python and NumPy metaphors, like dictionaries
and NumPy arrays. For example, you can iterate over datasets in a file, or
check the .shape or .dtype attributes of datasets. You don't need to know
anything special about HDF5 to
[get started](http://docs.h5py.org/en/latest/quick.html). H5py rests on an
object-oriented Cython wrapping of the HDF5 C API. Almost anything you can do
in HDF5 from C, you can do with h5py from Python.

## Availability at NERSC
If you want to use H5py with MPI-IO, then parallel h5py is recommended;
If you only use H5py for serial I/O (as most users do), you can load either
h5py, or NERSC's Python Anaconda. 

* Parallel H5Py, built with HDF5 1.10.2.0:
    * h5py-parallel/2.7.1
    * h5py-parallel/2.8.0
    * h5py-parallel/2.9.0 (Default, Recommended)	
* Serial H5py
    * python/2.7-anaconda-4.4 (default python 2.7), built with HDF5 1.8.17 
    * python3/3.6-anaconda-4.4 (default python 3.6), built with HDF5 1.10.1 (Recommended)

## How to Use H5Py

### Loading the Module
Serial H5Py
```
module load python
```

Parallel H5Py
```
module load python
module load h5py-parallel
```

### Using H5Py in the Codes

Serial H5Py
```
import h5py
fx = h5py.File('output.h5', 'w')
fx.close()
```

Parallel H5Py
```
from mpi4py import MPI
import h5py
fx = h5py.File('output.h5', 'w', driver = 'mpio', comm = MPI.COMM_WORLD)
fx.close()
```

## Basic Usage
### Different IO drivers
There are several HDF5 drivers available in h5py: sec2: unbuffered I/O;
stdio: buffered I/O; core: memory-mapped I/O; family: fixed-length file slices.
We recommend the default driver, which is sec2 on Unix.

H5py has several different I/O modes for opening files: r: readonly, file
must exist; r+: read/write, file must exist; w: create file, truncate if exists;
w- or x: create file, fail if exists; a: read/write if exists, create otherwise
(default)

```
import h5py
fx = h5py.File('output.h5', driver = <driver name>, 'w')
```

### Slice like Numpy

```
dx = fx['4857/55711/4/coadd'][('FLUX','IVAR')] # read 2 columns in the 'coadd' table dataset
dx = fx['4857/55711/4/coadd'][()] # read the whole 'coadd' dataset in the group '4857/55711/4'
dx = fx['path_to_dataset'][0:10] # slice the first 10 in the dataset
```

### Caution, Implicit Write! 
There is no explicit write function in h5py, all writes happen implicitly when
you do the assignment or dataset creation
```
# Initialize the dataset with existing numpy array
arr = np.arange(100)
dset = f.create_dataset('mydset', data = arr) # write happens here

# Rewrite h5py dataset with numpy array
dset = f.create_dataset('mydset', (10, 10), dtype = 'f8')
temp = np.random.random((2, 10))
dset[0:2, :] = temp # write happens here
```

## Common Errors

### Unknown Error 524
```
Unable to create file (unable to lock file, errno = 524, error message =
'Unknown error 524')
```
This usually happens on Burst Buffer or Project file systems. Root cause is
documented at HDF5 'known issues'. Simple fix to this is to disable file locking
in HDF5:
```
export HDF5_USE_FILE_LOCKING=FALSE     
```
### Dtype Wrong Size
```
numpy.dtype has the wrong size, try recompiling. Expected 88, got 96
```
This happens when the loaded python module is not what h5py is built with. You
should load python/2.7-anaconda-4.4 for serial h5py, or h5py-parallel (after
loading python/2.7-anaconda-4.4) for parallel h5py.



