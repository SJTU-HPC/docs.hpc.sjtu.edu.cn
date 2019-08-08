The HPC I/O stack has multiple layers. From top to bottom: H5py, HDF5,
MPI-IO, and POSIX. Most applications tend to use high-level I/O libraries,
such as HDF5, which is built on top of MPI-IO. More and more frequently,
users prefer to use the HDF5 Python wrapper, H5py. There are also
many legacy
applications that were written directly with MPI-IO, or even POSIX I/O.
Different I/O tuning strategies are appropriate for
different layers. We recommend 
that users tune the layer that the application is directly built upon.
For example, if your application is built with H5py, please refer to H5py
I/O Tuning. 

## MPI I/O Tuning

### MPI-I/O Collective Mode

Collective I/O operations are a set of optimizations available in many
implementations of MPI-I/O that improve the performance of large-scale
I/O to shared files. To enable these optimizations, you must use the
collective calls in the MPI-I/O library, which end in `_all`, e.g.
`MPI_File_write_at_all()`. Also, all MPI tasks in the given
MPI communicator must participate in the collective call, even if they
are not performing any I/O themselves. The MPI-I/O library has a
heuristic to determine whether to enable collective buffering, the
primary optimization used in collective mode.

### Collective Buffering

Collective buffering, also called two-phase I/O, breaks an I/O
operation into two stages. For a collective read, the first stage uses
a subset of MPI tasks (called aggregators) to communicate with the I/O
servers (OSTs in Lustre) and read a large chunk of data into a
temporary buffer. In the second stage, the aggregators ship the data
from the buffer to its destination among the remaining MPI tasks using
point-to-point MPI calls. A collective write does the reverse,
aggregating data with MPI into buffers on the aggregator nodes,
then writing from the aggregator nodes to the I/O servers. The
advantage of collective buffering is that fewer nodes are
communicating with the I/O servers, which reduces contention while
still attaining high performance through concurrent I/O transfers. In
fact, Lustre performs best with a one-to-one mapping of aggregator nodes
to OSTs.

Since the release of mpt/3.3, Cray has included a Lustre-aware
implementation of the MPI-I/O collective buffering algorithm. This
implementation is able to buffer data on the aggregator nodes into
stripe-sized chunks so that all read and writes to the Lustre
filesystem are automatically stripe-aligned without requiring any
padding or manual alignment from an application. Because of the way
Lustre is designed, alignment is a key factor in achieving optimal
performance.


### MPI-I/O Hints

Several environment variables can be used to control the behavior of collective
buffering on Cori. The MPICH_MPIIO_HINTS variable specifies hints to the
MPI-I/O library that can, for instance, override the built-in heuristic and
force collective buffering on:

```shell
export MPICH_MPIIO_HINTS="*:romio_cb_write=enable:romio_ds_write=disable"
```

Placing this command in your batch file before calling aprun will
cause your program to use these hints. The `*` indicates that the hint
applies to any file opened by MPI-I/O, while `romio_cb_write` controls
collective buffering for writes and `romio_ds_write` controls data
sieving for writes, an older collective mode optimization that is no
longer used and can interfere with collective buffering. The options
for these hints are enabled, disabled, or automatic (the default
value, which uses the built-in heuristic).

It is also possible to control the number of aggregator nodes using
the `cb_nodes` hint, although the MPI-I/O library will default to
automatically setting this to the stripe count of your file.

When set to a value of 1, the `MPICH_MPIIO_HINTS_DISPLAY` variable causes your
program to dump a summary of the current MPI-I/O hints to stderr each
time a file is opened. This is useful for debugging and as a sanity
check against spelling errors in your hints.

More detail on MPICH runtime environment variables, including a full
list and description of MPI-I/O hints, is available from the `intro_mpi`
man page on Cori.

## HDF5 I/O Tuning
[Optimizations for HDF5 on Lustre](https://www.nersc.gov/users/training/online-tutorials/introduction-to-scientific-i-o/?show_all=1)

## H5py I/O Tuning

* Choose the most modern format in file creation [Sample Codes](https://github.com/valiantljk/h5boss/blob/master/h5boss_py/scripts/test_file_creation.py)

By default, every file that is created by HDF5 uses the most backwardly
compatible version of the file format, which is less-efficient than more
recent versions of the file format. By turning on the 'latest' format during
file creation, you may save I/O cost. Refer to
[version bounding](http://docs.h5py.org/en/latest/high/file.html#version-bounding) for more information. 

```
f = h5py.File('name.h5', libver='earliest') # most compatible
f = h5py.File('name.h5', libver='latest')   # most modern
```
_"Using 1 process to create 1 file with 8000 objects (including subgroups,
different datasets, etc), 'latest' version achieved 2.25X speedup."_ 

* Speedup application I/O with
[Collective I/O](https://www.nersc.gov/users/storage-and-file-systems/optimizing-io-performance-for-lustre/#toc-anchor-4)
(Sample Codes:
[Parallel_Read](https://github.com/valiantljk/h5py/blob/master/examples/parallel_read.py),
[Parallel_Write](https://github.com/valiantljk/h5py/blob/master/examples/parallel_write.py))

```
with dset.collective:
 dset[start:end,:]=temp
```
_"Using 1K processes to write a 1TB file, collective IO achieved a 2X speedup on Cori."_

### Avoid Datatype Conversions

By default, numpy uses 64-bit floating-point values. However, whenever HDF5
accesses a dataset with another datatype (through H5py), those elements must
must be converted, which is a costly operation. For example, if we create a
dataset with dtype 'f' (which indicates a 32-bit float), and then assign a
numpy array to this dataset, HDF5 will perform datatype conversion as it
performs the I/O. To avoid type conversion when writing numpy arrays, the 'f8'
type must be used when creating the H5py dataset.  For example: 

```
#slower performance
dset = f.require_dataset('field', shape, 'f')
dset = temp # temp is a numpy array, which by default is 64-bit floats
```
```
#faster performance
dset = f.require_dataset('field', shape, 'f8')
dset = temp
```

_This greatly reduces the I/O time, from 527 seconds to 1.3 seconds when writing a 100x100x100 array with 5x5x5 procs_

### Use Low-level API 

([Sample Codes](https://github.com/valiantljk/h5boss/blob/master/h5boss_py/scripts/test_lower_api.py%20))

H5py provides a nice object-oriented interface by hiding many details that are
available in the HDF5 C interface. Fortunately, users can still leverage the
flexible HDF5 C features for tuning I/O performance. H5py allows you to do so
by using the h5py low-level [API](http://api.h5py.org/). For examples:

```
space = h5py.h5s.create_simple((100,)) 
plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE) 
plist.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
```

Application developers should also consider disabling dataset element
pre-filling, using the low-level API.
The following code creates a 900GB dataset, and by using 'FILL_TIME_NEVER' 
reduces the I/O cost from 40 minutes to less than 1 second:

```
fx = h5py.File('test_nofill.h5', 'w') 
spaceid = h5py.h5s.create_simple((30000, 8000000))  
plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)  
plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)  
plist.set_chunk((60, 15625))  
datasetid = h5py.h5d.create(fx.id, "data", h5py.h5t.NATIVE_FLOAT, spaceid, plist)
dset = h5py.Dataset(datasetid)
fx.close()  
```

### Case Study with H5Boss

H5py has been used in many scientific applications. One use case is
from astronomy, in which the developers built a customized file structure
based on HDF5 and developed query/subsetting/updating functions for managing
BOSS spectral data, SDSS-II.
For more details: [H5Boss](http://portal.nersc.gov/project/das/jialin/h5boss/h5boss/h5boss_py/docs/build/html/index.html).

### Performance vs. Productivity

[NERSC Data Seminar May 26 2017](http://www.nersc.gov/assets/Uploads/H5py-2017-May26-public.pdf)

