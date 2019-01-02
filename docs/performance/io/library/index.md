
# MPI-I/O Collective Mode

Collective mode refers to a set of optimizations available in many
implementations of MPI-I/O that improve the performance of large-scale
I/O to shared files. To enable these optimizations, you must use the
collective calls in the MPI-I/O library that end in `_all`, for
instance `MPI_File_write_at_all()`. Also, all MPI tasks in the given
MPI communicator must participate in the collective call, even if they
are not performing any I/O operations. The MPI-I/O library has a
heuristic to determine whether to enable collective buffering, the
primary optimization used in collective mode.

## Collective Buffering

Collective buffering, also called two-phase I/O, breaks the I/O
operation into two stages. For a collective read, the first stage uses
a subset of MPI tasks (called aggregators) to communicate with the I/O
servers (OSTs in Lustre) and read a large chunk of data into a
temporary buffer. In the second stage, the aggregators ship the data
from the buffer to its destination among the remaining MPI tasks using
point-to-point MPI calls. A collective write does the reverse,
aggregating the data through MPI into buffers on the aggregator nodes,
then writing from the aggregator nodes to the I/O servers. The
advantage of collective buffering is that fewer nodes are
communicating with the I/O servers, which reduces contention while
still attaining high performance through concurrent I/O transfers. In
fact, Lustre prefers a one-to-one mapping of aggregator nodes to OSTs.

Since the release of mpt/3.3, Cray has included a Lustre-aware
implementation of the MPI-I/O collective buffering algorithm. This
implementation is able to buffer data on the aggregator nodes into
stripe-sized chunks so that all read and writes to the Lustre
filesystem are automatically stripe aligned without requiring any
padding or manual alignment from the developer. Because of the way
Lustre is designed, alignment is a key factor in achieving optimal
performance.


## MPI-I/O Hints

Several environment variables can be used to control the behavior of
collective buffering on Edison and Cori. The MPICH_MPII/O_HINTS
variable specifies hints to the MPI-I/O library that can, for instance,
override the built-in heuristic and force collective buffering on:

```shell
export MPICH_MPIIO_HINTS="*:romio_cb_write=enable:romio_ds_write=disable"
```

Placing this command in your batch file before calling aprun will
cause your program to use these hints. The * indicates that the hint
applies to any file opened by MPI-I/O, while romio_cb_write controls
collective buffering for writes and romio_ds_write controls data
sieving for writes, an older collective mode optimization that is no
longer used and can interfere with collective buffering. The options
for these hints are enabled, disabled, or automatic (the default
value, which uses the built-in heuristic).

It is also possible to control the number of aggregator nodes using
the cb_nodes hint, although the MPI-I/O library will automatically set
this to the stripe count of your file.

When set to 1, the `MPICH_MPIIO_HINTS_DISPLAY` variable causes your
program to dump a summary of the current MPI-I/O hints to stderr each
time a file is opened. This is useful for debugging and as a sanity
check against spelling errors in your hints.

More detail on MPICH runtime environment variables, including a full
list and description of MPI-I/O hints, is available from the intro_mpi
man page on Edison.
