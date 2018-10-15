# Optimizing I/O performanceOA

## Lustre

Edison and Cori use Lustre as their $SCRATCH file systems. For many
applications a technique called file striping will increase I/O
performance. File striping will primarily improve performance for
codes doing serial I/O from a single node or parallel I/O from multiple
nodes writing to a single shared file as with MPI-I/O, parallel HDF5 or
parallel NetCDF.

The Lustre file system is made up of an underlying set of I/O servers
and disks called Object Storage Servers (OSSs) and Object Storage
Targets (OSTs) respectively. A file is said to be striped when read
and write operations access multiple OST's concurrently. File striping
is a way to increase I/O performance since writing or reading from
multiple OST's simultaneously increases the available I/O bandwidth.

### NERSC Striping Shortcuts

After the scratch reformatting, the default striping is changed to 1
on Edison's $SCRATCH (backed by either /scratch1 and /scratch2), while
it is still 8 on Edison's $SCRATCH3, and is 1 on Cori's
$SCRATCH. (Edison's striping recommendation is under investigation)
This means that each file created with the default striping is split
across 1 OSTs on Edison's primary scratch filesystems, and 8 on
Edison's specialized $SCRATCH3. On Cori, the default striping
allocates 1 OST for the file.  Selecting the best striping can be
complicated since striping a file over too few OSTs will not take
advantage of the system's available bandwidth but striping over too
many will cause unnecessary overhead and lead to a loss in
performance.  NERSC has provided striping command shortcuts based on
file size to simplify optimization on both Edison and Cori.  Users who
want more detail should read the sections below or contact the
consultants at consult@nersc.gov.

Striping must be set on a file before is written. For example, one
could simultaneously create an empty file which will later be 10-100
GB in size and set its striping appropriately with the command:

```shell
nersc$ stripe_medium $output_file
```

This could be done before running a job which will later populate this
file. Striping of a file cannot be changed once the file has been
written to, aside from copying the existing file into a newly created
(empty) file with the desired striping.

`stripe_small` will set the number of ost as 8, stripe_medium will
have 24 ost and stripe_large will set as 72. In all cases, the stripe
size is 1MB.

Files inherit the striping configuration of the directory in which
they are created. Importantly, the desired striping must be set on the
directory before creating the files (later changes of the directory
striping are not inherited). When copying an existing striped file
into a striped directory, the new copy will inherit the directory's
striping configuration. This provides another approach to changing the
striping of an existing file.

Inheritance of striping provides a convenient way to set the striping
on multiple output files at once, if all such files are written to the
same output directory. For example, if a job will produce multiple
10-100 GB output files in a known output directory, the striping of
the latter can be configured before job submission:

```shell
nersc$ stripe_medium $output_directory
```

Or one could put the striping command directly into the job script:

```bash
#!/bin/bash
#SBATCH --qos=debug
#SBATCH -N 2
#SBATCH -t 00:10:00
#SBATCH -J my_job
#SBATCH -V

cd $SLURM_SUBMIT_DIR

stripe_medium myOutputDir

srun -n 10 ./a.out
```

### Shared file I/O

| File size (GB) | command                  |
|:--------------:|--------------------------|
| &lt; 1         | do nothing (use default) |
| 1 - 10         | `stripe_small`           |
| 10 - 100       | `stripe_medium`          |
| &gt; 100       | `stripe_large`           |

### File per processing element

If files are larger than 100GB contact consult@nersc.gov otherwise use
the default settings.

## Burst Buffer

## MPI-I/O Collective Mode

Collective mode refers to a set of optimizations available in many
implementations of MPI-I/O that improve the performance of large-scale
I/O to shared files. To enable these optimizations, you must use the
collective calls in the MPI-I/O library that end in `_all`, for
instance `MPI_File_write_at_all()`. Also, all MPI tasks in the given
MPI communicator must participate in the collective call, even if they
are not performing any I/O operations. The MPI-I/O library has a
heuristic to determine whether to enable collective buffering, the
primary optimization used in collective mode.

### Collective Buffering

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

### MPI-I/O Hints

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

## Burst Buffer

The NERSC Burst Buffer is based on Cray DataWarp that uses flash or
SSD (solid-state drive) technology to significantly increase the I/O
performance on Cori for all file sizes and all access patterns.

### Striping

Currently, the Burst Buffer granularity is 82GiB. If you request an
allocation smaller than this amount, your files will sit on one Burst
Buffer node. If you request more than this amount, then your files
will be striped over multiple Burst Buffer nodes. For example, if you
request 82GiB then your files all sit on the same Burst Buffer
server. This is important, because each Burst Buffer server has a
maximum possible bandwidth of roughly 6.5GB/s - so your aggregate
bandwidth is summed over the number of Burst Buffer servers. If other
people are accessing data on the same Burst Buffer server at the same
time, then you will share that bandwidth and will be unlikely to reach
the theoretical peak.

 * It is better to stripe your data over many Burst Buffer servers,
particularly if you have a large number of compute nodes trying to
access the data.

 * The number of Burst Buffer nodes used by an application should be
scaled up with the number of compute nodes, to keep the Burst Buffer
nodes busy but not over-subscribed. The exact ratio of compute to
Burst Buffer nodes will depend on the amount of I/O load produced by
the application.

### Use large transfer sizes

We have seen that using transfer sizes less than 512KiB results in
poor performance. In general, we recommend using as large a transfer
size as possible.

### Use more processes per Burst Buffer node

We have seen that the Burst Buffer cannot be kept busy with less than
4 processes writing to each Burst Buffer node - less than this will
not be able to achieve the peak potential performance of roughly 6.5
GB/s per node.
