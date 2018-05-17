# Optimizing I/O performance

## Lustre

Edison and Cori use Lustre as their $SCRATCH file systems. For many
applications a technique called file striping will increase IO
performance. File striping will primarily improve performance for
codes doing serial IO from a single node or parallel IO from multiple
nodes writing to a single shared file as with MPI-IO, parallel HDF5 or
parallel NetCDF.

The Lustre file system is made up of an underlying set of IO servers
and disks called Object Storage Servers (OSSs) and Object Storage
Targets (OSTs) respectively. A file is said to be striped when read
and write operations access multiple OST's concurrently. File striping
is a way to increase IO performance since writing or reading from
multiple OST's simultaneously increases the available IO bandwidth.

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

## MPI-IO Collective Mode

Collective mode refers to a set of optimizations available in many
implementations of MPI-IO that improve the performance of large-scale
IO to shared files. To enable these optimizations, you must use the
collective calls in the MPI-IO library that end in `_all`, for
instance `MPI_File_write_at_all()`. Also, all MPI tasks in the given
MPI communicator must participate in the collective call, even if they
are not performing any IO operations. The MPI-IO library has a
heuristic to determine whether to enable collective buffering, the
primary optimization used in collective mode.

### Collective Buffering

Collective buffering, also called two-phase IO, breaks the IO
operation into two stages. For a collective read, the first stage uses
a subset of MPI tasks (called aggregators) to communicate with the IO
servers (OSTs in Lustre) and read a large chunk of data into a
temporary buffer. In the second stage, the aggregators ship the data
from the buffer to its destination among the remaining MPI tasks using
point-to-point MPI calls. A collective write does the reverse,
aggregating the data through MPI into buffers on the aggregator nodes,
then writing from the aggregator nodes to the IO servers. The
advantage of collective buffering is that fewer nodes are
communicating with the IO servers, which reduces contention while
still attaining high performance through concurrent I/O transfers. In
fact, Lustre prefers a one-to-one mapping of aggregator nodes to OSTs.

Since the release of mpt/3.3, Cray has included a Lustre-aware
implementation of the MPI-IO collective buffering algorithm. This
implementation is able to buffer data on the aggregator nodes into
stripe-sized chunks so that all read and writes to the Lustre
filesystem are automatically stripe aligned without requiring any
padding or manual alignment from the developer. Because of the way
Lustre is designed, alignment is a key factor in achieving optimal
performance.

### MPI-IO Hints

Several environment variables can be used to control the behavior of
collective buffering on Edison and Cori. The MPICH_MPIIO_HINTS
variable specifies hints to the MPI-IO library that can, for instance,
override the built-in heuristic and force collective buffering on:

```shell
export MPICH_MPIIO_HINTS="*:romio_cb_write=enable:romio_ds_write=disable"
```

Placing this command in your batch file before calling aprun will
cause your program to use these hints. The * indicates that the hint
applies to any file opened by MPI-IO, while romio_cb_write controls
collective buffering for writes and romio_ds_write controls data
sieving for writes, an older collective mode optimization that is no
longer used and can interfere with collective buffering. The options
for these hints are enabled, disabled, or automatic (the default
value, which uses the built-in heuristic).

It is also possible to control the number of aggregator nodes using
the cb_nodes hint, although the MPI-IO library will automatically set
this to the stripe count of your file.

When set to 1, the `MPICH_MPIIO_HINTS_DISPLAY` variable causes your
program to dump a summary of the current MPI-IO hints to stderr each
time a file is opened. This is useful for debugging and as a sanity
check against spelling errors in your hints.

More detail on MPICH runtime environment variables, including a full
list and description of MPI-IO hints, is available from the intro_mpi
man page on Edison.

## Burst Buffer

The NERSC Burst Buffer is based on Cray DataWarp that uses flash or
SSD (solid-state drive) technology to significantly increase the I/O
performance on Cori for all file sizes and all access patterns.

Access to the BurstBuffer resource is integrated with the Scheduler of
the system (i.e. SLURM). The Scheduler provides the ability to
provision the BurstBuffer resource to be shared by a set of users or
jobs. Using the Burst Buffer on Cori can be as simple as adding a
single line to your slurm batch script. Here we give examples of how
to use the Burst Buffer as a scratch space and as a persistent
reservation, and how to stage data in and out of the SSD. You might
also want to see the Burst Buffer FAQs.

The fast IO available with the Burst Buffer makes it an ideal scratch
space to store temporary files during the execution of an IO-intensive
code. Note that all files on the Burst Buffer allocation will be
removed when the job ends - so you will need to stage_out any data
that you want to retain at the end of your job. You also do not need
to delete any unwanted data you leave on the BB - it will be removed
automatically during the tear down of your BB allocation (performed by
the DataWarp software after your job completes). To use the Burst
Buffer as scratch, you will need to add a "#DW jobdw" command to your
slurm batch script, and specify what type of allocation you require
(striped or private) and how much. The "jobdw" indicates that you are
requesting an allocation that will last for the duration of the
compute job.

### Access modes

#### Striped

Striped access mode means your data will be striped across several
DataWarp nodes. The Burst Buffer has two levels of granularity in two
different pools - 80GiB (wlm_pool, default) and 20GiB (sm_pool). This
is the minimum allocation on each Burst Buffer SSD. For example, if
you wish to use 1.2TiB of Burst Buffer space in the wlm_pool, this
will be striped across 15 BB SSDs, each holding 80GiB (note that you
may actually see more than one unit of granularity on the same BB
SSD - there is no guarantee that your allocation will be spread evenly
between SSDs). DataWarp nodes are allocated on a round-robin basis.

#### Private

Private access mode means each of your compute jobs will have their
own, private space on the BB that will not be visible to any other
compute job. Data will be striped across BB nodes in private
mode. Note that all compute nodes will share the allocation - so if
one compute node fills up all the space then the other compute nodes
will run out and you will see "out of space" errors.

!!! note
    We recommend using access_mode=striped and the default
	granularity pool.

The path to the Burst Buffer disk will then be available as
$DW_JOB_PRIVATE or $DW_JOB_STRIPED. If you want your code to access
the BB disk space, you will need to tell it (your code will not
automatically stage your data on the BB for you). For example, as an
option to your code:

```shell
#!/bin/bash
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:15:00
#DW jobdw capacity=10GB access_mode=striped type=scratch pool=sm_pool
srun a.out INSERT_YOU_CODE_OPTIONS_HERE
```

The following example demonstrates how to copy data in or out of the
Burst Buffer. The location of this data could be passed as an option
to your executable (note that your executable will not know where this
data is unless you tell it!). The only filesystem currently mounted on
the datawarp nodes is the Lustre scratch system, accessible via
$SCRATCH on Cori. Any files to be transferred to or from the Burst
Buffer must be located on this disk.

!!! note
    This is a different way of moving data compared to the Datawarp `stage_in` and `stage_out` commands, and may be significantly slower.

```shell
#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:15:00
#DW jobdw capacity=10GB access_mode=striped type=scratch
mkdir $DW_JOB_STRIPED/inputdir
mkdir $DW_JOB_STRIPED/outputdir
cp $SCRATCH/path/to/input/file $DW_JOB_STRIPED/inputdir/
srun a.out INSERT_YOUR_CODE_OPTIONS_HERE
cp -r $DW_JOB_STRIPED/outputdir/ $SCRATCH/path/to/output/
```

### Striping

Currently, the Burst Buffer granularity is 82GiB in the wlm_pool, and
20.14GiB in the sm_pool. If you request an allocation smaller than
this amount, your files will sit on one BB node. If you request more
than this amount, then your files will be striped over multiple BB
nodes. For example, if you request 82GiB in wlm_pool then your files
all sit on the same BB server - but if you request 82GiB in sm_pool
then your files will be striped over 5 BB nodes. This is important,
because each BB server has a maximum possible BW of roughly 6.5GB/s -
so your aggregate BW is summed over the number of BB servers. Of
course, if other people are accessing data on the same BB server at
the same time, then you will share that BW and will be unlikely to
reach the theoretical peak.

In general, it is better to stripe your data over many BB servers,
particularly if you have a large number of compute nodes trying to
access the data. The wlm_pool is the default - you can request the
sm_pool by adding "pool=sm_pool" to your #DW command, e.g.

```shell
#DW jobdw capacity=10GB access_mode=striped type=scratch pool=sm_pool
```

!!! note
    there are a total of 80 nodes in the sm_pool, so if you
	request more than (80*20.14Gib=) 1611GiB of BB allocation in this
	pool, then you will guarantee to have multiple stripes on the same
	BB server, thereby halving your maximum possible BW.

The following figure shows current IOR results for MPI shared file
(mssf) and posix file-per-process (pfpp) runs, using 100MB block size
and 1MB transfer size, with files striped over increasing numbers of
BB nodes, for both the wlm_pool and sm_pool. 16 compute nodes were
used, with 4 MPI ranks per node.  The benefit of striping over
[1,2,4,8,16,32] BB nodes is clear in both granularity pools, although
The scaling is less strong after 4 nodes, because the compute load is
staying constant. The maximum possible bandwidth is roughly 6.6 GiB/S
per BB node - but this can only be achieved if you are generating
enough IO load to keep the nodes busy.

As a general rule, the number of BB nodes used by an application
should be scaled up with the number of compute nodes, to keep the BB
nodes busy but not over-subscribed. The exact ratio of compute to BB
nodes will depend on the amount of IO load produced by the
application.

### Use large transfer sizes

We have seen that using transfer sizes less than 512KiB results in
poor performance. In general, we recommend using as large a transfer
size as possible. The following figure shows IOR results configured as
described above, using an 800GiB BB allocation in both the wlm_pool
and sm_pool, for 100MiB block size and varying transfer size. Optimal
performance is seen at 512KiB transfer size in the wlm_pool, with some
fall-off above that in most cases. Results for the sm_pool appear to
favour a larger transfer size.

### Use more processes per BB node

We have seen that the Burst Buffer cannot be kept busy with less than
4 processes writing to each BB node - less than this will not be able
to achieve the peak potential performance of roughly 6.6 GiB/S per
node.
