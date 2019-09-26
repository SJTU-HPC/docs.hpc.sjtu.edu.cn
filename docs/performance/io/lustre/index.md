Cori uses Lustre as its $SCRATCH file system. For many applications a technique
called file striping will increase I/O performance. File striping will
primarily improve performance for codes doing serial I/O from a single node or
parallel I/O from multiple nodes writing to a single shared file as with
MPI-I/O, parallel HDF5 or parallel NetCDF.

## Lustre File Striping
The Lustre file system is made up of an underlying set of I/O servers
and disks called Object Storage Targets (OSTs). A file is said to be
striped when its data is on multiple OSTs. Read and write operations
on striped files will access multiple OST's concurrently. File
striping is a way to increase I/O performance since writing or reading
from multiple OST's simultaneously increases the available I/O
bandwidth. Selecting the best striping can be complicated since
striping a file over too few OSTs will not take advantage of the
system's available bandwidth but striping over too many will cause
unnecessary overhead and lead to a loss in performance. The default
striping is set to 1 on Cori's $SCRATCH. This means that each file is
written to 1 OSTs on Cori by default.

##NERSC File Striping Recommendations
NERSC has provided striping command shortcuts based on file size and
I/O pattern to simplify optimization on Cori.

*   **Shared file I/O**: Either one processor does all the I/O for a simulation in
    serial or multiple processors write to a single shared file as
    with MPI-IO and parallel HDF5 or NetCDF
*   **File per process**: Each process writes to its own file resulting in as many files
    as number of processes

|                | Single Shared-File I/O   | File per Process     |
|:--------------:|--------------------------|----------------------|
| File size (GB) | command                  | 		           |
| &lt; 1         | keep default striping    | keep default striping|
| 1 - 10         | `stripe_small`           | keep default striping|
| 10 - 100       | `stripe_medium`          | keep default striping|
| &gt; 100       | `stripe_large`           | keep default striping|
| &gt; 1000      | `stripe_large`           | `stripe_large`       |


These helper scripts will set the number of OSTs to stripe across to
8, 24, and 72 for `stripe_small`, `stripe_medium` and `stripe_large`,
respectively. In all cases, the stripe size is 1MB.

!!! warn
    Files larger than 1 TB should be striped with the stripe_large script.

Striping must be set on a file before is written. For example, one
could simultaneously create an empty file which will later be 10-100
GB in size and set its striping appropriately with the command:

```shell
nersc$ stripe_medium output_file
```

This could be done before running a job which will later populate this
file. Striping of a file cannot be changed once the file has been
written to, aside from copying the existing file into a newly created
(empty) file with the desired striping.

Files inherit the striping configuration of the directory in which
they are created. Again, the desired striping must be set on the
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
nersc$ stripe_medium output_directory
```

### Restriping an Existing File

Currently the only way to restripe an existing file is to make a copy
of it.
```bash
nersc$ stripe_large tmp_my_big_file
nersc$ cp my_big_file tmp_my_big_file
nersc$ mv tmp_my_big_file my_big_file
```
If there are multiple files, you could create a directory with the
desired striping and copy the files into it to avoid doing the above
procedure multiple times.

## Custom Lustre Striping

To set striping for a file or directory use the command `lfs
setstripe`.

Each file and directory can have a separate striping pattern and a
directory's striping setting can be overridden for a particular file
by issuing the `lfs setstripe` command for individual files within
that directory. However, as noted above, striping settings for a file
must be set before it is created.  If the settings for an existing
file are changed, it will only get the new striping setting if the
file is recreated. If the settings for an existing directory are
changed, the files need to be copied elsewhere and then copied back to
the directory in order to inherit the new settings. The lfs setstripe
syntax is:

```shell
nersc$ lfs setstripe --size [stripe-size] --index [OST-start-index]
--count [stripe-count] filename
```

| Option         | Description              | Default              |
|:--------------:|--------------------------|----------------------|
| stripe-size    | Number of bytes write on one OST before cycling to the next. Use multiples of 1MB. Default has been most successful. |1MB|
| stripe-count   | Number of OSTs a file exists on|1 on Cori|
| OST-start-index| Starting OST. Default highly recommended| -1 (System follows a round robin procedure to optimize creation of files by all users.)|
