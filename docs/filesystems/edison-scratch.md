# Edison SCRATCH

Edison has three local scratch file systems named /scratch1,
/scratch2, and /scratch3. Users are assigned to either /scratch1 or
/scratch2 in a round-robin fashion, so a user will be able to use one
or the other but not both. The third file system is reserved for users
who need large IO bandwidth, and the access is granted
by
[request](https://www.nersc.gov/users/computational-systems/edison/file-storage-and-i-o/edison-scratch3-directory-request-form/).

| Filesystem | Total disk space | Bandwidth | #IO Servers (OSS) | OSTs |
|------------|:----------------:|:---------:|:-----------------:|:----:|
| /scratch1  | 2.1 PB           | 48 GB/s   | 24                | 24   |
| /scratch2  | 2.1 PB           | 48 GB/s   | 24      		| 24   |
| /scratch3  | 3.2 PB           | 72 GB/s   | 36      		| 36   |

All three scratch file systems are Lustre file systems. Lustre allows
files to be "striped" or split across multiple OSTs to increase IO
performance. By default files are striped across one OST, but this can
be changed.

The /scratch1 or /scratch2 file systems should always be referenced
using the environment variable $SCRATCH (which expands to
/scratch1/scratchdirs/YourUserName or
/scratch2/scratchdirs/YourUserName on Edison). The scratch file
systems are available from all nodes (login, and compute nodes) and
are tuned for high performance. We recommend that you run your jobs,
especially data intensive ones, from the scratch file systems.

The "myquota" command (with no options) will display your current
usage and quota.  NERSC sometimes grants temporary quota increases for
legitimate purposes. To apply for such an increase, please use the
Disk Quota Increase Form.

The scratch file systems (/scratch1 and /scratch2) are subject to
purging. Files in your $SCRATCH directory that are older than 12 weeks
(defined by last access time) are removed. The /scratch3 file system
is subject to purging as well, with files that haven't been accessed
in 8 weeks being removed. Please make sure to back up your important
files (e.g. to HPSS).


