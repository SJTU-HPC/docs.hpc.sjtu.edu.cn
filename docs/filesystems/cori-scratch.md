# Cori SCRATCH

Cori has one scratch file system named `/global/cscratch1` with 30 PB
disk space and >700 GB/sec IO bandwidth. Cori scratch is a Lustre
filesystem designed for high performance temporary storage of large
files. It contains 10000+ disks and 248 I/O servers called OSTs.

The `/global/cscratch1` file system should always be referenced using
the environment variable `$SCRATCH` (which expands to
`/global/cscratch1/sd/YourUserName`). The scratch file system is
available from all nodes and is tuned for high performance. We
recommend that you run your jobs, especially data intensive ones, from
the burst buffer or the scratch file system.

!!! warning
    For large files you should stripe your files across multiple
    OSTs. Please see our [Lustre striping
    page](../performance/io/lustre/index.md) for details.

If your `$SCRATCH` usage exceeds your quota, you will not be able to
submit batch jobs until you reduce your usage.  The batch job submit
filter checks the usage of `/global/cscratch1`.

!!! note
    See [quotas](quotas.md) for detailed information about inode,
    space quotas and file system purge policies.

The `myquota` command will display your current usage and quota. NERSC
sometimes grants temporary quota increases for legitimate purposes. To
apply for such an increase, please use
the
[Disk Quota Increase Form](https://www.nersc.gov/users/storage-and-file-systems/file-systems/data-storage-quota-increase-request/).

The scratch file system is subject to purging. Please make sure to
back up your important files (e.g. to HPSS). [Instructions for
HPSS](archive.md).

