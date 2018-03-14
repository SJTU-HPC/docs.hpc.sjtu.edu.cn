Edison has three local scratch file systems named /scratch1, /scratch2, and /scratch3. Users are assigned to either /scratch1 or /scratch2 in a round-robin fashion, so a user will be able to use one or the other but not both. The third file system is reserved for users who need large IO bandwidth, and the access is granted by request. 

## Usage

If you need large IO bandwidth to conduct more efficient computations and data analysis at NERSC, please submit your request by filling out the [SCRATCH3 Directory Request Form](http://www.nersc.gov/users/computational-systems/edison/file-storage-and-i-o/edison-scratch3-directory-request-form/).

The `/scratch1` or `/scratch2` file systems should always be referenced using the environment variable `$SCRATCH` (which expands to `/scratch1/scratchdirs/YourUserName` or `/scratch2/scratchdirs/YourUserName` on Edison). The scratch file systems are available from all nodes (login, and compute nodes) and are tuned for high performance. 

!!!tip
	We recommend that you run your jobs, especially data intensive ones, from the scratch file systems.

## Quotas

| Type   | Quota |
|--------|:-----:|
| Space  | 10 TB |
| inodes | 5 M   |

The `myquota` command will display your current usage and quota.  NERSC sometimes grants temporary quota increases for legitimate purposes. To apply for such an increase, please use the [Disk Quota Increase Form](http://www.nersc.gov/users/storage-and-file-systems/file-systems/data-storage-quota-increase-request/).

!!! note "Purge policy"
	Files in your $SCRATCH directory that are older than **12 weeks** (defined by la\
st access time) are removed. Files in `/scratch3` older than **8 weeks** are subject to purging.

!!!warning
	If your `$SCRATCH` usage exceeds your quota, you will not be able to submit batch jobs until you reduce your usage. We have not set the quotas on the /scratch3 file system. The batch job submit filter checks only the usage of the /scratch1 or /scratch2, but not /scratch3.


## Performance

| Filesystem | Total disk space | Bandwidth |
|------------|:----------------:|:---------:|
| /scratch1  | 2.1 PB           | 48 GB/s   |
| /scratch2  | 2.1 PB           | 48 GB/s   |
| /scratch3  | 3.2 PB           | 72 GB/s   |

## Backup/Restore

No managed backups or project directories are done by NERSC.

!!! warning
	 All NERSC users should back up important files to HPSS on a regular basis.  Ultimately, it is your responsibility to protect yourself from data loss.
