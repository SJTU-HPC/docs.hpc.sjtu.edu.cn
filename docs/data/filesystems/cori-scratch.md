Cori has one scratch file system named `/global/cscratch1`. 

## Usage

The `/global/cscratch1` file system should always be referenced using the environment variable $SCRATCH (which expands to `/global/cscratch1/sd/$USER`). The scratch file system is available from all nodes (login, MOM, and compute nodes) and is tuned for high performance. 

!!!tip
	We recommend that you run your jobs, especially data intensive ones, from the burst buffer or the scratch file system.
	
!!!tip
	`/global/cscratch1` is also mounted on Edison!

## Quotas

| Type   | Quota |
|--------|:-----:|
| Space  | 20 TB |
| inodes | 10 M  |

The `myquota` command will display your current usage and quota.  NERSC sometimes grants temporary quota increases for legitimate purposes. To apply for such an increase, please use the [Disk Quota Increase Form](http://www.nersc.gov/users/storage-and-file-systems/file-systems/data-storage-quota-increase-request/).

!!! note "Purge policy"
	Files in your $SCRATCH directory that are older than **12 weeks** (defined by last access time) are removed.
	
!!!warning
    If your `$SCRATCH` usage exceeds your quota, you will not be able to submit batch jobs until you reduce your usage.

## Performance

The file system has 30 PB disk space and >700 GB/sec IO bandwidth.

## Backup/Restore

No managed backups or project directories are done by NERSC.

!!! warning
	 All NERSC users should back up important files to HPSS on a regular basis.  Ultimately, it is your responsibility to protect yourself from data loss.
