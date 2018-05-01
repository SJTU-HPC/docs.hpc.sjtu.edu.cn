
## Overview

| file system     | space | inodes | purge time | backed up | access          |
|-----------------|-------|--------|------------|-----------|-----------------|
| Project         | 1 TB  | 1 M    | -          | yes       | repo            |
| Global HOME     | 40 GB | 1 M    | -          | yes       | user            |
| Global common   | 10 GB | 1 M    | -          | no        | repo            |
| Cori SCRATCH    | 20 TB | 10 M   | 12 weeks   | no        | user            |
| Edison SCRATCH  | 10 TB | 5 M    | 12 weeks   | no        | user            |
| Edison SCRATCH3 | -     | -      | 8 weeks    | no        | special request |

## Quotas

!!! warning
	When a quota is reached writes to that filesystem may fail.

!!! note
	If your `$SCRATCH` usage exceeds your quota, you will not be
	able to submit batch jobs until you reduce your usage.
	
### Current usage

NERSC provides a `myquota` command which displays applicable quotas
and current usage.

To see current usage for home and available scratch filesystems:
```
nersc$ myquota
```

For project and global common the full path to the directory
```
nersc$ myquota --path=/project/projectdirs/<project_name>
```

or

```
nersc$ myquota --path=/global/common/software/<project_name>
```

### Increases

If you or your project needs additional space you may request it via
the
[Disk Quota Increase Form](https://nersc.service-now.com/nav_to.do?uri=catalog_home.do).

## Backups

!!! danger 
	All NERSC users should back up important files to HPSS on
    a regular basis.  **Ultimately, it is your responsibility to
    protect yourself from data loss.**

### Snapshots

Global homes and project use a *snapshot* capability to provide users
a seven-day history of their directories. Every directory and
sub-directory in global homes contains a ".snapshots" entry.

* `.snapshots` is invisble to `ls`, `ls -a`, `find` and similar
  commands
* Contents are visible through `ls -F .snapshots`
* Can be browsed normally after `cd .snapshots`
* Files cannot be created, deleted or edited in snapshots
* Files can *only* be copied *out* of a snapshot

### Backup/Restore

Global homes are backed up to HPSS on a regular basis.  If you require
a restoration of lost data that cannot be accomplished via the
snapshots capability, please contact NERSC Consulting with pathnames
and timestamps of the missing data.  Such restore requests may take a
few days to complete.

## Purging

Files in `$SCRATCH` directories may be purged if they are older than
12 weeks (defined by last access time).

!!! warning
	`$SCRATCH` directories are **not** backed up
