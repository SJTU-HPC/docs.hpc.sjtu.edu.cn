# Filesystem Quotas

## Overview

| file system     | space | inodes | purge time |
|-----------------|-------|--------|------------|
| Project         | 1 TB  | 1 M    | -          |
| Global HOME     | 40 GB | 1 M    | -          |
| Global common   | 10 GB | 1 M    | -          |
| Cori SCRATCH    | 20 TB | 10 M   | 12 weeks   |

## Policy

[NERSC data management policy](../data/policy.md).

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

For project you can use

```
nersc$ prjquota <project_name>
```

or use `myquota` with the full path to the directory

```
nersc$ myquota --path=/project/projectdirs/<project_name>
```

For global common software you can use

```
nersc$ cmnquota <project_name>
```

or use `myquota` with the full path to the directory

```
nersc$ myquota --path=/global/common/software/<project_name>
```

### Increases

If you or your project needs additional space you may request it via
the
[Disk Quota Increase Form](https://nersc.service-now.com/nav_to.do?uri=catalog_home.do).
