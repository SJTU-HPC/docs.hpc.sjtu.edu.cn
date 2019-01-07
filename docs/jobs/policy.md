# Queue Policies

This page details the charging and usage policies for Edison and
Cori. [Examples](examples/index.md) for each type of job are available.

## Edison

| QOS           | Max nodes | Max time (hrs) | Submit limit | Run limit | Priority | Charge |
|---------------|-----------|----------------|--------------|-----------|----------|--------|
| regular       | 5586      | 48             | 5000         | -         | 3        | 48     |
| shared[^1]    | 0.5       | 48             | 10000        | -         | 3        | 48     |
| debug         | 512       | 0.5            | 5            | 2         | 2        | 48     |
| premium       | 5586      | 48             | 5            | -         | 2        | 96     |
| scavenger[^2] | 5586      | 48             | 5000         | -         | 4        | 0      |
| xfer          | 1 (login) | 48             | 100          | 15        | -        | 0      |
| realtime[^3]  | custom    | custom         | custom       | custom    | 1        | custom |

## Cori

### Haswell

| QOS             | Max nodes | Max time (hrs) | Submit limit | Run limit | Priority | Charge |
|-----------------|-----------|----------------|--------------|-----------|----------|--------|
| regular         | 1932      | 48             | 5000         | -         | 4        | 80     |
| shared[^1]      | 0.5       | 48             | 10000        | -         | 4        | 80     |
| interactive[^4] | 64        | 4              | 1            | 1         | -        | 80     |
| debug           | 64        | 0.5            | 5            | 2         | 3        | 80     |
| premium         | 1932      | 48             | 5            | -         | 2        | 160    |
| scavenger[^2]   | 1932      | 48             | 5000         | -         | 5        | 0      |
| xfer            | 1 (login) | 48             | 100          | 15        | -        | 0      |
| bigmem          | 1 (login) | 48             | 100          | 1         | -        | 0      |
| realtime[^3]    | custom    | custom         | custom       | custom    | 1        | custom |
| special[^5]     | custom    | custom         | custom       | custom    | -        | custom |

### KNL

| QOS             | Max nodes | Max time (hrs) | Submit limit | Run limit | Priority | Charge |
|-----------------|-----------|----------------|--------------|-----------|----------|--------|
| regular         | 8991      | 48             | 5000         | -         | 4        | 96[^6] |
| interactive[^5] | 64        | 4              | 1            | 1         | -        | 96     |
| debug           | 512       | 0.5            | 5            | 2         | 3        | 96     |
| premium         | 8991      | 48             | 5            | -         | 2        | 192[^6]|
| scavenger[^2]   | 8991      | 48             | 5000         | -         | 5        | 0      |
| special[^4]     | custom    | custom         | custom       | custom    | -        | custom |

!!! tip
	Jobs using 1024 or more KNL nodes receive a 20% discount!

### JGI Accounts

There are 192 Haswell nodes reserved for the "genepool" and
"genepool_shared" QOSs combined.  Jobs run with the "genepool" QOS
uses these nodes exclusively. Jobs run with the "genepool_shared" QOS
can share nodes.

| QOS             | Max nodes | Max time (hrs) | Submit limit | Run limit | Priority | Charge |
|-----------------|-----------|----------------|--------------|-----------|----------|--------|
| genepool        | 192       | 72             | 500          | -         | 3        | 80     |
| genepool_shared | 0.5       | 72             | 500          | -         | 3        | 80     |

## Charging

Jobs are charged by the node-hour.

!!! example
	A job which ran for 35 minutes on 3 nodes on Edison with
	the regular qos would be charged:
	$$ (35/60)\\ \text{hours}*3\\ \text{nodes} * 48 = 84\\ \text{NERSC hours} $$

!!! example
	A job which ran for 12 hours on 4 physical cores (each core has 2 hyperthreads)
	on Edison with the shared qos would be charged:
	$$ 12\\ \text{hours} * (2*4\\ \text{cores}/48) * 48 = 96\\ \text{NERSC hours} $$

!!! note
    Jobs are only charged for the actual walltime used. That is, if a job uses less
    time than requested, the corresponding account is charged only for the actual job
    duration.

## Intended use

### Debug

The "debug" QOS is to be used for code development, testing, and
debugging. Production runs are not permitted in the debug QOS. User
accounts are subject to suspension if they are determined to be using
the debug QOS for production computing. In particular, job "chaining"
in the debug QOS is not allowed. Chaining is defined as using a batch
script to submit another batch script.

### Premium

The intent of the premium QOS is to allow for faster turnaround before
conferences and urgent project deadlines. It should be used with care.

### Scavenger

The intent of the scavenger QOS is to allow users with a zero or
negative balance in one of their repositories to continue to run jobs.
The scavenger QOS is not available for jobs submitted against
a repository with a positive balance. The charging rate for this QOS
is 0 and it has the lowest priority on all systems.

If you meet the above criteria, you can access the scavenger queue by
submitting with `-q scavenger` (`-q shared_scavenger` for the shared
queue). In addition, you must specify a minimum running time for this
job of 2 hours or less with the `--time-min` flag. This means that the
job could potentially be interrupted by a higher priority
(i.e. non-scavenger job) after the minimum time has elapsed. Because
of this, we recommend you implement checkpointing in your scavenger
jobs. Jobs submitted without these flags will be automatically
rejected by the batch system.

!!! example
        A scavenger job requesting a minimum time of 1.5 hours:
	`sbatch -q scavenger --time-min=01:30:00 my_batch_script.sl`

[^1]:
	Jobs in the "shared" QOS are only charged for the fraction of the
	node used.

[^2]:
	The "scavenger" QOS is *only* available when running a job would
    cause the *repository* (not only the user's allowed fraction)
    balance to go negative.

[^3]:
	The "realtime" QOS is only available via
    [special request](https://nersc.service-now.com/catalog_home.do?sysparm_view=catalog_default).

[^4]:
	The "interactive" QOS has 192 Haswell and 192 KNL nodes
    reserved. Batch job submission is not enabled and the 64 node
    limit applies **per repository** not per user.

[^5]:
	The "special" QOS is via special permission only.

[^6]:
	The "regular" and "premium" QOS charges on Cori KNL are discounted
    by 20% if the job uses 1024 or more nodes.
