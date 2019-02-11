# Queue Policies

This page details the charging and usage policies for Edison and
Cori. [Examples](examples/index.md) for each type of job are available.

## Edison

| QOS           | Max nodes | Max time (hrs) | Submit limit | Run limit | Priority | Charge |
|---------------|-----------|----------------|--------------|-----------|----------|--------|
| regular       | 5586      | 48             | 5000         | -         | 3        | 64     |
| shared[^1]    | 0.5       | 48             | 10000        | -         | 3        | 64     |
| debug         | 512       | 0.5            | 5            | 2         | 2        | 64     |
| premium       | 5586      | 48             | 5            | -         | 2        | 128    |
| scavenger[^2] | 5586      | 48             | 5000         | -         | 4        | 0      |
| xfer          | 1 (login) | 48             | 100          | 15        | -        | 0      |
| realtime[^3]  | custom    | custom         | custom       | custom    | 1        | custom |

## Cori

### Haswell

| QOS             | Max nodes | Max time (hrs) | Submit limit | Run limit | Priority | Charge |
|-----------------|-----------|----------------|--------------|-----------|----------|--------|
| regular         | 1932      | 48             | 5000         | -         | 4        | 90     |
| shared[^1]      | 0.5       | 48             | 10000        | -         | 4        | 90     |
| interactive[^4] | 64        | 4              | 1            | 1         | -        | 90     |
| debug           | 64        | 0.5            | 5            | 2         | 3        | 90     |
| premium         | 1772      | 48             | 5            | -         | 2        | 180    |
| scavenger[^2]   | 1772      | 48             | 5000         | -         | 5        | 0      |
| xfer            | 1 (login) | 48             | 100          | 15        | -        | 0      |
| bigmem          | 1 (login) | 72             | 100          | 1         | -        | 90     |
| realtime[^3]    | custom    | custom         | custom       | custom    | 1        | custom |
| special[^5]     | custom    | custom         | custom       | custom    | -        | custom |

### KNL

| QOS             | Max nodes | Max time (hrs) | Submit limit | Run limit | Priority | Charge |
|-----------------|-----------|----------------|--------------|-----------|----------|--------|
| regular         | 9489      | 48             | 5000         | -         | 4        | 90[^6] |
| interactive[^5] | 64        | 4              | 1            | 1         | -        | 90     |
| debug           | 512       | 0.5            | 5            | 2         | 3        | 90     |
| premium         | 9489      | 48             | 5            | -         | 2        | 180[^6]|
| low.            | 9489      | 48             | 5000         | -         | 5        | 45[^7] |
| scavenger[^2]   | 9489      | 48             | 5000         | -         | 6        | 0      |
| special[^4]     | custom    | custom         | custom       | custom    | -        | custom |

!!! tip
	Jobs using 1024 or more KNL nodes receive a 20% discount!

!!! note
	User held jobs that were submitted more than 12 weeks ago will be deleted.

### JGI Accounts

There are 192 Haswell nodes reserved for the "genepool" and
"genepool_shared" QOSs combined.  Jobs run with the "genepool" QOS
uses these nodes exclusively. Jobs run with the "genepool_shared" QOS
can share nodes.

| QOS             | Max nodes | Max time (hrs) | Submit limit | Run limit | Priority | Charge |
|-----------------|-----------|----------------|--------------|-----------|----------|--------|
| genepool        | 192       | 72             | 500          | -         | 3        | 90     |
| genepool_shared | 0.5       | 72             | 500          | -         | 3        | 90     |

## Charging

Jobs are charged by the node-hour.

!!! example
	A job which ran for 35 minutes on 3 nodes on Edison with
	the regular qos would be charged:
	$$ (35/60)\\ \text{hours}*3\\ \text{nodes} * 64 = 112\\ \text{NERSC hours} $$

!!! example
	A job which ran for 12 hours on 4 physical cores (each core has 2 hyperthreads)
	on Cori Haswell with the shared qos would be charged:
	$$ 12\\ \text{hours} * (2*4\\ \text{cores}/64) * 90 = 135\\ \text{NERSC hours} $$

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

### Low

The intent of the low QOS is to allow non-urgent jobs to run with a 
lower usage charge.

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

[^7]:
	The "low" QOS (available on Cori KNL only) is charged 50% as compared to 
	the "regular" QOS, but no extra large jobs discount.

