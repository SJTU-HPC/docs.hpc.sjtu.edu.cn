# Computational Systems

## Getting Started

You can log in to NERSC systems using SSH (Secure Shell) with the following
command from any UNIX, Linux, FreeBSD, etc. command shell or terminal:

```
ssh -l username cori.nersc.gov
```

or

```
ssh -l username denovo.nersc.gov
```

There are several SSH-capable clients available for Windows, Mac, and
UNIX/Linux machines. NERSC does not support or recommend any
particular client.

The login nodes are situated behind a load balancer, so you may
reach a different login node on different days.  If you make use of a
tool like **screen** or **tmux**, make sure to take note of which
login node you started it on.

To begin running jobs on Cori or Denovo, you should begin learning
about the Slurm batch scheduler system, and look at
the [example batch scripts](../jobs/examples/index.md) before submitting
your first job.

## [Cori Genepool](cori-genepool.md)

NERSC is pleased to provide compute capacity on its flagship
supercomputer, Cori, to JGI users. The Burst Buffer, Shifter, and all
other features available to Haswell Cori nodes are available via the
JGI-specific "quality of service" (QOS).

## [Denovo](denovo.md)

Denovo is the name of the system available exclusively to JGI users on the
NERSC Mendel cluster. **It is scheduled for retirement in July 2019.**

!!! warning "Deprecation"
     Although we provide documentation
     on usage of Denovo we **highly** recommend that most of your work be done
     on Cori genepool (the QOS dedicated for JGI workloads).

