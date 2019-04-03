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

!!! note
	 The login nodes are situated behind a load balancer, so you may
	 be connected to different login nodes at different times.  If you
	 make use of a tool like **screen** or **tmux**, note
	 which login node your sessions are on, and after initially
	 logging on to Cori you can **ssh** to a specific login node.
	 

Computational systems hosted at NERSC use the Slurm batch scheduler system.
Documentation and instructions for using Slurm can be found [here.](../../jobs/index.md)

## [Cori Genepool](cori-genepool.md)

A subset of nodes on Cori, the flagship supercomputer at NERSC, are reserved
for exclusive use by JGI by using JGI-specific "quality of service" (QOS) 
submissions. Full functionality of Cori is also available to JGI users such as
the Burst Buffer, Shifter, and more.
 
## [Denovo](denovo.md)

The Denovo system is exclusively available to JGI users.
**It is scheduled for retirement in July 2019.**

!!! warning "Deprecation"
     Although we provide documentation
     on usage of Denovo we **highly** recommend that most of your work be done
     on Cori genepool (the QOS dedicated for JGI workloads) or Cori ExVivo.

