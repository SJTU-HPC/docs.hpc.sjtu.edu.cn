## First Steps

You can log in to Genepool using SSH (Secure Shell) with the following command from any UNIX, Linux, FreeBSD, etc. command shell or terminal:

```ssh -l username denovo.nersc.gov```
or 
```ssh -l username cori.nersc.gov```

There are several SSH-capable clients available for Windows, Mac, and UNIX/Linux machines. NERSC does not support or recommend any particular client.

By ssh'ing to denovo.nersc.gov, you will access one of the two denovo login nodes. The Denovo system is dedicated solely to JGI use, and will generally be the most common choice of system until its' decommission in 2018. Cori is a much larger cluster, with 12 login nodes, but has a very little bioinformatics software available as modules. 
 
These login nodes are situated behind a load balancer, so you may reach a different login node on different days.  If you make use of a tool like **screen** or **tmux**, make sure to take note of which login node you started it on.

In addition to the login nodes, the "gpint/dint" systems are available for direct-ssh access.  The gpints/dints differ from the login nodes in that login nodes are intended purely for file management, batch job submission, software compilation, and other "light" use. The gp/dints can be used for heavier-weight interactive processing or for very long jobs.  If you want to run a short (and small) alignment or other calculation without using the batch system, use a gp/dint. Dints and gpints are assigned to specific groups of users. Please contact a consultant to identify which gp/dints you may access.


To begin running jobs on Denovo or Cori, you should begin learning about the Slurm batch scheduler system, and look at the [example batch scripts](genepool-software.md) before submitting your first job.
