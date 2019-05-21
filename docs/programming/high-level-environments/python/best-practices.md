# Best Practices

Here is a list of best practices for using Python at NERSC.

## Always Load a Python Module

Always use a version of Python provided by NERSC through "module load python" with an optional version suffix (2.7-anaconda, 3.5-anaconda, etc).
Never use the version of Python found at /usr/bin/python, it is an older version of Python that NERSC does not support for its users.

## Select the Right File System for Installing Your Own Packages

* If you mostly run serial Python jobs or use multiprocessing on a single node, you might be able to just install Python packages in your $HOME directory or on the /project file system without seeing any substantial performance issues.
* The best-performing shared file system for launching parallel Python applications is /global/common.
This file system is mounted read-only on Cori compute nodes with client-side caching enabled.
This is the file system that NERSC uses to install software modules (such as Python).
Contact NERSC if to see if your required packages can be made available on /global/common either as a NERSC-build module or through Anaconda.
* There are several interventions that can further improve Python package import times.
Users are advised to consider them and choose one that delivers the performance they desire at the level of invasiveness they are willing to accept.
