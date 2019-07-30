# Software

## Modules and Bioinformatics Software

Unlike previous systems, a limited and generic set of software modules are
maintained by NERSC and available for JGI users on Cori. Users should
themselves build and maintain any more specialized software they may need.

### Alternatives to using NERSC provided software modules are:

* Shifter, with the software built inside a Docker container
* Anaconda virtual environments
* A wide array of powerful compilers and tools are available
  to build software from source code. While scratch
  space is subject to NERSC's purge policy, many groups posess a
  "sandbox" allocation which is equally performant to \$BSCRATCH,
  and shared among group members. (Running software from $HOME is
  strongly discouraged.)

## Shifter

Shifter is a Docker-like software tool used to run containers on NERSC
systems. More information on Shifter is available
[here](../../programming/shifter/how-to-use.md), and Shifter
training slides are also available. The main advantages of using
Shifter are:

* Using containers makes your workflow portable between Cori, other
  HPC facilities and clusters, and cloud resources
* You no longer need to depend on system features, such as specific
  compiler versions, software libraries, or other tools
* Because Shifter uses Docker containers, you can build and debug
  containers on your laptop or desktop, then be confident they will
  run the same way on Cori or other platforms.
* Shifter exists because Docker does not incorporate the security
  requirements of a HPC facility. Docker requires an access model
  and permissions which are too broad; anyone who
  can run a container has access to the entire machine. Shifter
  implements a subset of the same functionality that
  Docker provides and can run Docker containers unmodified.

The process for building a container and running it with Shifter is
roughly as follows:

1. use Docker on a laptop or desktop machine to build a Docker
   container for your software
2. push that container to Dockerhub or another Docker container
   registry and use Shifter on Cori to pull that image to
   the NERSC Shifter registry
3. use Shifter on a batch node to run that container, and perform
   useful work

!!! note
	The JGI has a containerization project, intended to provide
	standardized containers for the majority of JGI use-cases. If you
	need a container for a tool or pipeline, check with your
	consultants - you may find it's already been done for
	you. Documentation for the containerization project will be made
	available soon.

## Anaconda

Anaconda virtual environments allow you to install software in a clean and
reproducible manner. See this the [training page](training.md) for more
information about using Anaconda at NERSC. Note that while Anaconda started
life as a python package manager, now R, Perl and many other computing
languages and associated modules can also be installed using it.
