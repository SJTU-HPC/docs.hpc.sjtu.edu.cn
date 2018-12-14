# Software

## Programming Environments

## Important Environment Variables

## Modules and Bioinformatics Software

Unlike Genepool, very few NERSC software modules are available for
user software. We encourage users to use alternatives wherever
possible, which will be more maintainable and portable to other
platforms. If you have Genepool module software that is critical to
migrating your workflow to Cori, please talk to a consultant.

### Alternatives to using software modules are:

* Shifter, with the software installed in a Docker container
* Anaconda virtual environments
* You are, as always, welcome to install and maintain any software you
  like on NERSC systems within your own user disk space. While scratch
  space is subject to NERSC's purge policy, many groups hold a
  "sandbox" space which is equally performant to $BSCRATCH and shared
  among group members. (Running software from $HOME is strongly
  discouraged.)

## Shifter

Shifter is Docker-like software for running containers on NERSC
systems. More information on Shifter is available here, and Shifter
training slides are also available. The main advantages of using
Shifter are:

* Using containers makes your workflow portable, across Cori, Denovo,
  Edison, and to cloud resources
* You no longer need to depend on system features, such as specific
  compiler versions, software libraries or other tools
* Because Shifter uses Docker containers, you can build and debug
  containers on your laptop or desktop, then be sure they will run the
  same way on Cori or other NERSC platforms.
* Shifter exists because Docker cannot be safely run on NERSC
  machines. Docker requires too much access to the system, anyone who
  can run a container can essentially access the entire
  machine. Shifter implements a subset of the same functionality that
  Docker provides, and can run Docker containers unmodified.

The process for building a container and running it with Shifter is
roughly as follows:

1. use Docker on a laptop or desktop machine to build a Docker
   container for your software
2. push that container to Dockerhub or another Docker container
registry use Shifter on Cori, Edison or Denovo to pull that image to
the NERSC Shifter registry
3. use Shifter on a batch node to run that container, and perform
useful work.  Note that Shifter is not available on Genepool, the
kernel version there is too old to support containers. This is just
one reason why we're commissioning Denovo.

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
