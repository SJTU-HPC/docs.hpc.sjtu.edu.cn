# Python on Cori KNL

The many-core Intel Xeon Phi (Knights Landing, KNL) architecture
presents new opportunities for science applications in terms of
scaling within a compute node from parallelism at the thread and
vector register level.  However, for Python applications, the KNL
architecture poses numerous challenges as well.

* Python applications that do not already take advantage of on-node
parallelism on Cori's Haswell processors can be
expected to deliver markedly worse performance on Cori KNL.  While the
KNLs are more energy-efficient, their lower clock rate and
instructions retired per cycle are much lower than those
architectures.
* Code written in Python that takes advantage of threaded
"performance" libraries written in C, C++, or Fortran (along the lines
of the Ousterhout Dichotomy) may be able to take advantage of the
larger number of CPUs per node.  Such libraries, such as numpy or
scipy built on top of Intel MKL, use OpenMP to deliver thread-level
parallelism and include specialized vectorization instructions.
* However, using threaded and vectorized performance library calls may
not be enough.  If Python code can make calls to performance libraries
then a computational bottleneck at the Python interpreter level arises
(see Amdahl's Law).  Therefore it is important that code "spend as
much time as possible" in doing computations in threaded/vectorized
performance libraries.
* Familiar and useful standard Python programming patterns and
libraries may not behave in ways that users are accustomed to on other
architectures.  One example is multiprocessing: Here, process spawning
is delegated to the operating system and processes are distributed
some load imbalance.  On the Haswell architecture, using a
multiprocessing Pool this is not really noticeable; on KNL the load
imbalance is severe.

The above issues (and others) should give Python developers pause.
Python is a powerful language for productive programming in the
sciences and data analysis.  Users of Python at NERSC may need to
consider the following, assuming that switching from Python to another
language is not (yet) feasible:

* How important is it for me to migrate Python code to KNL right now?
If the performance of your code on Ivy Bridge or Haswell is
satisfactory and the life cycle of your project is such that you will
be able to continue using those architectures for the next few years,
you may not need to worry about KNL right now at all.  However, if
plans for your code continued development and use beyond 2020, you may
want to think about what KNL and similar architectures means to your
application.
* Won't the developers of Python just fix all these problems for us
anyway?  Perhaps.  On the other hand, we suggest the risk from taking
such a cavalier attitude is too great for users of Python in HPC.  And
in general we always recommend future-proofing code.
* What skills and tools do I need to migrate to KNL?  It is an open
question just how easy it will be for Python codes to migrate to KNL
and, with any amount of work, deliver performance superior to (e.g.)
Haswell on Cori on a node-for-node basis.  However, there are tools
and techniques that Python programmers can learn that may help make
the transition.

Below we discuss tools, techniques, and skills that Python developers
can adopt or learn that may help them migrate code to Cori KNL.  As we
learn more about what the issues are with Python on KNL, workarounds,
and solutions, we will add to this page.  Much of the information
presented here comes from our input from the Intel Python team and
from work done in the NESAP for Data program.

## Anaconda Python and the Intel Distribution for Python

As documented elsewhere NERSC provides software modules for Anaconda
Python, including both Python 2 and Python 3.  The Anaconda Python
distribution includes a number of optimized libraries that leverage
Intel's expertise, particularly the Intel Math Kernel Library (MKL).
These libraries include numpy, scipy, numexpr, etc.  Most importantly
these libraries are threaded and include vector optimizations critical
for maximum performance on KNL.

In 2016 Intel released their own distribution of Python.  The
relationship between Intel Python and Anaconda Python is very close.
Python users should not view the two products as necessarily being
developed in competition --- rather Continuum Analytics (the company
behind Anaconda Python) and the Intel Python team work closely
together to deliver maximum performance of Intel's hardware to Python
users through a collaborative effort.

The Intel Python Distribution provides the above MKL optimizations but
in addition provides TBB (Thread Building Blocks library) and
interfaces to the DAAL (Data Analytics Acceleration Layer).  TBB in
particular enables users to compose threads across threaded library
calls and avoid thread oversubscription (see this blog post); with
particular emphasis on DASK and Joblib.

Users can try the Intel Distribution for Python through a conda
environment even!  This post provides more details, but at NERSC one
can use the following procedure:

    module load python/2.7-anaconda-2019.07
    conda create -n idp -c intel intelpython2_core python=2
    source activate idp

The conda tool can be further used to customize the Intel Python
environment in the usual manner (conda install, etc).

## Interactive Python and IPython in Intel Python Distribution

IPython is a popular interactive shell for Python providing command
history, tab-completion, ready access to documentation, syntax
highlighting and much more.  And Python itself provides an interactive
environment, without all those nice extras.  Many of these features
are provided by GNU readline, and because of licensing restrictions
Intel Python does not redistribute readline (see here).  Users can get
around this for IPython, for instance, by installing the Anaconda
defaults channel version of IPython:

    conda install -c defaults ipython

Mixing Intel packages and Anaconda packages, managed through the conda
tool, seems like the best option for now.

## Profiling Tools

Coming soon...

## Known Issues

Here we track known issues with Python on Cori KNL.  Check back
frequently to see what new issues have been added and what existing
ones have been resolved.

### Set Environment Variable: KMP_AFFINITY=disabled

For process-level threading (e.g., multiprocessing) to work in Python
on KNL, users are advised to set the KMP_AFFINITY variable to
"disabled" as follows in bash:

    export KMP_AFFINITY=disabled

This is especially important if there are any calls to performance
libraries with OpenMP regions in them.  The reason is that the first
OpenMP region creates a CPU affinity mask that later prevents
processes (not OpenMP threads) from spawning off of the master CPU.
The symptom is total lack of scaling as the number of requested
processes increases.

This is a known issue in Intel's OpenMP release that should be
addressed with the next release of Intel OpenMP later in 2017.  Until
that release is made and installed at NERSC we advise users to use the
setting above.
