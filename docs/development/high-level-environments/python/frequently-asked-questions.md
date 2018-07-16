
# Python at NERSC FAQ

Have a question about using Python on NERSC's supercomputers that you think others might have already asked?
This FAQ may be useful to you.
If you have suggestions or updates for the NERSC Python FAQ, contact consult@nersc.gov.

## Can NERSC install [some Python package] for me?

Users sometimes contact NERSC to ask if a Python package could be installed with NERSC-maintained Python (i.e., Python installed by NERSC staff at /usr/common/software and available via "module load").
 We consider three broad guidelines in making a decision:

* General utility.  It makes sense for NERSC to focus support on packages that are broadly useful to the most users. At the same time we are happy to help individual users install more specialized packages for their own use.  (See below.)
* Maintenance activity. We prefer to install packages that are actively maintained by community-engaged developers. This way, if we run into problems we can engage with developers to arrive at a solution quickly. Abandoned projects may also pose a security risk and we discourage users from such packages altogether.
* Ease of installation. Python packages are usually straightforward to install, but in cases where the build system is effectively broken and we cannot debug the problem, we may need to wait quite some time for the developer to address the issue.

Actively maintained, easy to install packages that a large number of users will find useful are the most likely candidates for NERSC support.
Packages that only a single user or a small number of users need are likely to be met with a suggestion that the requester manage installation themselves.
Abandoned packages will not be installed but we may suggest alternatives.

## Can I use virtualenv on Cori and Edison?

The virtualenv tool is not compatible with the conda tool used for maintaining Anaconda Python.
After the July 2017 Edison upgrade the default Python module on both systems has been Anaconda Python.
But this is not necessarily bad news as conda is an excellent replacement for virtualenv and addresses many of its shortcomings.
If you need help migrating a custom Python environment from virtualenv to a conda environment, contact consult@nersc.gov.

Of course there is nothing preventing you from doing a from-source installation of Python of your own, and then using virtualenv if you prefer.

## What if I can't find a conda package I need?

Conda package builds are provided through namespaces called channels.
At NERSC, we try to stick to packages from the defaults channel as much as possible.
Doing so helps us maintain a coherent Anaconda installation to guarantee that all the installed packages work together (a release).
But sometimes the defaults channel doesn't provide a version of a package we need.
In these cases we tend to use pip (see also) after installing dependencies using the conda tool where possible.

If you want to use another channel beyond the defaults channel, you can, but we suggest that you select your channel carefully.
We've found that there isn't much guidance in terms of which channels are actively maintained or exactly who is managing them.
Sometimes it's obvious from the name. Other times a developer community creates a channel of its own but you have to reach out to developers to find the right one.

To search for a package beyond the defaults channel, use the Anaconda client tool.
For example, to see channels providing AstroPy:

    module load python
    anaconda search -t conda astropy

Be sure to look for builds for the "linux-64" platform.

## Can I use "pip" to install my own packages?

Yes.
Pip is available under Anaconda Python.
If you create a conda environment but you are unable to find a conda build of whatever package (or version of that package) you want to install, then pip is one viable alternative.
The other alternative is to try a different channel (see also).

Users of the pip command may want to use the "--user" flag for per-user site-package installation following the PEP370 standard.
On Linux systems this defaults to `$HOME/.local`, and packages can be installed to this path with "pip install --user package_name."
This can be overridden by defining the PYTHONUSERBASE environment variable.

NOTE: To prevent per-user site-package installations from conflicting across machines and module versions, at NERSC we have configured our Python modules so that PYTHONUSERBASE is set to `$HOME/.local/$NERSC_HOST/version` where "version" corresponds to the version of the Python module loaded.

## Can I install my own Anaconda Python "from scratch?"

Yes.
One reason you might consider this is that you want to install Anaconda Python on $SCRATCH or in a Shifter image to improve launch-time performance for large-scale applications.
Or you might want more complete control over what versions of packages are installed and don't want to worry about whether NERSC will upgrade packages to versions that break backwards compatibility you depend on.
See here for more information on how you can do this.

## How can I use mpi4py from my Anaconda environment?

If you try to use mpi4py and you observe something like an apparent MPI_COMM_WORLD size of 1 and all processes report that they are rank 0, check to see if you have installed mpi4py from Anaconda with the conda tool.

If you have created an Anaconda environment through "conda create" or you have run the Anaconda/Miniconda installer script you will want to build mpi4py "from source" and not use "conda install mpi4py."
You will definitely want to build mpi4py using the Cray compiler wrappers described here and here that link in Cray MPICH libraries.
The procedure is described in more detail here.

## Why do I get an error using Matplotlib on compute nodes?

Using Matplotlib to interactively plot on the login nodes is easy, especially if you use NX.
But if you are running a Python script on compute nodes that imports Matplotlib, even if it doesn't make any plot files, it is important to specify an appropriate "backend.
" There are a few ways to do this, one is to simply tell Matplotlib to use a particular backend in your script as below:

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

The "Agg" backend is guaranteed to be available, but there are other choices.
If a backend is not specified in some way, then Matplotlib will seek out an X11 connection on the compute nodes in your job and the result is that it your job may simply wait until the wall-clock limit is reached, or try to connect back to the login node you submitted from with an error like " : cannot connect to X server cori09:28.0."
More technical details are available in the Matplotlib FAQ, "What is a Backend?" and the matplotlib.use API documentation.

## How do I use the Intel Distribution for Python at NERSC?

We recommend one of two methods.
Both methods use conda environments.
The first is to use one of NERSC's Anaconda modules, then create a conda environment based on the Intel Distribution for Python:

    module load python
    conda create -n idp -c intel intelpython2_core python=2
    source activate idp

The second method is to first run the Anaconda or Miniconda installer (see here) and then use the "conda create" and "source activate" lines from above.

## Should I use Anaconda or Intel Distribution for Python?

Intel Math Kernel Library (MKL), Data Analytics Acceleration Library (DAAL), Thread Building Blocks (TBB), and Integrated Performance Primitives (IPP) are available through Intel Community Licensing.
This enabled both Continuum Analytics and Intel to provide access to Intel's performance libraries through Python for free starting in late 2015 and early 2016.
In terms of performance the two distributions are about the same.

Python includes source code licensed under GPL and this constrains the Intel Python distribution somewhat.
Most importantly, interactive use of Python or IPython under the Intel Distribution is provided without GNU readline.
There doesn't yet appear to be a viable non-GPL alternative at this point in time.
We suggest that for the most part users may find that Anaconda Python provides the best of both worlds, but that Anaconda Python may lag slightly behind the Intel Distribution in terms of performance for short periods.
The two companies have a very strong relationship and we think this greatly benefits Python users in the long term.

## Unexpected error while saving and disk I/O error

If you try to save or create a new notebook and you see an error like `Unexpected error while saving file: <path-to-notebook> disk I/O error`
you may just be over quota.
Use ssh to log into Cori and run `myquota` to check.

## Coming Soon

How can I profile my Python code's performance?
