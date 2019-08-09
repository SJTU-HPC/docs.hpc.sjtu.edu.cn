# Python

[Python](https://www.python.org/about/) is an interpreted
general-purpose high-level programming language.  You can
use [Anaconda Python](https://docs.anaconda.com/anaconda/) on Cori
through software environment modules.  Do **not** use the
system-provided Python `/usr/bin/python`.

## Anaconda Python

Anaconda Python is a platform for large-scale data processing,
predictive analytics, and scientific computing.  It includes hundreds
of open source packages and Intel MKL optimizations throughout the
scientific Python stack.  Anaconda provides
the [`conda`](https://conda.io/)
command-line tool for managing packages, but also works well
with [`pip`.](https://pip.pypa.io/en/stable/user_guide/) The Anaconda
distribution also exposes access to
the
[Intel Distribution for Python.](https://software.intel.com/en-us/distribution-for-python/get-started)

Both Anaconda Python 2 and 3 are available.  For example, to load the
Python 3.7 Anaconda environment, type:

    module load python/3.7-anaconda-2019.07

The default is `python/2.7-anaconda-2019.07` so only `module load python`
is necessary to use it. Python 2.7 will remain
the
default
[through the end of 2019](https://github.com/python/devguide/pull/344)
but you are encouraged to migrate to Python 3 before then.

When you load a Python module you are placed into its default Anaconda
"root" environment.  This may be sufficient for your needs.  NERSC can
install Anaconda packages into the root environment upon request
subject to review of broad utility and maintainability.  Contact us to
find out about specific packages.  In general we recommend users
manage their own Python installations with "conda environments."

## Conda Environments

The `conda` tool lets you build your own custom Python installation
through "environments."  Conda environments replace and
surpass
[`virtualenv`](https://virtualenv.pypa.io/en/stable/userguide/)
virtual environments in many ways.  To create and start using a conda
environment you can use `conda create`.  Specify a name for the
environment and at least one Python package to install.  In particular
you should specify which Python interpreter you want installed.
Otherwise `conda` may make a decision that surprises you.

    module load python/3.7-anaconda-2019.07
    conda create -n myenv python=3 numpy
    [installation outputs]
    source activate myenv

!!! tip "Should I use `source activate\deactivate` or `conda activate\deactivate`?"
    You may have noticed messages like:

    ```
     To activate this environment, use
         $ conda activate <environment>
     To deactivate an active environment, use
         $ conda deactivate
    ```

    **Short answer**:

    - Use `source activate <environment>` or
    `source activate && conda activate <environment>`

    - Use `conda deactivate` instead of `source deactivate`

    **Long answer**:
    Using `conda activate <environment>` will fail with a message to
    run conda init <shell>. At NERSC, this is not possible because `conda init` will
    try to modify the shell initialization script (.bashrc for bash), which is
    owned by a restricted user, not by you. Running `source activate` (no environment
    specified) will effectively emulate the functionality of `conda init <shell>`.
    Once, conda is activated, either by `source activate && conda activate
    <environment>` or `source activate <environment>`, it is then possible (and
    recommended) to use the modern `conda deactivate` to deactivate the conda
    environment.

    Part of the issue is that conda
    does not provide a way for us to edit this configuration. The other part
    is how we have configured our own system. We are hoping that one or both
    of these issues can be resolved soon so we can switch to the more modern
    `conda activate`.

Before it installs anything `conda create` will show you what package
management steps it will take and where the installation will go.  You
will be asked for confirmation before installation proceeds.

!!! tip "The Life You Save May Be Your Own"
    Make it a habit to actually review `conda` tool reports and not just
    blithely punch the "y" key to approve create/install/update actions.
    Verify the installation is going where you think it should.  Make
    sure any proposed package downgrades are acceptable.

Once the environment is created you need to switch to it using `source
activate`.  After activation the name of the environment will appear
in your prompt.

To leave an environment behind use:

    source deactivate

!!! attention "Bad News for csh/tcsh Users"

    If you use csh or tcsh you will not be able to use the `source
    activate` syntax.  For csh users this is a shortcoming of the
    conda tool.  There are workarounds available on the web that work
    to varying degrees.  (We often find users are able to switch to /bin/bash
    without much difficulty, that is one solution.)

    If you are a csh user and you do not need to install or manage
    packages once a conda environment has been provisioned, you can
    simply set `PATH` to point to the path of the Python interpreter
    in the environment.

## Installing Packages

You can find packages and install them into your own environments
easily.

    conda search scipy
    [list of available versions of scipy]
    conda install scipy

If `conda search` fails to identify your desired package it may still
be installed via `pip.` Both `conda` and `pip` can be used in conda
environments.

!!! attention "Use conda to Install pip into Your Environment"
    To use `pip` in your own environment you may need to `conda install
    pip`.  Verify whether you need to by typing "`which pip`" at the
    command line.  If the path returned looks like
    `/usr/common/software/python/.../bin/pip` then do `conda install
    pip`.

If you consider `pip` a last resort you may want to search non-default
channels for builds of the package you want.  The syntax for that is a
little different:

    anaconda search -t conda <package-name>
    [list of channels providing the package]
    conda install -c <channel-name> <package-name>

Finally you can install packages "from source" and in some cases this
is recommended.  In particular any package that depends on the Cray
programming environment should be installed this way.  For Python this
usually boils down to `mpi4py` and `h5py` with MPI-IO support.

### `mpi4py`

Users creating their own conda environments should build `mpi4py`
using the Cray compiler wrappers instead of using `conda install
mpi4py`.  If you try to use `mpi4py` but you observe an
`MPI_COMM_WORLD` size of 1 and all processes report they are rank 0,
it could be because of a conda-installed `mpi4py`.

You can build `mpi4py` and install it into a conda environment on Cori
using a recipe like the following:

    wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.0.tar.gz
    tar zxvf mpi4py-3.0.0.tar.gz
    cd mpi4py-3.0.0
    module swap PrgEnv-intel PrgEnv-gnu
    module unload craype-hugepages2M
    python setup.py build --mpicc="$(which cc) -shared"
    python setup.py install

The MPI-enabled Python interpreter is not required (see
[this page](https://mpi4py.readthedocs.io/en/stable/appendix.html#mpi-enabled-python-interpreter) in
the mpi4py documentation).  To install it however, use these additional steps:

    python setup.py build_exe --mpicc="$(which cc) -dynamic"
    python setup.py install_exe

Initializing MPI on a login node will not work at NERSC.  This is what you will
see if you try to do it:

    nersc$ module load python
    nersc$ python -c 'from mpi4py import MPI'
    [Fri Aug  9 09:26:55 2019] [unknown] Fatal error in PMPI_Init_thread: Other MPI error, error stack:
    MPIR_Init_thread(537):
    MPID_Init(246).......: channel initialization failed
    MPID_Init(647).......:  PMI2 init failed: 1
    Aborted

If you see this kind of output from a batch job or in an interactive allocation
then it means something different.  It likely means that `MPI_Init()` exceeded
a timeout, perhaps due to I/O issues.  This is more likely to occur when the
file system you are importing packages from isn't optimized for serving up code
to the compute nodes.  Increasing the timeout is a temporary fix:

    export PMI_MMAP_SYNC_WAIT_TIME=300

but it just gives your job more time to start up.  What you want is for your
job to start up more quickly.  See the documentation on
[`/global/common/software`](../../../../filesystems/global-common)
or better yet,
[Shifter.](../../shifter/overview.md)

!!! tip "About Huge Memory Pages (As of 2019-08-02)"
    Note also that we recommend you unload craype-hugepages2M before
    compiling.  There's an issue with how Python and huge memory pages
    can work together, but Cray is working on a solution.  When that fix
    is in place we'll reconsider the guidance here, but for now
    compiling mpi4py without huge memory pages seems the easiest path
    forward for users.

### `h5py` with MPI-IO

[H5Py](../../libraries/hdf5/h5py.md)

## Collaborative Installations

Collaborations, projects, or experiments may wish to install a
shareable, managed Python stack to `/global/common/software`
independent of the NERSC modules. You are welcome to use the Anaconda
installer script for this purpose. In fact you may want to consider
the more "stripped-down" [Miniconda](https://conda.io/miniconda.html)
installer as a starting point. That option allows you to start with
only the bare essentials and build up. Be sure to select Linux
version in either case! For instance:

    https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b \
        -p /global/common/software/myproject/env
    [installation messages]
    source /global/common/software/myproject/env/bin/activate
    conda install <only-what-my-project-needs>

You can customize the path with the `-p` argument.  Ihe installation
above would go to `$HOME/miniconda3` without it.

!!! attention
    When using your own Anaconda/Miniconda installation be sure not to
    load any NERSC-provided Python modules.  Also take care to
    consider the `PYTHONSTARTUP` environment variable which you may
    wish to unset altogether.  It is mainly relevant to the system
    Python we advise against using.

Note that to activate the root environment, technically you should use
the `source` shell command.  Setting `PATH` to the root environment
`bin` directory works but the source/conda tool does more than that.
