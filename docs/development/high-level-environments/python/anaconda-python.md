
# Anaconda Python

Anaconda is a distribution of Python for large-scale data processing, predictive analytics, and scientific computing.
It includes a collection of about 200 open source packages and includes Intel MKL optimizations throughout the scientific Python stack.
Additional packages are available through contributed channels or through installation with pip.
The Anaconda distribution also exposes access to the Intel Distribution for Python (instructions here).

On Edison and Cori both Anaconda Python 2.7 and 3.5/6 are available through a module load command.
For example, to load the Python 2.7 Anaconda environment on Cori, type:

    module load python/2.7-anaconda-4.4

On Cori, 2.7-anaconda-4.4 has been made the default module so only "module load python" is necessary to load it there.
NERSC Anaconda modules include mpi4py.
All NERSC builds of Python on Cori and Edison have been retired.

The Anaconda distribution provides access to conda, an open source package management tool and environment management system for installing multiple versions of software packages and their dependencies and switching easily between them.

## Packages

When you load an Anaconda Python module, you are placed into the default Anaconda "root" environment.
For many users the packages provided by the root environment are sufficient for their needs.
NERSC can install Anaconda packages into the root environment upon request, subject to a quick review concerning broad utility and maintainability. 
Contact us to find out about specific packages.
The list of packages in a given environment can be obtained by entering at the prompt:

    conda list

You can create a "spec list" to be used to construct an identical environment across different systems or share environments with collaborators by typing "conda list -e."  
Using "conda create --file" with the output spec-list generates a copy of the original environment.
There are several ways to share/replicate environments with the conda tool.
For more information, see the conda documentation.

## Environments

Conda gives its users the power to manage entire software stacks through its environments.
At NERSC you can create, customize, update, share, and maintain their own environments this way.
To create and start using a conda environment, use the "conda create" tool, specifying a name for the environment and at least one Python package to install (like numpy).

    module load python/2.7-anaconda-4.4
    conda create -n myenv numpy python=2
    [installation messages]
    source activate myenv

Before it installs anything the "conda create" command will show you what package management steps it will take and where the installation will go.
By default conda will create environments in $HOME/.conda but this can be changed via the use of the .condarc file and "conda config" command (in particular by changing the "envs_dirs" setting, see documentation here).
Once the environment is created you need to switch to it using the "source activate" command.
After activation, the name of the environment will appear to the left of the user prompt.

To leave a created environment behind, use:

    source deactivate

NOTE: If you use csh or csh-derivative shells you will not be able to use the "source activate" syntax.
For csh users this is a shortcoming of the conda tool.
There are workarounds available on the web that work to varying degrees.
This is particularly an issue on Edison where the default shell setting is /bin/csh.
(We often find users are able to switch to /bin/bash without much difficulty, that is one solution.)
If you are a csh user and you don't need to install or manage packages once a conda environment has been provisioned, you can simply set $PATH to point to the path of that environment's Python interpreter via setenv.

An example .condarc file on Cori might look like the listing below:

    envs_dirs:
        - /global/homes/u/username/cori-envs
    channels:
        - defaults
    show_channel_urls: yes

## Installing Packages

Finding and installing packages into an environment is relatively easy.

    conda search scipy
    [list of available versions of scipy]
    conda install scipy

If the "conda search" command fails to identify a desired package, it may still be installed by "pip."
It should be straightforward but see documentation is here.
Certain packages (such as mpi4py or h5py with MPIIO support) may require special Cray-specific flags to build and install properly.
For assistance, contact NERSC consulting.

NOTE: The "pip" included with the python modules at NERSC installs packages into the default conda environment in /usr/common/software, so you will see "permission denied" messages.
The solution is to first install pip in your conda environment, with:

    conda install pip 
  
NOTE: Conda environments replace virtualenv and the two should not be used together, see this chart for a comparison of conda, pip, and virtualenv commands.

## User-Managed Installation

The root environment in NERSC's Anaconda modules provides a good starting point if you are new to Python, HPC, or both.
Eventually you may graduate to managing your own conda environments, or you may wish to install a Python stack to a specific file system (read here to understand why) completely independent of NERSC's Anaconda installation.
You are welcome to use the Anaconda installer script or better yet, the Miniconda installer script for this purpose.
Be sure to select Linux version in either case!
For instance:

    https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    [installation messages, default path is $HOME/miniconda3]
    source miniconda3/bin/activate

This installs a root Miniconda Python 3 environment to the default path "$HOME/miniconda3." 
The path can be changed through a "-p" argument.
Note that to activate the root environment, technically you should use the "source" shell command (not "source activate," that is for environments).
Environments under this install can be created as with the NERSC-maintained module.

This method allows users to be confident that they have total control of their Python environment, especially if they install it to a path like $SCRATCH.

NOTE: When using your own Anaconda/Miniconda installation be sure not to load any NERSC-provided Python modules.
Also take care to consider the PYTHONSTARTUP environment variable which you may wish to unset altogether; it is mainly relevant to system Python we advise against using.

## Building `mpi4py`

Users creating their own conda environments should probably build mpi4py using the Cray compiler wrappers instead of using "conda install mpi4py."
Assuming the user has activated their conda environment on Cori, one recipe is:

    wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-2.0.0.tar.gz
    tar zxvf mpi4py-2.0.0.tar.gz
    cd mpi4py-2.0.0
    module swap PrgEnv-intel PrgEnv-gnu
    python setup.py build --mpicc=$(which cc)
    python setup.py build_exe --mpicc="$(which cc) -dynamic"
    python setup.py install
    python setup.py install_exe

On Edison the procedure is slightly different.
The "build" command is:

    LDFLAGS="-shared" python setup.py build --mpicc=$(which cc)

The build_exe step builds python-mpi, and the install_exe step puts it into the conda path, with the name "python2.7-mpi" which can easily be relinked to "python-mpi" if the user prefers.
Of course, the MPI-enabled Python interpreter is not required (see this page in the mpi4py documentation) to use mpi4py.


