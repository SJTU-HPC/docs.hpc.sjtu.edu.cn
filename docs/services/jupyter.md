NERSC provides Jupyter (formerly known as IPython) services for
exploratory data analytics over the web. At NERSC, we view services
and others like them as the way to create new and exciting paths to
high-performance computing for interactive data analytics.

Currently, we are developing strategies and building infrastructure
that helps us provide services like these to users in a scalable
way. This effort requires a good deal of R&D on our part, so for the
time being we label these services as experimental.  For example, we
caution users not to consider Jupyter or RStudio as mission-critical
infrastructure for a new experiment or project without consulting with
NERSC staff first. But making this easier in the near future is our
plan.

At the same time, we do encourage users to experiment with these
services and give us feedback on how we can make them better. Please
report any issues to NERSC consultants at "consult at nersc dot gov."

## Jupyter

Jupyter (formerly IPython) is a flexible, popular literate-computing
web application for creating notebooks containing code, equations,
visualization, and text. Notebooks are documents that contain both
computer code and rich text elements (paragraphs, equations, figures,
widgets, links). They are human-readable documents containing analysis
descriptions and results but are also executable documents for data
analysis. Notebooks can be shared between researchers, or even
converted into static HTML documents. As such they are a powerful tool
for reproducible research and teaching.

A notebook is associated with one or more computational engines called
kernels. These kernels execute code passed to them from a notebook
process. When the Jupyter project was spun off from IPython, IPython
itself became one of many such kernels. A large number of kernels for
a number of languages and programming environments have been developed
to work with Jupyter.

On a laptop or workstation, a user typically starts up the Jupyter
notebook server application from the command line and uses a web
browser on the same system to author notebooks. JupyterHub is a web
application that enables a multi-user hub for spawning, managing, and
proxying multiple instances of single-user Jupyter notebook
servers. At NERSC, JupyterHub itself is run as a science gateway
application. Users authenticate to JupyterHub using their NERSC
credentials. Jupyter is available at NERSC through two services:

* The original Jupyter installation https://jupyter.nersc.gov/ runs as
  a science gateway application and is thus external to NERSC's Cray
  systems. Notebooks spawned by this service have access to the NERSC
  Global File System, in particular the /project and global $HOME file
  systems. They also use Python software environments and kernels that
  run on the science gateway hardware (not on e.g. Cori).
* The newer Jupyter installation at https://jupyter-dev.nersc.gov/
  actually spawns Jupyter notebooks on a reserved large-memory node of
  Cori. This means that these notebooks not only can access /project
  and global home directories, but also Cori $SCRATCH. The Python
  software environment used in these notebooks matches that found on
  Cori (Anaconda Python). Notebooks run on jupyter-dev can also submit
  jobs to the batch queues via simple Slurm Magic commands developed
  by NERSC staff. Users who have custom kernels that rely on code
  compiled on Cori will likely find that their kernels only work on
  jupyter-dev.  As we develop supporting infrastructure we plan to
  expand Jupyter offerings, make them more reliable, and create a
  well-defined service level guarantee. Ultimately we envision a
  single Jupyter URL at NERSC that will provide a variety of service
  levels to users.

### Customizing Kernels

You can customize your notebook experience to incorporate software you
have already installed at NERSC.  The following example demonstrates
starting up a local IPython kernel when `PATH` and `LD_LIBRARY_PATH`
need to be modified. This JSON file, a kernel spec should be placed
into `$HOME/.ipython/kernels/mykernel/kernel.json`:

```json
{
  "display_name": "mykernel",
  "language": "python",
  "argv": [
    "/global/homes/u/user/anaconda/bin/python",
    "-m",
    "ipykernel",
    "-f",
    "{connection_file}"
  ],
  "env": {
    "PATH":
    "/global/homes/u/user/anaconda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    "LD_LIBRARY_PATH":
    "/global/project/projectdirs/myproject/lib/:/global/homes/u/user/mylib"
  }
}
```

The argv field of the kernel specification can point to basically any
executable. So if setting up your kernel requires a number of settings
to be made to the environment or modules to be loaded, you might want
to place all that configuration in a shell script. Here is a further
example of how to do this. Create a shell script like:

```shell
#/bin/bash
module load ...
export SOMETHING=12345
/global/homes/u/user/anaconda/bin/python -m ipykernel $@
```

Suppose you call the above script "kernel-helper.sh."  This script
configures your environment and then launches the IPython kernel and
forwards along whatever arguments are passed to the script. Then a
corresponding kernel specification might be:

```json
{
  "display_name": "mykernel",
  "language": "python",
  "argv": [
    "/global/homes/u/user/kernel-helper.sh",
    "-f",
    "{connection_file}"
  ],
  "env": {
    "PATH":
    "/global/homes/u/user/anaconda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
  }
}
```
