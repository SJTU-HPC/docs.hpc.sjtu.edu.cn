# Jupyter

Jupyter is an essential component of NERSC's data ecosystem.
Use Jupyter at NERSC to:

* Perform exploratory data analytics and visualization of data stored on the NERSC Global Filesystem (NGF) or in databases at NERSC,
* Guide machine learning through distributed training, hyperparameter optimization, model validation, prediction, and inference,
* Manage workflows involving complex simulations and data analytics through the Cori batch queue,
* ... or do other things we haven't thought of yet.

[Jupyter](https://jupyter.readthedocs.io/en/latest/)
is a flexible, popular literate-computing web application for creating notebooks containing code, equations, visualization, and text.
Notebooks are documents that contain both computer code and rich text elements (paragraphs, equations, figures, widgets, links).
They are human-readable documents containing analysis descriptions and results but are also executable data analytics artifacts.
Notebooks are associated with *kernels*, processes that actually execute code.
Notebooks can be shared or converted into static HTML documents.
They are a powerful tool for reproducible research and teaching.

## JupyterHub

[JupyterHub](https://jupyterhub.readthedocs.io/en/stable/)
provides a multi-user hub for spawning, managing, and proxying multiple instances of single-user Jupyter notebook servers.
At NERSC, you authenticate to the JupyterHub instance we manage using your NERSC credentials and one-time password.
Here is a link to NERSC's JupyterHub service: https://jupyter.nersc.gov/

When you log into JupyterHub at NERSC, you will see a console or "home" page with some buttons.
These buttons allow you to manage notebook servers running on Cori or in Spin.
Which notebook server should you use?  It depends:

* Cori
    * Spawns Jupyter notebooks on special-purpose large-memory nodes of Cori (cori13, cori14, cori19)
    * Exposes GPFS and Cori `$SCRATCH` though not Edison `$SCRATCH`
    * Default Python software environment is the same as one of the modules found on Cori
    * Notebooks can submit jobs to Cori batch queues via simple Slurm Magic commands
* Spin
    * Runs as a [Spin](../services/spin/index.md) service and is thus external to NERSC's Cray systems
    * Notebooks spawned by this service have access to GPFS (e.g. `/project`, `$HOME`)
    * Python software environments and kernels run in the Spin service, not on Cori

We view the Cori notebook service as the production service users should normally use.
The Spin notebook service is a handy failover alternative if Cori login nodes are down.
Generally users should run a notebook service on Cori, unless there's a reason to fail over to Spin.

!!! tip
    The nodes used by <https://jupyter.nersc.gov/> are a shared resource, so
    please be careful not to use too many CPUs or too much memory.  Treat them
    like regular login nodes.

## JupyterLab

[JupyterLab](https://jupyterlab.readthedocs.io/en/stable/)
is the next generation of Jupyter.
It provides a way to use notebooks, text editors, terminals, and custom components together.
Documents and activities can be arranged in the interface side-by-side, and integrate with each other.

JupyterLab is new but [ready for use](https://blog.jupyter.org/jupyterlab-is-ready-for-users-5a6f039b8906) now.
With release 0.33 we have made JupyterLab the default interface to Jupyter on both hubs.
If you prefer to work with the "classic" interface select "Launch Classic Notebook" from the JupyterLab Help menu.
Alternatively you can also change the URL from `/lab` to `/tree`.

## Conda Environments as Kernels

You can use one of our default Python 2, Python 3, or R kernels.
If you have a Conda environment, depending on how it is installed, it may just show up in the list of kernels you can use.
If not, use the following procedure to enable a custom kernel based on a Conda environment.
Let's start by assuming you are a user with username `user` who wants to create a Conda environment on Cori and use it from Jupyter.

    cori$ module load python/3.6-anaconda-5.2
    cori$ conda create -n myenv python=3.6 ipykernel <further-packages-to-install>
    <... installation messages ...>
    cori$ source activate myenv
    cori$ python -m ipykernel install --user --name myenv --display-name MyEnv
    Installed kernelspec myenv52 in /global/u1/u/user/.local/share/jupyter/kernels/myenv52
    cori$

Be sure to specify what version of Python interpreter you want installed.
This will create and install a JSON file called a "kernel spec" in `kernel.json` at the path described in the install command output.

```json
{
	"argv": [
  		"/global/homes/u/user/.conda/envs/myenv52/bin/python",
  		"-m",
  		"ipykernel_launcher",
  		"-f",
  		"{connection_file}"
 	],
 	"display_name": "MyEnv52",
 	"language": "python"
}
```

## Customizing Kernels

Here is an example kernel spec where the user needs other executables from a custom `PATH` and shared libraries in `LD_LIBRARY_PATH`.
These are just included in an `env` dictionary:

```json
{
	"argv": [
  		"/global/homes/u/user/.conda/envs/myenv52/bin/python",
  		"-m",
  		"ipykernel_launcher",
  		"-f",
  		"{connection_file}"
 	],
 	"display_name": "MyEnv52",
 	"language": "python",
	"env": {
    	"PATH":
			"/global/homes/u/user/other/bin:/usr/local/bin:/usr/bin:/bin",
    	"LD_LIBRARY_PATH":
			"/global/project/projectdirs/myproject/lib:/global/homes/u/user/lib"
  	}
}
```

Note however that these environment variables do not prepend or append to existing `PATH` or `LD_LIBRARY_PATH` settings.
To use them you probably have to copy your entire path or library path, which is quite inconvenient.
Instead you can use this trick that takes advantage of a helper shell script:

```json
{
	"argv": [
    	"/global/homes/u/user/kernel-helper.sh",
  		"-f",
  		"{connection_file}"
 	],
 	"display_name": "Custom Env",
 	"language": "python"
}
```

The `kernel-helper.sh` script should be made executable (`chmod u+x kernel-helper.sh`).
The helper could be like:

```shell
#!/bin/bash
module load <some-module>
module load <some-other-module>
export SOME_VALUE=987654321
exec /global/homes/u/user/.conda/envs/myenv52/bin/python \
	-m ipykernel_launcher "$@"
```

You can put anything you want to configure your environment in the helper script.
Just make sure it ends with the `ipykernel_launcher` command.

## Shifter Kernels on Jupyter

Shifter works with Cori notebook servers, but not Spin notebook servers.
To make use of it, create a kernel spec and edit it to run `shifter`.
The path to Python in your image should be used as the executable.
Here's an example of how to set it up:

```shell
{
	"argv": [
    	"shifter",
        "--image=continuumio/anaconda3:latest",
        "/opt/conda/bin/python",
        "-m",
		"ipykernel_launcher",
        "-f",
		"{connection_file}"
	],
    "display_name": "my-shifter-kernel",
    "language": "python"
}
```

## Spark on Jupyter

You can run small instances (< 4 cores) of Spark on Cori with Jupyter.
You can even do it using Shifter too.
Create the following kernel spec (you'll need to make the `$SCRATCH/tmpfiles`, `$SCRATCH/spark/event_logs` directories first):

```shell
{
    "display_name": "shifter pyspark",
    "language": "python",
    "argv": [
        "shifter",
        "--image=nersc/spark-2.3.0:v1",
        "--volume=\"/global/cscratch1/sd/<your_dir>/tmpfiles:/tmp:perNodeCache=size=200G\"",
        "/root/anaconda3/bin/python",
        "-m",
        "ipykernel",
        "-f",
        "{connection_file}"
    ],
    "env": {
        "SPARK_HOME": "/usr/local/bin/spark-2.3.0/",
        "PYSPARK_SUBMIT_ARGS": "--master local[1] pyspark-shell
            --conf spark.eventLog.enabled=true
            --conf spark.eventLog.dir=file:///global/cscratch1/sd/<your_dir>/spark/event_logs
            --conf spark.history.fs.logDirectory=file:///global/cscratch1/sd/<your_dir>/spark/event_logs pyspark-shell",
        "PYTHONSTARTUP": "/usr/local/bin/spark-2.3.0/python/pyspark/shell.py",
        "PYTHONPATH": "/usr/local/bin/spark-2.3.0/python/lib/py4j-0.10.6-src.zip:/usr/local/bin/spark-2.3.0/python/",
        "PYSPARK_PYTHON": "/root/anaconda3/bin/python",
        "PYSPARK_DRIVER_PYTHON": "ipython3",
        "JAVA_HOME":"/usr" 
    }
}
```
