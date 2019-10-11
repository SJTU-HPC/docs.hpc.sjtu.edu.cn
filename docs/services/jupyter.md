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
    * Exposes GPFS and Cori `$SCRATCH`
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

## Using Jupyter at NERSC for Events

Jupyter at NERSC can be used for demos, tutorials, or workshops.
You can even use training accounts with Jupyter at NERSC.
If you plan to use Jupyter in this way, we ask that you observe the following guidelines:

* If 20 people or less at your event will be logging into jupyter.nersc.gov, there's no need to let us know ahead of time.
  We should be able to handle that level of increased load without any issues.
  Just be sure you don't schedule your event on a day when there is scheduled maintenance.
* For events where more than 20 people are logging in, please send us a heads up **at least 1 month in advance** via [ticket.](https://help.nersc.gov)
  We've been able to absorb events of 50-100 people without any issues but we still want to know about your event.
  This lets us keep an eye on things while your event is going and hopefully keep things going smoothly.
* In either case please let us know if you have any special requirements or would like to do something more experimental.
  That is likely to incur a need for more lead time, but we're willing to work with you if there aren't already similar events coming up.
  For this case, please contact us **at least 2 months in advance** via ticket.

These are not hard and fast rules, but we're more likely to be able to help if we have advanced notice.

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

    cori$ module load python/3.7-anaconda-2019.07
    cori$ conda create -n myenv python=3.7 ipykernel <further-packages-to-install>
    <... installation messages ...>
    cori$ source activate myenv
    cori$ python -m ipykernel install --user --name myenv --display-name MyEnv
    Installed kernelspec myenv in /global/u1/u/user/.local/share/jupyter/kernels/myenv
    cori$

Be sure to specify what version of Python interpreter you want installed.
This will create and install a JSON file called a "kernel spec" in `kernel.json` at the path described in the install command output.

```json
{
	"argv": [
  		"/global/homes/u/user/.conda/envs/myenv/bin/python",
  		"-m",
  		"ipykernel_launcher",
  		"-f",
  		"{connection_file}"
 	],
 	"display_name": "MyEnv",
 	"language": "python"
}
```

## Customizing Kernels

Here is an example kernel spec where the user needs other executables from a custom `PATH` and shared libraries in `LD_LIBRARY_PATH`.
These are just included in an `env` dictionary:

```json
{
	"argv": [
  		"/global/homes/u/user/.conda/envs/myenv/bin/python",
  		"-m",
  		"ipykernel_launcher",
  		"-f",
  		"{connection_file}"
 	],
 	"display_name": "MyEnv",
 	"language": "python",
	"env": {
    	"PATH":
			"/global/homes/u/user/other/bin:/usr/local/bin:/usr/bin:/bin",
    	"LD_LIBRARY_PATH":
			"/global/project/projectdirs/myproject/lib:/global/homes/u/user/lib"
  	}
}
```

## Customizing Kernels with a Helper Shell Script

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
export SOME_VALUE=987654321

module load python/3.7-anaconda-2019.07
source activate myenv

exec /global/homes/u/user/.conda/envs/myenv/bin/python \
	-m ipykernel_launcher "$@"
```

You can put anything you want to configure your environment in the helper script.
Just make sure it ends with the `ipykernel_launcher` command.

## Shifter Kernels on Jupyter

Shifter works with Cori notebook servers, but not Spin notebook servers.
To make use of it, create a kernel spec and edit it to run `shifter`.
The path to Python in your image should be used as the executable, and the
kernel spec should be placed at
`~/.local/share/jupyter/kernels/<my-shifter-kernel>/kernel.json` (you do not
need to create a Conda environment for this).


Here's an example of how to set up the kernel spec:

```json
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

```json
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

## Debugging Jupyter Problems

At NERSC, users launch Jupyter notebooks after authenticating to JupyterHub.
Logs from a user's notebook process appear in a file called `.jupyter.log` in the user's `$HOME` directory.
These logs can be very helpful when it comes to debugging issues with Jupyter, your custom kernels, or your Python environment.
One of the first things we do when investigating Jupyter tickets is consult this log file.

Need more information in the log file?
You can control how verbose the logging is by changing the value of `c.Application.log_level` in your Jupyter notebook config file.
You may not have a Jupyter notebook config file created yet.
You can create it by running

```shell
/usr/common/software/jupyter/19-09/bin/jupyter notebook --generate-config
```

Open the generated configuration file, uncomment `c.Application.log_level` and change the value to say, 0, for debug level information.
The logger used is Python's standard `logger` object.

!!! tip "Help Us Help You"
    You might save yourself a lot of time if you look at this log file yourself before opening a ticket.
    In fact, if you see anything that you think might be particularly important, you can highlight that in a ticket.

    And as always, be sure to be as specific as possible in tickets you file about Jupyter.
    For example, if you have an issue with a particular kernel or Conda environment, let us know which one it is.
