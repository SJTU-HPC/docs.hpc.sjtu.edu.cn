#Other Workflow Tools

#QDO

(Note that qdo is not actively supported or maintained by NERSC, but is
available on our systems.)

QDO (kew-doo) is a toolkit for managing many many small tasks within a larger
batch framework. QDO separates the queue of tasks to perform from the batch
jobs that actually perform the tasks. This simplifies managing tasks as a
group, and provides greater flexibility for scaling batch worker jobs up and
down or adding additional tasks to the queue even after workers have started
processing them.

QDO was designed by the astrophysics community to manage queues of high
throughput jobs.  It supports task dependencies, priorities, management of
tasks in aggregate, and flexibility such as adding additional tasks to a queue
even after the batch worker jobs have started.

The qdo module provides an API for interacting with task queues. The qdo script
uses this same API to provide a command line interface that can be used
interchangeably with the python API. Run "qdo -h" to see the task line options.

```
module use /project/projectdirs/cosmo/software/modules/$NERSC_HOST
module load qdo/0.7
```

qdo was developed to support workflows in processing images for cosmological
surveys. For more information please contact Stephen Bailey in the Cosmology
Group at Lawrence Berkeley Lab.

#Tigres

Tigres provides a C and Python programming library to compose and execute
large-scale data-intensive scientific workflows from desktops to
supercomputers. We offer a community supported module for the Tigres libraries.
For more information, refer to the Tigres
[documentation](http://tigres.lbl.gov/).

#Spark

Apache [Spark](../../analytics/spark.md)
 is a fast and general engine for large-scale data processing.



