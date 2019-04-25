# Spark Distributed Analytic Framework
## Description and Overview

Apache Spark is a fast and general engine for large-scale data processing.

### How to Use Spark

Because of its high memory and I/O bandwidth requirements, we
recommend you run your spark jobs on Cori. We recommend that you run
Spark inside of Shifter. This will improve performance and usability
by utilizing Shifter's per-node file cache for shuffle files and
temporary data files. Without this functionality, these files are
written either to your scratch directory (which is not optimized for
repeated accesses of many small files) or the RAM file system at /tmp
(which removes memory from the node doing the calculation and can lead
to the node crashing from lack of memory).

Follow the steps below to use spark, note that the order of the
commands matters. DO NOT load the spark module until you are inside a
batch job.

#### Interactive mode

Submit an interactive batch job with at least 2 nodes. Our setup for Spark puts the master on one node and the slaves on the other nodes.

``` 
salloc -N 2 -t 30 -C haswell -q interactive --image=nersc/spark-2.3.0:v1 --volume="/global/cscratch1/sd/<user_name>/tmpfiles:/tmp:perNodeCache=size=200G"
```

This will request a job with the Spark 2.3.0 Shifter image (if you
wish to run with an earlier version of spark, replace the version
numbers for the above and following commands). It also sets up an xfs
file on each node as a per node cache, which will be accessed inside
the Shifter image via /tmp. By default, Spark will use this as the
directory it caches temporary files. By default Spark will put event
logs in $SCRATCH/spark/spark_event_logs, you will need to create this
directory the first time you start up Spark.

Wait for the job to start. Once it does you will be on a compute node and you will need to load the Spark module:
```
export EXEC_CLASSPATH=path_to_any_extra_needed_jars #Only required if you're using external libraries or jarfiles
module load spark/2.3.0
```

You can start Spark with this command:
```
start-all.sh
```

To connect to the Python Spark Shell, do:
```
shifter pyspark
```
To connect to the Scala Spark Shell, do:
```
shifter spark-shell
```
To shutdown the Spark cluster, do:
```
stop-all.sh
```

#### Batch mode
Below are example batch scripts for Cori. You can change number of
nodes/time/queue accordingly (so long as the number of nodes is
greater than 1). On Cori you can use the debug queue for short,
debugging jobs and the regular queue for long jobs.  

Here's an example script for Cori called run.sl:

```
#!/bin/bash

#SBATCH -q regular
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH --image=nersc/spark-2.3.0:v1
#SBATCH --volume="/global/cscratch1/sd/<user_name>/tmpfiles:/tmp:perNodeCache=size=200G"

export EXEC_CLASSPATH=path_to_any_extra_needed_jars #Only required if you're using external libraries or jarfiles
module load spark/2.3.0

start-all.sh

shifter spark-submit $SPARK_EXAMPLES/python/pi.py

stop-all.sh
```

To submit the job:
```
sbatch run.sl
```

#### Example PySpark Script

Here's an example pyspark script that reads in a comma separated file,
does some aggregations, and writes out some filtered results:

```
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, udf, sum, count
from pyspark.sql.types import *

infile = "path_to_input_file"
outdir = "your_output_directory"

#Get a hook to the spark session
spark = SparkSession.builder.appName("DataMap").getOrCreate()

#Define data schema (define column names and their types, and whether a null field is allowed)
dataschema = StructType([ StructField("uid", LongType(),True), StructField("filepath", StringType(),True)])

#Read in data with given schema
dfdir = spark.read.csv(infile,sep=" ",schema=dataschema)

#what did we get?
dfdir.show()

#Count how many records
dfdir.count()

#Filter out a path names that contain "www"
#cache the dataset in memory because you'll be working with it again
dfdir = dfdir[(dfdir.filepath.rlike("www") == False)].cache()
dfdir.count()
```

You can run this script in an interactive or batch session (see above
for how to get one of those) with
```
shifter spark-submit --master $SPARKURL <path_to_python_script>
```

### Guide for Optimizing Your Spark Code
#### Scala or Python

When writing code for Spark, historically scala has out performed
python (via pySpark).However, as of Spark 2.0, the performance of
python code using dataframes has become roughly equivalent for most
operations. So we recommend that you write your code in python and
dataframes. You will get roughly the same performance and also benefit
from the fact that you can use a familiar language and a pandas-like
dataframe interface.

#### Memory Management and Input Data

Choose enough nodes that your input data can comfortably fit in the
aggregate memory of all the nodes. If you will be working with the
same dataframe over and over again, you can "cache" the dataframe to
make sure it stays in memory. This will cut down on recalculating
time. Just be sure to unpersist it when you're done using
it. Whenever possible, store your input data in your $SCRATCH
directory or in the Burst Buffer.