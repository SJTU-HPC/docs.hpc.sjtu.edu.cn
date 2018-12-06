# TensorFlow

## Description

TensorFlow is a deep learning framework developed by Google
in 2015. It is maintained and continuously updated by implementing
results of recent deep learning research. Therefore, TensorFlow
supports a large variety of state-of-the-art neural network layers,
activation functions, optimizers and tools for analyzing, profiling
and debugging deep neural networks. In order to deliver good
performance, the TensorFlow installation at NERSC utlizes the
optimized MKL-DNN library from Intel.  Explaining the full framework
is beyond the scope of this website. For users who want to get started
we recommend reading the
TensorFlow
[getting started page](https://www.tensorflow.org/get_started/). The
TensorFlow page also provides a
complete [API documentation](https://www.tensorflow.org/api_docs/).

## TensorFlow at NERSC

In order to use TensorFlow at NERSC load the TensorFlow module via

```bash
module load tensorflow/intel-<version>
```

where `<version>` should be replaced with the version string you are
trying to load. To see which ones are available use `module avail
tensorflow`.

Running TensorFlow on a single node is the same as on a local machine,
just invoke the script with

```bash
python my_tensorflow_program.py
```

## Distributed TensorFlow

By default, TensorFlow supports GRPC for distributed
training. However, this framework is tedious to use and very slow on
tightly couple HPC systems. Therefore, we recommend
using [Uber Horovod](https://github.com/uber/horovod) and thus also
pack it together with the TensorFlow module we provide. The version of
Horovod we provide is compiled against the optimized Cray MPI and thus
integrates well with SLURM. We will give a brief overview of how to
make an existing TensorFlow code multi-node ready but we recommend
inspecting the [examples on the Horovod page](https://github.com/uber/horovod/tree/master/examples).
We recommend using pure TensorFlow instead of Keras as it shows better
performance and the Horovod integration is more smooth.

In order to use TensorFlow, one needs to import the horovod module by
doing

```python
import horovod.tensorflow as hvd
```

One of the first statements should then be

```python
hvd.init()
```

which initializes the MPI runtime. Then, the user needs to wrap the
optimizers for distributed training using

```python
opt = hvd.DistributedOptimizer(opt)
```

To keep track of the global step a global step object has to be
created via `tf.train.get_or_create_global_step()` and passed to the
`minimize` (or `apply_gradients`) member functions of the optimizer
instance.

Furthermore, to ensure model consistency on all nodes it is mandatory
to register a broadcast hook via

```python
bcast_hook = [hvd.BroadcastGlobalVariablesHook(0)]
```

and pass it along with other hooks to the `MonitoredTrainingSession`
object. For example it is beneficial to register a stop hook via

```python
stop_hook = [tf.train.StopAtStepHook(last_step=num_steps_total)]
```

For example, a training code like

```python
import tensorflow as tf

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()

# Build model...
loss = ...
opt = tf.train.AdagradOptimizer(0.01 * hvd.size())

# Make training operation
train_op = opt.minimize(loss)

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.Session() as sess:
  for steps in range(num_steps_total):
    # Perform synchronous training.
    sess.run(train_op)
```

should read for distributed training

```python
import tensorflow as tf
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

# Build model...
loss = ...
opt = tf.train.AdagradOptimizer(0.01 * hvd.size())

# Add Horovod Distributed Optimizer
opt = hvd.DistributedOptimizer(opt)

# Add hook to broadcast variables from rank 0 to all other processes during
# initialization.
hooks = [hvd.BroadcastGlobalVariablesHook(0)]

# Add session stop hook
global_step = tf.train.get_or_create_global_step()
hooks.append(tf.train.StopAtStepHook(last_step=num_steps_total))

# Make training operation
train_op = opt.minimize(loss, global_step=global_step)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)
```

It is important to use `MonitoredTrainingSession` instead of the
regular `Session` because it keeps track of the number of global steps
and knows when to stop the training process when a correspondig hook
is installed. For more fine grained control over checkpointing,
a [`CheckpointSaverHook`](https://www.tensorflow.org/api_docs/python/tf/train/CheckpointSaverHook)
can be registered as well. Note that the graph has to be finalized before
the monitored training session context is entered. In case of the
regular session object, this is a limitation and can cause some
trouble with summary writers. Please see the
[distributed training recommendations](https://www.tensorflow.org/deploy/distributed)
for how to handle these cases.

## Splitting Data

It is important to note that splitting the data among the nodes is up
to the user and needs to be done besides the modifications stated
above. Here, utility functions can be used to determine the number of
independent ranks via `hvd.size()` and the local rank id via
`hvd.rank()`. If multiple ranks are employed per node,
`hvd.local_rank()` and `hvd.local_size()` return the node-local
rank-id's and number of ranks. If
the
[dataset API](https://www.tensorflow.org/programmers_guide/datasets)
is being used we recommend using the `dataset.shard` option to split
the dataset. In other cases, the data sharding needs to be done
manually and is application dependent.

## Frequently Asked Questions

### I/O Performance and Data Feeding Pipeline

For performance reasons, we recommend storing the data on the scratch
directory, accessible via the `SCRATCH` environment variable. At high
concurrency, i.e. when many nodes need to read the files we
recommend [staging them into burst buffer](). For efficient data
feeding we recommend using the `TFRecord` data format and using
the
[`dataset` API](https://www.tensorflow.org/programmers_guide/datasets)
to feed data to the CPU. Especially, please note that the
`TFRecordDataset` constructor takes `buffer_size` and
`num_parallel_reads` options which allow for prefetching and
multi-threaded reads. Those should be tuned for good performance, but
please note that a thread is dispatched for every independent
read. Therefore, the number of inter-threads needs to be adjusted
accordingly (see below). The `buffer_size` parameter is meant to be in
bytes and should be an integer multiple of the node-local batch size
for optimal performance.

### Potential Issues

For best MKL-DNN performance, the module already sets a set of OpenMP
environment variables and we encourage the user not changing those,
especially not changing the `OMP_NUM_THREADS` variable. Setting this
variable incorrectly can cause a resource starvation error which
manifests in TensorFlow telling the user that too many threads are
spawned. If that happens, we encourage to adjust the inter- and
intra-task parallelism by changing the `NUM_INTER_THREADS` and
`NUM_INTRA_THREADS` environment variables. Those parameters can also
be changed in the TensorFlow python script as well by creating a
session configs object via

```python
sess_config=tf.ConfigProto(inter_op_parallelism_threads=num_inter_threads,
                               intra_op_parallelism_threads=num_intra_threads)
```

and pass that to the session manager

```python
with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
  ...
```

Please note that
`num_inter_threads*num_intra_threads<=num_total_threads` where
`num_total_threads` is 64 on Haswell or 272 on KNL.
