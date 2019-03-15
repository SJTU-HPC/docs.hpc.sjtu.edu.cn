# Machine Learning at NERSC

NERSC supports a variety of software of Machine Learning and Deep Learning
on our systems.

These docs pages are still in progress, but will include details about how
to use our system optimized frameworks, multi-node training libraries, and
performance guidelines.

### Frameworks

We have prioritized support for the following Deep Learning frameworks on Cori:

* [TensorFlow](tensorflow/index.md)
* [PyTorch](pytorch.md)

### Deploying with Jupyter

Users can deploy distributed deep learning workloads to Cori from Jupyter
notebooks using parallel execution libraries such as IPyParallel. Jupyter
notebooks can be used to submit workloads to the batch system and also
provide powerful interactive capabilities for monitoring and controlling those
workloads.

We have some examples for running multi-node training and distributed
hyper-parameter optimization jobs from notebooks in this github repository:
https://github.com/sparticlesteve/cori-intml-examples

### Benchmarks

We track general performance of Deep Learning frameworks as well as some
specific scientific applications. See the [benchmarks](benchmarks.md) for details.

### Science use-cases

Machine Learning and Deep Learning are increasingly used to analyze scientific data, in diverse fields. We have gathered some examples of work ongoing at NERSC on the [science use-cases page](science-use-cases.md) including some code and datasets and how to run these at NERSC.
