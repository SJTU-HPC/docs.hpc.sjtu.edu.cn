# Parallelism

## On-node parallelism

On Cori, it will be necessary to specifically think both about inter-node parallelism as well as on-node parallelism. The baseline programming model for Cori is MPI+X where X represents some conscious level of on-node parallelism, which could also be expressed as MPI or a shared memory programming model like OpenMP, pthreads etc. For a lot of codes, running without changes on 72 (or up to 288) MPI tasks per node could be troublesome. Examples are codes that are MPI latency sensitive like 3D FFTs  and codes that duplicate data structures on MPI ranks (often MPI codes don't perfectly distribute every data structure across ranks) which could more quickly exhaust the HBM if running in pure MPI mode.

### Threads

### Vectorization

Modern CPUs have Vector Processing Units (VPUs) that allow the processor to do the same instruction on multiple data (SIMD) per cycle.

On KNL, the VPU will be capable of computing the operation on 8 rows of the vector concurrently. This is equivalent to computing 8 iterations of the loop at a time.

The compilers on Cori want to give you this 8x speedup whenever possible. However some things commonly found in codes stump the compiler and prevent it from vectorizing. The following figure shows examples of code the compiler won't generally choose to vectorize

## Off-node parallelism

### Strong scaling

How the time to solution varies with number of processing elements for a fixed problem size. 

### Weak scaling

How the time to solution varies with number of processing elements for a fixed problem size _per processor_.
