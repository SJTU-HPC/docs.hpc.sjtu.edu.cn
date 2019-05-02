# Getting Started and Optimization Strategy

There are several important differences between the Cori KNL ("Knight's
Landing", or Xeon Phi) node architecture and the Xeon architecture used
by Cori Haswell nodes or by Edison. This page will walk you through the
high-level steps to prepare an application to perform well on Cori KNL.


## How Cori KNL Differs From Edison or Cori Haswell

Cori KNL is a "many-core" architecture, meaning that instead of a few 
cores optimized for latency-sensitive code, Cori KNL nodes have many
(68) cores optimized for vectorized code. Some key differences are:

| Cori Intel Xeon Phi (KNL) | Cori Haswell (Xeon) | Edison (Xeon "Ivybridge") |
| --------------------------|---------------------|---------------------------|
| 68 physical cores on one socket | 16 physical cores on each of two sockets (32 total) | 12 physical cores on each of two sockets (24 total) |
| 272 virtual cores per node | 64 virtual cores per node | 48 virtual cores per node |
| 1.4 GHz                   | 2.3 GHz             | 2.4 GHz                   |
| 8 double precision operations per cycle | 4 double precision operations per cycle | 4 double precision operations per cycle |
| 96 GB of DDR memory and 16 GB of MCDRAM high-bandwidth memory | 128 GB of DDR memory | 64 GB of DDR memory |
| ~450 GB/sec memory bandwidth (MCDRAM) |         | ~100 GB/sec memory bandwidth |
| 512-bit wide vector units | 256-bit-wide vector units | 256-bit wide vector units |


## Important Aspects of an Application to Optimize for Cori KNL

There are three important areas of improvement to consider for Cori KNL:

1. Evaluating and improving your Vector Processing Unit (VPU) utilization and efficiency. As shown in the table above, the Cori processors have an 8 double-precision wide vector unit. Meaning, if your code produces scalar, rather than vector instructions, you miss on a potential 8x speedup. Vectorization is described in more detail in [Vectorization](../vectorization.md).
2. Identifying and adding more node-level [parallelism](../parallelism.md) and exposing it in your application. An MPI+X programming approach is encouraged where MPI represents a layer of internode communication and X represents a conscious intra-node parallelization layer where X could again stand for MPI or for OpenMP, pthreads, PGAS etc.
3. Evaluating and optimizing for your [memory bandwidth and latency](../mem_bw.md) requirements. Many codes run at NERSC are performance limited not by the CPU clock speed or vector width but by waiting for memory accesses. The memory hierarchy in Cori KNL nodes is different to that in Haswell nodes, so while memory bandwidth optimizations will benefit both, different optimizations will benefit each architecture differently.


