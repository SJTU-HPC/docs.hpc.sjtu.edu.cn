# Programming Models

A wide variety of programming models are used on NERSC systems. The
most common is MPI + OpenMP, but many others are supported.

## Parallel programming models at NERSC
Since the transition from vector to distributed memory (MPP) supercomputer architectures, the majority of HPC applications
deployed on NERSC resources have evolved to use MPI as their sole means of expressing parallelism. As single processor core
compute nodes on MPP architectures gave way to multicore processors, applying the same abstraction (processes passing
messages) to each available core remained an attractive alternative - no code changes were required, and vendors made an
effort to design optimized fast-paths for on-node communication.

However, as on-node parallelism rapidly increases and competition for shared resources per processing element (memory per
core, bandwidth per core, etc.) does as well, now is a good time to assess whether applications can benefit from a
different abstraction for expressing on-node parallelism. Examples of desirable functionality potentially available
through the latter include more efficient utilization of resources (e.g. through threading) or the ability to exploit
unique architectural features (e.g. vectorization).

## Cori, Perlmutter, and beyond: Performance and portability
Cori Phase II system, arrived in mid-2016, continues this trend toward greater intra-node parallelism. The Knights Landing
processor supports 68 cores per node, each supporting four hardware threads and possessing two 512-bit wide vector
processing units.

Perlmutter will arrive in 2020 and will have a mixture of CPU-only nodes and CPU + GPU nodes. Each CPU + GPU nodes will
have 4 GPUs per CPU node.

NERSC has made an effort to provide guidance on parallel programming approaches for Cori. Chief among these is the combination
of MPI for inter-node parallelism and OpenMP for intra-node parallelism (or potentially MPI per NUMA domain with OpenMP
within each). Many of the intra-node parallelism efforts made for Cori will propagate to Perlmutter.

## Why MPI + OpenMP?
The reasons we've chosen to suggest this approach to our users are many. Most importantly, it provides:

A model that allows application developers to think differently about inter- vs. intra-node parallelism
(which will be key to obtaining good performance);
A "gradual" onramp (in terms of application refactoring / modification) for existing pure-MPI applications to isolate
and express intra-node parallelism;
A convenient (compiler-directive agnostic) way of expressing SIMD parallelism at the loop or function level; and
A model that could potentially offer portability across a range of new supercomputing architectures characterized by
increased intra-node parallelism (especially with the OpenMP device directives introduced in 4.0, and subsequent
improvements in 4.5).
We must stress, however, that MPI + OpenMP might not be the right choice for all applications. For example, applications
that already make use of a different threading abstraction (e.g. Pthreads, C++11, TBB, etc.), or use a PGAS programming
model for one-sided communication, or use a thread-aware task-based runtime system, may already have chosen for model that
maps well to intra-node parallelism as well as inter-node.

Indeed, the key point we would like to make is that inter and intra-node parallelism must be understood and treated
differently in order to obtain good performance on many-core architectures like Cori.

## Combined Programming Models

Although not meant to be an exhaustive list, here we briefly examine a number of options that we see as potentially
attractive to our users. In general, a recommended mixture of parallelism models would be to limit the number of
processes (created via distributed memory parallelism, e.g. MPI) to a few instances per node where each process
implements a shared-memory parallelism model (e.g. threads) which creates a minimum of `nthreads = (ncores / nprocs)`.
Then, optionally when targetting GPUs, within the shared-parallelism model, there are additional threads of execution
that are reserved for asychronous communication with the GPU. While this certainly will not be a globally applicable
model to all codes using NERSC resources, the reasons for this recommendation are as follows:

- Limiting the number of processes within a node
  - Processes have separate memory address spaces
    - More separate memory spaces increases the total amount of memory used by the parallelism scheme
    - Communication between processes potentially requires using the network and an underlying copy and transfer of
    memory, whereas threads within the same memory address space can immediately "see" the updates
  - Separate memory address spaces can be emulated within threads using thread-local memory but other threads can
  still access that memory if given the address of that memory allocation.
    - E.g., in object-oriented codes, one can allocate entire objects in thread-local memory and have the master thread
    hold a list of the objects held by the worker threads.
- Shared-memory model creating `nthreads = (ncores / nprocs)`
  - If a node contains 48 cores and 4 processes are spawned on that node, each process creating 12 threads for
  execution will enable the workload to saturate all 48 cores.
- Additional threads of execution for asychronous communication with the GPU
  - Oversubscribing the number of threads (i.e. `nthreads = (4 * ncores) / nprocs`), the communication latency with
  the GPU can be hidden and the throughput can then be effectively increased.
    - When work is offloaded to the GPU, there are CPU cycles that are not being utilized while a thread waits on the
    communication with the GPU.
    - This can be achieved via one large thread-pool implementing `nthreads = (4 * ncores) / nprocs` or
    `nthreads = (ncores / nprocs)` each implementing an addition 4 threads.
  - Enabling asynchronous communication with GPU depends on programming model but for CUDA, it is achieved through
  CUDA streams.
    - In general, a single thread with multiple streams is less efficient that multiple CPU threads with
    one stream. In the former, the launching of kernels into different streams is a serial bottleneck where as in the
    latter, the coordination of launching of kernels is effectively parallelized.


### Distributed memory (inter-node) parallelism

These programming models create separate parallel processes with independent memory address spaces.

- [MPI](mpi/index.md)
- [PGAS models](https://en.wikipedia.org/wiki/Partitioned_global_address_space)
  - [UPC](https://upc.lbl.gov/)
  - [UPC++](upcxx.md)
  - [Coarrays](coarrays.md)
  - [Others](https://en.wikipedia.org/wiki/Partitioned_global_address_space)

### Shared memory (intra-node) parallelism

#### CPU

These programming models create separate parallel "threads" of execution on the CPU within a process and share the memory address space.

- [OpenMP](openmp/openmp.md)
- Raw threads ([pthreads](https://en.wikipedia.org/wiki/POSIX_Threads), [STL threads](https://en.cppreference.com/w/cpp/thread/thread))
- [TBB](https://www.threadingbuildingblocks.org/)
- [Kokkos](kokkos.md)
- [Raja](raja.md)

#### GPU

These programming models support parallelism on general-purpose GPUs.

- [OpenMP](openmp/openmp.md)
- [OpenACC](https://www.openacc.org/)
- [CUDA (NVIDIA GPUs)](https://developer.nvidia.com/about-cuda)
- [HIP (AMD and NVIDIA GPUs)](https://gpuopen.com/compute-product/hip-convert-cuda-to-portable-c-code/)
- [Kokkos](kokkos.md)
- [Raja](raja.md)


<!-- Old notes from old page that may potentially be integrated into the above

- [MPI](mpi/index.md)
  - While pure MPI using the classic two-sided (non-RMA components of MPI-2) messaging model indeed works on NERSC Cori,
  it fails to address many of the concerns we raised above. We do not expect that this approach will perform well for
  most applications.
- [MPI + MPI](mpi/index.md)
  - With the MPI-3 standard, shared memory programming on-node is possible via MPI's remote memory access (RMA) API,
  yielding an MPI + MPI model (RMA can also be used off-node). The upside of this approach is that one requires only one
  library for parallelism. For a program with shared memory parallelism at a very high level where most data is private
  by default, this is a powerful model.
- [MPI](mpi/index.md) + X (CPU)
  - While MPI + OpenMP was covered in some details above, we recognize that other options are possible under an MPI + X
  approach. For example, one could use a different threading model to express on-node parallelism, such as native C++
  concurrency primitives (available since C++11 and likely to improve considerably in 14 and 17) or Intel's TBB, as well
  as data container / execution abstractions like Kokkos.
  - [MPI](mpi/index.md) + [OpenMP](openmp/openmp.md)
    - As noted above, OpenMP provides a way to express on-node parallelism (including SIMD) with ease at a relatively fine
    level. In recent years, overhead due to thread team spin-up, fork and join operations and thread synchronization has
    been reduced drastically in common OpenMP runtimes.
  - [MPI](mpi/index.md) + [Kokkos](kokkos.md)
  - [MPI](mpi/index.md) + [Raja](raja.md)
  - [MPI](mpi/index.md) + [TBB](https://www.threadingbuildingblocks.org/)
- [MPI](mpi/index.md) + Z (GPU)
  - MPI can be combined with programming model "Z" that enables execution on the GPU
  - MPI + OpenMP
  - MPI + OpenACC
  - MPI + CUDA
  - MPI + HIP
  - MPI + [Kokkos](kokkos.md)
- [MPI](mpi/index.md) + X (CPU) + Z (GPU)
  - MPI can be combined with programming model "X" that enables threading and programming model "Y" that enables execution
  on the GPU
  - MPI + OpenMP (threading) + CUDA
  - MPI + pthreads + CUDA
  - MPI + STL threads + CUDA

## Efforts toward next-generation programming models
There are a number of ongoing efforts in the HPC research community to develop new programming systems geared toward
future exascale architectures. The DOE X-Stack program in particular is one such centralized effort that includes
projects integrating many of the key programming abstractions noted above, such as DAG-based execution and global
address space communication models.
-->
