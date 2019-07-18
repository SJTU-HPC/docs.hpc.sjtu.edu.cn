# UPC

Unified Parallel C (UPC) is an extension of the C programming language
designed for high performance computing on large-scale parallel machines. The
language provides a uniform programming model for both shared and distributed
memory hardware. The programmer is presented with a single shared, partitioned
address space, where variables may be directly read and written by any
processor, but each variable is physically associated with a single processor.
UPC uses a Single Program Multiple Data (SPMD) model of computation in which
the amount of parallelism is fixed at program startup time, typically with a
single thread of execution per processor. The [Berkeley
UPC](https://upc.lbl.gov/) project is one implementation.

Here is a good [training video tutorial on UPC](https://www.youtube.com/watch?v=Ey-inJ9Dz6Q).

## UPC at NERSC

UPC is supported on NERSC systems through two different implementations:
Berkeley UPC and Cray UPC.

### Berkeley UPC

[Berkeley UPC (BUPC)](https://upc.lbl.gov)
provides a portable UPC programming environment consisting
of a source translation front-end (which in turn relies on a user-supplied C
compiler underneath) and a runtime library based on
[GASNet](https://gasnet.lbl.gov/). The latter is able to take advantage of
advanced communications functionality of the Cray Aries interconnect on Cori,
such as remote direct memory access (RDMA).

BUPC is available via the `bupc` module on Cori, which provides both the `upcc`
compiler wrapper, as well as the `upcrun` launcher wrapper (which correctly
initializes the environment and calls `srun`). Further, all three supported
programming environments on Cori (Intel, GNU, and Cray) are supported by BUPC
for use as the underlying C compiler.

There are a number of flags and environment variables that affect the execution
environment of your UPC application compiled with BUPC, all of which can be
found in the [BUPC documentation](https://upc.lbl.gov/docs/). 
Both `upcc` and `upcrun` have `-help` options and man pages describing these.
One of the most important settings is the size of the shared symmetric heap
used to service shared memory allocations. This size can be controlled via the
`UPC_SHARED_HEAP_SIZE` envvar, or the `-shared-heap` flag to `upcc` or `upcrun`.
If you encounter errors related to shared memory allocation, you will likely
want to start by adjusting this quantity. 

Compiling and running a simple application with BUPC on Cori is fairly
straightforward. First, consider the following UPC source file:

```C
// Compute pi by approximating the area of a circle of radius 1. 
// Algorithm: generate random points in [0,1]x[0,1] and measure the fraction 
// of them falling in a circle centered at the origin (approximates pi/4)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <upc.h>

int hit() { // return non-zero for a hit in the circle
  double x = rand()/(double)RAND_MAX;
  double y = rand()/(double)RAND_MAX;
  return (x*x + y*y) <= 1.0;
}

// shared array for the results computed by each thread
shared int64_t all_hits[THREADS];

int main(int argc, char **argv) {
    int64_t trials = 100000000;
    if (argc > 1) trials = (int64_t)atoll(argv[1]);
    int64_t my_trials = (trials + THREADS - 1 - MYTHREAD)/THREADS;

    srand(MYTHREAD); // seed each thread's PRNG differently

    int64_t my_hits = 0;
    for (int64_t i=0; i < my_trials; i++)
        my_hits += hit(); // compute in parallel

    all_hits[MYTHREAD] = my_hits; // publish results
    upc_barrier;

    if (MYTHREAD == 0) { // fetch results from each thread
        // (could alternatively call upc_all_reduce())
        int64_t total_hits = 0;
        for (int i=0; i < THREADS; i++)
            total_hits += all_hits[i];
        double pi = 4.0*total_hits/(double)trials;
        printf("PI estimated to %10.7f from %lld trials on %d threads.\n",
               pi, (long long)trials, THREADS);
    }

    return 0;
}
```

To compile this file with BUPC:

```console
cori$ module load bupc
cori$ upcc mcpi.upc -o mcpi.x
```

And then run, in this case in a interactive `salloc` session:

```slurm
cori$ salloc -N 2 -t 10:00 --qos=interactive -C haswell
[...]
cori$ upcrun -n 4 ./mcpi.x
UPCR: UPC thread 2 of 4 on nid00707 (pshm node 1 of 2, process 2 of 4, pid=33268)
UPCR: UPC thread 0 of 4 on nid00705 (pshm node 0 of 2, process 0 of 4, pid=12390)
UPCR: UPC thread 3 of 4 on nid00707 (pshm node 1 of 2, process 3 of 4, pid=33269)
UPCR: UPC thread 1 of 4 on nid00705 (pshm node 0 of 2, process 1 of 4, pid=12391)
PI estimated to  3.1415196 from 100000000 trials on 4 threads.
```

### Cray UPC

UPC is directly supported under Cray's compiler environment through their PGAS
runtime library (providing similar performance-enabling RDMA functionality to
GASNet). To enable UPC support in your C code, simply switch to the Cray
compiler environment and supply the `-h upc` option when calling `cc`.

Because of its dependence on Cray's PGAS runtime, you may find the additional
documentation available on the `intro_pgas` man page valuable. Specifically,
two key environment variables introduced there are:

  - `XT_SYMMETRIC_HEAP_SIZE`: Limits the size of the symmetric heap used to
    service shared memory allocations, analogous to BUPC's `UPC_SHARED_HEAP_SIZE`
  - `PGAS_MEMINFO_DISPLAY`: Can be set to `1` in order to enable diagnostic
    output at launch regarding memory utilization.

In addition, there is one additional potential issue to be aware of: virtual
memory limits in interactive `salloc` sessions. If you encounter errors on
application launch similar to:

```console
PE 0: ERROR: failed to attach XPMEM segment (at or around line 23 in __pgas_runtime_error_checking() from file ...)
```

then you may need to release your virtual memory limits by running:

```console
cori$ ulimit -v unlimited
```

With all of this in mind, compiling and running a simple UPC application,
analogous to the above example for BUPC but now using the Cray compilers, would
look like:

```slurm
cori$ module swap PrgEnv-intel PrgEnv-cray
cori$ cc -h upc mcpi.upc -o mcpi.x
cori$ salloc -N 2 -t 10:00 --qos=interactive -C haswell
[...]
cori$ ulimit -v unlimited  # may not be necessary
cori$ srun -n 4 ./mcpi.x
PI estimated to  3.1414546 from 100000000 trials on 4 threads.
```
