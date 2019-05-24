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

## UPC at NERSC

UPC is supported on NERSC systems through two different implementations:
Berkeley UPC and Cray UPC.

### Berkeley UPC

Berkeley UPC (BUPC) provides a portable UPC programming environment consisting
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

There are a number of environment variables that affect the execution
environment of your UPC application compiled with BUPC, all of which can be
found in the BUPC documentation. One of the most important is
`UPC_SHARED_HEAP_SIZE`, which controls the size of the shared symmetric heap
used to service shared memory allocations. If you encounter errors at
application launch related to memory allocation, you will likely want to start
by adjusting this variable.

Compiling and running a simple application with BUPC on Cori is fairly
straightforward. First, consider the following UPC source file:

```C
/* The ubiquitous cpi program.
 Compute pi using a simple quadrature rule
 in parallel
 Usage: cpi [intervals_per_thread] */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <upc_relaxed.h>

#define INTERVALS_PER_THREAD_DEFAULT 100
/* Add up all the inputs on all the threads.
 When the collective spec becomes finalised this
 will be replaced */

shared double reduce_data[THREADS];
shared double reduce_result;
double myreduce(double myinput)
{
#if defined(__xlC__)
 // Work-around Bug 3228
 *(volatile double *)(&myinput);
#endif
 reduce_data[MYTHREAD]=myinput;
 upc_barrier;
 if(MYTHREAD == 0) {
 double result = 0;
 int i;
 for(i=0;i < THREADS;i++) {
 result += reduce_data[i];
 }
 reduce_result = result;
 }
 upc_barrier;
 return(reduce_result);
}

/* The function to be integrated */
double f(double x)
{
 double dfour=4;
 double done=1;
 return(dfour/(done + (x*x)));
}

/* Implementation of a simple quadrature rule */
double integrate(double left,double right,int intervals)
{
 int i;
 double sum = 0;
 double h = (right-left)/intervals;
 double hh = h/2;
 /* Use the midpoint rule */
 double midpt = left + hh;
 for(i=0;i < intervals;i++) {
 sum += f(midpt + i*h);
 }
 return(h*sum);
}

int main(int argc,char **argv)
{
 double mystart, myend;
 double myresult;
 double piapprox;
 int intervals_per_thread = INTERVALS_PER_THREAD_DEFAULT;
 double realpi=3.141592653589793238462643;
 /* Get the part of the range that I'm responsible for */
 mystart = (1.0*MYTHREAD)/THREADS;
 myend = (1.0*(MYTHREAD+1))/THREADS;
 if(argc > 1) {
 intervals_per_thread = atoi(argv[1]);
 }
 piapprox = myreduce(integrate(mystart,myend,intervals_per_thread));
 if(MYTHREAD == 0) {
 printf("Approx: %20.17f Error: %23.17f\n",piapprox,fabs(piapprox - realpi));
 }
 return(0);
}
```

To compile this file with BUPC:

```console
user@cori02:~$ module load bupc
user@cori02:~$ upcc cpi.c -o cpi.x
```

And then run, in this case in a interactive `salloc` session:

```slurm
user@cori02:~$ salloc -N 2 -t 10:00 -p debug -C haswell
[...]
user@cori02:~$ upcrun -n 4 ./cpi.x
UPCR: UPC thread 0 of 4 on nid01901 (pshm node 0 of 2, process 0 of 4, pid=36911)
UPCR: UPC thread 1 of 4 on nid01901 (pshm node 0 of 2, process 1 of 4, pid=36912)
UPCR: UPC thread 2 of 4 on nid01902 (pshm node 1 of 2, process 2 of 4, pid=35611)
UPCR: UPC thread 3 of 4 on nid01902 (pshm node 1 of 2, process 3 of 4, pid=35612)
Approx: 3.14159317442312691 Error: 0.00000052083333379
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
user@cori02:~$ ulimit -v unlimited
```

With all of this in mind, compiling and running a simple UPC application,
analogous to the above example for BUPC but now using the Cray compilers, would
look like:

```slurm
user@cori02:~$ module swap PrgEnv-intel PrgEnv-cray
user@cori02:~$ cc -h upc cpi.c -o cpi.x
user@cori02:~$ salloc -N 2 -t 10:00 -p debug -C haswell
[...]
user@cori02:~$ ulimit -v unlimited  # may not be necessary
user@cori02:~$ srun -n 4 ./cpi.x
Approx:  3.14159317442312691 Error:     0.00000052083333379
```
