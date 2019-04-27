# LIKWID

## Table of Contents

1. [How to load likwid on Cori](#How to load likwid on Cori)
2. [likwid-topology](#likwid-topology)
3. [likwid-pin](#likwid-pin)
4. [likwid-perfctr](#likwid-perfctr)
5. [likwid-mpirun](#likwid-mpirun)
6. [likwid-bench](#likwid-bench)

As modern architectures become more and more complex with deeper memory hierarchy and more levels of parallelism, it is important to understand the micro-architecture in order to take full advantage of the modern HPC systems such as Cori. There are a number of tools that are provided at NERSC that can be used by users for code profiling and code optimization. A few examples are [CrayPat](craypat.md), [Intel VTune](vtune.md) and [Allinea MAP](map.md). Compared to these tools, LIKWID from Erlangen Regional Computing Center is more lightweight. It doesn’t have a graphics user interface, and only provides a set of command line utilities. It doesn’t require code instrumentation or additional kernel modules to be installed or loaded at runtime. By simply reading the MSR (Model Specific Register) device files, it renders reports for various performance metrics, for example, FLOPS, bandwidth, load to store ratio, and energy.

[LIKWID](https://github.com/RRZE-HPC/likwid) is short for "Like I Knew What I’m Doing" and it addresses four problems that are frequently encountered by programmers during code migration and code optimization. First, it reports the thread/cache topology of a node so users understand what architecture exactly they are dealing with. Secondly, LIKWID can pin processes/threads to cores for a program in order to achieve thread affinity. Thirdly, LIKWID uses the Linux `msr` module, reads the MSR files from user space and reports the hardware performance counters for a number of performance metrics. Lastly, it provides a set of micro-benchmark kernels for users to quickly test some characteristics of an architecture.

The LIKWID suite contains 11 commands and on this page, we focus on 5 of them. For information on the other tools or more information about LIKWID itself, please refer to LIKWID’s own [wiki page](https://github.com/RRZE-HPC/likwid/wiki).

* **[likwid-topology](https://github.com/RRZE-HPC/likwid/wiki/likwid-topology)** prints the thread/core/cache/NUMA topology on a node for users to understand better the architecture they are running on.
* **[likwid-pin](https://github.com/RRZE-HPC/likwid/wiki/Likwid-Pin)** helps pin threads to cores so applications don’t migrate over the course of the job execution and loose cache locality; it supports POSIX threads and all threading models built on pthreads, such as Intel and GCC OpenMP.
* **[likwid-perfctr](https://github.com/RRZE-HPC/likwid/wiki/likwid-perfctr)** reports on hardware performance events, such as FLOPS, bandwidth, TLB misses and power; its Marker API provides focused examination of interested code regions; `likwid-perfctr` integrates the pinning functionality of `likwid-pin` and option `-C` can be used to specify the preferred affinity.
* **[likwid-mpirun](https://github.com/RRZE-HPC/likwid/wiki/Likwid-Mpirun)** is the performance profiler for MPI and hybrid applications as `likwid-perfctr` is only for purely threaded applications; `likwid-mpirun` calls `likwid-pin` and `likwid-perfctr` at the backend for pinning and performance counter reading, respectively.
* **[likwid-bench](https://github.com/RRZE-HPC/likwid/wiki/Likwid-Bench)** provides a set of micro-benchmark kernels, such as stream, triad and daxpy, allowing users to quickly benchmark a system on common characteristics such as bandwidth, FLOPS and vectorization efficiency. 

## How to load likwid on Cori

```
nersc$ module load likwid
nersc$ likwid-topology -h
likwid-topology -- Version 4.3.0 (commit: 233ab943543480cd46058b34616c174198ba0459)A tool to print the thread and cache topology on x86 CPUs.

Options:
-h, --help		 Help message
-v, --version		 Version information
-V, --verbose 	         Set verbosity
-c, --caches		 List cache information
-C, --clock		 Measure processor clock
-O			 CSV output
-o, --output 	         Store output to file. (Optional: Apply text filter)
-g			 Graphical output
```
To run LIKWID in a job, users need to specify the `--perf=likwid` flag to salloc or sbatch, for interactive jobs and batch jobs respectively. An example for the interactive job is:

```
nersc$ salloc -q interactive -C knl,quad,cache -N 1 -t 00:30:00 –perf=likwid
nersc$ module load likwid
```

The `--perf=likwid` flag ensures the Linux module `msr` loaded on the compute node when the job runs. 

The following sections will give a few examples on how to run the above mentioned 5 tools from the suite. **All examples are based on the Cori Haswell architecture. If other architectures are being used, please adjust the command line parameters accordingly.**

## likwid-topology

`likwid-topology` has several options, and the `-h` option lists all of them.

```
nersc$ likwid-topology -h
```

Without any option specified, `likwid-topology` prints out the basic thread/cache/NUMA information of a node.

```
nersc$ likwid-topology 
--------------------------------------------------------------------------------
CPU name:	Intel(R) Xeon(R) CPU E5-2698 v3 @ 2.30GHz
CPU type:	Intel Xeon Haswell EN/EP/EX processor
CPU stepping:	2
********************************************************************************
Hardware Thread Topology
********************************************************************************
Sockets:		2
Cores per socket:	16
Threads per core:	2
--------------------------------------------------------------------------------
HWThread	Thread		Core		Socket		Available
0		0		0		0		*
1		0		1		0		*
2		0		2		0		*
3		0		3		0		*

*snip*

60		1		28		1		*
61		1		29		1		*
62		1		30		1		*
63		1		31		1		*
--------------------------------------------------------------------------------
Socket 0:		( 0 32 1 33 2 34 3 35 4 36 5 37 6 38 7 39 8 40 9 41 10 42 11 43 12 44 13 45 14 46 15 47 )
Socket 1:		( 16 48 17 49 18 50 19 51 20 52 21 53 22 54 23 55 24 56 25 57 26 58 27 59 28 60 29 61 30 62 31 63 )
--------------------------------------------------------------------------------
********************************************************************************
Cache Topology
********************************************************************************
Level:			1
Size:			32 kB
Cache groups:		( 0 32 ) ( 1 33 ) ( 2 34 ) ( 3 35 ) ( 4 36 ) ( 5 37 ) ( 6 38 ) ( 7 39 ) ( 8 40 ) ( 9 41 ) ( 10 42 ) ( 11 43 ) ( 12 44 ) ( 13 45 ) ( 14 46 ) ( 15 47 ) ( 16 48 ) ( 17 49 ) ( 18 50 ) ( 19 51 ) ( 20 52 ) ( 21 53 ) ( 22 54 ) ( 23 55 ) ( 24 56 ) ( 25 57 ) ( 26 58 ) ( 27 59 ) ( 28 60 ) ( 29 61 ) ( 30 62 ) ( 31 63 )
--------------------------------------------------------------------------------
Level:			2
Size:			256 kB
Cache groups:		( 0 32 ) ( 1 33 ) ( 2 34 ) ( 3 35 ) ( 4 36 ) ( 5 37 ) ( 6 38 ) ( 7 39 ) ( 8 40 ) ( 9 41 ) ( 10 42 ) ( 11 43 ) ( 12 44 ) ( 13 45 ) ( 14 46 ) ( 15 47 ) ( 16 48 ) ( 17 49 ) ( 18 50 ) ( 19 51 ) ( 20 52 ) ( 21 53 ) ( 22 54 ) ( 23 55 ) ( 24 56 ) ( 25 57 ) ( 26 58 ) ( 27 59 ) ( 28 60 ) ( 29 61 ) ( 30 62 ) ( 31 63 )
--------------------------------------------------------------------------------
Level:			3
Size:			40 MB
Cache groups:		( 0 32 1 33 2 34 3 35 4 36 5 37 6 38 7 39 8 40 9 41 10 42 11 43 12 44 13 45 14 46 15 47 ) ( 16 48 17 49 18 50 19 51 20 52 21 53 22 54 23 55 24 56 25 57 26 58 27 59 28 60 29 61 30 62 31 63 )
--------------------------------------------------------------------------------
********************************************************************************
NUMA Topology
********************************************************************************
NUMA domains:		2
--------------------------------------------------------------------------------
Domain:			0
Processors:		( 0 32 1 33 2 34 3 35 4 36 5 37 6 38 7 39 8 40 9 41 10 42 11 43 12 44 13 45 14 46 15 47 )
Distances:		10 21
Free memory:		63120.5 MB
Total memory:		64301 MB
--------------------------------------------------------------------------------
Domain:			1
Processors:		( 16 48 17 49 18 50 19 51 20 52 21 53 22 54 23 55 24 56 25 57 26 58 27 59 28 60 29 61 30 62 31 63 )
Distances:		21 10
Free memory:		63388.7 MB
Total memory:		64506.9 MB
--------------------------------------------------------------------------------
```

With option `-c`, `likwid-topology` produces more information about caches, for example, the size, type, associativity and cache line sizes.

```
********************************************************************************
Cache Topology
********************************************************************************
Level:			1
Size:			32 kB
Type:			Data cache
Associativity:		8
Number of sets:		64
Cache line size:	64
Cache type:		Non Inclusive
Shared by threads:	2
Cache groups:		( 0 32 ) ( 1 33 ) ( 2 34 ) ( 3 35 ) ( 4 36 ) ( 5 37 ) ( 6 38 ) ( 7 39 ) ( 8 40 ) ( 9 41 ) ( 10 42 ) ( 11 43 ) ( 12 44 ) ( 13 45 ) ( 14 46 ) ( 15 47 ) ( 16 48 ) ( 17 49 ) ( 18 50 ) ( 19 51 ) ( 20 52 ) ( 21 53 ) ( 22 54 ) ( 23 55 ) ( 24 56 ) ( 25 57 ) ( 26 58 ) ( 27 59 ) ( 28 60 ) ( 29 61 ) ( 30 62 ) ( 31 63 )
--------------------------------------------------------------------------------
Level:			2
Size:			256 kB
Type:			Unified cache
Associativity:		8
Number of sets:		512
Cache line size:	64
Cache type:		Non Inclusive
Shared by threads:	2
Cache groups:		( 0 32 ) ( 1 33 ) ( 2 34 ) ( 3 35 ) ( 4 36 ) ( 5 37 ) ( 6 38 ) ( 7 39 ) ( 8 40 ) ( 9 41 ) ( 10 42 ) ( 11 43 ) ( 12 44 ) ( 13 45 ) ( 14 46 ) ( 15 47 ) ( 16 48 ) ( 17 49 ) ( 18 50 ) ( 19 51 ) ( 20 52 ) ( 21 53 ) ( 22 54 ) ( 23 55 ) ( 24 56 ) ( 25 57 ) ( 26 58 ) ( 27 59 ) ( 28 60 ) ( 29 61 ) ( 30 62 ) ( 31 63 )
--------------------------------------------------------------------------------
Level:			3
Size:			40 MB
Type:			Unified cache
Associativity:		20
Number of sets:		32768
Cache line size:	64
Cache type:		Inclusive
Shared by threads:	32
Cache groups:		( 0 32 1 33 2 34 3 35 4 36 5 37 6 38 7 39 8 40 9 41 10 42 11 43 12 44 13 45 14 46 15 47 ) ( 16 48 17 49 18 50 19 51 20 52 21 53 22 54 23 55 24 56 25 57 26 58 27 59 28 60 29 61 30 62 31 63 )
--------------------------------------------------------------------------------
```

## likwid-pin

`likwid-pin` provides thread-to-core pinning for an application, which helps avoid thread migration and the loss of cache locality. `likwid-pin` accepts 6 ways of specifying processor lists to its `-c` option. Users can choose the most convenient way on a given architecture for their specific application and affinity.

1. physical numbering: processors are numbered according to the numbering in the OS
1. logical numbering in node: processors are logical numbered over whole node (`N` prefix)
1. logical numbering in socket: processors are logical numbered in every socket (`S#` prefix, e.g. `S0`)
1. logical numbering in cache group: processors are logical numbered in last level cache group (`C#` prefix, e.g., `C1`)
1. logical numbering in memory domain: processors are logical numbered in NUMA domain (`M#` prefix, e.g. `M2`)
1. logical numbering within cpuset: processors are logical numbered inside Linux cpuset (`L` prefix)

With these 6 methods of numbering, `likwid-pin` always allocates physical cores first when hyper-threading is present, except for method 1 and 6. If environment variable `OMP_NUM_THREADS` is absent, `likwid-pin` can also figure out the number of threads based on the `-c` specification.

To understand what thread domains, i.e. the `N`/`S#`/`C#`/`M#`/`L` domains listed above, are available, use the `-p` option of `likwid-pin`.

```
nersc$ likwid-pin -p
Domain N:
	0,32,1,33,2,34,3,35,4,36,5,37,6,38,7,39,8,40,9,41,10,42,11,43,12,44,13,45,14,46,15,47,16,48,17,49,18,50,19,51,20,52,21,53,22,54,23,55,24,56,25,57,26,58,27,59,28,60,29,61,30,62,31,63

Domain S0:
	0,32,1,33,2,34,3,35,4,36,5,37,6,38,7,39,8,40,9,41,10,42,11,43,12,44,13,45,14,46,15,47

Domain S1:
	16,48,17,49,18,50,19,51,20,52,21,53,22,54,23,55,24,56,25,57,26,58,27,59,28,60,29,61,30,62,31,63

Domain C0:
	0,32,1,33,2,34,3,35,4,36,5,37,6,38,7,39,8,40,9,41,10,42,11,43,12,44,13,45,14,46,15,47

Domain C1:
	16,48,17,49,18,50,19,51,20,52,21,53,22,54,23,55,24,56,25,57,26,58,27,59,28,60,29,61,30,62,31,63

Domain M0:
	0,32,1,33,2,34,3,35,4,36,5,37,6,38,7,39,8,40,9,41,10,42,11,43,12,44,13,45,14,46,15,47

Domain M1:
	16,48,17,49,18,50,19,51,20,52,21,53,22,54,23,55,24,56,25,57,26,58,27,59,28,60,29,61,30,62,31,63
```

`xthi.c` code:

```
#define _GNU_SOURCE

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sched.h>
#include <mpi.h>
#include <omp.h>

/* Borrowed from util-linux-2.13-pre7/schedutils/taskset.c */
static char *cpuset_to_cstr(cpu_set_t *mask, char *str)
{
  char *ptr = str;
  int i, j, entry_made = 0;
  for (i = 0; i < CPU_SETSIZE; i++) {
    if (CPU_ISSET(i, mask)) {
      int run = 0;
      entry_made = 1;
      for (j = i + 1; j < CPU_SETSIZE; j++) {
        if (CPU_ISSET(j, mask)) run++;
        else break;
      }
      if (!run)
        sprintf(ptr, "%d,", i);
      else if (run == 1) {
        sprintf(ptr, "%d,%d,", i, i + 1);
        i++;
      } else {
        sprintf(ptr, "%d-%d,", i, i + run);
        i += run;
      }
      while (*ptr != 0) ptr++;
    }
  }
  ptr -= entry_made;
  *ptr = 0;
  return(str);
}

int main(int argc, char *argv[])
{
  int rank, thread;
  cpu_set_t coremask;
  char clbuf[7 * CPU_SETSIZE], hnbuf[64];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  memset(clbuf, 0, sizeof(clbuf));
  memset(hnbuf, 0, sizeof(hnbuf));
  (void)gethostname(hnbuf, sizeof(hnbuf));
  #pragma omp parallel private(thread, coremask, clbuf)
  {
    thread = omp_get_thread_num();
    (void)sched_getaffinity(0, sizeof(coremask), &coremask);
    cpuset_to_cstr(&coremask, clbuf);
    #pragma omp barrier
    printf("Hello from rank %d, thread %d, on %s. (core affinity = %s)\n",
            rank, thread, hnbuf, clbuf);
  }
  MPI_Finalize();
  return(0);
}
```

Take the code `xthi.c` as an example. To pin it on 4 cores, core 0, 8, 16, and 24, with 4 threads, a comma separated list can be provided to `likwid-pin`.

```
nersc$ likwid-pin -c 0,8,16,24 ./xthi.x
Running without Marker API. Activate Marker API with -m on commandline.
[pthread wrapper] 
[pthread wrapper] MAIN -> 0
[pthread wrapper] PIN_MASK: 0->8  1->16  2->24  
[pthread wrapper] SKIP MASK: 0x0
	threadid 46912611731328 -> core 8 - OK
	threadid 46912615929856 -> core 16 - OK
	threadid 46912620128384 -> core 24 - OK
Hello from rank 0, thread 0, on nid00028. (core affinity = 0)
Hello from rank 0, thread 1, on nid00028. (core affinity = 8)
Hello from rank 0, thread 2, on nid00028. (core affinity = 16)
Hello from rank 0, thread 3, on nid00028. (core affinity = 24)
```

`likwid-pin` also accepts dash separated processor lists. For example, `-c 0-3` will place 4 threads on core 0-3.

Equivalent ways of specifying physical cores 0,8,16,24 include:

```
likwid-pin -c N:0,8,16,24 ./xthi.x
likwid-pin -c S0:0,8@S1:0,8 ./xthi.x
likwid-pin -c E:N:4:1:16 ./xthi.x
likwid-pin -c E:S0:2:1:16@E:S1:2:1:16 ./xthi.x
```

The `@` sign packs multiple processor lists into one, and the `E` sign is for expression based syntax which has the following 2 variants. This is very useful when there are many cores and the number of software threads doesn’t agree with the number of hardware threads. For example, on Cori KNL, if users want to run 128 threads on a node with 2 software threads running on a core where 4 hardware threads are available. The pinning can be expressed as `E:N:128:2:4`.

1. -c E:<thread domain>:<number of threads>
1. -c E:<thread domain>:<number of threads>:<chunk size>:<stride>

To pin MPI or hybrid MPI/threaded applications, users can wrap `likwid-pin` with the MPI job launcher, which at NERSC is srun. The following example will place 2 MPI processes on a Cori Haswell node, with 2 threads per process, spread out as far as they can. 

```
nersc$ srun -n 2 -c 32 --cpu-bind=cores likwid-pin -c E:N:2:1:16 ./xthi.x
Running without Marker API. Activate Marker API with -m on commandline.
Running without Marker API. Activate Marker API with -m on commandline.
[pthread wrapper] 
[pthread wrapper] MAIN -> 16
[pthread wrapper] PIN_MASK: 0->24  
[pthread wrapper] SKIP MASK: 0x0
	threadid 46912611731328 -> core 24 - OK
[pthread wrapper] 
[pthread wrapper] MAIN -> 0
[pthread wrapper] PIN_MASK: 0->8  
[pthread wrapper] SKIP MASK: 0x0
	threadid 46912611731328 -> core 8 - OK
Hello from rank 0, thread 0, on nid00028. (core affinity = 0)
Hello from rank 0, thread 1, on nid00028. (core affinity = 8)
Hello from rank 1, thread 0, on nid00028. (core affinity = 16)
Hello from rank 1, thread 1, on nid00028. (core affinity = 24)
```

Another way to achieve this is
```
srun -n 2 -c 32 --cpu-bind=cores likwid-pin -c N:0,8 ./xthi.x
```

## likwid-perfctr

`likwid-perfctr` uses the Linux `msr` module to access the model specific registers stored in `/dev/cpu/*/msr` (which contain hardware performance counters), and calculates performance metrics, FLOPS, bandwidth, etc, based on the formula defined by LIKWID or customized by user.

To get a list of performance metrics/performance groups supported by LIKWID, run this command on that particular architecture. 

```
nersc$ likwid-perfctr -a
 Group name	Description
--------------------------------------------------------------------------------
FALSE_SHARE	False sharing
         HA	Main memory bandwidth in MBytes/s seen from Home agent
   RECOVERY	Recovery duration
  TLB_INSTR	L1 Instruction TLB miss rate/ratio
       SBOX	Ring Transfer bandwidth
     CACHES	Cache bandwidth in MBytes/s
  UOPS_EXEC	UOPs execution
UOPS_RETIRE	UOPs retirement
     BRANCH	Branch prediction miss rate/ratio
  FLOPS_AVX	Packed AVX MFLOP/s
    L2CACHE	L2 cache miss rate/ratio
         L3	L3 cache bandwidth in MBytes/s
        QPI	QPI Link Layer data
       UOPS	UOPs execution info
       NUMA	Local and remote memory accesses
     ENERGY	Power and Energy consumption
    L3CACHE	L3 cache miss rate/ratio
 UOPS_ISSUE	UOPs issueing
     ICACHE	Instruction cache miss rate/ratio
       CBOX	CBOX related data and metrics
CYCLE_ACTIVITY	Cycle Activities
   TLB_DATA	L2 data TLB miss rate/ratio
        MEM	Main memory bandwidth in MBytes/s
       DATA	Load to store ratio
         L2	L2 cache bandwidth in MBytes/s
      CLOCK	Power and Energy consumption
```

Command `likwid-perfctr -e` lists all the hardware events/counters available and `likwid-perfctr -E <perf group>` shows the events/counters used to calculate a particular performance group.

Please note that available performance groups may be different on login and compute nodes since they are of different architecture.

LIKWID derives performance metrics using performance counters based on a formula and to find out what formula is being used, type `likwid-perfctr -H -g <perf group>` for a particular performance group/metric. For example, for DRAM bandwidth metric:

```
nersc$ likwid-perfctr -H -g MEM
Group MEM:
Formulas:
Memory read bandwidth [MBytes/s] = 1.0E-06*(SUM(MBOXxC0))*64.0/runtime
Memory read data volume [GBytes] = 1.0E-09*(SUM(MBOXxC0))*64.0
Memory write bandwidth [MBytes/s] = 1.0E-06*(SUM(MBOXxC1))*64.0/runtime
Memory write data volume [GBytes] = 1.0E-09*(SUM(MBOXxC1))*64.0
Memory bandwidth [MBytes/s] = 1.0E-06*(SUM(MBOXxC0)+SUM(MBOXxC1))*64.0/runtime
Memory data volume [GBytes] = 1.0E-09*(SUM(MBOXxC0)+SUM(MBOXxC1))*64.0
-
Profiling group to measure memory bandwidth drawn by all cores of a socket.
Since this group is based on Uncore events it is only possible to measure on a
per socket base. Some of the counters may not be available on your system.
Also outputs total data volume transferred from main memory.
The same metrics are provided by the HA group.
```

The most commonly used option of likwid-perfctr is the `-C` option, which pins threads to and measures performance metrics on the specified processor lists. For example, the following command measures the DRAM bandwidth of the job execution of `xthi.x`. Since LIKWID only collects uncore counters on a per-socket basis, only core 0 and core 16 have `CAS_COUNT_RD`/`CAS_COUNT_WR` values whereas core 8 and core 16 don’t. Events `INSTR_RETIRED_ANY`, `CPU_CLK_UNHALTED_CORE`, and `CPU_CLK_UNHALTED_REF` are on-core counters and LIKWID collects them on a per-core basis so core 0,8,16 and 24 all have values for these counters.

```
nersc$ likwid-perfctr -C 0,8,16,24 -g MEM ./xthi.x 
--------------------------------------------------------------------------------
CPU name:	Intel(R) Xeon(R) CPU E5-2698 v3 @ 2.30GHz
CPU type:	Intel Xeon Haswell EN/EP/EX processor
CPU clock:	2.30 GHz
--------------------------------------------------------------------------------
Running without Marker API. Activate Marker API with -m on commandline.
Hello from rank 0, thread 0, on nid00171. (core affinity = 0)
Hello from rank 0, thread 1, on nid00171. (core affinity = 8)
Hello from rank 0, thread 3, on nid00171. (core affinity = 24)
Hello from rank 0, thread 2, on nid00171. (core affinity = 16)
--------------------------------------------------------------------------------
Group 1: MEM
+-----------------------+---------+-----------+-----------+-----------+-----------+
|         Event         | Counter |   Core 0  |   Core 8  |  Core 16  |  Core 24  |
+-----------------------+---------+-----------+-----------+-----------+-----------+
|   INSTR_RETIRED_ANY   |  FIXC0  | 118115454 | 156152998 | 176027742 | 204491583 |
| CPU_CLK_UNHALTED_CORE |  FIXC1  |  84800602 | 150120503 | 171989148 | 190897841 |
|  CPU_CLK_UNHALTED_REF |  FIXC2  |  54885567 | 108715434 | 114797646 | 126977020 |
|      CAS_COUNT_RD     | MBOX0C0 |     85204 |         0 |    105033 |         0 |
|      CAS_COUNT_WR     | MBOX0C1 |     58560 |         0 |    113572 |         0 |
|      CAS_COUNT_RD     | MBOX1C0 |    103562 |         0 |     80388 |         0 |
|      CAS_COUNT_WR     | MBOX1C1 |     79060 |         0 |     93672 |         0 |
|      CAS_COUNT_RD     | MBOX2C0 |         0 |         0 |         0 |         0 |
|      CAS_COUNT_WR     | MBOX2C1 |         0 |         0 |         0 |         0 |
|      CAS_COUNT_RD     | MBOX3C0 |         0 |         0 |         0 |         0 |
|      CAS_COUNT_WR     | MBOX3C1 |         0 |         0 |         0 |         0 |


* snip *

+----------------------------------------+------------+-----------+-----------+-----------+
|                 Metric                 |     Sum    |    Min    |    Max    |    Avg    |
+----------------------------------------+------------+-----------+-----------+-----------+
|        Runtime (RDTSC) [s] STAT        |     1.5580 |    0.3895 |    0.3895 |    0.3895 |
|        Runtime unhalted [s] STAT       |     0.2600 |    0.0369 |    0.0830 |    0.0650 |
|            Clock [MHz] STAT            | 13633.1396 | 3175.9463 | 3553.5720 | 3408.2849 |
|                CPI STAT                |     3.5899 |    0.7179 |    0.9771 |    0.8975 |
|  Memory read bandwidth [MBytes/s] STAT |   123.1928 |         0 |   62.4888 |   30.7982 |
|  Memory read data volume [GBytes] STAT |     0.0479 |         0 |    0.0243 |    0.0120 |
| Memory write bandwidth [MBytes/s] STAT |   113.4400 |         0 |   68.1522 |   28.3600 |
| Memory write data volume [GBytes] STAT |     0.0441 |         0 |    0.0265 |    0.0110 |
|    Memory bandwidth [MBytes/s] STAT    |   236.6327 |         0 |  130.6409 |   59.1582 |
|    Memory data volume [GBytes] STAT    |     0.0922 |         0 |    0.0509 |    0.0231 |
+----------------------------------------+------------+-----------+-----------+-----------+
```

For MPI or hybrid applications, users can wrap MPI launcher (srun on Cori) around `likwid-perfctr`, and the following example runs `xthi.x` on 2 MPI processes, each with 2 threads. The placeholders, `%h`, `%p` and `%r`, help distinguish the output from different processes: `%h` for hostname, `%p` for process ID, and `%r` for MPI rank.

```
nersc$ srun -n 2 -c 32 --cpu-bind=cores likwid-perfctr -C 0,8 -g MEM -o test_%h_%p_%r.txt ./xthi.x
INFO: You are running LIKWID in a cpuset with 32 CPUs. Taking given IDs as logical ID in cpuset
INFO: You are running LIKWID in a cpuset with 32 CPUs. Taking given IDs as logical ID in cpuset
Running without Marker API. Activate Marker API with -m on commandline.
Running without Marker API. Activate Marker API with -m on commandline.
Hello from rank 0, thread 0, on nid00171. (core affinity = 0)
Hello from rank 0, thread 1, on nid00171. (core affinity = 8)
Hello from rank 1, thread 0, on nid00171. (core affinity = 16)
Hello from rank 1, thread 1, on nid00171. (core affinity = 24)
```

Another way to measure performance for MPI or hybrid applications is to use `likwid-mpirun`, which we will cover in the next section.

`likwid-perfctr` can also be run in stethoscope mode and timeline mode. Stethoscope mode allows users to listen to all applications running on a node without distinguishing which is what, while timeline mode allows users to listen to a specific application on the node with a specified sampling frequency. Examples of running `likwid-perfctr` in these two modes are:

```
likwid-perfctr -c N:0-7 -g BRANCH  -S 10s
likwid-perfctr -c N:0-7 -g BRANCH -t 2s > out.txt
```

LIKWID also provides a Marker API allowing users to specify a start and stop point in the code so they can only measure performance metrics for that specific region. Multiple regions in the same code are supported too. An example of using Marker API is (`test.c`):

```
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
// This block enables compilation of the code with and without LIKWID in place
#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

#define N 10000

int main(int argc, char* argv[])
{
    int i;
    double data[N];
    LIKWID_MARKER_INIT;
#pragma omp parallel
{
    LIKWID_MARKER_THREADINIT;
}
#pragma omp parallel
{
    LIKWID_MARKER_START("foo");
    #pragma omp for
    for(i = 0; i < N; i++)
    {
        data[i] = omp_get_thread_num();
    }
    LIKWID_MARKER_STOP("foo");
}
    LIKWID_MARKER_CLOSE;
    return 0;
}
```

To compile the code, one needs to provide the correct include and library path, for example, for Intel compilers,
```
cc -qopenmp -DLIKWID_PERFMON -I$LIKWID_INCLUDE -L$LIKWID_LIB -llikwid -dynamic test.c -o test.x
```
To run the code, specify `-m` to likwid-perfctr and LIKWID will only collect performance data when the code is in that particular region. Results will be distinguished between the threads since LIKWID reads each thread’s device files separately. 
```
likwid-perfctr -C 0-3 -g MEM -m ./test.x
```
For more information on LIKWID’s Marker API, please refer to [this page](https://github.com/RRZE-HPC/likwid/wiki/TutorialMarkerC).

## likwid-mpirun

As mentioned above, `likwid-perfctr` only supports purely threaded applications while likwid-mpirun can detect the MPI environment and wrap the job launcher (`srun` on Cori) around likwid-perfctr to measure performance for MPI and hybrid applications. It also integrates the functionality of `likwid-pin` so when option `-g` is not specified, it can also be used as a pinning tool for MPI and hybrid applications.
```
Options:
-h, --help	Help message
-v, --version	Version information
-d, --debug	Debugging output
-n/-np 		Set the number of processes
-nperdomain 	Set the number of processes per node by giving an affinity domain and count
-pin 		Specify pinning of threads. CPU expressions like likwid-pin separated with '_'
-s, --skip 	Bitmask with threads to skip
-mpi 		Specify which MPI should be used. Possible values: openmpi, intelmpi and mvapich2
		If not set, module system is checked
-omp 		Specify which OpenMP should be used. Possible values: gnu and intel
		Only required for statically linked executables.
-hostfile	Use custom hostfile instead of searching the environment
-g/-group 	Set a likwid-perfctr conform event set for measuring on nodes
-m/-marker	Activate marker API mode
-O		Output easily parseable CSV instead of fancy tables
-f		Force overwrite of registers if they are in use. You can also use environment variable LIKWID_FORCE
```

The following example pins 2 MPI ranks on a dual-socket Haswell node with 2 threads each. Since there are uncore events, `CAS_COUNT_RD` and `CAS_COUNT_WR`, `likwid-mpirun` will only collect these data on core 0 and 16. For other events, `likwid-mpirun` collects information on all cores that are being run on. The `_` sign in `-pin` specification separates the processor lists for different ranks. `likwid-mpirun` shares the same performance groups as supported by `likwid-perfctr`.

```
nersc$ likwid-mpirun -pin S0:0,8_S1:0,8 -g MEM ./xthi.x 
Running without Marker API. Activate Marker API with -m on commandline.
Running without Marker API. Activate Marker API with -m on commandline.
Hello from rank 0, thread 0, on nid00191. (core affinity = 0)
Hello from rank 0, thread 1, on nid00191. (core affinity = 8)
Hello from rank 1, thread 0, on nid00191. (core affinity = 16)
Hello from rank 1, thread 1, on nid00191. (core affinity = 24)
Group: 1
+-----------------------+---------+--------------+--------------+---------------+---------------+
|         Event         | Counter | nid00191:0:0 | nid00191:0:8 | nid00191:1:16 | nid00191:1:24 |
+-----------------------+---------+--------------+--------------+---------------+---------------+
|   INSTR_RETIRED_ANY   |  FIXC0  |     41864172 |    586591930 |      42177525 |     478905078 |
| CPU_CLK_UNHALTED_CORE |  FIXC1  |     31442281 |    457123681 |      32258448 |     457217238 |
|  CPU_CLK_UNHALTED_REF |  FIXC2  |     20882252 |    312420615 |      21439634 |     301527102 |
|      CAS_COUNT_RD     | MBOX0C0 |       211530 |            0 |         71031 |             0 |
|      CAS_COUNT_WR     | MBOX0C1 |        90610 |            0 |         90279 |             0 |
|      CAS_COUNT_RD     | MBOX1C0 |       216769 |            0 |         95775 |             0 |
|      CAS_COUNT_WR     | MBOX1C1 |       106912 |            0 |        106699 |             0 |
|      CAS_COUNT_RD     | MBOX2C0 |            0 |            0 |             0 |             0 |
|      CAS_COUNT_WR     | MBOX2C1 |            0 |            0 |             0 |             0 |
|      CAS_COUNT_RD     | MBOX3C0 |            0 |            0 |             0 |             0 |
|      CAS_COUNT_WR     | MBOX3C1 |            0 |            0 |             0 |             0 |


* snip *

+----------------------------------------+------------+-----------+-----------+-----------+-----------+-----------+-----------+
|                 Metric                 |     Sum    |    Min    |    Max    |    Avg    |  %ile 25  |  %ile 50  |  %ile 75  |
+----------------------------------------+------------+-----------+-----------+-----------+-----------+-----------+-----------+
|        Runtime (RDTSC) [s] STAT        |     1.4542 |    0.3633 |    0.3638 |    0.3636 |    0.3633 |    0.3633 |    0.3638 |
|        Runtime unhalted [s] STAT       |     0.4253 |    0.0137 |    0.1988 |    0.1063 |    0.0137 |    0.0140 |    0.1988 |
|            Clock [MHz] STAT            | 13776.4838 | 3365.2590 | 3487.5572 | 3444.1209 | 3365.2590 | 3460.5984 | 3463.0692 |
|                CPI STAT                |     3.2499 |    0.7511 |    0.9547 |    0.8125 |    0.7511 |    0.7648 |    0.7793 |
|  Memory read bandwidth [MBytes/s] STAT |   213.6791 |         0 |  153.2342 |   53.4198 |         0 |         0 |  153.2342 |
|  Memory read data volume [GBytes] STAT |     0.0777 |         0 |    0.0557 |    0.0194 |         0 |         0 |    0.0220 |
| Memory write bandwidth [MBytes/s] STAT |   138.6333 |         0 |   69.6328 |   34.6583 |         0 |         0 |   69.0005 |
| Memory write data volume [GBytes] STAT |     0.0504 |         0 |    0.0253 |    0.0126 |         0 |         0 |    0.0251 |
|    Memory bandwidth [MBytes/s] STAT    |   352.3124 |         0 |  222.2347 |   88.0781 |         0 |         0 |  130.0777 |
|    Memory data volume [GBytes] STAT    |     0.1280 |         0 |    0.0807 |    0.0320 |         0 |         0 |    0.0473 |
+----------------------------------------+------------+-----------+-----------+-----------+-----------+-----------+-----------+
```

## likwid-bench

`likwid-bench` provides a list of benchmark kernels for users to quickly test some characteristics of an architecture. 

```
nersc$ likwid-bench -a | head
clcopy - Double-precision cache line copy, only touches first element of each cache line.
clload - Double-precision cache line load, only loads first element of each cache line.
clstore - Double-precision cache line store, only stores first element of each cache line.
copy - Double-precision vector copy, only scalar operations
copy_avx - Double-precision vector copy, optimized for AVX
copy_avx512 - Double-precision vector copy, optimized for AVX-
copy_mem - Double-precision vector copy, only scalar operations but with non-temporal stores
copy_mem_avx - Double-precision vector copy, uses AVX and non-temporal stores
copy_mem_sse - Double-precision vector copy, uses SSE and non-temporal stores
copy_sse - Double-precision vector copy, optimized for SSE
```

For example, to run the stream benchmark on a Cori Haswell node,
```
nersc$ likwid-bench -t stream -w S0:20kB:2 -w S1:20kB:2
Allocate: Process running on core 0 (Domain S0) - Vector length 833/6664 Offset 0 Alignment 512
Allocate: Process running on core 0 (Domain S0) - Vector length 833/6664 Offset 0 Alignment 512
Allocate: Process running on core 0 (Domain S0) - Vector length 833/6664 Offset 0 Alignment 512
Allocate: Process running on core 16 (Domain S1) - Vector length 833/6664 Offset 0 Alignment 512
Allocate: Process running on core 16 (Domain S1) - Vector length 833/6664 Offset 0 Alignment 512
Allocate: Process running on core 16 (Domain S1) - Vector length 833/6664 Offset 0 Alignment 512
--------------------------------------------------------------------------------
LIKWID MICRO BENCHMARK
Test: stream
--------------------------------------------------------------------------------
Using 2 work groups
Using 4 threads
--------------------------------------------------------------------------------
Group: 0 Thread 1 Global Thread 1 running on core 32 - Vector length 416 Offset 416
Group: 0 Thread 0 Global Thread 0 running on core 0 - Vector length 416 Offset 0
Group: 1 Thread 1 Global Thread 3 running on core 48 - Vector length 416 Offset 416
Group: 1 Thread 0 Global Thread 2 running on core 16 - Vector length 416 Offset 0
--------------------------------------------------------------------------------
Cycles:			6925952823
CPU Clock:		2299995373
Cycle Clock:		2299995373
Time:			3.011290e+00 sec
Iterations:		33554432
Iterations per thread:	8388608
Inner loop executions:	104
Size (Byte):		39936
Size per thread:	9984
Number of Flops:	27917287424
MFlops/s:		9270.87
Data volume (Byte):	335007449088
MByte/s:		111250.48
Cycles per update:	0.496177
Cycles per cacheline:	3.969413
Loads per update:	2
Stores per update:	1
Load bytes per element:	16
Store bytes per elem.:	8
Load/store ratio:	2.00
Instructions:		66303557649
UOPs:			90731184128
--------------------------------------------------------------------------------
```
