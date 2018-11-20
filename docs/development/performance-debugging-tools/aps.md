# Application Performance Snapshot (APS)

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Analyzing Shared Memory Applications](#analyzing-shared-memory-applications)
4. [Analyzing MPI Applications](#analyzing-mpi-applications)
5. [Next Steps](#next-steps)
6. [Quick Metrics Reference](#quick-metrics-reference)
7. [Documentation and Resources](#documentation-and-resources)

## Introduction

Application Performance Snapshot (APS) is a lightweight open source profiling
tool developed by the Intel VTune developers. 

Use Application Performance Snapshot for a quick view into a shared memory or
MPI application's use of available hardware (CPU, FPU, and memory). Application
Performance Snapshot analyzes your application's time spent in MPI, MPI and
OpenMP imbalance, memory access efficiency, FPU usage, and I/O and memory
footprint. After analysis, it displays basic performance enhancement
opportunities for systems using Intel platforms. Use this tool as a first step
in application performance analysis to get a simple snapshot of key
optimization areas and learn about profiling tools that specialize in
particular aspects of application performance.

Application Performance Snapshot is available as a free product download from
the Intel Developer Zone at https://software.intel.com/performance-snapshot and
is also available pre-installed as part of Intel Parallel Studio or Intel VTune
Amplifier.

## Prerequisites

**Optional:** Use the following software to get an advanced metric set when
running Application Performance Snapshot:

* Recommended compilers: Intel C/C++ or Fortran Compiler (other compilers can
  be used, but information about OpenMP imbalance is only available from the
  Intel OpenMP library)
* Use Intel MPI library version 2017 or later. Other MPICH-based MPI
  implementations can be used, but information about MPI imbalance is only
  available from the Intel MPI library. There is no support for OpenMPI.

**Optional:** Enable system-wide monitoring to reduce collection overhead and
collect memory bandwidth measurements. Use one of these options to enable
system-wide monitoring:

* Set the `/proc/sys/kernel/perf_event_paranoid` value to 0 (or less), or
* Install the Intel VTune Amplifier drivers. Driver sources are available in
  `<APS_install_dir>/internal/sepdk/src`. Installation instructions are
  available online at
  https://software.intel.com/en-us/vtune-amplifier-help-building-and-installing-the-sampling-drivers-for-linux-targets.

Before running the tool, set up your environment appropriately:

Run `<install-dir>/apsvars.sh`, where `<install-dir>` is the location where
Application Performance Snapshot is installed either as a standalone package or
as part of Intel Parallel Studio or Intel VTune Amplifier.

Example:

```
nersc$ /opt/intel/performance_snapshots/apsvars.sh
```

## Analyzing Shared Memory Applications

Run the following command:

```
nersc$ aps <my app> <app parameters>
```

where `<my app>` is the location of your application and `<app parameters>` are
your application parameters.

Application Performance Snapshot launches the application and runs the data
collection.

After the analysis completes, a report appears in the command window. You can
also open an HTML report with the same information in a supported browser. The
path to the HTML report is included in the command window.

Analyze the data shown in the report. See the metric descriptions below or
hover over a metric in the HTML report for more information.

Determine appropriate next steps based on result analysis. Common next steps
may include application tuning or using another performance analysis tool for
more detailed information, such as Intel VTune Amplifier or Intel Advisor.

## Analyzing MPI Applications

Run the following command to collect data about your MPI application:

```
nersc$ <mpi launcher> <mpi parameters>aps <my app> <app parameters>
```

where:

* `<mpi launcher>` is an MPI job launcher such as `mpirun`, `srun`, or `aprun`
* `<mpi parameters>` are the MPI launcher parameters
* `<my app>` is the location of your application
* `<app parameters>` are your application parameters

!!! note
    `aps` must be the last `<mpi launcher>` parameter.

Application Performance Snapshot launches the application and runs the data
collection. After the analysis completes, an `aps_result_<date>` directory is
created. To complete the analysis, run:

```
nersc$ aps --report=aps_result_<date>
```

After the analysis completes, a report appears in the command window. You can
also open a HTML report with the same information in a supported browser.

Analyze the data shown in the report. See the metric descriptions below or
hover over a metric in the HTML report for more information.

!!! tip
    If your application is MPI-bound, run the following command to get more
    details about message passing such as message sizes, data transfers between
    ranks or nodes, and time in collective operations:

    ```
    nersc$ aps-reports <option> app_result_<date>
    ```

    Use `aps-reports --help` to see the available options.

Determine appropriate next steps based on result analysis. Common next steps
may include communication tuning with the `mpitune` utility or using another
performance analysis tool for more detailed information, such as Intel Trace
Analyzer and Collector or Intel VTune Amplifier.

## Next Steps

* [Intel Trace Analyzer and Collector](https://software.intel.com/en-us/intel-trace-analyzer)
  is a graphical tool for understanding MPI application behavior, quickly
  identifying bottlenecks, improving correctness, and achieving high
  performance for parallel cluster applications running on Intel architecture.
  Improve weak and strong scaling for applications.
  [Get started](https://software.intel.com/en-us/get-started-with-itac-for-linux).
* [Intel VTune Amplifier](https://software.intel.com/en-us/intel-vtune-amplifier-xe)
  provides a deep insight into node-level performance including algorithmic
  hotspot analysis, OpenMP threading, general exploration microarchitecture
  analysis, memory access efficiency, and more. It supports C/C++, Fortran,
  Java, Python, and profiling in containers.
  [Get started](https://software.intel.com/en-us/get-started-with-vtune-linux-os).
* [Intel Advisor](https://software.intel.com/en-us/intel-advisor-xe) provides
  two tools to help ensure your Fortran, C, and C++ applications realize full
  performance potential on modern processors.
  [Get started](https://software.intel.com/en-us/get-started-with-advisor).
    - Vectorization Advisor is an optimization tool to identify loops that will
      benefit most from vectorization, analyze what is blocking effective
      vectorization, and forecast the benefit of alternative data
      reorganizations
    - Threading Advisor is a threading design and prototyping tool to analyze,
      design, tune, and check threading design options without disrupting a
      regular environment

## Quick Metrics Reference

The following metrics are collected with Application Performance Snapshot.
Additional detail about each of these metrics is available in the
[Intel VTune Amplifier online help](https://software.intel.com/en-us/vtune-amplifier-help-cpu-metrics-reference).

**Elapsed Time**: Execution time of specified application in seconds.

**SP GFLOPS**: Number of single precision giga-floating point operations
calculated per second. All double operations are converted to two single
operations. SP GFLOPS metrics are only available for 3rd Generation Intel Core
processors, 5th Generation Intel processors, and 6th Generation Intel
processors.

**Cycles per Instruction Retired (CPI)**: The amount of time each executed
instruction took measured by cycles. A CPI of 1 is considered acceptable for
high performance computing (HPC) applications, but different application
domains will have varied expected values. The CPI value tends to be greater
when there is long-latency memory, floating-point, or SIMD operations,
non-retired instructions due to branch mispredictions, or instruction
starvation at the front end.

**MPI Time**: Average time per process spent in MPI calls. This metric does not
include the time spent in `MPI_Finalize`. High values could be caused by high
wait times inside the library, active communications, or sub-optimal settings
of the MPI library. The metric is available for MPICH-based MPIs.

**MPI Imbalance**: CPU time spent by ranks spinning in waits on communication
operations. A high value can be caused by application workload imbalance
between ranks, or non-optimal communication schema or MPI library settings.
This metric is available only for Intel MPI Library version 2017 and later.

**OpenMP Imbalance**: Percentage of elapsed time that your application wastes
at OpenMP synchronization barriers because of load imbalance. This metric is
only available for the Intel OpenMP Runtime Library.

**CPU Utilization**: Estimate of the utilization of all logical CPU cores on
the system by your application. Use this metric to help evaluate the parallel
efficiency of your application. A utilization of 100% means that your
application keeps all of the logical CPU cores busy for the entire time that it
runs. Note that the metric does not distinguish between useful application work
and the time that is spent in parallel runtimes.

**Memory Stalls**: Indicates how memory subsystem issues affect application
performance. This metric measures a fraction of slots where pipeline could be
stalled due to demand load or store instructions. If the metric value is high,
review the Cache and DRAM Stalls and the percent of remote accesses metrics to
understand the nature of memory-related performance bottlenecks. If the average
memory bandwidth numbers are close to the system bandwidth limit, optimization
techniques for memory bound applications may be required to avoid memory
stalls.

**FPU Utilization**: The effective FPU usage while the application was running.
Use the FPU Utilization value to evaluate the vector efficiency of your
application. The value is calculated by estimating the percentage of operations
that are performed by the FPU. A value of 100% means that the FPU is fully
loaded. Any value over 50% requires additional analysis. FPU metrics are only
available for 3rd Generation Intel Core processors, 5th Generation Intel
processors, and 6th Generation Intel processors.

**I/O Operations**: The time spent by the application while reading data from
the disk or writing data to the disk. **Read** and **Write** values denote mean
and maximum amounts of data read and written during the elapsed time. This
metric is only available for MPI applications.

**Memory Footprint**: Average per-rank and per-node consumption of both virtual
and resident memory.

## Documentation and Resources

* [Intel Performance Snapshot User Forum](https://software.intel.com/en-us/forums/intel-performance-snapshot):
  User forum dedicated to all Intel Performance Snapshot tools, including
  Application Performance Snapshot
* [Application Performance Snapshot](https://software.intel.com/sites/products/snapshots/application-snapshot/):
  Application Performance Snapshot product page, see this page for support and
  online documentation
* [Application Performance Snapshot User's Guide](https://software.intel.com/en-us/application-snapshot-user-guide):
  Learn more about Application Performance Snapshot, including details on specific metrics and best practices for application optimization
