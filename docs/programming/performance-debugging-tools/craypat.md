# CrayPat

## Description

CrayPat is a performance analysis tool offered by Cray for the XC platform.
CrayPat has a large feature set.

`perftools-lite` is a simplified and easy-to-use version of the CrayPat tool.
It provides basic performance analysis information automatically with simple
steps. Users can decide whether to use the full CrayPat tools after trying
`perftools-lite`.

Here we will highlight basic usage and point to other relevant documentation.

## How to use `perftools-lite`

The general workflow for getting performance data using `perftools-lite` is as
follows:

  1. Unload the darshan module  (`perftools-lite` conflicts with the default
     loaded `darshan` I/O performance monitoring tool).
  1. Load the `perftools-base` and `perftools-lite` modules.
  1. Build your application as normal.
  1. Run as normal. Performance data is summarized at the end of the job STDOUT.
  1. More detailed information can also be gathered with `pat_report` or Cray
     Apprentice2 after the run.

## Outputs from `perftools-lite`

  - In the job's stdout file, basic information from the default
    `sample_profile` option: execution time, memory high-water mark, aggregate
    FLOPS rate, top time-consuming user functions, MPI information, etc.
  - A `.rpt` text file with the same info as above.
  - A `.ap2` file that can be used with `pat_report` for more detailed
    informaition, and with `app2` for graphic visualization.
  - Possibly one or more suggested `MPICH_RANK_ORDER_FILE` files.

## How to Use CrayPat

The general workflow for getting performance data using CrayPat is as follows:

  - Unload the `darshan` module if it is loaded.
  - Load the `perftools-base` and `perftools` modules.
  - Build your application; keep `.o` files.
  - Instrument the application using `pat_build`.
  - Run the instrumented executable to get a performance data (`.xf`) file.
  - Run `pat_report` on the data file to view the results.

[Darshan](./darshan.md) is an I/O profiling tool that is loaded by default.
When using CrayPat, the `darshan` module needs to be unloaded first. Otherwise,
the following error will occur:

```console
user@cori02:~$ pat_build myprogram
FATAL: The program '/.../myprogram' directly or indirectly uses the MPI profiling mechanism.
```

If you still see this error after unloading the `darshan` module, then it is
necessary to unload the darshan module in your `.bashrc.ext` (bash shell) or
`.cshrc.ext` (tcsh or csh shell) dotfile and login again.

The `perftools` module needs to be loaded before you start building your
application. Otherwise, the following error message appears during the link
stage.

```console
ERROR: Missing required ELF section 'link information' from the file ....
Load the correct 'craypat' module and rebuild the program.
```

Object files (`.o` files) need to be made available to CrayPat to correctly
build an instrumented executable for profiling or tracing. In other words,
compile and link stage should be separated by using the `-c` compile flag.
Otherwise, one will see the warning message:

```console
user@cori02:~$ module load perftools-base perftools
user@cori02:~$ ftn mytest.f90
WARNING: CrayPat is saving object files from temporary locations into directory '/global/homes/...'
```

Please note that the Cray compiler wrappers (`ftn`, `cc` and `CC`) must be used
for building an executable, instead of native compiler commands (`ifort`,
`icc`, `icpc`, `gcc`, etc.) since `pat_build` cannot build an instrumented
executable from an executable built with a native compiler.

Try to run a CrayPat-instrumented executable in the `$SCRATCH` or `$SCRATCH2`
space or make sure that CrayPat writes its performance data to those spaces via
the `PAT_RT_EXPDIR_NAME` environment variable (see the `intro_craypat` man
page):
```slurm
#!/bin/bash
#SBATCH -N 2
...
export PAT_RT_EXPDIR_NAME=$SCRATCH/data_dir  # the directory must exist
srun -n 48 ./myprogram+pat
```

## Sampling Experiments

Sampling (sometimes called "asynchronous") is to sample the program counter
(PC) or the call stack at given time intervals or when specified counter
overflows. The default experiment type is to sample the PC at a time interval
(i.e., `samp_pc_time`). There are other sampling experiment types available,
and the type can be set by the environment variable `PAT_RT_EXPERIMENT` (see
the `intro_craypat` man page).

```console
user@cori02:~$ module load perftools
user@cori02:~$ ftn -c myprogram.f90
user@cori02:~$ ftn -o myprogram myprogram.o
user@cori02:~$ pat_build -S myprogram  (or simply pat_build myprogram)
```

This generates a new executable, `myprogram+pat`. Run this executable on
compute nodes, as you would with the regular executable. This will generate a
performance data file with the suffix `.xf` (e.g.,
`myprogram+pat+5511-2558sdot.xf`). To view the contents, run `pat_report` on it:

```console
user@cori02:~$ pat_report myprogram+pat+5511-2558sdot.xf
```

This commands prints ASCII text report to your terminal and creates files with
different suffices, `.ap2` and `.apa`. The first file is used to view
performance data graphically with the Cray Apprentice2 tool, and the latter is
for suggested `pat_build` options for more detailed tracing experiments.  To
see source line information use the `-O ca+src` option to `pat_report`.

A more detailed source line-by-line profile can be obtained by the following:

```console
pat_build a.out [don't use any pat_build options]
srun -n ... a.out
pat_report -O samp_profile+src ....
```

This will produce an output similar to the following:

```console
Table 1: Profile by Group, Function, and Line
Samp% | Samp | Imb. | Imb. |Group
      |      | Samp | Samp% | Function
 |    |      |              | Source
 |    |      |              | Line
100.0% | 3654.0 | -- | -- |Total
|---------------------------------------------------------------
| 99.6% | 3640.0 | -- | -- |USER
||--------------------------------------------------------------
|| 82.5% | 3015.0 | -- | -- |dim3_sweep_module_dim3_sweep_
3|       |        |    |    | HOPPER/src/./dim3_sweep.f90
||||------------------------------------------------------------
4|||  1.3% |  47.0 | -- | -- |line.122
4|||  6.2% | 226.0 | -- | -- |line.217
4||| 11.0% | 402.0 | -- | -- |line.218
4||| 11.1% | 407.0 | -- | -- |line.228
4|||  7.5% | 274.0 | -- | -- |line.229
4|||  7.3% | 266.0 | -- | -- |line.238
4|||  4.3% | 158.0 | -- | -- |line.240
4|||  5.8% | 212.0 | -- | -- |line.241
4|||  5.4% | 198.0 | -- | -- |line.242
4|||  6.7% | 245.0 | -- | -- |line.243
4|||  3.3% | 120.0 | -- | -- |line.322
4|||  4.5% | 164.0 | -- | -- |line.371
4|||  3.9% | 142.0 | -- | -- |line.377
```

which shows that the function `dim3_sweep` takes up nearly all the time (~82%)
and source lines 218 and 228 comprise the bulk of that.  Note that the
individual source lines shown do not add to 82%, probably because some source
lines have fallen below the `pat_report` printing threshold.

## Tracing Experiments

`pat_build` also can instrument an executable to trace calls to user-defined
functions and Cray-provided library functions (e.g., MPI functions). Again, to
generate an instrumented executable, one needs to load the perftools module
first, and, then, compile and link in separate steps.

```console
user@cori02:~$ module load perftools
user@cori02:~$ ftn -c myprogram.f90
user@cori02:~$ ftn -o myprogram myprogram.o
```
The `-w` flag enables tracing. If only this flag is used, the entire program is
traced as a whole (as `main`), with no individual function being traced.

```console
user@cori02:~$ pat_build -w myprogram
```

To instrument user-defined functions `func1` and `func2`, use the `-T` option,
together with the `-w` option:

```console
user@cori02:~$ pat_build -w -T func1,func2 myprogram
```

Be careful to choose the `func1` and `func2` names properly; the compiler may
have appended underscore characters to the Fortran routine name.

To trace a group of functions you list in a text file, tracefile, use the `-t`
option:

```console
user@cori02:~$ pat_build -w -t tracefile myprogram
```

where the file, tracefile, contains the function names to be traced.

To trace all the user-defined function, use the `-u` option.

```console
user@cori02:~$ pat_build -u myprogram
```

Tracing the entire user functions can slow down the code significantly if it
contains many small and frequently called functions. To avoid such excessive
overhead, one can restrict only to the functions with a certain text size or
larger, by using the directive, trace-text-size (see the `pat_build` man page):

```console
user@cori02:~$ pat_build -u -Dtrace-text-size=800 myprogram    # to trace those with text size >= 800 bytes
```

To trace a Cray-provided library function group (e.g., MPI, OpenMP, ...),
specify the function group name after the `-g` flag. The supported function
groups are listed in the `pat_build` man page. For example, to trace the MPI,
OpenMP and heap memory related functions as well as all the user functions, one
can do:

```console
user@cori02:~$ pat_build -g mpi,omp,heap -u myprogram
```

After running the instrumented executable on compute nodes via aprun, run
`pat_report` on the generated data file (`.xf` file). This prints ASCII text
output to terminal, and creates a file with the same basename but with the
`.ap2` suffix, which is to be viewed with the Cray Apprentice2 tool.

## Automatic Program Analysis (APA)

Since a sampling experiment runs with little overhead and a detailed tracing
experiment in general comes with large overhead, a good strategy to get
performance analsys for a code that a user doesn't know about its performance
characteristics would to run a sampling experiment first to identify routines
that need to be instrumented for a more detailed tracing experiment later.

CrayPat's Automatic Program Analysis (APA) feature provides an easy way for
such a purpose. Using this feature, one can generate an instrumented executable
for a sampling experiment. When the binary is executed, it generates an ASCII
text file that contains CrayPat's suggestion for `pat_build` tracing options,
which can be used to re-instrument the executable for detailed tracing
experiments.

The general workflow for using APA is as follows.

  1. Generate the executable for sampling, using the special `-O apa` flag. It
     will generate an instrumented executable, `myprogram+pat`.
     ```console
     user@cori02:~$ pat_build -O apa myprogram    # generates myprogram+pat
     ```
  1. Running the executable on compute nodes via `srun` generates a `.xf` file.
     Let's call it `myprogram+pat+4571-19sdot.xf` in this example.
  1. Run `pat_report` on the data file:
     ```console
     user@cori02:~$ pat_report myprogram+pat+4571-19sdot.xf
     ```
     It will generate the `myprogram+pat+4571-19sdot.ap2` and
     `myprogram+pat+4571-19sdot.apa1. The latter contains suggested `pat_build`
     options for building an executable for tracing experiments.
  1. Examine the `myprogram+pat+4571-19sdot.apa` file and, if necessary,
     customize it for your need using your favorite text editor.
  1. Rebuild an executable using `pat_build -O` option with the `.apa` file
     name as the argument. It generates a new instrumented executable,
     `myprogram+apa`.
     ```console
     user@cori02:~$ pat_build -O myprogram+pat+4571-19sdot.apa    # generates myprogram+apa
     ```
  1. Run the new executable, `myprogram+apa`, for a tracing experiment.
  1. Run `pat_report` on the newly created `.xf` file. This is the tracing result.
     ```console
     user@cori02:~$ pat_report myprogram+apa+4590-19tdot.xf
     ```

## Monitoring Hardware Performance Counters

One can monitor hardware performance counter (HWPC) events while running
sampling or tracing experiments (however, doing this with sampling experiment
is discouraged). Supported PAPI standard and Intel native event names that can
be monitored can be found by running the `papi_avail` and `papi_native_avail`
commands on a compute node using a batch script:

```slurm
#!/bin/bash
#SBATCH -N 1
...
module load perftools
papi_avail
papi_native_avail
```

By default, hardware performance counters are not monitored during sampling or
tracing experiments. To enable monitoring, one has to explicitly specify up to
four event names using the `PAT_RT_PERFCTR` environment variable before aprun
command is executed. For example, one can monitor floating point operations and
L1 data cache misses with the following:

```console
export PAT_RT_PERFCTR="PAPI_FP_OPS,PAPI_L1_DCM"
```

Or one can set the environment variable to a predefined hardware counter group
number:

```console
export PAT_RT_PERFCTR=1
```

The meaning of each counter group (1, 2, 3, ...) depends on the Cray system
your application is running. E.g., on some systems, group 1 collects
floating-point and cache metrics. These groups and the meaings are explained in
the `hwpc` man page on each machine (accessible when the `perftools` module is
loaded).

## Some Advanced CrayPat Topics

By default, CrayPat will aggregate values from multiple processing elements
during a run (and there are options to control the aggregation).  If you want
to look at HWPC values on a per-processing element (PE) basis, you can just do
the following:

```console
pat_report -s pe=ALL file.ap2 ...
```

The following should work as well:

```console
pat_report -d counters -b pe -s aggr_pe_counters=select0 ...
```

If you want to sort the report by PEs, you can add

```console
-s sort_by_pe=yes
```

If you'd like HPWC values for just the whole program (not per function, etc),
you can do

```console
pat_report -O hwpc -s pe=ALL ...
```

## Measuring Load Imbalance

You can use CrayPat to measure load imbalance in programs instrumented to trace
MPI functions. By default CrayPat causes the trace wrapper for each MPI
collective subroutine to measure the time for a barrier call prior to entering
the collective. This time is reported by `pat_report` in as `MPI_SYNC`, which
is separate from the MPI function group itself. The `MPI_SYNC` time essentially
represents the time spent waiting for the MPI call to synchronize; it
determines if the MPI ranks arrive at the collectives together or not.  If the
environment variable `PAT_RT_MPI_SYNC` is set to `1` (which is the default),
the time spent waiting at a barrier and synchronizing processes is reported
under `MPI_SYNC`, while the time spent executing after the barrier is reported
under `MPI`. Default values are 1 for tracing experiments and 0 for sampling
experiments.

## Cray Apprentice2

Cray Apprentice2 is a tool used to visualize performance data instrumented with
the CrayPat tool. There are many options for viewing results. Please refer to
the `app2` man page or [Cray's documentation](https://pubs.cray.com/) for more
details.

```console
user@cori02:~$ module load perftools
user@cori02:~$ app2 myprogram+pat+####-####tdot.ap2
```

To enable the "Mosaic" and "Traffic Report" views you need to set an
environment variable as follows:

```console
export PAT_RT_SUMMARY=0
```

Doing this may create enormous CrayPat data files and may take a long time.

## Cray Reveal

Cray Reveal is a tool developed by Cray to help developing the hybrid
MPI/OpenMP programming model. It is part of the Cray Perftools software
package. It utilizes the Cray CCE program library (hence it only works under
`PrgEnv-cray`) for loopmark and source code analysis, combined with performance
data collected from CrayPat. Reveal helps to identify top consuming loops, with
compiler feedback on dependency and vectorization. Its loop scope analysis
provides variable scope and compiler directive suggestions for inserting OpenMP
parallelism to a serial or pure MPI code. Please see [this](./reveal.md) page
with detailed steps for using Reveal.

## Known Issues

### Intel Fortran and nounderscore

If you are using the Intel Fortran compilers with `-assume nounderscore` to
facilitate linking with code written in C, you may encounter issues when
executing your Craypat-instrumented (i.e. `+pat`) executable related to multuple
initializations of MPI. This would produce errors similar to:

```console
Rank 0 [<date>] [<node>] Fatal error in MPI_Init: Other MPI error, error stack:
MPI_Init(141): Cannot call MPI_INIT or MPI_INIT_THREAD more than once
```
If possible, avoid using the `-assume nounderscore` option when profiling with
Craypat.

## Further Information

NERSC has prepared a detailed tutorial on Cray's perftools.  You can view the
presentation material
[here](http://www.nersc.gov/assets/Uploads/05-craypat-reveal-20170609.pdf).

Please refer to [Cray's documentation](https://pubs.cray.com/browse/xc) for
more details.  Man pages are available but only after you load the `perftools`
modulefile.  Try `man pat_help` for a tutorial.  Other man pages include:

  - `craypat`
  - `pat_build`
  - `pat_report`
  - `app2`
  - `hwpc`
  - `intro_perftools`
  - `papi`
  - `papi_counters`

For questions on using CrayPat at NERSC, please contact the [NERSC help
desk](https://help.nersc.gov/).
