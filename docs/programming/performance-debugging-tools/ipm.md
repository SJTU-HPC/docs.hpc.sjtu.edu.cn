# IPM

## Description and Overview

IPM is a portable profiling infrastructure which provide a high level report on
the execution of a parallel job. IPM reports hardware counters data, MPI
function timings, and memory usage. It provides a low overhead means to
generate scaling studies or performance data for ERCAP submissions. When you
run a job using the IPM module you will get a performance summary (see below)
to STDOUT as well as a web accessible summary of all your IPM jobs. The two
main objectives of IPM are ease-of-use and scalability in performance analysis.

## Usage

```console
user@cori02:~$ module load ipm
```

On HPC architectures that support shared libraries, that's all you need to do.
Once the module is loaded you can run as you normally and get a performance
profile once the job has successfully completed. You do not need to relink your
code. For static executables and architectures which do not support shared
libraries, a relink is required. You simply load the `ipm` module, add `$(IPM)`
to your link line, and run as you normally would.

## Using IPM on Cori


!!! warning "Note on Darshan and IPM"
    Currently if a program is linked with IPM, no Darshan IO statics will be
    collected.  For more information about Darshan, please see [this
    page](./darshan.md).

You must link your code against IPM. Here is a simple example compile and link
for the default Intel programming environment.

```console
user@cori02:~$ module load ipm
user@cori02:~$ ftn -c mycode.f90
user@cori02:~$ ftn -o mycode.x mycode.o $IPM
```

The `$IPM` reference needs to be the last argument on the link line.

Note that IPM is currently available for `PrgEnv-intel` and `PrgEnv-gnu`. by
specifying `$IPM` at link time it will automatically link against the correct
libraries if the corresponding `PrgEnv` is loaded.

## Output and Results

Once the module has been loaded each parallel code will, upon completion, print
a concise report to standard out. In addition, detailed results are available
the day after the job completed from the [Completed
Jobs](https://www.nersc.gov/users/job-logs-statistics/completed-jobs/) page.

More detailed reports are possible, for example, a more detailed report looks
like:

```console
##IPMv0.8######################################################################
#
# code   : ./bin/cg.B.32 (completed)
# host   : s05601/006035314C00_AIX        mpi_tasks : 32 on 2 nodes
# start  : 11/30/04/14:35:34              wallclock : 29.975184 sec
# stop   : 11/30/04/14:36:00              %comm     : 27.72
# gbytes : 6.65863e-01 total              gflop/sec : 2.33478e+00 total
#
#
#                           [total]         <avg>           min           max
# wallclock                  953.272       29.7897       29.6092       29.9752
# user                        837.25       26.1641         25.71         26.92
# system                        60.6       1.89375          1.52          2.59
# mpi                        264.267       8.25834       7.73025       8.70985
# %comm                                    27.7234       25.8873       29.3705
# gflop/sec                  2.33478     0.0729619      0.072204     0.0745817
# gbytes                    0.665863     0.0208082     0.0195503     0.0237541
# PM_FPU0_CMPL           2.28827e+10   7.15084e+08   7.07373e+08   7.30171e+08
# PM_FPU1_CMPL           1.70657e+10   5.33304e+08   5.28487e+08   5.42882e+08
# PM_FPU_FMA             3.00371e+10    9.3866e+08   9.27762e+08   9.62547e+08
# PM_INST_CMPL           2.78819e+11   8.71309e+09   8.20981e+09   9.21761e+09
# PM_LD_CMPL             1.25478e+11   3.92118e+09   3.74541e+09   4.11658e+09
# PM_ST_CMPL             7.45961e+10   2.33113e+09   2.21164e+09   2.46327e+09
# PM_TLB_MISS            2.45894e+08   7.68418e+06   6.98733e+06   2.05724e+07
# PM_CYC                  3.0575e+11   9.55467e+09   9.36585e+09   9.62227e+09
#
#                            [time]       [calls]        <%mpi>      <%wall>
# MPI_Send                   188.386        639616         71.29        19.76
# MPI_Wait                   69.5032        639616         26.30         7.29
# MPI_Irecv                  6.34936        639616          2.40         0.67
# MPI_Barrier              0.0177442            32          0.01         0.00
# MPI_Reduce              0.00540609            32          0.00         0.00
# MPI_Comm_rank           0.00465156            32          0.00         0.00
# MPI_Comm_size          0.000145341            32          0.00         0.00
###############################################################################
```

The amount of detail reported information can be obtained using the options
described in the [next section](#options).

## Options

The interface to IPM is through environment variables and `MPI_Pcontrol`. The
environment variable interface is selected at execute/submit time while the
later allows for dynamic control of IPM. A description of environment variables
supported is given below. A description of the `MPI_Pcontrol` interface is
included in the main [IPM documentation](https://github.com/nerscadmin/IPM).

Variable     | Values            | Description
-------------|-------------------|------------
`IPM_REPORT` | `terse` (default) | Aggregate wallclock time, memory usage and flops are reported along with the percentage of wallclock time spent in MPI calls.
             | `full`            | Each HPM counter is reported as are all of wallclock, user, system, and MPI time. The contribution of each MPI call to the communication time is given.
             | `none`            | No report.

## IPM XML Log

IPM can also generate a detailed report in the form of an XML file. By default,
this file is placed in a system directory. The data is imported to the web and
available in your MyNERSC section of the website 24 hours after your job
completed. You can override this behavior by modifying the `IPM_LOGDIR`
environment variable (e.g. `export IPM_LOGDIR=.` in bash). Additionally,
setting the `IPM_LOG` environment variable to `full` provides additional
information as in the `IPM_REPORT` variable options above.

## Displaying IPM results as HTML

By default, IPM records its profiling data in an XML file. One can convert this
to an HTML page using the following command

```console
user@cori02:~$ ipm_parse -html <ipm_file_name>.xml
```

This produces a new directory in the present working directory, in which is
contained several HTML files. One can then view the IPM results in a web
browser by opening the `index.html` file. A portion of this page is shown
below:

![IPM graphical report](images/IPM_output.png)
