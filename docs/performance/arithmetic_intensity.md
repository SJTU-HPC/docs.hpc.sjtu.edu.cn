# Measuring Arithmetic Intensity

Arithmetic intensity is a measure of floating-point operations (FLOPs)
performed by a given code (or code section) relative to the amount of
memory accesses (Bytes) that are required to support those
operations. It is most often defined as a FLOP per Byte ratio
(F/B). This application note provides a methodology for determining
arithmetic intensity using
Intel's
[Software Development Emulator Toolkit (SDE)](https://software.intel.com/en-us/articles/intel-software-development-emulator) and
[VTune Amplifier (VTune)](https://software.intel.com/en-us/intel-vtune-amplifier-xe) tools. A
NERSC tutorial on using VTune can be
found
[here](../programming/performance-debugging-tools/vtune.md). This
method can also be used to determine arithmetic intensity for use in
the
[Roofline Performance Model](https://crd.lbl.gov/departments/computer-science/PAR/research/roofline/).

Historically, processor manufacturers have provided counters for FLOPs
and/or Bytes and profiling tools to support the F/B calculation. Some
modern processors such as Intel's Haswell (used in Cori) do not
provide counters for FLOPs. However, Intel's SDE can be used to count
floating-point instructions in addition to core-level memory accesses,
and VTune can be used to count data accesses to the uncore (off-chip
DRAM DIMMs).

The SDE dynamic instruction tracing capability, and in particular the
mix histogram tool, captures dynamic instructions executed,
instruction length, instruction category and ISA extension
grouping. Intel has developed a methodology
for
[calculating FLOPs with SDE](https://software.intel.com/en-us/articles/calculating-flop-using-intel-software-development-emulator-intel-sde).
In general the following uses the method "Instructions to Count
Unmasked FLOP" from Intel.

This application note provides additional instruction on how to only
capture traces around certain key segments of a code. This is critical
for real applications as both SDE and VTune collect traces that can
use large amounts of disk space if tracing is enabled for more than a
few minutes. And maybe more importantly, post-processing the traces
can take an intractable amount of time.

An example command line for SDE is:

```console
$ srun -n 4 -c 6 sde -ivb -d -iform 1 -omix my_mix.out -i -global_region -start_ssc_mark 111:repeat -stop_ssc_mark 222:repeat -- foo.exe
```

where:

  - for Cori use `-hsw` for Haswell or `-knl` for KNL processors
  - `-d` specifies to only collect dynamic profile information
  - `-iform 1` turns on compute ISA iform mix
  - `-omix` specifies the output file (and turns on `-mix`)
  - `-i` specifies that each process will have a unique file name based on
     process ID (needed for MPI)
  - `-global_region` will include any threads spawned by a process (needed for
    OpenMP)

An example command line for VTune is:

```slurm
$ srun -n 4 amplxe-cl -start-paused -r my_vtune -collect memory-access -finalization-mode=none -trace-mpi -- foo.exe
```

SDE allows tracing to only occur around specified sections of code
using `__SSC_MARK` macros.  The `-start_ssc_mark` and `-stop_ssc_mark`
flags tell SDE to only trace sections of code between these marks. In
addition, VTune has calls to start and stop tracing. This is
illustrated below for an example kernel from the STREAM
benchmark. Note that both use a double underscore prefix which may not
be obvious when viewed in the web browser.

```C
// Code must be built with appropriate paths for VTune include file
// (ittnotify.h) and library (-littnotify)

#include <ittnotify.h>

__SSC_MARK(0x111); // start SDE tracing, note it uses 2 underscores
__itt_resume(); // start VTune, again use 2 underscores

for (k=0; k<NTIMES; k++) {
 #pragma omp parallel for
 for (j=0; j<STREAM_ARRAY_SIZE; j++)
 a[j] = b[j]+scalar*c[j];
}

__itt_pause(); // stop VTune
__SSC_MARK(0x222); // stop SDE tracing
```

In addition to limiting the collection of traces to certain code segments or
kernels, it is also desirable to use start and stop markers to limit the amount
of data collected as SDE and VTune tracing analysis can generate a large volume
of data if the total execution time is excessive. This may require trial and
error to determine where in the code to enable tracing, and to limit the amount
of time tracing is enabled. It's desirable that tracing is enabled for only a
few minutes or less. To use VTune markers in Fortran codes, see [this Intel
article](https://software.intel.com/en-us/articles/how-to-call-resume-and-pause-api-from-fortran-code).

SDE will create a file for every process that is created by the application.
For example, in an MPI code SDE will create a file for each MPI process (one
per rank) and if the application contains threads (e.g. OpenMP) those will be
encapsulated into the same file (`-global_region` enables this).

!!! note "Example"

    All of the above is best illustrated with an example found
    [here](https://bitbucket.org/dwdoerf/stream-ai-example). This example uses a
    modified version of the STREAM benchmark (it also contains a directory with an
    example Jacobi Method code utilizing fortran bindings). The example can be
    accessed using git, e.g.:

    ```console
    $ git clone https://bitbucket.org/dwdoerf/stream-ai-example.git
    $ cd stream-ai-example
    $ module load sde
    $ module load vtune
    $ make
    ```

    You may have to modify the Makefile to suit your needs. Once you've
    successfully built the executable, `stream.exe`, (an MPI+OpenMP code) you can
    use the example batch script `stream-ai.sh` (which may need to be modified to
    suit your target and submit a job which executes `stream.exe` in three modes: 
	without instrumentation that can be used for accurate timing estimates, 
	then under the control of SDE, and finally under the control of VTune.

    ```console
    $ sbatch stream-ai.sh
    < wait for job to finish, this may take a few to several minutes depending on demand>
    ```

When the job completes, SDE will have created several files, one for
each rank, starting with `sde_`. VTune will have created one or more
directories (one for each node used) starting with `vtbw_`.

SDE provides a wealth of information in each of its respective output
files.  For arithmetic intensity the floating-point instruction and
data access instructions are of primary interest. You can use the
links to the Intel documentation provided above to better understand
these details, or you can use the script provided in the example to
parse the output files. The script prints the instruction counts
followed by a summary of total floating-point operations and total
bytes.

!!! tip "Parsing SDE output"
    You want to pass the script all files generated by SDE (one
    per rank) on the command line. E.g.:

    ```console
    $ ./parse-sde.sh sde_2p16t*
    Search stanza is "EMIT_GLOBAL_DYNAMIC_STATS"
    elements_fp_single_1 = 0
    elements_fp_single_2 = 0
    elements_fp_single_4 = 0
    elements_fp_single_8 = 0
    elements_fp_single_16 = 0
    elements_fp_double_1 = 2960
    elements_fp_double_2 = 0
    elements_fp_double_4 = 999999360
    elements_fp_double_8 = 0
    --->Total single-precision FLOPs = 0
    --->Total double-precision FLOPs = 4000000400
    --->Total FLOPs = 4000000400
    mem-read-1 = 8618384
    mem-read-2 = 1232
    mem-read-4 = 137276433
    mem-read-8 = 149329207
    mem-read-16 = 1999998720
    mem-read-32 = 0
    mem-read-64 = 0
    mem-write-1 = 264992
    mem-write-2 = 560
    mem-write-4 = 285974
    mem-write-8 = 14508338
    mem-write-16 = 0
    mem-write-32 = 499999680
    mem-write-64 = 0
    --->Total Bytes read = 33752339756
    --->Total Bytes written = 16117466472
    --->Total Bytes = 49869806228
    ```

`amplxe-cl` stores its trace data in directories (one per node). The example
script includes `-finalization-mode=none` as the finalize step is I/O intensive
and can take a long time. It is best done on an external login node. (Please
see the [VTune](../programming/performance-debugging-tools/vtune.md) page for
more information about using VTune effectively. The following `amplxe-cl`
command creates a summary report which is then redirected to another file. The
`parse-vtune.sh` script is provided to extract the uncore counter data for all
nodes and then print a summary of the total data traffic to the DDR memory.
Note that each uncore count is a 64 byte cache line, which is reflected in the
total bytes reports. Also note that you will need to run a summary report for
each directory (one per node) created during data collection. In this example,
only a single directory was created. Even though there is only a single summary
file for this example, in the 2nd step of executing `parse-vtune.sh` a wildcard
is used to illustrate that all summary files need to be specified on the
command line for a multi-node result.

```console
$ amplxe-cl -report hw-events -group-by=package -r vtbw_2p16t_13568698.nid00619 -column=UNC_M_CAS_COUNT -format=csv -csv-delimiter=comma > vtbw_2p16t_13568698.summary
--> lots of VTune output ....
--> Repeat for each directory created during data collection, one per node
$ ./parse-vtune2018.sh vtbw_2p16t*.summary
Search stanza is "Uncore"
UNC_M_CAS_COUNT.RD[UNIT0] = 0
UNC_M_CAS_COUNT.RD[UNIT1] = 0
UNC_M_CAS_COUNT.RD[UNIT2] = 127252047
UNC_M_CAS_COUNT.RD[UNIT3] = 126829175
UNC_M_CAS_COUNT.RD[UNIT4] = 0
UNC_M_CAS_COUNT.RD[UNIT5] = 0
UNC_M_CAS_COUNT.RD[UNIT6] = 126861782
UNC_M_CAS_COUNT.RD[UNIT7] = 127247700
UNC_M_CAS_COUNT.WR[UNIT0] = 0
UNC_M_CAS_COUNT.WR[UNIT1] = 0
UNC_M_CAS_COUNT.WR[UNIT2] = 62611982
UNC_M_CAS_COUNT.WR[UNIT3] = 62274525
UNC_M_CAS_COUNT.WR[UNIT4] = 0
UNC_M_CAS_COUNT.WR[UNIT5] = 0
UNC_M_CAS_COUNT.WR[UNIT6] = 62389886
UNC_M_CAS_COUNT.WR[UNIT7] = 62519044
--->Total Bytes read = 32524205056
--->Total Bytes written = 15986907968
--->Total Bytes = 48511113024
```

Arithmetic intensity (AI) can now be calculated. Nominally, it's the ratio of
"Total FLOPs" as reported by SDE to "Total Bytes" as reported by VTune.

AI (DRAM) = 4000000400 / 48511113024 = 0.0825

Alternatively, it can be calculated using the "Total Bytes" as seen by the core
L1 caches and reported by SDE.

AI (L1) = 4000000400 / 49869806228 = 0.0802

Since STREAM has very little reuse of data, the AI of the two is approximately
the same. For real codes, the AI (L1) will most likely be significantly lower.
AI (L1) divided by AI (DRAM) can be used as a "bandwidth bound" figure of
merit, the closer to 1.0 the more bandwidth bound the application.

## Change Log

- June 9, 2018
    - Brought up to date to reflect the latest system software installed at
      NERSC. In particular the example script is now compatible with the latest
      configuration of Slurm and the amplxe-cl arguments are compatible with VTune
      Amplifier 2018. In addition, the STREAM example and all associated scripts to
      automate the method is now available via git on
      [BitBucket](https://bitbucket.org/dwdoerf/stream-ai-example).
- May 18, 2016
    - Updated the example with a new `parse-vtune.sh` file which supports
      parsing for the Knights Landing MCDRAM counter output of VTune. You
      need to use at least VTune version 2016 Update 3 to collect MCDRAM counters.
- March 23, 2016
    - Added link to Fortran bindings for `SSC_MARK`
    - Fixed bug in `parse_sde.sh` which gave incorrect total FLOP count when using
      `single_precision`
- March 15, 2016
    - Example download now supports Slurm
    - Example download now supports Haswell's (Cori) fused multiply-add,
      multiply-subtract instructions
    - Updated for use with VTune Amplifier XE 2016
    - Example download is now a single MPI+OpenMP code and supports multi-node
      execution with SDE and VTune
- March 6, 2016
    - Updated SDE parse script in the example download to handle fused
      multiply-add and multiply-subtract instructions
