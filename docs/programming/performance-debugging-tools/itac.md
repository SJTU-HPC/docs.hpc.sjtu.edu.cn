# Intel Trace Analyzer and Collector

Intel [Trace
Analyzer](https://software.intel.com/en-us/ita-user-and-reference-guide) and
[Collector](https://software.intel.com/en-us/itc-user-and-reference-guide)
(ITAC) are two tools used for analyzing MPI behavior in parallel applications.
ITAC identifies MPI load imbalance and communication hotspots in order to help
developers optimize MPI parallelization and minimize communication and
synchronization in their applications. Using Trace Collector on Cori must be
done with a command line interface, while Trace Analyzer supports both a
command line and graphical user interface which analyzes the data from Trace
Collector. Currently ITAC is not available on Edison.

Although ITAC works in principle with any MPICH distribution (including Cray
MPI), it functions best when used on applications compiled with Intel MPI. Its
basic features also work with Cray MPI, but some others are currently
unavailable.

## Performing a collection with Trace Collector

To perform a collection with Trace Collector, one can compile her application
as normal, but must link the appropriate Trace Collector libraries to the
application executable at the end of compilation.

### Using Cray MPI

With the Cray compiler wrappers, an application may be linked either statically
or dynamically to Trace Collector. All three programming environments (Intel,
GCC, and Cray) support some (but not all) functionality of Trace Collector.

#### Static Linking

By default, the Cray compiler wrappers (`cc`, `CC`, and `ftn`) link
applications statically. To link an application with Trace Collector
statically, load the "itac" module and amend the link line to include the
appropriate tracing library. E.g., the standard MPI tracing library is `libVT`,
so the link like would look like the following:

```console
cc -o foo.ex foo.c -L$VT_LIB_DIR -lVT $VT_ADD_LIBS
```

where the environment variables `VT_LIB_DIR` and `VT_ADD_LIBS` are defined by
the `itac` module. Trace Collector supports several different collection types,
which are stored in different libraries; to perform a different collection, one
can replace the `-lVT` line with a link to a different library, e.g., `-lVTim`
for tracing MPI load imbalance. The available collections are listed
[here](https://software.intel.com/en-us/node/561272).

After the application has been linked statically to Trace Collector, one can
run the application as normal on a compute node, and upon completion Trace
Collector will generate a collection database in the directory from which the
application was launched.

#### Dynamic Linking

One can also link an application to Trace Collector dynamically. In this case,
the link line will look like the following:

```console
cc -o foo.ex -dynamic foo.c
```

Before the application is launched, one must load the `itac` module and set

```console
LD_PRELOAD=$VT_ROOT/slib/libVT.so
```

where `libVT.so` can be replaced with any of the available collection libraries
(see above). Then the application can be launched as normal and Trace Collector
will produce a collection database after the application completes.

As mentioned above, some features of Trace Collector currently do not work with
the Cray MPI wrappers. Please see the [known issues](#known-issues) section
below for a description of these.

### Using Intel MPI

Instructions for compiling applications with Intel MPI on Cori in general
(without ITAC) can be found
[here](../../programming/compilers/wrappers.md#intel-compiler-wrappers). In
contrast to the Cray wrappers, the Intel MPI wrappers link applications
dynamically by default. As a result, no extra link flags are necessary to link
an application to Trace Collector:

```console
module load impi
mpiicc -o foo.ex foo.c
```

To run Trace Collector on an application compiled with Intel MPI, one then
loads the `itac` module and sets the `LD_PRELOAD` environment variable to one
of Trace Collector's available collection libraries, as with the case of
dynamically linked Cray MPI applications shown above:

```console
LD_PRELOAD=$VT_ROOT/slib/libVT.so
```

where `VT_ROOT` is defined by the `itac` module. As with all applications
compiled with Intel MPI on Cori, one must set the following environment
variable as well:

```console
I_MPI_PMI_LIBRARY=/usr/lib64/slurmpmi/libpmi.so
```

(See [this](../../programming/compilers/wrappers.md#intel-compiler-wrappers)
page for more details regarding running applications compiled with Intel MPI.)

Occasionally, when an application compiled with Intel MPI begins, the following
message will be printed to `STDOUT`:

```console
Unidentified node: Error detected by libibgni.so. Subsequent operation may be unreliable. IAA did not recognize this as an MPI process
```

However, the application should continue to execute normally.

Following code completion, Trace Collector will produce several files in the
directory from which the job was launched, including a file named
`<executable>.stf`, which is the primary collection database. Steps for
analyzing this database are provided
[below](#analyzing-a-trace-collection-with-trace-analyzer).

### Additional options for trace collections

Trace Collector supports a few non-default collection options.

#### Recording source code location and call stacks

Trace Collector can save MPI call stacks and source locations via the
`VT_PCTRACE` environment variable. For example, to collect a function call
stack with a depth of 5, set

```console
VT_PCTRACE=5
```

More information about recording source code location is provided
[here](https://software.intel.com/en-us/node/561289). Note that this feature
does not work with applications compiled with Cray MPI (see the [known
issues](#known-issues) section below).

#### Consolidating collection database files

By default, Trace Collector will produce a large number of files along with the
`.stf` file, which is the primary collection database. However, Trace Collector
can consolidate the entire database into a single `.stf` file by using the
following environment variable:

```console
VT_LOGFILE_FORMAT=STFSINGLE
```

#### Capturing OpenMP behavior

Trace Collector can also capture an application's OpenMP behavior, if the
application is compiled dynamically with the Intel compiler suite (either via
the Cray or Intel MPI wrappers). To do so, one should set the following
environment variables at run time:

```console
INTEL_LIBITTNOTIFY64=$VT_ROOT/slib/libVT.so
KMP_FORKJOIN_FRAMES_MODE=0
```

where `libVT.so` should match the same collection library used in `LD_PRELOAD`.

### Analyzing a trace collection with Trace Analyzer

One can analyze the contents of a trace collection with Trace Analyzer, which
is included in the `itac` module. To launch the GUI on a login node:

```console
traceanalyzer /path/to/collection/database/foo.stf
```

Trace Analyzer will begin by showing a summary page indicating what fraction of
the run time the application spent in user code vs MPI calls. If the
application was traced without OpenMP tracing support (see
[above](#capturing-openmp-behavior)), then Trace Analyzer will report that the
total time spent in OpenMP regions is zero, even if the application was
compiled and run with OpenMP threading enabled.

[IMAGE]

From here one can navigate to various windows which display detailed
information about the MPI communication in the application. For example, the
"event timeline" depicts the MPI traffic over time among all process, which can
help with identifying regions where the application experiences load imbalance
or a communication hotspot among MPI processes.

[IMAGE]

Similarly, the "quantitative timeline" shows the fraction of time spent in MPI
vs user code over the duration of the run:

[IMAGE]

Clicking "Show advanced ..." in the pane on the right expands the description
of MPI bottlenecks in the application (e.g., late sender, early receiver), and
also provides an illustration and explanation of how that particular bottleneck
tends to occur:

[IMAGE]

Trace Analyzer also supports a command line interface, which can produce
machine-readable text files from a `.stf` collection file for further analysis.
For example, one can compute statistics regarding messages, collectives, and
functions via

```console
traceanalyzer --cli --messageprofile -o messages.txt ./foo.stf   # messages
traceanalyzer --cli --collopprofile -o messages      ./foo.stf   # collective operations
traceanalyzer --cli --functionprofile -o messages    ./foo.stf   # functions
```

More information about using the Trace Analyzer CLI can be found
[here](https://software.intel.com/en-us/node/561584).

## Known issues

Currently ITAC has a few issues when running on Cori:

- While the tracing itself slows down application execution only mildly, the
conclusion of the tracing collection - writing the data to the `.stf` file(s) -
is extremely slow, and can take much longer than the application itself. If
possible, consider tracing a small, representative problem, in order to keep
the `.stf` file generation short.

- OpenMP behavior can be captured only if the application is compiled with the
Intel compilers (either via the Cray or Intel MPI wrappers) and if the
environment variables shown [above](#capturing-openmp-behavior) are set.

- If attempting to capture call stack information with the `VT_PCTRACE`
environment variable, an application compiled with the Cray MPI compiler
wrappers will crash upon completion, when Trace Collector attempts to write the
collection database to disk. To capture call stack and source code location
information, one must compile and run the application with Intel MPI (see
[above](#using-intel-mpi)). Attempting to use `VT_PCTRACE` with a Cray MPI
application will result in the following error message:

```console
[0] Intel(R) Trace Collector INFO: Writing tracefile foo.stf in /global/cscratch1/sd/user/foo
xpmem_attach error: : No such file or directory
```

## Resources

- [Intel Trace Collector User and Reference Guide](https://software.intel.com/en-us/itc-user-and-reference-guide)
- [Intel Trace Analyzer User and Reference Guide](https://software.intel.com/en-us/ita-user-and-reference-guide)
- [Intel Cluster Tools on NERSC Systems (PDF)](https://www.nersc.gov/assets/Uploads/Intel-Cluster-Tools-Mar2017.pdf)
- [Intel Cluster Tools in a Cray Environment](https://software.intel.com/en-us/articles/intel-mpi-itac-and-mps-in-a-cray-environment)
