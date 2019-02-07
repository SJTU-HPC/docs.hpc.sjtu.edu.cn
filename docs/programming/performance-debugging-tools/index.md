# Performance and Debugging Tools

NERSC provides many popular debugging and profiling tools. Some of them are
general-purpose tools and others are geared toward more specific tasks.

A quick guideline on when to use which debugging tool is as follows:

* **DDT** and **TotalView**: general purpose parallel debuggers allowing users
  to interactively control the pace of execution of a program using a graphical
  user interface
* **gdb**: a serial command line mode debugger; can be useful in quickly
  examining core files to see where the code crashed (DDT and TotalView can be
  used for this purpose as well)
* **STAT**: used for obtaining call backtraces for all parallel tasks from a
  live parallel application and displaying a call tree graphically, showing
  where each task is executing; useful in debugging a hung application
* **ATP**: used for generating call backtraces for all parallel tasks when a
  code crashes; useful in debugging a hung application; can be a good starting
  point if a code crashes with little hint left behind
* **CCDB** and **lgdb**: run two versions of a code (e.g., one working version
  and an incorrect version, or a code run with two different numbers of tasks)
  side by side to find out where the two runs start to generate diverging
  results
* **Valgrind**: a suite of debugging and profiling tools; the best known tool
  is memcheck which can detect memory errors or memory leaks; other tools
  include cache profiling, heap memory profiling tools, and more
* **Intel Inspector**: a memory and threading error-checking tool for users
  developing serial and multithreaded applications on Windows and Linux
  operating systems

Two "getting started" tutorials on some debugging tools are:

* [Parallel Debugging Tools, New User Training, February 2017](http://www.nersc.gov/assets/Uploads/ParallelDebuggingTools-201702.pdf)
* [Debugging Tools, NUG 2014](http://www.nersc.gov/assets/pubs_presos/12a-DebuggingTools-NUG2014.pdf)

A quick guideline for performance analysis tools is as follows:

* **IPM**: a low-overhead easy-to-use tool for getting hardware counters data,
  MPI function timings, and memory usage
* **CrayPat**: a suite of sophisticated Cray tools for a more detailed
  performance analysis which can show routine-based hardware counter data, MPI
  message statistics, I/O statistics, etc; in addition to getting performance
  data deduced from a sampling method, tracing of certain routines (or library
  routines) can be performed for better understanding of performance statistics
  associated with the selected routines
* **MAP**: a sampling tool for performance metrics; time series of the
  collected data for the entire run of the code is displayed graphically, and
  the source code lines are annotated with performance metrics
* **Intel VTune Amplifier XE**: a GUI-based tool that can find performance
  bottlenecks

A "getting started" tutorial on some performance analysis is:

* [More Profiling Tools at NERSC, NUG 2015](http://www.nersc.gov/assets/Uploads/NUG-hackathon-Profiling-Tools.pdf)

For more information about how to use a tool, click on the relevant item below:

[DDT](ddt.md): DDT is a parallel debugger that can be run with up to 8,192
processors. It has features similar to TotalView and and a similarly intuitive
user interface. Read more [here](ddt.md).

[TotalView](totalview.md): TotalView, from Rogue Wave Software, is a parallel
debugging tool that can be run with up to 512 processors. It provides an X
Windows-based Graphical User Interface and a command line interface. Read more
[here](totalview.md).

[GDB](gdb.md): GDB can be used to quickly and easily examine a core file that
was produced when an execution crashed to give an approximate traceback. Read
more [here](gdb.md)

[STAT and ATP](stat_atp.md): STAT (the Stack Trace Analysis Tool) is a highly
scalable, lightweight tool that gathers and merges stack traces from all of the
processes of a parallel application. ATP (Abnormal Termination Processing)
automatically runs STAT when the code crashes. Read more [here](stat_atp.md).

[CCDB and lgdb](lgdb_ccdb.md): lgdb (Cray Line Mode Parallel Debugger) is a
GDB-based parallel debugger, developed by Cray. It allows programmers to either
launch an application or attach to an already-running application that was
launched with `srun` in order to debug the parallel code in command-line mode.
Read more [here](lgdb_ccdb.md).

[Valgrind](valgrind.md): The Valgrind tool suite provides several debugging and
profiling tools that can help make your programs faster and more correct. The
most popular tool is Memcheck, which can detect many memory-related errors that
are common in C and C++ programs. Read more [here](valgrind.md).

[IPM](ipm.md): IPM is a portable profiling infrastructure which provides a
high-level report on the execution of a parallel job. IPM reports hardware
counter data, MPI function timings, and memory usage. Read more [here](ipm.md).

[CrayPat](craypat.md): CrayPat is a performance analysis tool provided by Cray
for the XT and XE platforms. Read more [here](craypat.md).

[MAP](map.md): Allinea MAP is a parallel profiler with a simple GUI. It can be
run with up to 512 processors. Read more [here](map.md).

[VTune](vtune.md): Intel VTune is a GUI-based tool for identifying performance
bottlenecks and getting performance metrics. Read more [here](vtune.md).

[Application Performance Snapshot (APS)](aps.md): Application Performance
Snapshot (APS) is a lightweight open source profiling tool developed by the
Intel VTune developers. Use APS for a quick view into a shared memory or MPI
application's use of available hardware (CPU, FPU, and memory). APS analyzes
your application's time spent in MPI, MPI and OpenMP imbalance, memory access
efficiency, FPU usage, and I/O and memory footprint. Read more [here](aps.md).

[LIKWID](likwid.md): LIKWID ("Like I Knew What I'm Doing") is a lightweight
suite of command line utilities. By reading the the MSR (Model Specific
Register) device files, it renders reports for various performance metrics such
as FLOPS, bandwidth, load to store ratio, and energy. Read more
[here](likwid.md). Read more [here]().

[darshan](darshan.md): Darshan is a light weight I/O profiling tool capable of
profiling POSIX I/O, MPI I/O and HDF5 I/O. Read more [here](darshan.md).

[Intel Advisor](advisor.md): Intel Advisor provides two workflows to help
ensure that Fortran, C and C++ applications can make the most of today's
processors: vectorization advisor and threading advisor. Read more
[here](advisor.md).

[Intel Inspector](inspector.md): Intel Inspector is a memory and threading
error-checking tool for users developing serial and multithreaded applications
on Windows and Linux operating systems. Read more [here](inspector.md).

[Intel Trace Analyzer and Collector](itac.md): Intel Trace Analyzer and
Collector (ITAC) are two tools used for analyzing MPI behavior in parallel
applications. ITAC identifies MPI load imbalance and communication hotspots in
order to help developers optimize MPI parallelization and minimize
communication and synchronization in their applications. Read more
[here](itac.md).
