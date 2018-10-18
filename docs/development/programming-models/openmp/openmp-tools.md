# Tools for OpenMP

Tools for tuning OpenMP codes to get better performance include:

* [Intel Advisor for Threading Design and Vectorization](#intel-advisor-for-threading-design-and-vectorization)
* [Intel Inspector to Detect Threading and Memory Issues](#intel-inspector-to-detect-threading-and-memory-issues)
* [Intel VTune for Performance Tuning](#intel-vtune-for-performance-tuning)
* [Use Cray Reveal to Insert OpenMP Directives](#use-cray-reveal-to-insert-openmp-directives)

## Intel Advisor for Threading Design and Vectorization

Intel Advisor provides two tools to help the user to modify Fortran, C, and C++
applications to take full performance advantage of today's processors:

* **Vectorization Advisor** is a vectorization optimization tool that lets the
  user identify loops that will benefit most from vectorization and locates
  what is blocking effective vectorization
* **Threading Advisor** is a threading design and prototyping tool that lets
  the user analyze, design, tune, and check threading design options

### Threading Advisor

Key Threading Advisor features include the following:

* **Survey Report** shows the loops and functions where the application spends
  the most time
* **Trip Counts** analysis shows the minimum, maximum, and median number of
  times a loop body will execute, as well as the number of times a loop is
  invoked
* **Annotations** can be inserted by the programmer to mark places in the
  application that are good candidates for later replacement with parallel
  framework code that enables parallel execution using threads
* **Suitability Report** predicts the maximum speedup of the application based
  on the inserted annotations and a variety of modeling parameters
* **Dependencies Report** predicts parallel data sharing problems based on the
  inserted annotations

For information on how to use Intel Advisor and how to execute it on NERSC
systems, visit the page on [Advisor](../../performance-debugging-tools/advisor.md).

## Intel Inspector to Detect Threading and Memory Issues

Intel Inspector is a memory and threading error checking tool for users
developing serial and multithreaded applications. Intel Inspector offers:

* Visibility into individual problems, problem occurrences, and call stack
  information
* Interactive debugging capability
* On-demand memory leak detection
* Memory growth measurement to help ensure that the application uses no more
  memory than expected
* Data race, deadlock, lock hierarchy violation, and cross-thread stack access
  error detection

For information on using Intel Inspector, detecting data races and executing
Inspector on the NERSC systems, visit the page on [Inspector](../../performance-debugging-tools/inspector.md).

## Intel VTune for Performance Tuning

Intel VTune Amplifier is a performance analysis tool for users developing
serial and multithreaded applications. VTune Amplifier helps to identify where
and how an application can benefit from available hardware resources.

VTune Amplifier can be used for several purposes including:

* Finding the most time-consuming functions in the application
* Identifying the best sections of code to optimize to get performance benefits
* Finding synchronization objects that affect the application performance
* Identifying hardware-related issues in the code such as false sharing, cache
  misses, branch misprediction, etc.

For information on detection of false sharing using VTune Amplifier, visit the
page on
[False Sharing Detection using VTune Amplifier](sharing-vtune.md).

For information on how to use VTune on NERSC systems, visit the page on
[VTune](../../performance-debugging-tools/vtune.md).

## Use Cray Reveal to Insert OpenMP Directives

Reveal is an integrated performance analysis and code optimization tool.

Reveal provides:

* Source code navigation using the analysis data of the entire program
* Performance data collected during program execution
* Dependency information for targeted loops
* Variable scoping feedback and suggested compiler directives

For information on how to use Cray Reveal and how to use it on NERSC systems,
visit the page on [Reveal](../../performance-debugging-tools/reveal.md).
