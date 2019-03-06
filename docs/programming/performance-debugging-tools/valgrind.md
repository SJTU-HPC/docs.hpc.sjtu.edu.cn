# VALGRIND

## Description

The Valgrind tool suite provides a number of debugging and profiling
tools that help you make your programs faster and more correct. The
most popular of these tools is called Memcheck which can detect
many memory-related errors and memory leaks.

## Using Valgrind

### Prepare Your Program

Compile your program with -g to include debugging information so
that Memcheck's error messages include exact line numbers. Using
-O0 is also a good idea, if you can tolerate the slowdown. With -O1
line numbers in error messages can be inaccurate, although generally
speaking running Memcheck on code compiled at -O1 works fairly well,
and the speed improvement compared to running -O0 is quite significant.
Use of -O2 and above is not recommended as Memcheck occasionally
reports uninitialised-value errors which don't really exist.

### Load Module

```shell
nersc$ module load valgrind
```

### Running Serial Programs

If you normally run your program like this:

```shell
nersc$ ./myprog arg1 arg2
```

Use this command line:

```shell
nersc$ valgrind --leak-check=yes ./myprog arg1 arg2
```

Memcheck is the default tool. The --leak-check option turns on the
detailed memory leak detector.

Your program will run much slower (eg. 20 to 30 times) than normal,
and use a lot more memory. Memcheck will issue messages about memory
errors and leaks that it detects.

### Running Parallel Programs

In your batch script, simply 1. load the module; 2. add "valgrind"
in front of your command. For example, your srun line will be
replaced by the following:

```shell
nersc$ module load valgrind
nersc$ srun -n 24 valgrind --leak-check=yes ./myprog arg1 arg2
```

## Unrecognized Instructions

When using Valgrind to debug your code, you may occasionally encounter
error messages of the form:

```shell
nersc$ valgrind: Unrecognised instruction at address 0x6b2f2b
```

accompanied by your program raising SIGILL and exiting. While this
may be bug in your program (which caused it to jump to a non-code
location), it may also be an instruction that is not correctly
handled by Valgrind. For example, when using the Intel compilers
on Edison, you may find this error is raised within the
`__intel_sse4_strtok()` library function, which is in turn using the
currently (Nov. 2014) unhandled pcmpistrm SSE4.2 instruction.

There are a couple of ways to work around issues related to
unrecognized instructions. The simplest is often to make sure that
the code you are debugging is compiled with the minimum level of
optimization necessary in order to reproduce the bug you are
investigating. This is in general good practice, and will avoid the
use of more obscure (typically SIMD) instructions which are more
likely to be unhandled.

If you find that this does not work, you may wish to try a different
compiler - this can affect both the nature of the optimizations
peformed on your code, as well as the libraries to which your code
is linked. In the specific example above with `__intel_sse4_strtok`,
switching to the GNU programming environment and recompiling the
code being debugged remedied this situation.

## Link to Outside Documentation

This page is based on the "Valgrind Quick Start Page". For more
information about valgrind, please refer to
[http://valgrind.org/](http://valgrind.org/).

For questions on using Valgrind at NERSC
contact [NERSC Consulting](https://help.nersc.gov).
