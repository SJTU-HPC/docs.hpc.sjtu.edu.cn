# LGDB and CCDB

## Parallel Debugging with lgdb

lgdb (Cray Line Mode Parallel Debugger) is a GDB-based parallel
debugger, developed by Cray. It allows programmers to either launch
an application or attach to an already-running application that was
launched with aprun, to debug the parallel code in command-line
mode. These features can be useful, but you will probably want to
use a more powerful GUI-based debuggers instead.

Below is an example of running lgdb for a parallel application:

```shell
nersc$ salloc -N 1 -t 30:00 -q debug
[snip]
nersc$ module rm altd                       # Remove altd because it interferes
nersc$ module load cray-lgdb
nersc$ lgdb
[snip]
dbg_all> launch $pset{8} ./hello_mpi   # Launch 'hello_mpi' using 8 tasks which I name '$pset'
dbg_all> break hello_mpi.c:21          # Set a breakpoint at line 21 of hello_mpi.c
dbg_all> continue                      # Run

dbg_all> print $pset::myRank           # Print the value of 'myRank' for all processes in $pset
pset[0]: 0
[snip]
pset[7]: 7
dbg_all> print $pset{3}::myRank        # Print the value of 'myRank' for process 3 ($pset[3]) only
pset[3]: 3
```

## Comparative Debugging

What makes lgdb (and CCDB) unique is the comparative debugger
technology, which enables programmers to run two executing applications
side by side and compare data structures between them. This allows
users to run two versions of the same application simultaneously,
one that you know generates the correct results and another that
gives incorrect results, to identify the location where the two
codes start to deviate from each other.

CCDB is a GUI tool for comparative debugging. It runs lgdb underneath.
Its interface makes it easy for users to interact with lgdb for
debugging. Users are advised to use CCDB over lgdb.

To compare something between two applicaions, you need to let lgdb
and CCDB know the name of the variable, and the location where a
comparison is to be made, and how the data is distributed over MPI
processes. For these, lgdb and CCDB use 3 entities:

-   __PE set__: A set of MPI processes
-   __Decomposition__: How a variable is distributed over the MPI processes in a PE set
-   __Assertion script__: A collection of mathematical relationships
    (e.g., equality) to be tested

Please see the man page `man lgdb` for usage information about
lgdb's comparative debugging feature. Cray's 'XC Series Programming
Environment User Guide,' available from [here]((https://pubs.cray.com/))
provides info on how to use the tool.  The tutorial manual uses
example codes that are provided in the lgdb distribution package.
You can build executables using the provided script as follows:

```shell
nersc$ module load cray-lgdb
nersc$ cp -R $CRAY_LGDB_DIR/demos/hpcc_demo .  # copy the entire directory to the current directory
nersc$ cd hpcc_demo
nersc$ module swap PrgEnv-intel PrgEnv-cray    # its Makefile uses the Cray compiler
nersc$ ./build_demo.sh
```

This will build two binaries, `hpcc_working` and `hpcc_broken`.

### CCDB Example

To use:

```shell
nersc$ salloc -N 2 -t 30:00 -q debug           # request enough nodes for launching two applications
nersc$ module load cray-ccdb
nersc$ ccdb
```

Then, launch two applications from the CCDB window.

Below is an assertion script which tests whether the 6 variables
have the same values between the applications, at line 418 of
`HPL_pdtest.c`. It shows that `resid0` and `XmormI` have different values
between the applications and therefore both applications have stopped
at line 418.

![ccdbpass1assertresid12](images/ccdbpass1assertresid12.png)
