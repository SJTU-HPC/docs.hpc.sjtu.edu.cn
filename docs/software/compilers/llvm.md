#LLVM (C/C++)

##Description
The LLVM project is a collection of modular compiler and toolchain technologies. Note that the support for LLVM at NERSC is experimental.


##Availability
The llvm core libraries along with the clang compiler is only available on Cori at the moment. It is compiled against the gcc and thus cannot be used with intel-based programming environments.

##Using the Clang Compiler Cori
In order to enable clang compiler, first make sure to load the gnu programming environment

```Shell
module load PrgEnv-gnu
module load gcc
module load llvm/<version>
```

where `module avail llvm` displays which versions are currently installed.

LLVM does not only provide a compiler infrastructure, it also provides a language independent instruction set and type system. When code is compiled with clang, it is first translated into a powerful intermediate representation, similar to assembly code but architecture independent. 
That way, LLVM is for example able to translate all object files and libraries of a large project into the intermediate representation and then perform an optimization step at link time across modules, also known as link-time-optimization (LTO). 

###Hints for Building Applications
The plain use of the clang compiler is not much different from using the GNU or Intel compilers.
For optimizations, the common compiler flags such as `-On` can be used, where $n=0,1,2,3$ is the optimization level. 

The clang compiler/llvm framework also supports a variety of other linker flags which can help debugging code:

* AddressSanitizer (`-fsanitize address`): this enables the memory error detector. It provides out-of-bounds, use-after-free, use-after-return, double-free checks as well as some rudimentary memory leak detector. The slowdown caused by the AddressSanitizer is approximately 2x.
* ThreadSanitizer (`-fsanitize-threads`): this option aims at detecting possible deadlocks and race conditions. The feature is currently in beta stage. Note that the slowdown caused by the ThreadSanitizer is 5x-15x. Furthermore, memory consumption of ThreadSanitized codes is higher than for un-sanitized codes.
When the above mentioned options are enabled, make sure to use clang/clang++ as the final linker and not ld, so that all necessary libraries are linked. The memory overhead is approximately 5x plus 1MB per thread!
* MemorySanitizer (`-fsanitize-memory`): this option tries to detect uninitialized reads.  The slowdown is currently 3x. The memory consumption is 2x compared to the un-sanitized code.
For all mentioned options, in order to enable a nicer stack trace, add `-fno-omit-frame-pointer`. For enabling a complete stack trace, add `-fno-optimize-sibling-calls` and avoid using optimization levels higher than `-O1`.

##Fortran Support with Flang
The module `llvm/5.0.0-gnu-flang` contains `flang` in addition to `clang`. This compiler is supposed to compile fortran code and does support OpenMP. However, please note that this compiler is even more experimental than `clang` itself. Furthermore, it does not find the standard headers and fortran modules by default. Therefore, those need to be added manually to the compilation flags using `-I`. Please use `module show llvm/5.0.0-gnu-flang` to find the corresponding include paths. 

##Documentation
For questions about using the Intel compilers at NERSC contact the [consulting services](consult@nersc.gov).

For more information, see [LLVM](http://llvm.org/) and [Clang](http://clang.llvm.org/) websites.