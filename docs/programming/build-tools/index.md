# Build tools

## GNU Autoconf

GNU Autoconf is a tool for producing configure scripts for building,
installing and packaging software on computer systems where a Bourne
shell is available.

At NERSC one should typically replace `./configure` with

```shell
./configure CC=cc CXX=CC FC=ftn F77=ftn
```

in order to use the [compiler wrappers](../compilers/wrappers.md).

It is often useful to see what additional options are available:

```shell
./configure --help | less
```

Examples of common options:

* `--enable-mpi`
* `--enable-hdf5`

## GNU Make

[Make](https://en.wikipedia.org/wiki/Make_(software)) is a common
build automation tool in wide use by Unix like systems. (GNU
Make)[https://www.gnu.org/software/make/] is the most widespread
implementation and the default for Mac OS X and is the default for
most Linux distributions.

A typical Makefile:

```make
TARGET = test
LIBS =
CC = cc
CFLAGS = -g -Wall -qopenmp

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.c, %.o, $(wildcard *.c))
HEADERS = $(wildcard *.h)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -Wall $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
```

!!! tip
	MPI codes distributed via make have lines like `CC = mpicc`. In
	most cases it is sufficient to change these lines to `CC = cc`.

## CMake

[CMake](https://cmake.org) is an open-source, cross-platform family of
tools designed to build, test, and package software. It is build-system generator
-- on NERSC machines, CMake will generate UNIX Makefiles, by default -- and there is
no need to enable CMake in cross-compilation mode, e.g.
`cmake -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment <etc>`.

### CMake Recommendations

Use modern CMake interface libraries. "INTERFACE" libraries are libraries that do not
create an output but instead hold specific build features, e.g. include paths, build flags,
link flags, link libraries, compile definitions, etc.

In order to "use" interface libraries, just "link" the interface library against the target you want
the properties to build properties to be applied to.

NOTE: in below, replace `foo-` with your project name or custom prefix. A prefix is not necessary but
prevents potential conflicts with other targets, e.g. if you use target `compile-options` in one of
your projects named `foo` and also use `compile-options` as a target in another project named `bar`, then
when using `foo` and `bar` in the same project, one of the `add_library(compile-options INTERFACE)` commands
will fail because the target `compile-option` already exists.

#### MPI

- Provided by Cray compiler wrappers
    - There is no need to do anything unique
    - If you have an existing `find_package(MPI)`, this will succeed (i.e. `MPI_FOUND` will be true)
but variables such as `MPI_CXX_FLAGS`, etc. will be empty.

For general MPI support when not using the Cray compiler wrappers, you can use the following:

```cmake
# helper macro to convert string to a CMake list (semi-colon delimited)
macro(to_list _VAR _STR)
    STRING(REPLACE "  " " " ${_VAR} "${_STR}")
    STRING(REPLACE " " ";" ${_VAR} "${_STR}")
endmacro(to_list _VAR _STR)

# FindMPI.cmake is provided by CMake, located in "<install-prefix>/share/cmake-<version>/Modules"
find_package(MPI)

if(MPI_FOUND)
    # include directories
    target_include_directories(foo-mpi INTERFACE ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

    # link libraries
    target_link_libraries(foo-mpi INTERFACE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

    if(MPI_EXTRA_LIBRARY)
        target_link_libraries(foo-mpi INTERFACE ${MPI_EXTRA_LIBRARY})
    endif()

    # convert these strings to CMake lists
    to_list(MPI_C_COMPILE_OPTIONS   "${MPI_C_COMPILE_FLAGS}")
    to_list(MPI_CXX_COMPILE_OPTIONS "${MPI_CXX_COMPILE_FLAGS}")
    to_list(MPI_C_LINK_OPTIONS      "${MPI_C_LINK_FLAGS}")
    to_list(MPI_CXX_LINK_OPTIONS    "${MPI_CXX_LINK_FLAGS}")

    # set interface target compile options
    target_compile_options(foo-mpi INTERFACE
        $<$<COMPILE_LANGUAGE:C>:${MPI_C_COMPILE_OPTIONS}>
        $<$<COMPILE_LANGUAGE:CXX>:${MPI_CXX_COMPILE_OPTIONS}>))

    # set interface target link options
    set_target_properties(foo-mpi PROPERTIES
        INTERFACE_LINK_OPTIONS
            $<$<COMPILE_LANGUAGE:C>:${MPI_C_LINK_OPTIONS}>
            $<$<COMPILE_LANGUAGE:CXX>:${MPI_CXX_LINK_OPTIONS}>)
endif()

# later, when creating actual library or executable:
add_executable(bar bar.cpp)
target_link_libraries(bar PUBLIC foo-mpi)
```

#### OpenMP

- Standard package finding for OpenMP:

```cmake
find_package(OpenMP)
add_library(foo-openmp INTERFACE)

if(OpenMP_FOUND)
    target_compile_options(foo-openmp INTERFACE
        $<$<COMPILE_LANGUAGE:C>:${OpenMP_C_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    target_link_libraries(foo-openmp INTERFACE
        ${OpenMP_C_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
endif()

# later, when creating actual library or executable:
add_library(bar SHARED bar.cpp)
target_link_libraries(bar PUBLIC foo-openmp)
```

#### Threading

- If using non-OpenMP threading, the Intel compilers require adding `-pthread` compilation flag

```cmake
# FindThreads.cmake options
set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# provided by CMake
find_package(Threads)

# interface library for threading
add_library(foo-threading INTERFACE)

if(THREADS_FOUND)
    if(THREADS_HAVE_PTHREAD_ARG)
        target_compile_options(foo-threading INTERFACE
            $<$<COMPILE_LANGUAGE:C>:-pthread>
            $<$<COMPILE_LANGUAGE:CXX>:-pthread>)
    endif()
    target_link_libraries(foo-threading INTERFACE ${CMAKE_THREAD_LIBS_INIT})
endif()

# later, when creating actual library or executable:
add_library(bar SHARED bar.cpp)
target_link_libraries(bar PUBLIC foo-threading)
```

#### Language Standards and Features

CMake knows the appropriate build flag(s) for the different language standards of "first-class" languages,
e.g. C, C++, Fortran, and (in CMake 3.8+) CUDA.

These can be set globally:

```cmake
# helpful for setting the language standard
set(CMAKE_C_STANDARD    11 CACHE STRING "C language standard")
set(CMAKE_CXX_STANDARD  11 CACHE STRING "C++ language standard")
set(CMAKE_CUDA_STANDARD 11 CACHE STRING "CUDA language standard")

set(CMAKE_C_STANDARD_REQUIRED    ON CACHE BOOL "Require the C language standard to set")
set(CMAKE_CXX_STANDARD_REQUIRED  ON CACHE BOOL "Require the C++ language standard to set")
set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL "Require the CUDA language standard to set")

set(CMAKE_C_EXTENSIONS    OFF CACHE BOOL "Enable/disable extensions, e.g. -std=gnu11 vs. -std=c11")
set(CMAKE_CXX_EXTENSIONS  OFF CACHE BOOL "Enable/disable extensions, e.g. -std=gnu++11 vs. -std=c++11")
set(CMAKE_CUDA_EXTENSIONS OFF CACHE BOOL "Enable/disable extensions")
```

Or on a per-target basis:

```cmake
add_library(foo SHARED foo.cu foo.cpp foo.c)
set_target_properties(foo PROPERTIES
    C_STANDARD                    99
    C_STANDARD_REQUIRED           ON
    C_EXTENSIONS                  OFF
    CXX_STANDARD                  11
    CXX_STANDARD_REQUIRED         ON
    CXX_EXTENSIONS                OFF
    CUDA_STANDARD                 11
    CUDA_STANDARD_REQUIRED        ON
    CUDA_EXTENSIONS               OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS   ON
    CUDA_SEPARABLE_COMPILATION    ON
    CUDA_PTX_COMPILATION          OFF)
)
```

#### Build flag recommendations

- CMake has the ability to check whether compiler flags are supported
    - Use this in combination with INTERFACE libraries and you can use
    `target_link_libraries(foo PUBLIC foo-compile-options)`
    to ensure other CMake projects linking against `foo` will inherit the compile options or
    `target_link_libraries(foo PRIVATE foo-compile-options)` to ensure that the compile options are not
    inherited when linking against `foo`.
    - Use the most recent CMake release to ensure that the CMake version was released after the compiler version

- The following macros are useful

```cmake
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

# create interface target with compiler flags
add_library(foo-compile-options INTERFACE)

#----------------------------------------------------------------------------------------#
# macro that checks if flag if supported for C, if so add to foo-compile-options
#----------------------------------------------------------------------------------------#
macro(ADD_C_FLAG_IF_AVAIL FLAG)
    if(NOT "${FLAG}" STREQUAL "")
        # create a variable for checking the flag if supported, e.g.:
        #   -fp-model=precise --> c_fp_model_precise
        string(REGEX REPLACE "^-" "c_" FLAG_NAME "${FLAG}")
        string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")

        check_c_compiler_flag("${FLAG}" ${FLAG_NAME})
        if(${FLAG_NAME})
            target_compile_options(foo-compile-options INTERFACE
                $<$<COMPILE_LANGUAGE:C>:${FLAG}>)
        endif()
    endif()
endmacro()

#----------------------------------------------------------------------------------------#
# macro that checks if flag if supported for C++, if so add to foo-compile-options
#----------------------------------------------------------------------------------------#
macro(ADD_CXX_FLAG_IF_AVAIL FLAG)
    if(NOT "${FLAG}" STREQUAL "")
        # create a variable for checking the flag if supported, e.g.:
        #   -fp-model=precise --> cxx_fp_model_precise
        string(REGEX REPLACE "^-" "cxx_" FLAG_NAME "${FLAG}")
        string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")

        # runs check to see flag is supported by compiler
        check_cxx_compiler_flag("${FLAG}" ${FLAG_NAME})
        if(${FLAG_NAME})
            target_compile_options(foo-compile-options INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
        endif()
    endif()
endmacro()

#----------------------------------------------------------------------------------------#
# macro that checks if flag if supported for C and C++
#----------------------------------------------------------------------------------------#
macro(ADD_FLAGS_IF_AVAIL)
    foreach(FLAG ${ARGN})
        add_c_flag_if_avail("${FLAG}")
        add_cxx_flag_if_avail("${FLAG}")
    endforeach()
endmacro()
```

- Provide options for enable AVX-512 flags and leak-detection flags

```cmake
# ---------------------------------------------------------------------------- #
# options
option(USE_AVX512 "Enable AVX-512 architecture flags" OFF)
option(USE_SANTITIZER "Enable leak detection" OFF)
```

- With the above macros and options, check the flags and when available, the
macro appends these flags to `foo-compile-options` which
are later "linked" to your library and/or executable which inherits the compile-options

```cmake
# standard flags for C and C++
add_flags_if_avail("-W" "-Wall" "-Wextra" "-Wshadow")

# "new" keyword doesn't exist in C so no need to check
add_cxx_if_avail("-faligned-new")

# OpenMP SIMD-only (supported by GCC)
add_flags_if_avail("-fopenmp-simd")

# enable runtime leak detection
if(USE_SANITIZER)
    add_flags_if_avail("-fsanitize=leak")

    # emit warnings that this feature is not available
    if(NOT c_fsanitize_leak)
        message(WARNING "Sanitizer is not available for selected C compiler")
    endif()

    if(NOT cxx_fsanitize_leak)
        message(WARNING "Sanitizer is not available for selected C++ compiler")
    endif()
endif()

# check for AVX-512 flags
if(USE_AVX512)
    if(CMAKE_C_COMPILER_ID MATCHES "Intel")
        add_flags_if_avail("-xMIC-AVX512")
    else()
        # these flags are supported by newer GCC versions
        add_flags_if_avail("-mavx512f" "-mavx512pf" "-mavx512er" "-mavx512cd")
    endif()
endif()
```

### Example

```shell
## sample project using features described above
$ git clone https://github.com/jrmadsen/TiMemory.git ${SCRATCH}/timemory

## go to source directory
$ cd ${SCRATCH}/timemory

## create a separate build directory
$ mkdir -p build-timemory/intel-test

## go into build directory
$ cd build-timemory/intel-test

## in below, "../.." is relative path to source tree at ${SCRATCH}/mypackage
$ cmake -DCpuArch_TARGET=knl -DCMAKE_BUILD_TYPE=Release ../..

-- TiMemory version 3.0.0
-- The C compiler identification is Intel 18.0.3.20180410
-- The CXX compiler identification is Intel 18.0.3.20180410
-- Cray Programming Environment 2.5.15 C
-- Check for working C compiler: /opt/cray/pe/craype/2.5.15/bin/cc
-- Check for working C compiler: /opt/cray/pe/craype/2.5.15/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Cray Programming Environment 2.5.15 CXX
-- Check for working CXX compiler: /opt/cray/pe/craype/2.5.15/bin/CC
-- Check for working CXX compiler: /opt/cray/pe/craype/2.5.15/bin/CC -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Performing Test c_W
-- Performing Test c_W - Success
-- Performing Test c_Wall
-- Performing Test c_Wall - Success
-- Performing Test c_Wextra
-- Performing Test c_Wextra - Success
-- Performing Test cxx_W
-- Performing Test cxx_W - Success
-- Performing Test cxx_Wall
-- Performing Test cxx_Wall - Success
-- Performing Test cxx_Wextra
-- Performing Test cxx_Wextra - Success
-- Performing Test cxx_Wno_c++17_extensions
-- Performing Test cxx_Wno_c++17_extensions - Failed
-- Performing Test c_funroll_loops
-- Performing Test c_funroll_loops - Success
-- Performing Test cxx_funroll_loops
-- Performing Test cxx_funroll_loops - Success
-- Performing Test c_ftree_vectorize
-- Performing Test c_ftree_vectorize - Success
-- Performing Test cxx_ftree_vectorize
-- Performing Test cxx_ftree_vectorize - Success
-- Performing Test c_finline_functions
-- Performing Test c_finline_functions - Success
-- Performing Test cxx_finline_functions
-- Performing Test cxx_finline_functions - Success
-- Performing Test c_ftree_loop_optimize
-- Performing Test c_ftree_loop_optimize - Failed
-- Performing Test cxx_ftree_loop_optimize
-- Performing Test cxx_ftree_loop_optimize - Failed
-- Performing Test c_ftree_loop_vectorize
-- Performing Test c_ftree_loop_vectorize - Failed
-- Performing Test cxx_ftree_loop_vectorize
-- Performing Test cxx_ftree_loop_vectorize - Failed
-- Performing Test c_finline_limit_2048
-- Performing Test c_finline_limit_2048 - Success
-- Performing Test cxx_finline_limit_2048
-- Performing Test cxx_finline_limit_2048 - Success
-- Performing Test cxx_faligned_new
-- Performing Test cxx_faligned_new - Failed
-- Performing Test cxx_ftls_model_initial_exec
-- Performing Test cxx_ftls_model_initial_exec - Success
-- Found CpuArch: 'knl' with features: 'mmx;sse;sse2;ssse3;sse4.1;sse4.2;avx;avx2;mic-avx512'
-- Found CpuArch: knl
-- Performing Test c_timemory_arch_xmmx
-- Performing Test c_timemory_arch_xmmx - Failed
-- Performing Test cxx_timemory_arch_xmmx
-- Performing Test cxx_timemory_arch_xmmx - Failed
-- Performing Test c_timemory_arch_xsse
-- Performing Test c_timemory_arch_xsse - Failed
-- Performing Test cxx_timemory_arch_xsse
-- Performing Test cxx_timemory_arch_xsse - Failed
-- Performing Test c_timemory_arch_xsse2
-- Performing Test c_timemory_arch_xsse2 - Success
-- Performing Test cxx_timemory_arch_xsse2
-- Performing Test cxx_timemory_arch_xsse2 - Success
-- Performing Test c_timemory_arch_xssse3
-- Performing Test c_timemory_arch_xssse3 - Success
-- Performing Test cxx_timemory_arch_xssse3
-- Performing Test cxx_timemory_arch_xssse3 - Success
-- Performing Test c_timemory_arch_xsse4.1
-- Performing Test c_timemory_arch_xsse4.1 - Success
-- Performing Test cxx_timemory_arch_xsse4.1
-- Performing Test cxx_timemory_arch_xsse4.1 - Success
-- Performing Test c_timemory_arch_xsse4.2
-- Performing Test c_timemory_arch_xsse4.2 - Success
-- Performing Test cxx_timemory_arch_xsse4.2
-- Performing Test cxx_timemory_arch_xsse4.2 - Success
-- Performing Test c_timemory_arch_xavx
-- Performing Test c_timemory_arch_xavx - Success
-- Performing Test cxx_timemory_arch_xavx
-- Performing Test cxx_timemory_arch_xavx - Success
-- Performing Test c_timemory_arch_xavx2
-- Performing Test c_timemory_arch_xavx2 - Success
-- Performing Test cxx_timemory_arch_xavx2
-- Performing Test cxx_timemory_arch_xavx2 - Success
-- Performing Test c_timemory_arch_xmic_avx512
-- Performing Test c_timemory_arch_xmic_avx512 - Success
-- Performing Test cxx_timemory_arch_xmic_avx512
-- Performing Test cxx_timemory_arch_xmic_avx512 - Success
-- Performing Test c_timemory_avx512_xMIC_AVX512
-- Performing Test c_timemory_avx512_xMIC_AVX512 - Success
-- Performing Test cxx_timemory_avx512_xMIC_AVX512
-- Performing Test cxx_timemory_avx512_xMIC_AVX512 - Success
-- Found MPI_C: /opt/cray/pe/craype/2.5.15/bin/cc (found version "3.1")
-- Found MPI_CXX: /opt/cray/pe/craype/2.5.15/bin/CC (found version "3.1")
-- Found MPI: TRUE (found version "3.1")
-- Found PythonInterp: /usr/common/software/python/3.6-anaconda-4.4/bin/python3.6 (found version "3.6.8")
-- Found PythonLibs: /usr/common/software/python/3.6-anaconda-4.4/lib/libpython3.6m.so
-- pybind11 v2.3.dev0
-- [interface] PAPI not found. timemory-papi interface will not provide PAPI...
-- [interface] coverage not found. timemory-coverage interface will not provide coverage...
-- [interface] CUDA not found. timemory-cuda interface will not provide CUDA...
-- [interface] CUPTI not found. timemory-cupti interface will not provide CUPTI...
-- [interface] gperftools not found. timemory-gperftools interface will not provide gperftools...
-- Performing Test cxx_tls_model_global_dynamic_ftls_model_global_dynamic
-- Performing Test cxx_tls_model_global_dynamic_ftls_model_global_dynamic - Success
-- Performing Test HAS_INTEL_IPO
-- Performing Test HAS_INTEL_IPO - Success
-- LTO enabled
--
-- The following features are defined/enabled (+):
     CMAKE_BUILD_TYPE: Build type (Debug, Release, RelWithDebInfo, MinSizeRel) -- ["Release"]
     CMAKE_CUDA_STANDARD: CUDA language standard -- ["11"]
     CMAKE_CUDA_STANDARD_REQUIRED: Require C++ language standard
     CMAKE_CXX_FLAGS_RELEASE: C++ compiler build type flags -- ["-O3 -DNDEBUG"]
     CMAKE_CXX_STANDARD: C++ language standard -- ["11"]
     CMAKE_CXX_STANDARD_REQUIRED: Require C++ language standard
     CMAKE_C_FLAGS_RELEASE: C compiler build type flags -- ["-O3 -DNDEBUG"]
     CMAKE_C_STANDARD: C language standard -- ["11"]
     CMAKE_C_STANDARD_REQUIRED: Require C language standard
     CMAKE_INSTALL_PREFIX: Installation prefix -- ["/usr/local"]
     CMAKE_INSTALL_RPATH_USE_LINK_PATH: Embed RPATH using link path
     PYBIND11_PYTHON_VERSION: PyBind11 Python version -- ["3.6"]
     TIMEMORY_BUILD_C: Build the C compatible library
     TIMEMORY_BUILD_EXTERN_TEMPLATES: Pre-compile list of templates for extern
     TIMEMORY_BUILD_PYTHON: Build Python binds for TiMemory
     TIMEMORY_BUILD_TOOLS: Enable building tools
     TIMEMORY_COMPILED_LIBRARIES: Compiled libraries -- ["timemory-cxx-shared;timemory-cxx-static;timemory-c-shared;timemory-c-static"]
     TIMEMORY_INSTALL_PREFIX: TiMemory installation -- ["/usr/local"]
     TIMEMORY_INTERFACE_LIBRARIES: Interface libraries -- ["timemory-compile-options;timemory-arch;timemory-avx512;timemory-headers;timemory-cereal;timemory-extern-templates;timemory-mpi;timemory-threading;timemory-papi;timemory-cuda;timemory-cupti;timemory-cudart;timemory-cudart-device;timemory-cudart-static;timemory-gperftools;timemory-coverage;timemory-exceptions;timemory-extensions;timemory-analysis-tools;timemory-cxx;timemory-c"]
     TIMEMORY_TLS_MODEL: Thread-local static model: 'global-dynamic', 'local-dynamic', 'initial-exec', 'local-exec' -- ["initial-exec"]
     TIMEMORY_USE_MPI: Enable MPI usage
     TiMemory_CXX_FLAGS: C++ compiler flags -- ["-W;-Wall;-Wextra;-funroll-loops;-ftree-vectorize;-finline-functions;-finline-limit=2048;-ftls-model=initial-exec"]
     TiMemory_C_FLAGS: C compiler flags -- ["-W;-Wall;-Wextra;-funroll-loops;-ftree-vectorize;-finline-functions;-finline-limit=2048"]

-- The following features are NOT defined/enabled (-):
     CMAKE_CUDA_EXTENSIONS: CUDA language standard (e.g. gnu++11)
     CMAKE_CXX_EXTENSIONS: C++ language standard (e.g. gnu++11)
     CMAKE_C_EXTENSIONS: C language standard extensions (e.g. gnu++11)
     TIMEMORY_BUILD_EXAMPLES: Build the examples
     TIMEMORY_BUILD_LTO: Enable link-time optimizations in build
     TIMEMORY_DEVELOPER_INSTALL: Python developer installation from setup.py
     TIMEMORY_DOXYGEN_DOCS: Make a `doc` make target
     TIMEMORY_USE_CLANG_TIDY: Enable running clang-tidy
     TIMEMORY_USE_COVERAGE: Enable code-coverage
     TIMEMORY_USE_CUDA: Enable CUDA option for GPU measurements
     TIMEMORY_USE_CUDA_ARCH: Enable CUDA architecture flags
     TIMEMORY_USE_CUPTI: Enable CUPTI profiling for NVIDIA GPUs
     TIMEMORY_USE_EXCEPTIONS: Signal handler throws exceptions (default: exit)
     TIMEMORY_USE_GPERF: Enable gperf-tools
     TIMEMORY_USE_PAPI: Enable PAPI
     TIMEMORY_USE_SANITIZER: Enable -fsanitize flag (=leak)

-- Configuring done
-- Generating done
-- Build files have been written to: /global/cscratch1/sd/jrmadsen/timemory/build-timemory/intel-test

## run make
$ make -j8
```
