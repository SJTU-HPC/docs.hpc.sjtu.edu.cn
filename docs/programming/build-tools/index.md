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

#### MPI

- Provided by Cray compiler wrappers
    - There is no need to do anything unique
    - If you have an existing `find_package(MPI)`, this will succeed (i.e. `MPI_FOUND` will be true)
but variables such as `MPI_CXX_FLAGS`, etc. will be empty

#### OpenMP

- Standard package finding for OpenMP works

```cmake
find_package(OpenMP)
if(OpenMP_FOUND)
    # Add the OpenMP-specific compiler and linker flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
```
#### Threading

- If using non-OpenMP threading, the Intel compilers require adding `-pthread` compilation flag
- In general, just use below and include `${EXTERNAL_LIBRARIES}` when linking targets

```cmake
# Threading
set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)
if(THREADS_FOUND)
    list(APPEND EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
endif(THREADS_FOUND)
```

#### Build flag recommendations

- CMake has the ability to check whether compiler flags are supported
    - Use this in combination with options
    - Use the most recent CMake release to ensure that the CMake version was released after the compiler version

- The following macros are useful

```cmake
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

# ---------------------------------------------------------------------------- #
# macro that checks if flag if supported for C, if so append to
# PROJECT_C_FLAGS
macro(add_c_if_avail FLAG)
    # create unique name for flag check by prefixing with language and
    # replacing spaces, dashes, and equal signs with underscores)
    string(REGEX REPLACE "^-" "c_" FLAG_NAME "${FLAG}"
    string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
    string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
    string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
    # provided by CheckCCompilerFlag.cmake (included above)
    check_c_compiler_flag("${FLAG}" ${FLAG_NAME})
    # if the flag if supported, append to PROJECT_C_FLAGS
    if(${FLAG_NAME})
        set(PROJECT_C_FLAGS "${PROJECT_C_FLAGS} ${FLAG}")
    endif()
endmacro()

# ---------------------------------------------------------------------------- #
# macro that checks if flag if supported for CXX, if so append to
# PROJECT_CXX_FLAGS
macro(add_cxx_if_avail FLAG)
    # create unique name for flag check by prefixing with language and
    # replacing spaces, dashes, and equal signs with underscores
    string(REGEX REPLACE "^-" "cxx_" FLAG_NAME "${FLAG}")
    string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
    string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
    string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
    # provided by CheckCXXCompilerFlag.cmake (included above)
    check_cxx_compiler_flag("${FLAG}" ${FLAG_NAME})
    # if the flag if supported, append to PROJECT_CXX_FLAGS
    if(${FLAG_NAME})
        set(PROJECT_CXX_FLAGS "${PROJECT_CXX_FLAGS} ${FLAG}")
    endif()
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
macro appends these flags to either `PROJECT_C_FLAGS` or `PROJECT_CXX_FLAGS` which
are later appended to `CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS`

```cmake
# ---------------------------------------------------------------------------- #
# standard flags for C
add_c_if_avail("-W")
add_c_if_avail("-Wall")
add_c_if_avail("-Wextra")
add_c_if_avail("-std=c11")
if(NOT c_std_c11)
    add_c_if_avail("-std=c99")
endif()

# ---------------------------------------------------------------------------- #
# OpenMP SIMD-only (supported by GCC)
add_c_if_avail("-fopenmp-simd")
add_cxx_if_avail("-fopenmp-simd")

# ---------------------------------------------------------------------------- #
# standard flags for CXX
# general warnings
add_cxx_if_avail("-W")
add_cxx_if_avail("-Wall")
add_cxx_if_avail("-Wextra")
add_cxx_if_avail("-Wshadow")
add_cxx_if_avail("-faligned-new")

# ---------------------------------------------------------------------------- #
# enable runtime leak detection
if(USE_SANITIZER)
    add_c_if_avail("-fsanitize=leak")
    add_cxx_if_avail("-fsanitize=leak")

    # emit warning that this feature is not available
    if(NOT c_fsanitize_leak)
        message(WARNING "Sanitizer is not available for C")
    endif()
    # emit warning that this feature is not available
    if(NOT cxx_fsanitize_leak)
        message(WARNING "Sanitizer is not available for CXX")
    endif()
endif()

# ---------------------------------------------------------------------------- #
# check for AVX-512 flags
if(USE_AVX512)
    if(CMAKE_C_COMPILER_ID MATCHES "Intel")
        add_c_if_avail("-axMIC-AVX512")
    else()
        # these flags are supported by newer GCC versions
        add_c_if_avail("-mavx512f")
        add_c_if_avail("-mavx512pf")
        add_c_if_avail("-mavx512er")
        add_c_if_avail("-mavx512cd")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        add_cxx_if_avail("-axMIC-AVX512")
    else()
        # these flags are supported by newer GCC versions
        add_cxx_if_avail("-mavx512f")
        add_cxx_if_avail("-mavx512pf")
        add_cxx_if_avail("-mavx512er")
        add_cxx_if_avail("-mavx512cd")
    endif()
endif()

# ---------------------------------------------------------------------------- #
# Append all the flags that are enabled and supported to the build flags
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${PROJECT_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PROJECT_CXX_FLAGS}")
```

### Example

```shell
## sample project using features described above
$ git clone https://github.com/jrmadsen/PTL.git ${SCRATCH}/PTL

## go to source directory
$ cd ${SCRATCH}/PTL

## create a separate build directory
$ mkdir -p build-PTL/Release

## go into build directory
$ cd build-PTL/Release

## in below, "../.." is relative path to source tree at ${SCRATCH}/mypackage
$ cmake -DCMAKE_BUILD_TYPE=Release -DPTL_USE_AVX512=ON -DPTL_USE_ARCH=ON -DPTL_USE_TBB=OFF ../..

-- PTL version 0.0.1
-- The C compiler identification is Intel 18.0.3.20180410
-- The CXX compiler identification is Intel 18.0.3.20180410
-- Cray Programming Environment 2.5.14 C
-- Check for working C compiler: /opt/cray/pe/craype/2.5.14/bin/cc
-- Check for working C compiler: /opt/cray/pe/craype/2.5.14/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Cray Programming Environment 2.5.14 CXX
-- Check for working CXX compiler: /opt/cray/pe/craype/2.5.14/bin/CC
-- Check for working CXX compiler: /opt/cray/pe/craype/2.5.14/bin/CC -- works
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
-- Performing Test c_std_c11
-- Performing Test c_std_c11 - Success
-- Performing Test c_pthread
-- Performing Test c_pthread - Success
-- Performing Test cxx_pthread
-- Performing Test cxx_pthread - Success
-- Performing Test c_fopenmp_simd
-- Performing Test c_fopenmp_simd - Failed
-- Performing Test cxx_fopenmp_simd
-- Performing Test cxx_fopenmp_simd - Failed
-- Performing Test cxx_W
-- Performing Test cxx_W - Success
-- Performing Test cxx_Wall
-- Performing Test cxx_Wall - Success
-- Performing Test cxx_Wextra
-- Performing Test cxx_Wextra - Success
-- Performing Test cxx_Wshadow
-- Performing Test cxx_Wshadow - Success
-- Performing Test cxx_faligned_new
-- Performing Test cxx_faligned_new - Failed
-- Performing Test c_xHOST
-- Performing Test c_xHOST - Success
-- Performing Test cxx_xHOST
-- Performing Test cxx_xHOST - Success
-- Performing Test c_axMIC_AVX512
-- Performing Test c_axMIC_AVX512 - Success
-- Performing Test cxx_axMIC_AVX512
-- Performing Test cxx_axMIC_AVX512 - Success
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - found
-- Found Threads: TRUE
--
-- The following features are defined/enabled (+):
     BUILD_SHARED_LIBS: Build shared library
     BUILD_STATIC_LIBS: Build static library
     CMAKE_BUILD_TYPE: Build type (Debug, Release, RelWithDebInfo, MinSizeRel) -- ["Release"]
     CMAKE_CXX_FLAGS: C++ compiler flags -- ["-pthread -W -Wall -Wextra -Wshadow -xHOST -axMIC-AVX512"]
     CMAKE_CXX_FLAGS_RELEASE: C++ compiler build-specific flags -- ["-O3 -DNDEBUG"]
     CMAKE_CXX_STANDARD: C++11 STL standard -- ["11"]
     CMAKE_C_FLAGS: C compiler flags -- ["-W -Wall -Wextra -std=c11 -pthread -xHOST -axMIC-AVX512"]
     CMAKE_C_FLAGS_RELEASE: C compiler build-specific flags -- ["-O3 -DNDEBUG"]
     CMAKE_INSTALL_PREFIX: Installation prefix -- ["/usr/local"]
     PTL_USE_ARCH: Enable architecture specific flags
     PTL_USE_AVX512: Enable AVX-512 flags (if available)

-- The following features are NOT defined/enabled (-):
     PTL_BUILD_DOCS: Build documentation with Doxygen
     PTL_BUILD_EXAMPLES: Build examples
     PTL_BUILD_TESTING: Enable testing
     PTL_USE_CLANG_TIDY: Enable running clang-tidy on
     PTL_USE_COVERAGE: Enable code coverage
     PTL_USE_GPERF: Enable gperftools
     PTL_USE_GPU: Enable GPU preprocessor
     PTL_USE_ITTNOTIFY: Enable ittnotify library for VTune
     PTL_USE_PROFILE: Enable profiling
     PTL_USE_SANITIZER: Enable -fsanitize=leak -fsanitize=address
     PTL_USE_TBB: Enable TBB
     PTL_USE_TIMEMORY: Enable TiMemory for timing+memory analysis

-- Configuring done
-- Generating done
-- Build files have been written to: /global/cscratch1/sd/jrmadsen/PTL/build-PTL/Release

## run make
$ make -j8
```
