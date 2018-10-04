# Using NERSC's Intel Docker Containers

## Overview

-   Intel Parallel Studio XE 2018 Docker containers are now available
    at
    [registry.services.nersc.gov](https://registry.services.nersc.gov)
    in the “nersc” directory

-   Access to the NERSC private registry is required

    -   Follow instructions from [Using NERSC's Private Registry](#using-nerscs-private-registry)

## Image Naming Convention

NERSC has built a number of images to support building and running applications
using Intel Compiler suite.  The images have a number of combinations of components,
versions, and image use.

-   The image follow the following naming scheme.: <span style="color:red">registry.services.nersc.gov/nersc/intel_{toolset}_{image}:{tag}</span>
-   The toolset defines the components included in the image.
-   The Image defines whether the image is intended for building and compiling
    applications or running applications.
-   The tag defines the version of the Intel compiler suite to use.
-   The table below shows the different options for each portion.  All combinations
    are availble (7x2x6)
-   Like all docker images, if a tag(version) isn't supplied it defaults to _latest_.

### Image Matrix

| Toolset Variants (7)    | Image     | Version/tag |
|---                      |---        |---        |
| cxx                     |devel    |latest     |
| fort                    |runtime    |2018       |
| cxx_fort                |           |2018.1     |
| cxx_mpi                 |           |2018.2     |
| cxx_fort_mpi            |           |2018.1.163 |
| cxx_fort_mpi_mkl        |           |2018.2.199 |
| cxx_fort_mpi_mkl_ipp    |           |           |

Note:

-   All cxx variants include TBB and PSTL
-   mpi has libfabric

Here are examples of a full image name:

-   *registry.services.nersc.gov/nersc/intel_cxx_devel* (the latest version of the devel image with the cxx compiler)
-   *registry.services.nersc.gov/nersc/intel_cxx_mpi_devel:2018.2* (the 2018.2 version of the devel image with cxx and MPI)

### Image types: “devel” Image versus "runtime" Images

-   The devel images are relatively large and contain all the components for
    the set of Intel tools.  These are intended for compiling codes.
-   The runtime images are much smaller and optimized for running previously compiled
    applications.  These images contain only the shared libraries.  All bin directories and static libraries have been removed.  They can be used with a staged Docker build (described below) to create compact images that can be pushed to public registries
    without violating the Intel License since the libraries can be distributed.

| Image | Devel Size | Runtime Size |
|--- |--- :|--- :|
| nersc/intel\_cxx\_{devel,runtime}   | 1.3 GB | 0.7 GB|
| nersc/intel\_fort\_{devel,runtime}  | 1.2 GB | 0.7 GB|
| nersc/intel\_cxx\_fort\_{devel,runtime}       | 1.4 GB | 0.7 GB|
| nersc/intel\_cxx\_mpi\_{devel,runtime}        | 2.9 GB | 1.2 GB|
| nersc/intel\_cxx\_fort\_mpi\_{devel,runtime}  | 2.9 GB | 1.2 GB|
| nersc/intel\_cxx\_fort\_mpi\_mkl\_{devel,runtime} | 4.7 GB | 2.0 GB|
| nersc/intel\_cxx\_fort\_mpi\_mkl\_ipp\_{devel,runtime} | 7 GB | 2.8 GB|

## Using the Intel Compilers with NERSC's License Server

Using the Intel compiler tools requires a license.  The NERSC license server can
be used to support this, but the NERSC License server is reserved for NERSC users and
is not publicly accessible.  

-   The “devel” docker images require a valid Intel license to use the
    compilers
-   If the Intel compilers are not required, there is no need to connect
    to the NERSC license server
-   Intel compilers in the NERSC Intel images are configured to request a license at
    [intel.licenses.nersc.gov](mailto:intel.licenses.nersc.gov) @ port 28519

In order to use the server, you must tunnel the connection
via SSH and provide a DNS entry during the Docker build.  Here are the basic steps to use the NERSC License server remotely.

1.  Configure docker to
        resolve [intel.licenses.nersc.gov](mailto:intel.licenses.nersc.gov)
        to your local IP address

2.  Configure a local SSH connection to:

    -   allow remote hosts (i.e., docker container) to connect to
        local forwarded ports

    -   forward local port 28519 requests to port 28519
        @ cori.nersc.gov

### SSH session

-   Obtain the local IP address of your system (macOS)

    ```
    $ ipconfig getifaddr `route get nersc.gov | grep 'interface:' | awk '{print $NF}'`
    ```

-   Obtain local IP address (Linux)

    ```
    $ ip route ls | tail -n 1 | awk '{print $NF}'
    ```

-   SSH command

    ```
    $ ssh -g -L 28519:intel.licenses.nersc.gov:28519 <username>@cori.nersc.gov
    ```

-   SSH config

    ```
    Host intel-docker-cori
      Hostname cori.nersc.gov
      user <username>
      Port 22
      GatewayPorts yes
      LocalForward <ip_addr>:28519 intel.licenses.nersc.gov:28519
    ```

### Docker build flag

-   Run docker build with flag “–add-host” flag that directs
    [intel.licenses.nersc.gov](mailto:intel.licenses.nersc.gov) to your IP
    address

Below is a BASH script that be used to automate these steps.

```Shell
#!/bin/bash
build-intel-docker()
{
  if [ "$(uname)" = "Darwin" ]; then
    LOCAL_IP=$(ipconfig getifaddr $(route get nersc.gov | grep 'interface:' | awk '{print $NF}'))
  else
    LOCAL_IP=$(ip route ls | tail -n 1 | awk '{print $NF}')
  fi

  local BUILD_TAG=""
  local DOCKERFILE=""
  if [ ! -z "${1}" ]; then BUILD_TAG=${1}; else BUILD_TAG=$(basename ${PWD}); fi
  if [ ! -z "${2}" ]; then DOCKERFILE=${2}; else DOCKERFILE=Dockerfile; fi

  docker build -f ${DOCKERFILE} --tag=${BUILD_TAG} \
      --add-host intel.licenses.nersc.gov:${LOCAL_IP} ${PWD}
}
```

## Using Multistage Docker Builds

Multistage builds can be used to create compact images optimized for running
applications.  Smaller images can be pulled and converted more quickly by Docker and Shifter.
The images also remove binaries that require an Intel license which means that they
can be distributed without violating the License terms of the compilers.  As always,
consult the license terms of any software you use in your images before distributing
them on a public site such as DockerHub.

### Build Staging

The Basic steps in using Multi-stage builds with the Intel compilers is:

-   Use first “FROM” statement with “devel” image variant and label with
    “as” keyword (e.g. as build)

-   Use second “FROM” statement with “runtime” image variant and copy
    from first image

-   Note: Multiple “as” images can be used

### Build Staging Example using NERSC's Intel Docker images

Here is an example of a multi-stage build used to compile an MPI application written
in C.

```Shell
######## Stage 1 ########
FROM registry.services.nersc.gov/nersc/intel_cxx_mpi_devel as builder
# ... build your code with "devel" image variant ...
# ... recommended to use a common install prefix, such as "/opt/local", e.g.
#

ENV CC /opt/intel/bin/icc
ENV CXX /opt/intel/bin/icpc

RUN cd ${HOME}/package_a && \
    ./configure --prefix=/opt/local && \
    make install -j8 && \
    cd ${HOME}/package_b && \
    mkdir build_package_b && \
    cd build_package_b && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/local .. && \
    make install -j8

######## Stage 2 ########
# ... don't have to clean above, just copy over installation
FROM registry.services.nersc.gov/nersc/intel_cxx_mpi_runtime
COPY --from=builder /opt/local/ /opt/local
RUN echo '/opt/local/lib' > /etc/ld.so.conf.d/package-libs.conf && \
    ldconfig
```

If you are interested in how the runtime images are built, you can expand the
section below.
<details><summary>EXPAND</summary>

```Shell
#################################################################
# Stage 1 from image with compilers
#################################################################
FROM devel_psxe_2018_cxx as builder

WORKDIR /root
USER root
ENV HOME /root
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US
ENV LC_ALL C
ENV SHELL /bin/bash
ENV BASH_ENV /etc/bash.bashrc
ENV DEBIAN_FRONTEND noninteractive

# funcs defined in /etc/bash.bashrc
#   configs link paths and deletes static libs
RUN write-ld-config intel-libs.conf && \
    intel-runtime-cleanup

#################################################################
# Stage 2 from base image
#################################################################
FROM debian:latest

COPY --from=builder /usr /usr/
COPY --from=builder /etc /etc/
COPY --from=builder /opt/intel/ /opt/intel

WORKDIR /root
USER root
ENV HOME /root
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US
ENV LC_ALL C
ENV SHELL /bin/bash
ENV BASH_ENV /etc/bash.bashrc
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update --fix-missing && \
    apt-get -y --no-install-recommends install --reinstall \
        libc6 libcc1-0 libgcc1 libgmp10 libisl15 libmpc3 libmpfr4 libstdc++6 && \
    apt-get -y --purge autoremove && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

#################################################################
# Entry point
#################################################################

COPY config/runtime-entrypoint.sh /intel-runtime-entrypoint.sh
SHELL [ "/bin/bash", "--login", "-c" ]
ENTRYPOINT [ "/intel-runtime-entrypoint.sh" ]
```
</details>

### Build Staging Recommendations

-   It is not recommended to copy over `/usr`

-   Manipulating `LD_LIBRARY_PATH` at NERSC can cause issues with MPI since Shifter
    must manipulate the LD_LIBRARY_PATH at runtime to inject the optimized MPI libraries.
    Avoid changing the LD_LIBRARY_PATH after image launch.

-   Runtime link path options in containers

    1.  Install compiled binaries into an isolated directory (e.g. `/opt/local`) in the builder stage and relocate this to `/usr` in final stage

        -   Note some libraries will hard-code link path so relocating to `/usr` in
            staged build causes runtime error

    2.  Add custom `LD_LIBRARY_PATH` paths then prefix `LD_LIBRARY_PATH` with original (i.e. `/opt/udiImage/lib`)

        -   May cause system library to be found instead of custom built library

    3.  Create a file ending in “.conf” in the dynamic linker configuration directory and list all the library paths needed at runtime (recommended).  This can furhter
    help optimize loading required libraries at runtime.

        - E.g., <span style="color:red">/etc/ld.so.conf.d/intel-libs.conf</span>
        - Add a RUN /sbin/ldconfig at the end of the Dockerfile to refresh the
          cache used by the loader

```Shell
  # docker run -it registry.services.nersc.gov/nersc/intel_cxx_fort_mpi_mkl_ipp_runtime:2018.1

  root@ec7c62623bb9:~# ls -1 /etc/ld.so.conf.d/
    intel-libs.conf
    intel-mpi-2018.1.163.conf
    libc.conf
    x86_64-linux-gnu.conf

  root@ec7c62623bb9:~# cat /etc/ld.so.conf.d/intel-libs.conf
    /opt/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64
    /opt/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin
    /opt/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64
    /opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin
    /opt/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib
    /opt/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib
    /opt/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7
```
