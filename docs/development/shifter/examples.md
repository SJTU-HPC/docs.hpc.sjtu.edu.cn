# Example Dockerfiles for Shifter

The easiest way to run your software in a Shifter container is to
create a Docker image, push it to Docker Hub then pull it to Shifter
ImageGateway which creates the corresponding Shifter
image. See [How to use](#how-to-use.md) for more details.  Below we
will discuss some examples in more details.

## HEP/HENP Software Stacks

### DESI

[DESI](http://desi.lbl.gov/) jobs use community standard publicly
available software, it is independent of the Linux distro flavor. We
show below how to build a Docker image with Ubuntu base.

This Dockerfile is also available on github for download.  Docker hub
image is available
under [`mmustafa/desi`](https://hub.docker.com/r/mmustafa/desi/tags/)
image. Shifter image is available at Edison and Cori under
`mmustafa/desi:v0`.

You need the following prerequesites:

1. Start with an Ubuntu base image
2. Install all the needed standard packages from Ubuntu repositories
3. Compile
   [Astrometry.net](https://github.com/dstndstn/astrometry.net),
   [Tractor](https://github.com/dstndstn/tractor),
   [TMV](https://github.com/rmjarvis/tmv/)
   and [Galsim](https://github.com/GalSim-developers/GalSim).
4. Setup the needed environment variables along the way

Build the image:

```Shell
# Build DESI software environment ontop an Ubuntu base
FROM ubuntu:16.04
MAINTAINER Mustafa Mustafa <mmustafa@lbl.gov>

# install astrometry and tractor dependencies
RUN apt-get update && \
    apt-get install -y wget make git python python-dev python-matplotlib \
                       gcc swig python-numpy libgsl2 gsl-bin pkg-config \
                       zlib1g-dev libcairo2-dev libnetpbm10-dev netpbm \
                       libpng12-dev libjpeg-dev python-pyfits zlib1g-dev \
                       libbz2-dev libcfitsio3-dev python-photutils python-pip && \
    pip install fitsio

ENV PYTHONPATH /desi_software/astrometry_net/lib/python:$PYTHONPATH
ENV PATH /desi_software/astrometry_net/lib/python/astrometry/util:$PATH
ENV PATH /desi_software/astrometry_net/lib/python/astrometry/blind:$PATH

# ------- install astrometry
RUN mkdir -p /desi_software/astrometry_net && \
         git clone https://github.com/dstndstn/astrometry.net.git && \
         cd astrometry.net && \
         make install INSTALL_DIR=/desi_software/astrometry_net &&\
         cd / && \
         rm -rf astrometry.net

# ------- install tractor
RUN mkdir -p /desi_software/tractor && \
         git clone https://github.com/dstndstn/tractor.git && \
         cd tractor && \
         make && \
         python setup.py install --prefix=/desi_software/tractor/

ENV PYTHONPATH /desi_software/tractor/lib/python2.7/site-packages:$PYTHONPATH

# ------- install missing GalSim dependencies (others have been installed above)
RUN apt-get install -y python-future python-yaml python-pandas scons fftw3-dev libboost-all-dev

# ------- install TMV
RUN wget https://github.com/rmjarvis/tmv/archive/v0.73.tar.gz -O tmv.tar.gz && \
         gunzip tmv.tar.gz && \
         mkdir tmv && tar xf tmv.tar -C tmv --strip-components 1 && \
         cd tmv && \
         scons && \
         scons install && \
         cd / && \
         rm -rf tmv.tar tmv

# ------- install GalSim
RUN wget https://github.com/GalSim-developers/GalSim/archive/v1.4.2.tar.gz -O GalSim.tar.gz && \
         gunzip GalSim.tar.gz && \
         mkdir GalSim && tar xf GalSim.tar -C GalSim --strip-components 1 && \
         cd GalSim && \
         scons && \
         scons install && \
         cd / && \
         rm -rf GalSim.tar GalSim
```

### STAR

The [STAR](http://www.star.bnl.gov/) experiment software stack is
typically built and run on Scientific Linux.

There are two ways we can build the STAR image, the first is to
compile all the stack components one by one. The other is to install
the compiled libraries by copying them into the image. We chose to do
the latter in this example.

We use an SL6.4 docker base image that
is [publicly](https://hub.docker.com/r/ringo/scientific/tags/)
available, install the needed rpms, extract pre-compiled binaries
tarballs into the image and finally install some software that needed
to run STAR jobs at Cori. The latest image is available at Cori and
Edison (`mmustafa/sl64_sl16d:v1_pdsf6`).

```Shell
# Example Dockerfile to show how to build STAR
# environment image from binuaries tarballs. Not necessarily
# the one currently used for STAR docker image build
FROM ringo/scientific:6.4
MAINTAINER Mustafa Mustafa <mmustafa@lbl.gov>

# RPMs
RUN yum -y install libxml2 tcsh libXpm.i686 libc.i686 libXext.i686 \
                   libXrender.i686 libstdc++.i686 fontconfig.i686 \
                   zlib.i686 libgfortran.i686 libSM.i686 mysql-libs.i686 \
                   gcc-c++ gcc-gfortran glibc-devel.i686 xorg-x11-xauth \
                   wget make libxml2.so.2 gdb libXtst.{i686,x86_64} \
                   libXt.{i686,x86_64} glibc glibc-devel gcc-c++# Dev Tools
RUN wget -O /etc/yum.repos.d/slc6-devtoolset.repo \
     http://linuxsoft.cern.ch/cern/devtoolset/slc6-devtoolset.repo && \
 yum -y install devtoolset-2-toolchain
COPY enable_scl /usr/local/star/group/templates/

# untar STAR OPT
COPY optstar.sl64_gcc482.tar.gz /opt/star/
COPY installstar /
RUN python installstar SL16c && \
 rm -f installstar &&         \
 rm -f optstar.sl64_gcc482.tar.gz

# untar ROOT
COPY rootdeb-5.34.30.sl64_gcc482.tar.gz /usr/local/star/
COPY installstar /
RUN python installstar SL16c && \
 rm -f installstar && \
 rm -f rootdeb-5.34.30.sl64_gcc482.tar.gz

# untar STAR library
COPY SL16d.tar.gz /usr/local/star/packages/
COPY installstar /
RUN python installstar SL16d && \
 rm -f installstar && \
 rm -f /usr/local/star/packages/SL16d.tar.gz

# DB load balancer
COPY dbLoadBalancerLocalConfig_generic.xml /usr/local/star/packages/SL16d/StDb/servers/

# production pipeline utility macros
COPY Hadd.C /usr/local/star/packages/SL16d/StRoot/macros/
COPY lMuDst.C /usr/local/star/packages/SL16d/StRoot/macros/
COPY checkProduction.C /usr/local/star//packages/SL16d/StRoot/macros/

# Special RPMs for production at Cori; OpenMpi, mysql-server
RUN yum -y install libibverbs.x86_64 environment-modules infinipath-psm-devel.x86_64 \
 librdmacm.x86_64 opensm.x86_64 papi.x86_64 && \
 wget http://mirror.centos.org/centos/6.8/os/x86_64/Packages/openmpi-1.10-1.10.2-2.el6.x86_64.rpm && \
 rpm -i openmpi-1.10-1.10.2-2.el6.x86_64.rpm && \
 rm -f openmpi-1.10-1.10.2-2.el6.x86_64.rpm && \
 yum -y install glibc-devel devtoolset-2-libstdc++-devel.i686 && \
 yum -y install mysql-server mysql && \

# add open mpi library to LD Path
ENV LD_LIBRARY_PATH /usr/lib64/openmpi-1.10/lib/
```

#### STAR MySQL DB

STAR jobs need access to a read-only MySQL server which provides
conditions and calibration tables.

We have found that job scalability is not ideal if the MySQL server is
outside Cori network. Our solution was to run a local MySQL server on
each node, the server services all the threads running on the node
(e.g. 32 core threads). We chose to overcommit the cores, i.e. 32
production threads + 1 mysql server running on 32 cores.

The DB payload (~30GB) resides on Lustre. We have found out that the
server with the payload accessed directly from Lustre FS doesn't
perform well for this IO pattern, it takes more than 30 minutes to
cache the first few requests. In this case the XFS image mount
capability (perCacheNode) came in handy. As soon as the job starts we
copy the payload from Lustre FS into an XFS file mount, then we set
the DB server to use this copy. Copying the 30 GB payload takes 1-3
minutes. The performance was stunning, caching time dropped down from
30 minutes to less than 1 minute, it also provided us with trivial
scalability of the number of concurrent jobs.

Below are the relevant lines from our slurm batch file:

Request a perCacheNode of 50GB and mount it to `/mnt` in the Shifter
image.

```Shell
#!/bin/bash
#SBATCH --image=mmustafa/sl64_sl16d:v1_pdsf6
#SBATCH --volume=/global/cscratch1/sd/mustafa/:/mnt:perNodeCache=size=50G
```

Launch the Shifter container:

```Shell
shifter /bin/csh <<EOF
```

Copy the payload to the /mnt then launch the DB sever:

```Shell
#Prepare DB...
cd /mnt
cp -r -p /global/cscratch1/sd/mustafa/mysql51VaultStar6/ .
/usr/bin/mysqld_safe --defaults-file=/mnt/mysql51VaultStar6/my.cnf --skip-grant-tables &
sleep 30
```

## Using Non-system MPI

Some applications are hard coded to require a certain version of
openMPI (e.g. ORCA). These applications can be run on our system in a
Shifter image. However, please keep in mind that this will **not
perform as well as the system MPI**. This is because it must use ssh
to communicate, which is not as fast as the native libraries. Where
ever possible it is recommended you recompile your executable to use
the system libraries.

Here's an example Dockerfile to build an image with openmpi. It
downloads the openmpi tarball, installs it in `/usr`, and configures
MPI to communicate via ssh.

```Shell
FROM ubuntu:16.10

RUN apt-get update && apt-get install -y build-essential apt-utils ssh

RUN cd / && wget https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.1.tar.bz2 \
    && tar xvjf openmpi-2.1.1.tar.bz2 && cd openmpi-2.1.1 \
    && ./configure --prefix=/usr && make && make install \
    && rm -rf /openmpi-2.1.1 && rm -rf openmpi-2.1.1.tar.bz2

RUN echo "--mca plm ^slurm" > /usr/etc/openmpi-mca-params.conf
```

Build and upload this image to NERSC. Below is an example script for
running ORCA (a chemistry package). In this particular case, the ORCA
binaries are precompiled and released in a tar ball. Since this
tarball is nearly 20GB, we chose to install this onto the scratch
directory instead of into the image. For smaller packages, we
recommend installing them into the shifter image.

```Shell
#!/bin/bash
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -t 00:10:00
#SBATCH -C haswell
#SBATCH -L SCRATCH
#SBATCH --image=lgerhardt/openmpi_2.1.1:v1 (or your openmpi image)

#populate the node list
scontrol show hostnames $SLURM_JOB_NODELIST > your_input_file_name.nodes
shifter $SCRATCH/orca_4_0_0_2_linux_x86-64/orca <your_input_file_name>.inp
```
