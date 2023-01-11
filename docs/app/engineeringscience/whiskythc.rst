.. _WhiskyTHC:

WhiskyTHC
====================

简介
----

WhiskyTHC includes the sophisticated microphysics framework developed at the Max Planck Institute for Gravitational Physics for the original Whisky code and uses advanced C++ programming techniques to generate optimized numerical code for a variety of numerical methods (Templated Hydrodynamics).




WhiskyTHC安装说明
-----------------------------

在思源一号上自行安装WhiskyTHC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 首先在自己的家目录下新建一个目录作为安装目录,并进入该目录：

.. code::

    mkdir WhiskyTHC
    cd  WhiskyTHC


2. 申请计算资源并加载所需模块：

.. code::

    srun -p 64c512g -n 40 --pty /bin/bash
    module purge
    module load gcc/9.3.0
    module load hwloc/2.6.0-gcc-9.3.0
    module load openmpi/4.1.1-gcc-9.3.0


3. 复制源码包到当前目录并解压,然后进入FreeTHC目录：

.. code::

    cp /dssg/share/Sourcefile/hpcpzz/WhiskyTHC-20221221.tar  ./
    tar xvpf ./WhiskyTHC-20221221.tar
    cd FreeTHC


4. 在Cactus/batchtools/templates/cactus目录下编写如下sjtusy.cfg配置文件：

.. code::

    # Whenever this version string changes, the application is configured
    # and rebuilt from scratch
    VERSION = 2019-09-03

    CPP = cpp -DFORTRAN_DISABLE_IEEE_ARITHMETIC
    FPP = cpp -DFORTRAN_DISABLE_IEEE_ARITHMETIC
    CC  = gcc
    CXX = g++
    F77 = gfortran
    F90 = gfortran

    FPPFLAGS = -traditional

    CPPFLAGS =
    FPPFLAGS =
    CFLAGS   = -g -std=gnu99
    CXXFLAGS = -g -std=gnu++11
    F77FLAGS = -g -fcray-pointer -m128bit-long-double -ffixed-line-length-none -fno-range-check
    F90FLAGS = -g -fcray-pointer -m128bit-long-double -ffixed-line-length-none -fno-range-check

    LDFLAGS = -rdynamic

    DEBUG           = no
    CPP_DEBUG_FLAGS = -DCARPET_DEBUG -DHRSCC_DEBUG -DTHC_DEBUG -DCPPUTILS_DEBUG
    FPP_DEBUG_FLAGS = -DCARPET_DEBUG -DHRSCC_DEBUG -DTHC_DEBUG -DCPPUTILS_DEBUG
    C_DEBUG_FLAGS   = -O0
    CXX_DEBUG_FLAGS = -O0
    F77_DEBUG_FLAGS = -O0
    F90_DEBUG_FLAGS = -O0

    OPTIMISE           = yes
    CPP_OPTIMISE_FLAGS =
    FPP_OPTIMISE_FLAGS =
    C_OPTIMISE_FLAGS   = -O2
    CXX_OPTIMISE_FLAGS = -O2
    F77_OPTIMISE_FLAGS = -O2
    F90_OPTIMISE_FLAGS = -O2

    PROFILE           = no
    CPP_PROFILE_FLAGS =
    FPP_PROFILE_FLAGS =
    C_PROFILE_FLAGS   = -pg
    CXX_PROFILE_FLAGS = -pg
    F77_PROFILE_FLAGS = -pg
    F90_PROFILE_FLAGS = -pg

    OPENMP           = yes
    CPP_OPENMP_FLAGS = -fopenmp
    FPP_OPENMP_FLAGS = -D_OPENMP
    C_OPENMP_FLAGS   = -fopenmp
    CXX_OPENMP_FLAGS = -fopenmp
    F77_OPENMP_FLAGS = -fopenmp
    F90_OPENMP_FLAGS = -fopenmp

    WARN           = yes
    CPP_WARN_FLAGS = -Wall
    FPP_WARN_FLAGS = -Wall
    C_WARN_FLAGS   = -Wall
    CXX_WARN_FLAGS = -Wall
    F77_WARN_FLAGS = -Wall
    F90_WARN_FLAGS = -Wall

    #BLAS_DIR = /usr
    #BOOST_DIR = /usr
    #FFTW3_DIR = /usr
    #GSL_DIR = /usr
    #HDF5_DIR = /usr
    #HWLOC_DIR = /usr
    #LAPACK_DIR = /usr

    MPI_DIR =/dssg/opt/icelake/linux-centos8-icelake/gcc-9.3.0/openmpi-4.1.1-usre7vgur4rv6jllqd4yuf5gg57kothm
    BLAS_DIR = BUILD
    BOOST_DIR = BUILD
    FFTW3_DIR = BUILD
    GSL_DIR =  BUILD
    HDF5_DIR = BUILD
    HWLOC_DIR =/dssg/opt/icelake/linux-centos8-icelake/gcc-9.3.0/hwloc-2.6.0-mkqkyei3gxxtri4uaafreqteyhyj2exl
    LAPACK_DIR = BUILD

    PTHREADS_DIR = NO_BUILD


5. 构建 thc 依赖库:

.. code::

  cd Cactus
  yes | make thc THORNLIST=thornlists/full.th     options=batchtools/templates/cactus/sjtusy.cfg


6. 构建 thc :


.. code::

 make -j40 thc








在pi2.0上自行安装WhiskyTHC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；
2. 申请计算资源并加载所需模块：

.. code::

  srun -N 1 -p cpu --exclusive --pty /bin/bash
  module purge
  module load gcc/9.2.0
  module load hwloc/1.11.11-gcc-9.2.0
  module load openmpi/3.1.5-gcc-9.2.0


3. 复制源码包到当前目录并解压,然后进入FreeTHC目录：

.. code::

    cp /lustre/share/Sourcefile/hpcpzz/WhiskyTHC-20221221.tar  ./
    tar xvpf ./WhiskyTHC-20221221.tar
    cd FreeTHC


4. 在Cactus/batchtools/templates/cactus目录下编写如下sjtupi.cfg配置文件：

.. code::

    # Whenever this version string changes, the application is configured
    # and rebuilt from scratch
    VERSION = 2019-09-03

    CPP = cpp -DFORTRAN_DISABLE_IEEE_ARITHMETIC
    FPP = cpp -DFORTRAN_DISABLE_IEEE_ARITHMETIC
    CC  = gcc
    CXX = g++
    F77 = gfortran
    F90 = gfortran

    FPPFLAGS = -traditional

    CPPFLAGS =
    FPPFLAGS =
    CFLAGS   = -g -std=gnu99
    CXXFLAGS = -g -std=gnu++11
    F77FLAGS = -g -fcray-pointer -m128bit-long-double -ffixed-line-length-none -fno-range-check
    F90FLAGS = -g -fcray-pointer -m128bit-long-double -ffixed-line-length-none -fno-range-check

    LDFLAGS = -rdynamic

    DEBUG           = no
    CPP_DEBUG_FLAGS = -DCARPET_DEBUG -DHRSCC_DEBUG -DTHC_DEBUG -DCPPUTILS_DEBUG
    FPP_DEBUG_FLAGS = -DCARPET_DEBUG -DHRSCC_DEBUG -DTHC_DEBUG -DCPPUTILS_DEBUG
    C_DEBUG_FLAGS   = -O0
    CXX_DEBUG_FLAGS = -O0
    F77_DEBUG_FLAGS = -O0
    F90_DEBUG_FLAGS = -O0

    OPTIMISE           = yes
    CPP_OPTIMISE_FLAGS =
    FPP_OPTIMISE_FLAGS =
    C_OPTIMISE_FLAGS   = -O2
    CXX_OPTIMISE_FLAGS = -O2
    F77_OPTIMISE_FLAGS = -O2
    F90_OPTIMISE_FLAGS = -O2

    PROFILE           = no
    CPP_PROFILE_FLAGS =
    FPP_PROFILE_FLAGS =
    C_PROFILE_FLAGS   = -pg
    CXX_PROFILE_FLAGS = -pg
    F77_PROFILE_FLAGS = -pg
    F90_PROFILE_FLAGS = -pg

    OPENMP           = yes
    CPP_OPENMP_FLAGS = -fopenmp
    FPP_OPENMP_FLAGS = -D_OPENMP
    C_OPENMP_FLAGS   = -fopenmp
    CXX_OPENMP_FLAGS = -fopenmp
    F77_OPENMP_FLAGS = -fopenmp
    F90_OPENMP_FLAGS = -fopenmp

    WARN           = yes
    CPP_WARN_FLAGS = -Wall
    FPP_WARN_FLAGS = -Wall
    C_WARN_FLAGS   = -Wall
    CXX_WARN_FLAGS = -Wall
    F77_WARN_FLAGS = -Wall
    F90_WARN_FLAGS = -Wall

    #BLAS_DIR = /usr
    #BOOST_DIR = /usr
    #FFTW3_DIR = /usr
    #GSL_DIR = /usr
    #HDF5_DIR = /usr
    #HWLOC_DIR = /usr
    #LAPACK_DIR = /usr

    MPI_DIR =/lustre/opt/cascadelake/linux-centos7-cascadelake/gcc-9.2.0/openmpi-3.1.5-gtpczurejutqns55psqujgakh7vpzqot
    BLAS_DIR = BUILD
    BOOST_DIR = BUILD
    FFTW3_DIR = BUILD
    GSL_DIR =  BUILD
    HDF5_DIR = BUILD
    HWLOC_DIR =/lustre/opt/cascadelake/linux-centos7-cascadelake/gcc-9.2.0/hwloc-1.11.11-whc7fqivalihbihdrueouc4pcnzbcror
    LAPACK_DIR = BUILD

    PTHREADS_DIR = NO_BUILD


5. 构建 thc 依赖库:

.. code::

  cd Cactus
  yes | make thc THORNLIST=thornlists/full.th     options=batchtools/templates/cactus/sjtupi.cfg


6. 构建 thc :


.. code::

 make -j40 thc



参考资料
----------------

-  `WhiskyTHC官网 <http://personal.psu.edu/dur566/whiskythc.html>`__


