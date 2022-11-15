.. _dealii:

Deal.II
========

Deal.II作为一个开源有限元软件，其本质上是一个C++软件库，用以支持创建有限元代码。
DEAL的全称为Differential Equations Analysis Library（微分方程分析库），而以II作为后缀则是因为这一软件库是微分方程分析库的后续工作，其主要的功能是创建C++软件库，使用自适应有限元来解决针对偏微分方程的计算。
它使用最先进的编程技术为用户提供所需的复杂数据结构和算法的现代接口

可用版本
------------------------

+----------+----------------+----------+--------------------------------------------+
| 版本     | 平台           | 构建方式 | 名称                                       |
+==========+================+==========+============================================+
| 9.3.3    |  |cpu|         | spack    | dealii/9.3.3-gcc-11.2.0-hdf5-openblas 思源 |
+----------+----------------+----------+--------------------------------------------+

Deal.II可提供的内容
--------------------


- 使用统一的接口来支持一、二、三个空间维度，该接口允许编写几乎与唯独无关的程序。
- 处理局部的细化网格，包括基于局部误差指示器和误差估计器的不同自适应网格策略。
- 支持多种有限元元素：连续和不连续的任意阶拉格朗日单元(Lagrange elements)、内德勒克和拉维亚特-托马斯单元（Nedelec and Raviart-Thomas elements）以及其他单元。
- 通过MPI跨界点并行计算。
- 允许快速访问的所需信息，包含教程、报告以及一些接口文档，其中解释了所有的类、函数以及变量，所有的文件与库同时提供，安装后即可在计算机中获取。
- 使访问复杂数据结构和算法尽可能透明的软件技术。
- 一个完整的独立线性代数库。
- 支持多种输出格式。
- 对各种计算机和编译器的可移植支持。
- 在开源许可下免费获取源代码。

思源一号上Deal.II的 ``lib`` 和  ``include`` 库的路径
----------------------------------------------------

.. code:: bash

   /dssg/opt/icelake/linux-centos8-icelake/gcc-11.2.0/dealii-9.3.3-pyqxfz27lwca6qu5r6mc6zoh65pl2jad/lib
   /dssg/opt/icelake/linux-centos8-icelake/gcc-11.2.0/dealii-9.3.3-pyqxfz27lwca6qu5r6mc6zoh65pl2jad/include

如何在个人目录下自编译Deal.II软件
----------------------------------

.. code:: bash

   srun -p 64c512g -N 1 -n 6 --pty /bin/bash
   mkdir -p ~/src/dealii/src
   mkdir -p ~/src/dealii/install

   module load mpich/3.4.1-gcc-9.2.0 gcc/9.2.0  boost/1.70.0-gcc-9.2.0
   export CC=mpicc
   export CXX=mpicxx
   export FC=mpif90

   cd ~/src/dealii/src
   tar xf petsc-3.13.1.tar.gz
   cd petsc-3.13.1
   export PETSC_DIR=`pwd`
   export PETSC_ARCH=x86_64
   python3 config/configure.py --with-shared-libararies=1 --with-x=0 --with-mpi=1 --with-mpi-dir=/lustre/opt/cascadelake/linux-centos7-cascadelake/gcc-9.2.0/mpich-3.4.1-76cdhzxhzr7wp5kqkwirehvwpo7oe6lc --download-hypre=1 --download-fblaslapack=1
   make PETSC_DIR=~/src/dealii/src/petsc-3.13.1 PETSC_ARCH=x86_64 all

   export PETSC_DIR=~/src/dealii/src/petsc-3.13.1
   export PETSC_ARCH=x86_64
   export PATH=$PATH:~/src/dealii/src/petsc-3.13.1/x86_64/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/src/dealii/src/petsc-3.13.1/x86_64/lib:~/src/dealii/src/petsc-3.13.1/x86_64/externalpackages/git.fblaslapack

   cd ~/src/dealii/src
   tar xf p4est-2.2.tar.gz
   cd p4est-2.2/
   ./configure --prefix=~/src/dealii/install/p4est BLAS_LIBS=~/src/dealii/src/petsc-3.13.1/x86_64/externalpackages/git.fblaslapack/libfblas.a LAPACK_LIBS=~/src/dealii/src/petsc-3.13.1/x86_64/externalpackages/git.fblaslapack/libflapack.a --enable-mpi
   make
   make install

   export PATH=$PATH:~/src/dealii/src/petsc-3.13.1/x86_64/bin:~/src/dealii/install/p4est/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/src/dealii/src/petsc-3.13.1/x86_64/lib:~/src/dealii/src/petsc-3.13.1/x86_64/externalpackages/git.fblaslapack:~/src/dealii/install/p4est/lib

   cd ~/src/dealii/src
   unzip Trilinos.zip
   cd Trilinos/Trilinos-trilinos-release-12-4-1/
   mkdir build && cd build
   cmake -DTrilinos_ENABLE_Amesos=ON -DTrilinos_ENABLE_Epetra=ON -DTrilinos_ENABLE_EpetraExt=ON -DTrilinos_ENABLE_Ifpack=ON  -DTrilinos_ENABLE_AztecOO=ON -DTrilinos_ENABLE_Sacado=ON -DTrilinos_ENABLE_Teuchos=ON -DTrilinos_ENABLE_MueLu=ON -DTrilinos_ENABLE_ML=ON -DTrilinos_ENABLE_ROL=ON -DTrilinos_ENABLE_Tpetra=ON -DTrilinos_ENABLE_COMPLEX_DOUBLE=ON -DTrilinos_ENABLE_COMPLEX_FLOAT=ON -DTrilinos_ENABLE_Zoltan=OFF -DTrilinos_VERBOSE_CONFIGURE=OFF -DTPL_ENABLE_MPI=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_VERBOSE_MAKEFILE=OFF -DCMAKE_BUILD_TYPE=RELEASE -DBLAS_LIBRARY_NAMES:STRING=libfblas.a -DBLAS_LIBRARY_DIRS:STRING=~/src/dealii/src/petsc-3.13.1/x86_64/externalpackages/git.fblaslapack -DLAPACK_LIBRARY_NAMES:STRING=libflapack.a -DLAPACK_LIBRARY_DIRS:STRING=~/src/dealii/src/petsc-3.13.1/x86_64/externalpackages/git.fblaslapack -DCMAKE_INSTALL_PREFIX=~/src/dealii/install/trilinos ..
   make install

   cd ~/src/dealii/src      
   tar xf zlib-1.2.12.tar.gz
   cd zlib-1.2.12/
   ./configure --prefix=~/src/dealii/install/zlib
   make install

   export PATH=$PATH:~/src/dealii/src/petsc-3.13.1/x86_64/bin:~/src/dealii/install/p4est/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/src/dealii/src/petsc-3.13.1/x86_64/lib:~/src/dealii/src/petsc-3.13.1/x86_64/externalpackages/git.fblaslapack:~/src/dealii/install/p4est/lib:~/src/dealii/install/zlib/lib

   cd ~/src/dealii/src
   tar xf metis-5.1.0.tar.gz
   cd metis-5.1.0/
   make config prefix=~/src/dealii/install/metis
   make install

   export PATH=$PATH:~/src/dealii/src/petsc-3.13.1/x86_64/bin:~/src/dealii/install/p4est/bin:~/src/dealii/install/metis/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/src/dealii/src/petsc-3.13.1/x86_64/lib:~/src/dealii/src/petsc-3.13.1/x86_64/externalpackages/git.fblaslapack:~/src/dealii/install/p4est/lib:~/src/dealii/install/zlib/lib:~/src/dealii/install/metis/lib

   cd ~/src/dealii/src
   tar xf dealii-9.2.0.tar.gz
   cd dealii-9.2.0
   mkdir build
   cd build
   cmake -DDEAL_II_WITH_MPI=ON -DMETIS_DIR=~/src/dealii/install/metis -DP4EST_DIR=~/src/dealii/install/p4est -DDEAL_II_WITH_P4EST=ON -DZLIB_DIR=~/src/dealii/install/zlib -DPETSC_DIR=~/src/dealii/src/petsc-3.13.1 -DPETSC_ARCH=x86_64 -DTRILINOS=~/src/dealii/install/trilinos -DCMAKE_INSTALL_PREFIX=~/src/dealii/install/deal ..
   make --jobs=8 install
   make test

   export PATH=$PATH:~/src/dealii/src/petsc-3.13.1/x86_64/bin:~/src/dealii/install/p4est/bin:~/src/dealii/install/metis/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/src/dealii/src/petsc-3.13.1/x86_64/lib:~/src/dealii/src/petsc-3.13.1/x86_64/externalpackages/git.fblaslapack:~/src/dealii/install/p4est/lib:~/src/dealii/install/zlib/lib:~/src/dealii/install/metis/lib:~/src/dealii/install/deal/lib

参考资料
--------

- 官方网站 https://www.dealii.org/
