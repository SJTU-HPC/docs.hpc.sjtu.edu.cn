.. _spooles:

SPOOLES
=======

SPOOLES的全称是SParse Object Oriented Linear Equations Solver，中文名大概就是面向对象的稀疏线性等式解析器。顾名思义，就是可以用来解稀疏矩阵为参数的线性方程组的数学函数库。所谓面向对象是指的应用了面向对象的封装思想，但实际上SPOOLES是用非面向对象的C语言来写的。

最新版是2.2，支持单线程，多线程和MPI三种计算模式。

安装教程
------------------------

首先，下载并解压软件包
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   srun -p 64c512g -N 1 -n 2 --pty /bin/bash
   cd
   mkdir spooles
   cd spooles
   wget http://www.netlib.org/linalg/spooles/spooles.2.2.tgz
   tar xf spooles.2.2.tgz && rm -rf spooles.2.2.tgz

然后，导入MPI环境
~~~~~~~~~~~~~~~~~

.. code:: bash

   module load openmpi/4.1.1-gcc-8.3.1


接下来，修改 ``Make.inc`` 文件即可  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

主要修改 ``CC`` 和 ``MPI_INSTALL_DIR`` 的位置即可

.. code:: bash

   将 
   # CC = /opt/mpi/bin/mpicc
   修改为
   CC = /dssg/opt/icelake/linux-centos8-icelake/gcc-8.3.1/openmpi-4.1.1-me4z4iiamxv3l6efci5wcmjd2pk4rvye/bin/mpicc

   将 
   MPI_INSTALL_DIR = /usr/local/mpich-1.0.13
   修改为
   MPI_INSTALL_DIR = /dssg/opt/icelake/linux-centos8-icelake/gcc-8.3.1/openmpi-4.1.1-me4z4iiamxv3l6efci5wcmjd2pk4rvye

   将   
   MPI_LIB_PATH = -L$(MPI_INSTALL_DIR)/lib/solaris/ch_p4
   修改为
   MPI_LIB_PATH = -L/dssg/opt/icelake/linux-centos8-icelake/gcc-8.3.1/openmpi-4.1.1-me4z4iiamxv3l6efci5wcmjd2pk4rvye/lib

   将 
   #  MPI_LIBS = $(MPI_LIB_PATH) -lmpi -lpthread
   修改为
   MPI_LIBS = $(MPI_LIB_PATH) -lmpi -lpthread

最后，执行编译命令
~~~~~~~~~~~~~~~~~~

.. code:: bash

   make lib
   make

成功的标志
~~~~~~~~~~

生成 ``spooles.a`` 等库文件，既代表编译成功

参考资料
--------

- 官方网站 https://netlib.org/linalg/spooles/spooles.2.2.html
