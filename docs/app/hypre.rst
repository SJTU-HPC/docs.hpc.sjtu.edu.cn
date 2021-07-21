Hypre
=====

Hypre是运行在多核处理器上，借助目前性能较好的预处理矩阵（preconditioner）对于大型稀疏线性方程组使用迭代法求解的一个c语言库。 其基本目标是让用户可以借助于多核处理器的并行性能，并行存储矩阵不同范围的信息，并行地进行迭代法求解，从而达到事半功倍的效果。

可用OpenBLAS版本
----------------

+----------+-------+----------+-----------------------------------+
| 版本     | 平台  | 构建方式 | 模块名                            |
+==========+=======+==========+===================================+
| 2.18.0   | |cpu| | Spack    | 2.20.0-gcc-9.2.0-openblas-openmpi |
+----------+-------+----------+-----------------------------------+
| 2.20.0   | |cpu| | Spack    | 2.18.0-gcc-8.3.0-openblas-openmpi |
+----------+-------+----------+-----------------------------------+
| 其他版本 | |cpu| | 源码编译 | 用户家目录                        |
+----------+-------+----------+-----------------------------------+

链接Hypre库
-----------

我们推荐您链接平台上预安装的Hypre模块，您也可以自行编译Hypre库然后将程序链接到这个库上。
我们使用Hypre仓库提供的示例代码 https://github.com/hypre-space/hypre/tree/master/src/examples ，首先下载这份代码，进入示例代码目录：

.. code:: bash

   $ wget https://github.com/hypre-space/hypre/archive/refs/tags/v2.22.0.tar.gz
   $ tar xzvpf v2.22.0.tar.gz
   $ cd hypre-2.22.0/src/examples

在CPU平台上链接Hypre库
~~~~~~~~~~~~~~~~~~~~~~

在这个示例中我们使用 ``hypre/2.18.0-gcc-9.2.0-openblas-openmpi`` 模块，这个模块使用GCC 9.2.0构建，还需要载入匹配的编译器和MPI库后才能：

.. code:: bash

   $ module load hypre/2.20.0-gcc-9.2.0-openblas-openmpi gcc/9.2.0-gcc-4.8.5 openmpi/3.1.5-gcc-9.2.0 
   $ make


在Hypre库的源代码文件夹中，有一个 ``examples`` 文件夹存放了Hypre库的测试文件。

进入该文件夹：

.. code:: bash
   
   $ cd ~/hypre/src/examples
   $ make

该文件夹下将生成多个可执行测试文件。使用 ``mpirun`` 命令运行这些文件，对Hypre库进行测试：

.. code:: bash

   $ mpirun -n 2 --mca mpi_cuda_support 0 ./ex1

编译Hypre库
-----------

自行在X86平台上编译Hypre库
~~~~~~~~~~~~~~~~~~~~~~~~~~

首先申请计算资源：

.. code:: bash

   $ srun -p small -n 4 --pty /bin/bash

Hypre库的编译需要OpenMPI。请根据自己的需要选择合适的OpenMPI及GCC版本。这里我们选择加载CPU及GPU平台上全局部署的 ``openmpi/3.1.5-gcc-8.3.0``：

.. code:: bash
    
   $ module purge
   $ module load openmpi/3.1.5-gcc-8.3.0

进入Hypre的github中clone源代码

.. code:: bash

   $ git clone https://github.com/hypre-space/hypre.git

进入 ``hypre/src`` 文件夹并进行编译:

.. code:: bash

   $ cd hypre/src
   $ ./configure -prefix=/lustre/home/$YOUR_ACCOUNT/$YOUR_USERNAME/mylibs/hypre
   $ make install -j 4

编译完成之后，在家目录下会出现一个 ``mylibs`` 文件夹，Hypre库的头文件以及库文件分别在这 ``mylibs/hypre/include`` 以及 ``mylibs/hypre/lib`` 中。

.. code:: bash

   $ ls mylibs/hypre
   include  lib

参考资料
--------
- Hypre主页 https://github.com/hypre-space/hypre
- Hypre与Petsc安装文档及性能测试 https://www.jianshu.com/p/6bfadd9d6d64
