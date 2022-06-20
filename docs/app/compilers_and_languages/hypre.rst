.. _Hypre:

Hypre
======

简介
------

Hypre是运行在多核处理器上，借助目前性能较好的预处理矩阵(preconditioner)对于大型稀疏线性方程组使用迭代法求解的一个c语言库。 其目标是让用户可以借助于多核处理器的并行性能，并行存储矩阵不同范围的信息，并行地进行迭代法求解，从而达到事半功倍的效果。

可用的Hypre版本
----------------------

+----------+-------+----------+-----------------------------------------+
| 版本     | 平台  | 构建方式 | 模块名                                  |
+==========+=======+==========+=========================================+
| 2.18.0   | |cpu| | Spack    | hypre/2.18.0-gcc-11.2.0-openblas-openmpi|
+----------+-------+----------+-----------------------------------------+
| 2.20.0   | |cpu| | Spack    | hypre/2.20.0-gcc-11.2.0-openblas-openmpi|
+----------+-------+----------+-----------------------------------------+
| 2.23.0   | |cpu| | Spack    | hypre/2.23.0-gcc-11.2.0-openblas-openmpi|
+----------+-------+----------+-----------------------------------------+
| 其他版本 | |cpu| | 源码编译 | 用户家目录                              |
+----------+-------+----------+-----------------------------------------+


Hypre使用说明
-----------------------------

思源一号上的Hypre
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 从Github上下载相关文件并解压，然后进入example目录：

.. code:: bash

  wget https://github.com/hypre-space/hypre/archive/refs/tags/v2.22.0.tar.gz
  tar xzvpf v2.22.0.tar.gz
  cd hypre-2.22.0/src/examples

2. 在该目录下可看到如下文件(其中以ex为前缀的文件均为示例文件)：

.. code:: bash

  CMakeLists.txt
  docs
  ex10.cxx
  ex11.c
  ex12.c
  ex12f.f
  ex13.c
  ex14.c
  ex15big.c
  ex15.c
  ex16.c
  ex17.c
  ex18.c
  ex18comp.c
  ex1.c
  ex2.c
  ex3.c
  ex4.c
  ex5big.c
  ex5.c
  ex5f.f
  ex6.c
  ex7.c
  ex8.c
  ex9.c
  ex.h
  Makefile
  Makefile_gpu
  vis
  vis.c


3. 在该目录下编写如下hypretest.slurm脚本文件(编译ex1.c文件并运行)：

.. code:: bash

  #!/bin/bash

  #SBATCH --job-name=hypretest
  #SBATCH --partition=64c512g
  #SBATCH --ntasks-per-node=2
  #SBATCH -n 2
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load openmpi/4.1.1-gcc-11.2.0
  module load hypre/2.23.0-gcc-11.2.0-openblas-openmpi

  mpicc ex1.c -lHYPRE -lm -o ex1

  mpirun -np 2 ./ex1

4. 使用如下命令提交作业：

.. code:: bash

  sbatch hypretest.slurm

5. 作业完成后可在.out文件中得到如下结果：

.. code:: bash

  <C*b,b>: 1.800000e+01

  Iters       ||r||_C     conv.rate  ||r||_C/||b||_C
  -----    ------------    ---------  ------------ 
    1    2.509980e+00    0.591608    5.916080e-01
    2    9.888265e-01    0.393958    2.330686e-01
    3    4.572262e-01    0.462393    1.077693e-01
    4    1.706474e-01    0.373223    4.022197e-02
    5    7.473022e-02    0.437922    1.761408e-02
    6    3.402624e-02    0.455321    8.020061e-03
    7    1.214929e-02    0.357057    2.863616e-03
    8    3.533113e-03    0.290808    8.327628e-04
    9    1.343893e-03    0.380371    3.167586e-04
   10    2.968745e-04    0.220906    6.997400e-05
   11    5.329671e-05    0.179526    1.256215e-05
   12    7.308483e-06    0.137128    1.722626e-06
   13    7.411552e-07    0.101410    1.746920e-07


pi2.0上的Hypre
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全一样。


2. 此步骤和上文完全一样。



3. 在该目录下编写如下hypretest.slurm脚本文件(编译ex1.c文件并运行)：

.. code:: bash

  #!/bin/bash

  #SBATCH --job-name=hypretest
  #SBATCH --partition=small
  #SBATCH --ntasks-per-node=2
  #SBATCH -n 2
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load openmpi/4.0.5-gcc-9.2.0
  module load hypre/2.20.0-gcc-9.2.0-openblas-openmpi

  mpicc ex1.c -lHYPRE -lm -o ex1

  mpirun -np 2 ./ex1

4. 使用如下命令提交作业：

.. code:: bash

  sbatch hypretest.slurm

5. 作业完成后可在.out文件中得到如下结果：

.. code:: bash

  <C*b,b>: 1.800000e+01

  Iters       ||r||_C     conv.rate  ||r||_C/||b||_C
  -----    ------------    ---------  ------------ 
    1    2.509980e+00    0.591608    5.916080e-01
    2    9.888265e-01    0.393958    2.330686e-01
    3    4.572262e-01    0.462393    1.077693e-01
    4    1.706474e-01    0.373223    4.022197e-02
    5    7.473022e-02    0.437922    1.761408e-02
    6    3.402624e-02    0.455321    8.020061e-03
    7    1.214929e-02    0.357057    2.863616e-03
    8    3.533113e-03    0.290808    8.327628e-04
    9    1.343893e-03    0.380371    3.167586e-04
   10    2.968745e-04    0.220906    6.997400e-05
   11    5.329671e-05    0.179526    1.256215e-05
   12    7.308483e-06    0.137128    1.722626e-06
   13    7.411552e-07    0.101410    1.746920e-07


编译Hypre库
-----------

自行在X86平台上编译Hypre库
~~~~~~~~~~~~~~~~~~~~~~~~~~

首先申请计算资源：

.. code:: bash

   $ srun -p small -n 4 --pty /bin/bash

Hypre库的编译需要OpenMPI。请根据自己的需要选择合适的OpenMPI及GCC版本。这里我们选择加载CPU及GPU平台上全局部署的 ``openmpi/3.1.5-gcc-8.3.0``：

.. code:: bash
    
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
---------

-  `Hypre github主页 <https://github.com/hypre-space/hypre>`__
-  `Hypre与Petsc安装文档及性能测试 <https://www.jianshu.com/p/6bfadd9d6d64>`__


